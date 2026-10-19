import copy
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name, gamma)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos and "episode" in infos["final_info"]:
            episode = infos["final_info"]["episode"]
            for i in np.where(infos["final_info"]["_episode"])[0]:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={episode['r'][i]}")
                episodic_returns += [episode["r"][i]]
        obs = next_obs

    return episodic_returns



def find_wrapper(env: gym.Env, wrapper_cls: type):
    """Return the first wrapper of type `wrapper_cls` in the env's wrapper chain, or None."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_cls):
            return env
        env = env.env
    return None


def evaluate_agent(
    agent: torch.nn.Module,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    obs_rms,
    seed: int,
    device: torch.device = torch.device("cpu"),
    gamma: float = 0.99,
):
    """Evaluate the in-memory policy with deterministic (mean) actions.

    The observation normalization stats of the training env (`obs_rms`) are copied into the eval env and frozen,
    so the agent sees observations scaled the same way as during training. Episode `i` is reset with `seed + i`,
    so every evaluation (and every run) uses the same set of start states. Returns are raw (unnormalized) rewards.

    For LunarLander also returns per-episode landing outcomes as (landed, between_flags) tuples, else None:
    `landed` means the episode terminated with the lander at rest (the env's +100 case), `between_flags` means
    it came to rest with its x position between the two helipad flags.
    """
    env = make_env(env_id, 1, False, "", gamma)()
    norm_obs = find_wrapper(env, gym.wrappers.NormalizeObservation)
    norm_obs.obs_rms = copy.deepcopy(obs_rms)
    norm_obs.update_running_mean = False

    is_lunar_lander = type(env.unwrapped).__name__ == "LunarLander"
    episodic_returns, landings = [], []
    for episode in range(eval_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            with torch.no_grad():
                action = agent.actor_mean(torch.Tensor(obs).unsqueeze(0).to(device))
            obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            done = terminated or truncated
        episodic_returns.append(float(info["episode"]["r"]))
        if is_lunar_lander:
            lander = env.unwrapped
            landed = terminated and not lander.lander.awake
            between_flags = landed and lander.helipad_x1 <= lander.lander.position.x <= lander.helipad_x2
            landings.append((landed, between_flags))
    env.close()
    return episodic_returns, (landings if is_lunar_lander else None)

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ppo_continuous_action import Agent, make_env

    model_path = hf_hub_download(
        repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )
