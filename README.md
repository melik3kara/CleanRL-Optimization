# SAGE-RL-OPT

Experiments on optimizing PPO reinforcement learning processes, built on CleanRL (LunarLanderContinuous-v3, gymnasium 1.2.3), tracked with Weights & Biases.

## Docker

```bash
cp .env.example .env    # then paste your key from https://wandb.ai/authorize into WANDB_API_KEY
docker compose build
docker compose run --rm rl bash    # interactive shell
```

`.env` is git- and docker-ignored; compose passes it into the container as environment variables, so no `wandb login` is needed.
Check the connection with:

```bash
docker compose run --rm rl python -c "import wandb; print(wandb.Api().viewer.username)"
```

## Run

```bash
docker compose run --rm rl python cleanrl/ppo_continuous_action.py --track
```

| Flag | What it does | Saved to |
|---|---|---|
| `--track` | log metrics to W&B (project `lunar-lander-ppo-sage`) | `wandb/` + wandb.ai |
| `--capture-video` | record agent episodes (uploaded to W&B when tracking) | `videos/<run_name>/` |
| `--save-model` | save weights + run 10 eval episodes | `runs/<run_name>/*.cleanrl_model` |
| (always) | TensorBoard logs | `runs/<run_name>/` |

Periodic evaluation (`--eval-freq`, default every 25k steps, `--eval-episodes` per eval) runs the deterministic policy
on fixed seeds and logs `eval/episodic_return_mean`, `eval/episodic_return_std` and, for LunarLander,
`eval/landed_rate` (episodes ending at rest), `eval/between_flags_rate` (at rest between the flags) and
`eval/between_flags_share_of_landings`. `--run-dir` overrides the TensorBoard/model directory.

## Baseline

```bash
docker compose run --rm rl python cleanrl/ppo_continuous_action.py --track --no-cuda \
  --exp-name ppo-baseline --seed 1 --run-dir runs/ppo/baseline/seed1 \
  --num-envs 16 --num-steps 1024 --num-minibatches 256 --update-epochs 4 \
  --gamma 0.999 --gae-lambda 0.98 --ent-coef 0.01
```

Other useful flags: `--seed`, `--total-timesteps`, `--no-cuda`. See all with `--help`.
