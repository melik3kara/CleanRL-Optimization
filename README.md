# SAGE-RL-OPT

Experiments on optimizing PPO reinforcement learning processes, built on CleanRL (MountainCarContinuous-v0), tracked with Weights & Biases.

## Docker

```bash
cp .env.example .env    # set WANDB_API_KEY
docker compose build
docker compose run --rm rl bash    # interactive shell
```

## Run

```bash
docker compose run --rm rl python cleanrl/ppo_continuous_action.py --track
```

| Flag | What it does | Saved to |
|---|---|---|
| `--track` | log metrics to W&B (project `SAGE-RL-OPT`) | `wandb/` + wandb.ai |
| `--capture-video` | record agent episodes (uploaded to W&B when tracking) | `videos/<run_name>/` |
| `--save-model` | save weights + run 10 eval episodes | `runs/<run_name>/*.cleanrl_model` |
| (always) | TensorBoard logs | `runs/<run_name>/` |

Other useful flags: `--seed`, `--total-timesteps`, `--no-cuda`. See all with `--help`.
