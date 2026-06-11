# RL Experiment Management

## Pre-flight Information Collection

Understand the user's experiment management needs:
- What logging tool are they using? (WandB / TensorBoard / CSV / none)
- How many experiments need comparison?
- What is the purpose of comparison? (A/B test / ablation / hyperparameter search)

---

## Experiment Configuration Templates

Recommended pattern: dataclass + YAML configuration,
ensuring every experiment's config is traceable and reproducible.

### PyTorch Custom Version

```python
from dataclasses import dataclass, field
from typing import Optional, List
import yaml

@dataclass
class RLConfig:
    # Algorithm
    algorithm: str = "ppo"              # ppo, dqn, sac, td3
    # Environment
    env_id: str = "CartPole-v1"
    env_kwargs: dict = field(default_factory=dict)
    # Training
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    seed: int = 42
    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    # Optimizer
    lr: float = 3e-4
    batch_size: int = 256
    # Algorithm-specific
    gamma: float = 0.99
    gae_lambda: float = 0.95        # PPO
    clip_range: float = 0.2         # PPO
    ent_coef: float = 0.01          # PPO / SAC
    tau: float = 0.005              # SAC / TD3
    # Logging
    log_dir: str = "./logs"
    log_interval: int = 1000        # steps
    eval_interval: int = 10000      # steps
    eval_episodes: int = 10
    # WandB
    use_wandb: bool = False
    wandb_project: str = "rl-project"
    wandb_group: str = "default"
    # Reward
    reward_components: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "RLConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
```

### Stable-Baselines3 Version

```python
# SB3 users just manage a hyperparameter dict; SB3 handles config persistence
from stable_baselines3.common.utils import set_random_seed

ppo_params = {
    "policy": "MlpPolicy",
    "env": "CartPole-v1",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "verbose": 1,
    "tensorboard_log": "./logs/",
    "seed": 42,
}
```

### Configuration Management Best Practices

1. **Save a config copy per experiment** to the log directory for reproducibility
2. **git commit hash + config → fully reproducible**
3. **seed must be configurable**, and always run at least 3 different seeds (ideally ≥ 5)
4. **Each reward component must be logged independently** — critical for debugging

---

## Logging Standards

### Minimum Required Metrics

Regardless of logging tool, the following are the "vital signs" of an RL experiment:

| Metric | Frequency | Description |
|--------|----------|-------------|
| `episode_reward` | per episode | Total reward, most important metric |
| `episode_length` | per episode | Episode length changes may indicate exploration/exploitation shifts |
| `reward_<component>` | per episode | **Log each reward component independently** (essential for debugging!) |
| `policy_loss` | every N steps | Policy / actor loss |
| `value_loss` | every N steps | Value / critic loss |
| `entropy` | every N steps | Policy entropy, measures exploration |
| `approx_kl` | every N steps | (PPO) KL divergence; too large means constraint failure |
| `q_values` (mean/max) | every N steps | (DQN/SAC) Q-values; spiking signals non-convergence |
| `fps` | every log_interval | Training speed; degrading signals performance issues |

### WandB Best Configuration

```python
import wandb

wandb.init(
    project="rl-project",
    group="ppo-cartpole",       # same experiment group
    name="lr3e-4_bs256_seed42", # descriptive name
    config=config.__dict__,     # save all hyperparameters
    save_code=True,
)
```

### CSV Log Format Requirements

If not using WandB/TensorBoard, log via CSV.
**Must satisfy the following format for `scripts/analyze_reward.py` to parse:**

```csv
episode,reward_total,reward_goal,reward_bonus,reward_penalty,policy_loss,value_loss,entropy,episode_length,fps,timestamp
1,12.3,8.1,4.2,0.0,0.034,0.128,1.45,200,823,2026-06-11T10:00:00
2,9.7,5.2,4.5,0.0,0.029,0.115,1.32,195,845,2026-06-11T10:00:02
...
```

Column naming requirements:
- Each reward component column must start with `reward_`
- First column is episode number
- Encoding: UTF-8

---

## Experiment Result Analysis

### Single Experiment Analysis

After obtaining output from `scripts/analyze_reward.py`, focus on:

1. **Learning speed**: steps/episodes needed for reward to rise from initial to converged
2. **Asymptotic performance**: converged reward level (mean ± std of last N episodes)
3. **Stability**: post-convergence fluctuation (CV of last N episodes)
4. **Efficiency**: whether fps is stable, any gradual decline (memory leak signal)

### Multi-Experiment Comparison

Run the comparison script:
```bash
python scripts/compare_runs.py --runs logs/exp1 logs/exp2 logs/exp3 --metric reward_total
```

Output includes:
- Per-experiment mean ± std (last N episodes)
- Pairwise comparison with statistical test results
- Learning curve comparison summary

#### Criteria for "A is better than B"

```
1. Difference > 2× std_pooled → likely real
2. Consistency across seeds > 80% → reliable
3. Difference persists late in training → asymptotic advantage
4. (Optional) paired t-test p < 0.05 → statistically significant

All 4 criteria must be satisfied before claiming "A is better than B";
at minimum criteria 1-3 must hold.
```

#### Ablation Experiment Analysis

When analyzing reward/component ablations, focus on:
- After removing a component, **direction** and **magnitude** of target metric change
- If removing a component gives unchanged performance → component contributes nothing, consider deleting to reduce variance
- If removing a component significantly reduces variance → component was a noise source

### Experiment Report Template

```markdown
## Experiment: <brief description>

### Configuration
- Algorithm: [PPO/DQN/...]
- Environment: [env name]
- Key hyperparameters: [lr, batch_size, gamma, ...]
- Seeds: [N]

### Results
- Asymptotic performance: mean ± std (last 20% episodes)
- Learning speed: steps to reach X performance
- Stability: post-convergence CV = X%

### Comparison (if applicable)
- vs baseline: +X% / -Y%
- Statistical significance: [p-value / bootstrap CI]

### Key Findings
1. ...
2. ...

### Next Steps
- ...
```
