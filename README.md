# RL Assistant

A Claude Code skill for full-lifecycle reinforcement learning development —
debug training issues, manage experiments, review code quality, and analyze RL projects.
Chinese README.md([rl-assistant/README_CN.md](https://github.com/051error/rl-assistant-skills/blob/main/rl-assistant/README_CN.md))

## Features

### 🔍 Training Debugging
- **Reward diagnosis framework** — 5-step statistical analysis of reward components
- **Loss diagnosis** — value loss divergence, policy loss oscillation, NaN detection
- **Exploration diagnosis** — entropy collapse detection for discrete and continuous action spaces
- **Hyperparameter diagnosis** — symptom-based root cause analysis
- **Code-and-data cross-validation** — find hidden bugs by cross-referencing training logs with source code

### 📊 Experiment Management
- **Configuration templates** — dataclass + YAML config for both custom PyTorch and SB3
- **Logging standards** — minimum required metrics for RL experiments
- **Multi-experiment comparison** — statistical tests, effect sizes, rankings
- **Ablation analysis** — identify which reward components actually drive learning

### 🛡️ Code Review
- **Project-level review** — automatic scanning of RL projects to detect framework, algorithm, file roles, and potential issues
- **10-class bug checklist** — target network, gradient flow, replay buffer, Gymnasium API, initialization, advantage computation, gradient clipping, entropy, bootstrapping, seed management
- **7-point performance optimization** — vectorization, CPU/GPU transfers, render removal, replay buffer, AMP, torch.compile, pin_memory
- **Reward function specialist review** — static analysis + design pattern checks

### 📈 Analysis Scripts (run locally, zero LLM token cost)

| Script | Purpose |
|--------|---------|
| `scripts/scan_project.py` | Scan RL project structure — detect framework, algorithm, file roles, patterns, issues |
| `scripts/analyze_reward.py` | Generate statistical profile of reward components from training logs |
| `scripts/compare_runs.py` | Compare multiple experiment runs with statistical tests |
| `scripts/profile_bottleneck.py` | Identify training performance bottlenecks |

## Installation

```bash
# Clone to your Claude Code skills directory
git clone git@github.com:051error/051repo.git
cp -r 051repo/rl-assistant ~/.claude/skills/rl-assistant
```

Or install via the `.skill` file directly in Claude Code.

## Quick Start

### Debug why reward isn't increasing

```
> My PPO training reward has been flat for 50k steps. Here's my training log CSV.

The skill will:
1. Have you run `analyze_reward.py` on your log
2. Read the JSON summary (~300 tokens)
3. Diagnose based on CV, trends, cross-seed stability, and correlations
4. Ask for code if the data suggests code-level issues
```

### Analyze an RL project

```
> Help me review this RL project for bugs.

The skill will:
1. Ask permission to run `scan_project.py`
2. Present a project structure summary
3. Read key files in priority order
4. Cross-reference against the RL bug checklist
5. Output a review report — but never modify code without your explicit consent
```

### Compare two experiment runs

```
> Is my new reward function actually better than the old one?

The skill will:
1. Have you run `compare_runs.py` on both experiment directories
2. Check if the difference exceeds 2σ
3. Report statistical significance and effect sizes
4. Warn if data is insufficient for a reliable conclusion
```

## Design Principles

1. **No statistics, no definitive conclusions** — never diagnose from single data points
2. **Offload computation to scripts, not the LLM** — scripts compute locally, LLM reads summaries
3. **Signal vs. noise** — default assumption: changes < 2σ are noise
4. **Scan with permission** — project scanning requires explicit user consent
5. **Modify with permission** — code changes require explicit user consent after review

## Requirements

- Python 3.8+
- numpy, scipy (for analysis scripts)
- tensorboard (optional, for TensorBoard log parsing)

## Project Structure

```
rl-assistant/
├── SKILL.md                    # Main entry point — scene routing, core principles
├── references/
│   ├── debug.md                # Training issue diagnosis + cross-validation
│   ├── experiment.md           # Experiment management + logging standards
│   └── code-review.md          # Project & per-file review + bug checklist
└── scripts/
    ├── scan_project.py         # RL project scanner
    ├── analyze_reward.py       # Reward statistical profile generator
    ├── compare_runs.py         # Multi-experiment comparison
    └── profile_bottleneck.py   # Training bottleneck profiler
```
