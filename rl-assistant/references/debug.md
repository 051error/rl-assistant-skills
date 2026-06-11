# RL Training Issue Diagnosis

## Pre-flight Information Collection

Before starting diagnosis, collect the following by priority. **Refuse to give definitive conclusions when information is insufficient.**

### Minimum Information Requirements

| Information | Minimum Standard | How to Obtain |
|-------------|-----------------|---------------|
| Training curves (reward + each loss) | ≥ 3 seeds, ≥ 10 episodes/seed | `scripts/analyze_reward.py` |
| Algorithm type | DQN / PPO / SAC / TD3 / ... | User provides |
| Environment type | discrete/continuous action, obs dim | User provides |
| Reward function code | Full code or pseudocode | User provides |

### Ideal Information

- ≥ 5 seeds, ≥ 50 episodes/seed (post-convergence)
- Time-series data for each reward component logged independently
- Hyperparameter configuration
- Network architecture

If information doesn't meet minimum requirements:
1. Clearly state "Missing X, cannot give definitive diagnosis"
2. Only give "hypothetical analysis" with assumptions labeled
3. Suggest user supplement data or run relevant scripts

---

## Diagnosis Decision Tree

```
User reports issue
    │
    ├─ Reward-related (not increasing / oscillating / dropping)
    │   ├─ First run scripts/analyze_reward.py
    │   ├─ Check CV → CV > 1.0 means signal is unreliable
    │   ├─ Check cross-seed consistency → inconsistent means training itself is unstable
    │   ├─ Check each component → find the laggard term
    │   └─ See "Reward Diagnosis Framework" below
    │
    ├─ Loss-related (diverging / NaN / not decreasing)
    │   ├─ Value Loss diverging → lr too high / reward scale too large / target update issues
    │   ├─ Policy Loss diverging → clip range issues (PPO) / entropy tuning
    │   ├─ Loss = NaN → gradient explosion / log(0) / sqrt(negative) / reward unbounded
    │   └─ Loss not moving → lr too small / vanishing gradients / network capacity insufficient
    │
    ├─ Exploration-related (premature policy convergence / single action)
    │   ├─ Discrete action → ε decay too fast / entropy coef too small
    │   ├─ Continuous action → σ collapse / entropy coef too small
    │   └─ See "Exploration Issues Diagnosis" below
    │
    └─ Performance-related (slow training / OOM / high CPU)
        └─ Guide user to run scripts/profile_bottleneck.py
           then load references/code-review.md
```

---

## Reward Diagnosis Framework

### Step 1: Per-Component Statistical Profile (must run script)

Have the user run:
```bash
python scripts/analyze_reward.py --log <path> [--components reward_goal reward_bonus reward_penalty]
```

The script outputs a statistical profile for each component. **When reading the JSON summary, judge by these rules:**

#### Signal Quality Criteria

```
CV = std / |mean|

CV < 0.3   → clear signal, single observation trustworthy
0.3 ≤ CV < 1.0 → moderate noise, need rolling average
CV ≥ 1.0   → high noise, never cite single observation values
```

#### Cross-Seed Stability Criteria

```
cross_seed_std < episode_std → consistent across seeds, algorithm is stable
cross_seed_std ≈ episode_std → large seed-to-seed variance, randomness dominates
cross_seed_std > episode_std → training is unstable, seed impact > algorithm impact
```

### Step 2: Per-Component Correlation Analysis

For each reward component, check its correlation with total_reward:

```
|r| > 0.5   → component actually drives optimization
0.2 < |r| < 0.5 → weak correlation, likely background noise
|r| < 0.2   → component may be "decorative," has no real impact on policy
```

If a reward term has large weight but small correlation → **reward shaping is ineffective**:
the shaping term doesn't guide policy learning, it only adds variance.

### Step 3: Reward Range Estimation Verification

When the user provides reward function code, estimate per-component ranges and cross-validate against actual observations:

```
Theoretical estimation:
  reward = α·term_A + β·term_B + γ·term_C
  Compute per-term min / max / typical

Observed comparison:
  min_obs vs min_theo: large gap → possibly missing a random factor
  max_obs vs max_theo: max_obs ≪ max_theo → theoretical extreme nearly impossible, use typical_range
  mean_obs vs typical: large gap → assumed random variable distribution is wrong
```

**Common traps**:
- `term_A = distance_to_goal`: theoretical upper bound is `sqrt(world_width² + world_height²)`, but the agent rarely starts from the farthest corner → actual upper bound much smaller than theoretical
- `term_B = 1.0 if success else 0.0`: sparse reward with naturally extremely high CV — don't look at mean, look at success rate
- Multiplicative interactions: `term_A * term_B` amplifies variance from either side

### Step 4: Magnitude Consistency Check

```
Find max(|α·term_A|, |β·term_B|, |γ·term_C|)
Compute ratio of each term's typical value to this maximum

If a term < 5% of max → drowned out, cannot influence gradient direction
If a term > 10× others → others are drowned out
Ideal: all terms at the same order of magnitude, or scaled by coefficients to reach parity
```

### Step 5: Formulate Diagnosis Based on Above

**Common problems and suggestions**:

| Symptom | Possible Cause | Suggestion |
|---------|---------------|------------|
| Total reward CV > 1.0 | High environment randomness / sparse reward | Increase seeds, use rolling average, consider reward normalization |
| One component CV extremely high, others normal | Contains rare events (e.g., goal reaching) | Analyze separately, look at success rate instead of mean |
| Component correlation with total reward ≈ 0 | Component doesn't drive learning | Consider removing or redesigning |
| Reward dropping late in training | Overfitting to reward / reward hacking | Check if policy found a reward loophole |
| Cross-seed std continuously increasing | Training instability | Lower lr, increase batch size, check for random seed contamination across runs |

---

## Exploration Issues Diagnosis

### Discrete Action Space

| Symptom | Diagnostic Method | Suggestion |
|---------|------------------|------------|
| Action distribution entropy rapidly → 0 | Log per-step entropy | Increase entropy coefficient / slow ε decay |
| Some action never selected | Log per-action frequency | Check if Q initialization is symmetric |
| Policy converges too early | entropy < 0.1 * log(n_actions) | Add entropy bonus / raise ε start value |

### Continuous Action Space

| Symptom | Diagnostic Method | Suggestion |
|---------|------------------|------------|
| σ → 0 (deterministic output) | Log log_std trend | Set σ lower bound (min_log_std), increase entropy coef |
| Actions always at boundaries | Log per-dim mean | Redesign action space / use tanh + rescale |
| Action distribution collapses to single point | Log per-dim variance | Increase entropy coef / check if overtrained |

---

## Hyperparameter Issues Diagnosis

| Symptom | Most Likely Cause | How to Check |
|---------|------------------|-------------|
| Reward rises rapidly then crashes | lr too high | Halve lr and retry |
| Loss steadily decreases but reward unchanged | reward scale issue | reward normalization |
| Q-values keep increasing without converging | γ too large + no terminal | Check for infinite horizon, lower γ |
| Performance spike after target update | τ too large (hard update) | Switch to polyak averaging (τ=0.005) |
| High inter-batch loss variance | batch size too small | Increase batch size |

---

## Code-and-Data Cross-Validation

**Motivation**: Looking only at training logs/charts without code misses many hidden problems.
When the user has both training data and project code, perform cross-validation.

### Cross-Validation Workflow

```
1. First run scripts/analyze_reward.py to get data profile
2. Run scripts/scan_project.py to get project structure (requires user permission)
3. For each type of issue, cross-check code and data for contradictions
```

### Cross-Validation Checklist

#### Data vs Reward Function Consistency

| Data Pattern | What to Look for in Code | Example Hidden Problem |
|-------------|------------------------|----------------------|
| Reward components have wildly different CVs | Are there random factors in the reward function? | `reward += np.random.random()` adds noise but isn't logged |
| Observed range of a component << theoretical | Does the reward function depend on conditions never triggered? | `if distance < 0.1: reward += 100` but agent never gets that close |
| Component mean ≈ constant | Is the reward term hardcoded as a constant? | `reward = 0.1` instead of `reward = 0.1 * progress` |
| Total reward decreasing per episode | Is there a mechanism that gets harder over time? | `curriculum_penalty += 0.01 * episode` |
| Reward spikes mid-episode | Are there phase/step-dependent conditional branches? | `if step > 100: reward += bonus` |

#### Data vs Training Loop Consistency

| Data Pattern | What to Look for in Code | Example Hidden Problem |
|-------------|------------------------|----------------------|
| Loss not decreasing but reward rising | Is the logged loss the true loss or scaled? | `loss = policy_loss + 0.001 * value_loss` — value loss dominates but is ignored |
| Entropy slowly rising instead of falling | Is entropy coefficient negative? | `loss = policy_loss - 0.01 * entropy` but intent was to add entropy bonus |
| Q-values jump at some point | Does environment reset change reward scale? | `env.reset()` zeros out reward normalization counters |
| Performance curves across seeds diverge at a checkpoint | Are there non-deterministic ops (e.g., dropout) not in `eval()`? | Dropout in train mode introduces seed-dependent noise |

#### Data vs Network Architecture Consistency

| Data Pattern | What to Look for in Code | Example Hidden Problem |
|-------------|------------------------|----------------------|
| Policy loss oscillates at high frequency | Is the network too small? | 2-layer 64-dim network can't fit complex reward landscape |
| Value loss >> policy loss | Shared backbone with mismatched learning rates per head? | shared encoder → gradient conflict |
| Initial Q-values extreme (near 0 or very large) | Is the final layer properly initialized? | Bias initialized to large positive → initial Q too high, insufficient exploration in early replay buffer steps |

#### Data vs Hyperparameter Consistency

| Data Pattern | What to Look for in Code | Example Hidden Problem |
|-------------|------------------------|----------------------|
| Periodic dips in training curve | Are there periodic hyperparameter changes? | lr scheduler decays every N steps but decays too fast |
| Training flat early on | Are warmup steps configured? | No warmup → random policy from early steps contaminates buffer |
| Performance suddenly collapses after fixed steps | Is batch normalization properly adapted for RL? | BN running stats break when on-policy data distribution shifts |

### Cross-Validation Diagnosis Template

When finding contradictions between code and data, report in this format:

```markdown
## Cross-Validation Finding

### Observation (from data)
[Anomaly observed from analyze_reward.py output]

### Hypotheses (pointing to code)
[N possible causes in code that could produce this observation]

### Verification
- Read [file:line] confirms: the code actually [description]
- Data prediction: if hypothesis correct, should see [prediction]
- Actual data: shows [result], [consistent / inconsistent] with hypothesis

### Conclusion
[Confirmed or ruled out the hypothesis]
```

---

## Final Output Format

After completing diagnosis, output in this structure:

```markdown
## Diagnosis Conclusion

### Data Sufficiency Assessment
- Seeds: [sufficient / insufficient], Episodes: [sufficient / insufficient]
- Signal quality: [clear / moderate noise / high noise]

### Key Findings
1. **[Issue name]**: description + evidence (cite specific statistics)
2. ...

### Recommended Fixes (by priority)
1. **[High priority]** specific suggestion + expected effect
2. **[Medium priority]** ...
3. **[Low priority / further exploration]** ...

### Uncertainty Notes
- Which conclusions are hypotheses based on incomplete data
- What additional data is suggested to verify
```
