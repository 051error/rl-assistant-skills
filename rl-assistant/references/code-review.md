# RL Code Review

## Project-Level Review (using scan_project.py)

Use this workflow when the user wants to analyze an entire RL project rather than a single file.

### Step 1: Obtain Permission and Scan

Run the scan script (must obtain user permission first):
```bash
python scripts/scan_project.py --dir <project_directory>
```

### Step 2: Read the Scan Results

The scan outputs compact JSON. Understand the structure as follows:

```json
{
  "framework": "stable_baselines3",      // RL framework used
  "algorithm": "PPO",                    // Algorithm type
  "file_roles": {                        // File count per role
    "training_entry": 1,
    "network_definition": 2,
    "environment": 1,
    "config": 1
  },
  "rl_patterns_found": [                 // Detected RL patterns
    "target_network",
    "advantage_gae",
    "gradient_clipping"
  ],
  "potential_issues": [                  // ⚠️ Auto-detected project-level concerns
    {"severity": "warning", "message": "PPO + target network — PPO doesn't need a target network"}
  ],
  "suggested_review_order": [            // 📋 Suggested review order
    {"priority": 1, "role": "training_entry", "path": "train.py",
     "reason": "Training loop — review first"}
  ]
}
```

### Step 3: Review Files by Priority

**Don't read all files at once.** Follow the `suggested_review_order` priority:
1. Read the training entry first to understand the overall training flow
2. Then the network definitions
3. Then the reward function (most important)
4. Then the environment
5. Finally the config

For each file read, cross-reference against the "RL Common Bug Checklist" below.

After reading each file, decide:
- Have enough issues been found already?
- Could the remaining files change the diagnosis?
- Based on the user's question, should we continue to the next file?

### Step 4: Validate Project-Level Issues

The scanner's `potential_issues` are based on **static regex matches** and may be false positives.
After reading actual code, verify each `potential_issue`:

| Auto-Detected Issue | Verification Method |
|--------------------|--------------------|
| "PPO + target network" | Read training code to confirm if target network is really used |
| "No seed management" | Search for `seed` keyword; maybe managed centrally in another file |
| "No gradient clipping" | Read training loop; confirm if gradient clipping is needed |
| "Off-policy but no target network" | Read training code and target network definition |

### Step 5: Output Project-Level Report

```markdown
## Project Review Report

### Project Overview
- Framework: [SB3 / CleanRL / Custom PyTorch]
- Algorithm: [PPO / SAC / ...]
- RL-related files: [N]

### 🔴 Critical Issues
- [file:line] issue description + impact + fix suggestion

### 🟡 Suggested Improvements
- [file:line] issue description + improvement plan

### 🟢 Performance Optimizations
- [file:line] bottleneck + optimization suggestion

### Reward Function Review
- Range estimation / design issues

### Missing Modules
- Project lacks [seed management / config management / eval script / ...] — suggest adding

### Code Structure Suggestions
- Architecture-level improvement suggestions

---
⚠️ **Which of these issues would you like me to fix? Please let me know the specific items.**
```

---

## Per-File Review (when user pastes code)

### Pre-flight Information Collection

When reviewing RL code, confirm the following:
- Training script (training loop)
- Network definition code
- Reward function code (important!)
- Hyperparameter configuration

If the code base is large (> 500 lines), locate the key modules first and review module by module rather than reading everything at once.

---

## RL Common Bug Checklist

### 🔴 Critical (causes training to fail completely or silently)

#### 1. Target Network Updates

```python
# ❌ Wrong: forgot to update target network
def train_step(self, batch):
    loss = self.compute_loss(batch)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    # Missing self.update_target() !!

# ❌ Wrong: hard update every step (target and online always in sync)
def train_step(self, batch):
    self.update_network(batch)
    self.target.load_state_dict(self.online.state_dict())  # every single step

# ✅ Correct: Polyak averaging (SAC/TD3)
def train_step(self, batch):
    self.update_network(batch)
    for target_param, online_param in zip(self.target.parameters(),
                                           self.online.parameters()):
        target_param.data.copy_(self.tau * online_param.data
                                + (1 - self.tau) * target_param.data)

# ✅ Correct: Periodic hard update (DQN)
def train_step(self, batch):
    self.update_network(batch)
    if self.step_count % self.target_update_freq == 0:
        self.target.load_state_dict(self.online.state_dict())
```

**Checklist**:
- Is `tau` set correctly? (SAC/TD3 typically `tau=0.005`, not `tau=0.5` or `tau=1.0`)
- PPO doesn't need a target network! If PPO code has one, it's redundant
- Is the target network running in `eval()` mode?

#### 2. Broken Gradient Flow

```python
# ❌ Wrong: .item() / .detach() misuse
q_values = self.critic(states, actions)
loss = F.mse_loss(q_values, targets.detach())  # targets already detached, OK
# But if target computation depends on online network output...
next_q = self.target_critic(next_states, next_actions).detach()  # detach prevents gradient to target critic, correct
# But the following is wrong:
next_actions = self.actor(next_states)  # no .detach() — actor gradients flow back through target!
target = reward + self.gamma * self.target_critic(next_states, next_actions)

# ❌ Wrong: converting tensor to float breaks gradient
loss_value = loss.item()  # returns Python float, no grad
# If loss_value is used in subsequent computation, gradient flow is broken

# ✅ Correct: use .item() for logging only, never for computation
self.logger.log("loss", loss.item())  # OK, just logging
```

#### 3. Replay Buffer Implementation

```python
# ❌ Wrong 1: Device mismatch
class ReplayBuffer:
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = self.buffer[indices]  # self.buffer is CPU numpy array
        return batch  # returns numpy; training loop must manually .to(device)

# ❌ Wrong 2: Tensor replay buffer in-place update
self.buffer[self.ptr] = transition_tensor  # if buffer is CPU tensor but transition is on GPU...

# ❌ Wrong 3: SAC needs (s, a, r, s', done) — forgetting done means can't mask terminal states correctly

# ✅ Recommended: pre-allocated numpy arrays, convert to tensor on sampling
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.capacity = capacity

    def sample(self, batch_size, device):
        indices = np.random.choice(self.size, batch_size)
        return (
            torch.FloatTensor(self.states[indices]).to(device),
            torch.FloatTensor(self.actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device),
        )
```

#### 4. Gymnasium API: terminated vs truncated

```python
# ❌ Wrong: confusing terminated and truncated
next_obs, reward, done, info = env.step(action)  # old API
# Gymnasium >= 0.26:
next_obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# ❌ Wrong: treating truncated as terminal
# truncated = timeout, episode cut short but not truly terminal
# When bootstrapping, truncated states still have value (not 0)
target = reward + self.gamma * (1 - float(terminated)) * next_value
# Note: when truncated, (1 - float(terminated)) = 1, so bootstrap continues
# This is correct!
```

#### 5. Fully Connected Layer Initialization

```python
# ❌ Default init can cause gradient vanishing/explosion
# PyTorch default Kaiming init is good for ReLU but may be poor for tanh

# ✅ Recommended: explicit init for the final layer (so initial policy is near zero)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

# ✅ For continuous control Actor's final layer (outputting mean):
nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)  # small gain → initial actions near zero
nn.init.constant_(self.mean_layer.bias, 0.0)
```

### 🟡 Medium (training may run but be unstable)

#### 6. Advantage Computation (PPO / A2C)

```python
# ❌ Wrong: forgot to normalize advantage
advantages = (returns - values).detach()
# Should normalize:
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

#### 7. Gradient Clipping

```python
# ❌ Wrong: one-size-fits-all gradient clipping
# PPO typically doesn't need clip_grad_norm_ (already has clip_range)
# DQN/SAC typically do need it

# ✅ PPO relies on clip_range constraint; generally no extra grad clip needed
# ✅ SAC recommendation:
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
```

#### 8. Entropy Coefficient

```python
# ❌ Common mistake: entropy coefficient wrong order of magnitude
# For SAC: ent_coef is typically auto-tuned (alpha)
# For PPO discrete: 0.01 is a common starting point
# For PPO continuous: may need 0.0 or extremely small value (continuous action entropy can tend to -inf)
```

#### 9. Bootstrapping Errors

```python
# ❌ DQN: using target network to select actions
next_q = self.target_net(next_states).max(dim=1)[0]  # DDQN separates selection and evaluation
# DDQN approach:
next_actions = self.online_net(next_states).argmax(dim=1)
next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))

# ❌ SAC: forgetting the entropy term in the target critic
# SAC target formula includes entropy bonus:
# y = r + γ * (min_q_next - α * log_prob_next)
# Missing -α * log_prob_next is a common omission
```

### 🟢 Minor (doesn't affect functionality but affects maintainability)

#### 10. Random Seed Management

```python
# ❌ Common issue: set torch seed but forget numpy/env seeds
# ✅ Complete seed setup:
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # slows things down, generally not needed
```

---

## Performance Optimization Checklist

### 🔴 Most Common Performance Killers

#### 1. Environment Stepping Is the Bottleneck (GPU Idle)

```python
# ❌ Single environment + synchronous interaction → GPU utilization < 20%
obs = env.reset()
for _ in range(total_steps):
    action = model(obs)      # GPU (2ms)
    obs, reward, done, info = env.step(action)  # CPU (10ms) ← bottleneck

# ✅ Vectorized environments
from gymnasium.vector import AsyncVectorEnv
envs = AsyncVectorEnv([lambda: make_env() for _ in range(n_envs)])

# ✅ SB3: increase n_envs
model = PPO("MlpPolicy", env, n_envs=8, ...)
```

#### 2. Frequent CPU ↔ GPU Transfers

```python
# ❌ Transferring one-by-one in a loop
for transition in trajectory:
    tensor = torch.tensor(transition).to(device)
    buffer.add(tensor)

# ✅ Batch them together
batch = np.array(trajectory_buffer)
tensor_batch = torch.from_numpy(batch).to(device)
```

#### 3. Calling env.render() in Training Loop

```python
# ❌ render() can consume 50%+ of training time
# ✅ Only render during eval; completely disable during training
```

#### 4. Replay Buffer Using list + Append One-by-One

```python
# ❌ Python list + looped sampling
self.buffer = []
self.buffer.append(transition)  # append each transition individually
indices = np.random.choice(len(self.buffer), batch_size)
batch = [self.buffer[i] for i in indices]  # fetch one at a time

# ✅ Pre-allocated numpy array (see replay buffer implementation above)
```

### 🟡 Further Optimizations

#### 5. Mixed Precision Training (AMP)

```python
# ✅ Suitable for large-batch network updates
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = compute_loss(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 6. torch.compile (PyTorch ≥ 2.0)

```python
# ✅ Compile networks, typically 5-20% speedup
self.actor = torch.compile(self.actor)
self.critic = torch.compile(self.critic)
```

#### 7. pin_memory + non_blocking

```python
# ✅ For DataLoader / replay buffer sampling
batch = tuple(b.pin_memory().to(device, non_blocking=True) for b in batch)
```

### Performance Analysis Flow

When facing a "training is too slow" complaint:

1. **First have the user run the performance analysis script**:
   ```bash
   python scripts/profile_bottleneck.py --script train.py --steps 1000
   ```

2. **Read the bottleneck analysis output**; locate the biggest bottleneck by time consumption

3. **Give targeted suggestions based on the checklist above**; don't generically say "optimize it"

---

## Reward Function Code Review (Specialized)

### Static Checklist

- [ ] Are all reward component dtypes consistent? (int + float mixing can cause truncation)
- [ ] Does every branch path have an explicit reward assignment? (missing else → undefined behavior)
- [ ] Does reward computation depend on `env` internal state? (may persist stale state across env resets)
- [ ] In vectorized environments, are per-env rewards correctly aggregated?
- [ ] Are reward-related values mixed into the `info` dict? (info is for diagnostics, should not affect reward)
- [ ] Are `terminated` and `truncated` rewards handled differently?

### Common Design Issues

| Issue | Example | Risk |
|-------|---------|------|
| Unbounded positive reward | `reward += speed` no upper bound | Q-value explosion |
| Unbounded negative penalty | `reward -= distance * 1000` | Policy learns only to avoid penalty |
| Discrete reward insufficient precision | `reward = int(distance)` truncation | Loses gradient information |
| Sign inconsistency | `reward_goal` positive, `reward_penalty` also positive | Terms fight each other |
| Sparse reward + no shaping | `reward = 1 if success else 0` | Extremely hard to learn |

### ⚠️ Code Modification Requires Permission

**Never modify code directly after review.** Strictly follow this process:

1. Output the review report (using the template below)
2. At the end of the report, explicitly ask: "Which of these issues would you like me to fix?"
3. Wait for the user to confirm which specific issues to fix
4. Only execute edits after explicit user specification

Reasons:
- LLMs can misdiagnose (e.g., flagging correct implementation patterns as bugs)
- Users may have their own reasons for keeping certain "issues"
- Performance optimization suggestions may not suit all use cases

### Review Output Format

```markdown
## Code Review Results

### 🔴 Critical Issues (must fix)
- [file:line] issue description + fix suggestion

### 🟡 Suggested Improvements
- [file:line] issue description + improvement plan

### 🟢 Performance Optimization Suggestions
- [file:line] bottleneck description + optimization direction

### Reward Function Special
- Range estimation: [theoretical range]
- Potential risks: [reward hacking / magnitude inconsistency / ...]

---
⚠️ **Which of these issues would you like me to fix? Please let me know the specific items.**
```
