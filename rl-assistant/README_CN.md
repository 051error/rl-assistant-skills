# RL Assistant（强化学习助手）

一个 Claude Code skill，提供强化学习开发全流程支持——
调试训练问题、管理实验、审查代码质量、分析 RL 项目。

## 功能特性

### 🔍 训练问题调试
- **Reward 诊断框架** — 5 步统计分析法分析 reward 各组分
- **Loss 诊断** — value loss 发散、policy loss 震荡、NaN 检测
- **探索问题诊断** — 离散/连续动作空间的 entropy 坍塌检测
- **超参数诊断** — 症状导向的根因分析
- **代码与数据交叉验证** — 通过对比训练日志和源代码发现隐藏 Bug

### 📊 实验管理
- **配置模板** — dataclass + YAML，同时支持手写 PyTorch 和 SB3
- **日志标准** — RL 实验的最小必要指标集
- **多实验对比** — 统计检验、效应量、排名
- **Ablation 分析** — 识别哪些 reward 组分真正驱动了学习

### 🛡️ 代码审查
- **项目级审查** — 自动扫描 RL 项目，检测框架、算法、文件角色和潜在问题
- **10 类 Bug 检查清单** — target network、梯度流、replay buffer、Gymnasium API、初始化、advantage 计算、梯度裁剪、entropy、bootstrapping、种子管理
- **7 项性能优化** — 向量化、CPU/GPU 搬运、render 移除、replay buffer、AMP、torch.compile、pin_memory
- **Reward 函数专项审查** — 静态分析 + 设计模式检查

### 📈 分析脚本（本地运行，零 LLM token 消耗）

| 脚本 | 功能 |
|------|------|
| `scripts/scan_project.py` | 扫描 RL 项目结构，检测框架、算法、文件角色、模式、问题 |
| `scripts/analyze_reward.py` | 从训练日志生成 reward 各组分的统计画像 |
| `scripts/compare_runs.py` | 多实验统计对比 |
| `scripts/profile_bottleneck.py` | 识别训练性能瓶颈 |

## 安装

```bash
# 克隆到 Claude Code skills 目录
git clone git@github.com:051error/rl-assistant-skills.git
cp -r rl-assistant-skills/rl-assistant ~/.claude/skills/rl-assistant
```

或通过 `.skill` 文件直接在 Claude Code 中安装。

## 快速开始

### 调试 reward 不涨的问题

```
> 我的 PPO 训练 reward 已经 5 万步没涨了，这是训练日志 CSV。

Skill 会：
1. 让你运行 analyze_reward.py 分析日志
2. 读取 JSON 摘要（约 300 tokens）
3. 基于 CV、趋势、跨 seed 稳定性、相关性给出诊断
4. 如果数据层面发现问题，会进一步要求查看代码
```

### 分析一个 RL 项目

```
> 帮我审查这个 RL 项目有没有 Bug。

Skill 会：
1. 先征求你的同意运行 scan_project.py
2. 展示项目结构摘要
3. 按优先级读取关键文件
4. 与 RL Bug 清单交叉比对
5. 输出审查报告 —— 但在你明确同意前绝不会修改代码
```

### 对比两个实验结果

```
> 我的新 reward 函数真的比旧的好吗？

Skill 会：
1. 让你运行 compare_runs.py 对比两个实验目录
2. 检查差异是否超过 2σ
3. 报告统计显著性和效应量
4. 如果数据不足以得出可靠结论，会明确告知
```

## 设计原则

1. **没有统计量，不给确定性结论** — 绝不用单点数据做诊断
2. **计算卸载到脚本，不靠 LLM 硬算** — 脚本本地计算，LLM 只读摘要
3. **区分信号和噪声** — 默认假设 < 2σ 的变化是噪声
4. **扫描需获许可** — 项目扫描前必须征得用户同意
5. **修改需获许可** — 审查后修改代码前必须征得用户同意

## 环境要求

- Python 3.8+
- numpy, scipy（分析脚本需要）
- tensorboard（可选，用于解析 TensorBoard 日志）

## 项目结构

```
rl-assistant/
├── SKILL.md                    # 主入口 — 场景分流、核心原则
├── references/
│   ├── debug.md                # 训练问题诊断 + 交叉验证
│   ├── experiment.md           # 实验管理 + 日志标准
│   └── code-review.md          # 项目级和逐文件审查 + Bug 清单
└── scripts/
    ├── scan_project.py         # RL 项目扫描器
    ├── analyze_reward.py       # Reward 统计画像生成器
    ├── compare_runs.py         # 多实验对比工具
    └── profile_bottleneck.py   # 训练瓶颈分析工具
```
