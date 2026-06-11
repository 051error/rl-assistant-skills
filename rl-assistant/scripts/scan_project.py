#!/usr/bin/env python3
"""
RL project scanner — discovers and classifies RL-related files without LLM involvement.

Scans a project directory for Python files, detects the RL framework and algorithm,
classifies files by role in the training pipeline, finds key RL patterns,
and outputs a compact JSON summary designed for LLM consumption (~200-500 tokens).

Usage:
    python scan_project.py --dir /path/to/project
    python scan_project.py --dir . --exclude tests docs

Output:
    JSON to stdout with keys:
    - project_summary: framework, algorithm, file roles
    - files: per-file classification and key lines
    - patterns_found: key RL patterns and their locations
    - potential_issues: auto-detected project-level concerns
    - suggested_review_order: priority-ordered files to read
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ── Pattern definitions ──────────────────────────────────────────

FRAMEWORK_PATTERNS = {
    "stable_baselines3": [
        (r"from\s+stable_baselines3", "import"),
        (r"import\s+stable_baselines3", "import"),
    ],
    "sb3_contrib": [
        (r"from\s+sb3_contrib", "import"),
        (r"import\s+sb3_contrib", "import"),
    ],
    "cleanrl": [
        (r"cleanrl", "any"),
        (r"argparse.*--track", "cli"),  # CleanRL's distinctive CLI
        (r"SyncVectorEnv|RecordEpisodeStatistics", "env_wrapper"),
    ],
    "rllib": [
        (r"from\s+ray\s+import\s+tune", "import"),
        (r"from\s+ray\.rllib", "import"),
        (r"import\s+ray\.rllib", "import"),
    ],
    "tianshou": [
        (r"from\s+tianshou", "import"),
        (r"import\s+tianshou", "import"),
    ],
    "torchrl": [
        (r"from\s+torchrl", "import"),
        (r"import\s+torchrl", "import"),
    ],
}

ALGORITHM_PATTERNS = {
    "PPO": [
        (r"\bPPO\b", "class_or_string"),
        (r"\bppo\b", "string"),
        (r"clip_range|gae_lambda|GAE", "param"),
    ],
    "SAC": [
        (r"\bSAC\b", "class_or_string"),
        (r"\bsac\b", "string"),
        (r"alpha.*auto|entropy.*temperature|log_alpha", "param"),
    ],
    "TD3": [
        (r"\bTD3\b", "class_or_string"),
        (r"\btd3\b", "string"),
        (r"policy_delay|target_policy_noise", "param"),
    ],
    "DQN": [
        (r"\bDQN\b", "class_or_string"),
        (r"\bdqn\b", "string"),
        (r"epsilon_start|epsilon_end|exploration_fraction", "param"),
    ],
    "A2C": [
        (r"\bA2C\b", "class_or_string"),
        (r"\ba2c\b", "string"),
    ],
    "DDPG": [
        (r"\bDDPG\b", "class_or_string"),
        (r"\bddpg\b", "string"),
    ],
    "TRPO": [
        (r"\bTRPO\b|trpo\b", "any"),
    ],
    "Dreamer": [
        (r"\bDreamer\b|dreamer\b", "any"),
        (r"world_model|RSSM", "arch"),
    ],
}

FILE_ROLE_PATTERNS = {
    "training_entry": [
        (r"if\s+__name__\s*==\s*['\"]__main__['\"]", "main_guard"),
        (r"\.learn\(|\.train\(|train\(\)", "train_call"),
        (r"argparse|ArgumentParser|click\.command|hydra\.main", "cli"),
    ],
    "network_definition": [
        (r"class\s+\w*(?:Network|Net|Model|Policy|Actor|Critic|QNet|ValueNet)\w*", "class_def"),
        (r"nn\.Module", "pytorch"),
        (r"forward\(self", "forward_method"),
    ],
    "reward_function": [
        (r"def\s+\w*(?:reward|compute_reward|get_reward)\w*\(", "function_def"),
        (r"reward\s*\+?=\s*", "reward_accum"),
        (r"reward\s*=\s*(?!.*\.backward)", "reward_assign"),
    ],
    "environment": [
        (r"class\s+\w*(?:Env|Environment|Gym)\w*", "env_class"),
        (r"gym\.make|gymnasium\.make", "gym_make"),
        (r"class\s+\w*Env\w*\(.*gym(?:nasium)?\.Env\)", "gym_inherit"),
    ],
    "config": [
        (r"@dataclass|@attr\.s|omegaconf\.DictConfig", "structured_config"),
        (r"yaml\.safe_load|yaml\.load|json\.load.*config", "config_loader"),
        (r"^learning_rate|^batch_size|^gamma", "config_keys"),
    ],
    "replay_buffer": [
        (r"class\s+\w*(?:ReplayBuffer|Buffer|Memory|Experience)\w*", "buffer_class"),
        (r"def\s+(?:sample|push|add|store)\(", "buffer_methods"),
        (r"capacity|buffer_size|replay_size", "buffer_params"),
    ],
}

RL_PATTERNS = {
    "target_network": [
        (r"target(?:_net|_network|_critic|_actor)", "naming"),
        (r"load_state_dict|polyak|tau\s*=", "update_mechanism"),
    ],
    "advantage_gae": [
        (r"gae|generalized_advantage|compute_returns|compute_advantage", "naming"),
        (r"gae_lambda|lambda\s*=", "param"),
    ],
    "replay_buffer": [
        (r"class\s+\w*ReplayBuffer|class\s+\w*Buffer", "class"),
        (r"deque|collections\.deque", "deque"),
    ],
    "entropy_bonus": [
        (r"ent_coef|entropy_coef|entropy_bonus|entropy_loss", "param"),
        (r"\.entropy\(\)", "method"),
    ],
    "gradient_clipping": [
        (r"clip_grad_norm|clip_grad_value|grad_clip", "call"),
        (r"max_grad_norm|max_norm\s*=", "param"),
    ],
    "lr_schedule": [
        (r"lr_schedule|learning_rate.*schedule|linear_schedule|cosine_schedule", "naming"),
        (r"LambdaLR|StepLR|CosineAnnealingLR|ReduceLROnPlateau", "pytorch"),
    ],
    "reward_normalization": [
        (r"reward_norm|running_mean.*reward|RewardNormalizer|VecNormalize", "naming"),
        (r"normalize.*reward|reward.*normalize", "usage"),
    ],
    "seed_management": [
        (r"set_seed|seed_everything|manual_seed|set_random_seed", "function"),
        (r"torch\.manual_seed|np\.random\.seed|random\.seed", "individual"),
    ],
    "amp": [
        (r"torch\.cuda\.amp|autocast|GradScaler", "amp"),
    ],
    "compile": [
        (r"torch\.compile", "compile"),
    ],
}

# Issues to flag at project level.
# Each check receives a (patterns, algorithm) tuple so it can cross-reference
# detected RL patterns with the inferred algorithm for smarter diagnostics.
PROJECT_ISSUE_CHECKS = [
    {
        "id": "no_seed_management",
        "check": lambda p, a: "seed_management" not in p,
        "severity": "warning",
        "message": "No seed management found — results may not be reproducible.",
    },
    {
        "id": "no_gradient_clipping",
        "check": lambda p, a: "gradient_clipping" not in p,
        "severity": "info",
        "message": "No gradient clipping detected. DQN/SAC/TD3 typically need it; PPO usually doesn't.",
    },
    {
        "id": "single_env_no_vectorization",
        "check": lambda p, a: "env_parallelism" not in str(p),
        "severity": "info",
        "message": "Check if vectorized environments are used — single env → GPU underutilization.",
    },
    {
        "id": "no_config_file",
        "check": lambda p, a: "config" not in p,
        "severity": "info",
        "message": "No config file or structured config detected — hyperparameters may be hardcoded.",
    },
    {
        "id": "no_target_network_but_off_policy",
        "check": lambda p, a: "target_network" not in p and any(
            algo in a for algo in ["SAC", "TD3", "DQN", "DDPG"]
        ),
        "severity": "warning",
        "message": "Off-policy algorithm detected but no target network found — critical omission.",
    },
    {
        "id": "target_network_in_ppo",
        "check": lambda p, a: "target_network" in p and "PPO" in a,
        "severity": "warning",
        "message": "PPO does not use a target network — this may be an architectural mistake.",
    },
]


def find_python_files(root: Path, exclude_dirs: Set[str]) -> List[Path]:
    """Find all Python files in project, excluding specified directories."""
    exclude = {".git", "__pycache__", ".venv", "venv", ".tox", "node_modules",
               ".mypy_cache", ".pytest_cache", "build", "dist", ".eggs"}
    exclude.update(exclude_dirs)

    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fname in filenames:
            if fname.endswith(".py"):
                py_files.append(Path(dirpath) / fname)
    return py_files


def scan_file(filepath: Path) -> dict:
    """Scan a single Python file for RL patterns."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"path": str(filepath), "error": "unreadable"}

    lines = content.split("\n")
    result = {
        "path": str(filepath),
        "lines": len(lines),
        "roles": [],
        "frameworks": [],
        "algorithms": [],
        "patterns": [],
        "key_line_numbers": {},  # role → [line numbers]
    }

    for line_no, line in enumerate(lines, start=1):
        # Framework detection
        for fw, patterns in FRAMEWORK_PATTERNS.items():
            for pattern, ptype in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if fw not in result["frameworks"]:
                        result["frameworks"].append(fw)

        # Algorithm detection
        for algo, patterns in ALGORITHM_PATTERNS.items():
            for pattern, ptype in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if algo not in result["algorithms"]:
                        result["algorithms"].append(algo)

        # File role detection
        for role, patterns in FILE_ROLE_PATTERNS.items():
            for pattern, ptype in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if role not in result["roles"]:
                        result["roles"].append(role)
                    if role not in result["key_line_numbers"]:
                        result["key_line_numbers"][role] = []
                    result["key_line_numbers"][role].append(line_no)

        # RL pattern detection
        for pat_name, patterns in RL_PATTERNS.items():
            for pattern, ptype in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if pat_name not in result["patterns"]:
                        result["patterns"].append(pat_name)

    return result


def classify_files(scan_results: List[dict]) -> dict:
    """Organize scanned files by role for easy navigation."""
    by_role = {}
    for r in scan_results:
        if r.get("error"):
            continue
        for role in r.get("roles", []):
            by_role.setdefault(role, []).append(r["path"])

    # Determine primary training entry — prefer files with "train" in name
    # over generic files that just happen to have __main__ (e.g. voice_controller.py)
    training_files = by_role.get("training_entry", [])
    primary_entry = None
    # Tier 1: filename contains "train" (case-insensitive)
    for f in training_files:
        if "train" in Path(f).stem.lower():
            primary_entry = f
            break
    # Tier 2: filename is main.py or run.py
    if not primary_entry:
        for f in training_files:
            if Path(f).name in ("main.py", "run.py"):
                primary_entry = f
                break
    # Tier 3: file in a "scripts/" directory
    if not primary_entry:
        for f in training_files:
            if "scripts" in str(Path(f).parent):
                primary_entry = f
                break
    # Tier 4: first available
    if not primary_entry and training_files:
        primary_entry = training_files[0]

    return {
        "by_role": {k: v for k, v in sorted(by_role.items())},
        "primary_training_entry": primary_entry,
    }


def collect_project_patterns(scan_results: List[dict]) -> Set[str]:
    """Collect all unique RL patterns found across the project."""
    patterns = set()
    for r in scan_results:
        for p in r.get("patterns", []):
            patterns.add(p)
    return patterns


def detect_framework(scan_results: List[dict]) -> str:
    """Determine the primary RL framework."""
    fw_counts = {}
    for r in scan_results:
        for fw in r.get("frameworks", []):
            fw_counts[fw] = fw_counts.get(fw, 0) + 1
    if not fw_counts:
        return "custom_pytorch"  # default assumption
    # SB3 contrib is part of SB3
    if "sb3_contrib" in fw_counts:
        fw_counts["stable_baselines3"] = fw_counts.get("stable_baselines3", 0) + fw_counts.pop("sb3_contrib")
    return max(fw_counts, key=fw_counts.get)


def detect_algorithm(scan_results: List[dict]) -> str:
    """Determine the RL algorithm(s) used."""
    algo_counts = {}
    for r in scan_results:
        for algo in r.get("algorithms", []):
            algo_counts[algo] = algo_counts.get(algo, 0) + 1
    if not algo_counts:
        return "unknown"
    # Return all found, sorted by prevalence
    sorted_algos = sorted(algo_counts, key=algo_counts.get, reverse=True)
    if len(sorted_algos) == 1:
        return sorted_algos[0]
    return " + ".join(sorted_algos[:3])  # top 3


def generate_review_order(file_classification: dict, algorithm: str, user_concern: str = "") -> List[dict]:
    """Generate a priority-ordered list of files to review."""
    by_role = file_classification.get("by_role", {})
    primary = file_classification.get("primary_training_entry")

    order = []

    # Priority 1: Training entry
    if primary:
        order.append({"priority": 1, "role": "training_entry",
                       "path": primary, "reason": "Training loop — review first"})

    # Priority 2: Network definition
    for f in by_role.get("network_definition", [])[:2]:
        if f not in {o["path"] for o in order}:
            order.append({"priority": 2, "role": "network_definition",
                           "path": f, "reason": "Network architecture"})

    # Priority 3: Reward function
    for f in by_role.get("reward_function", [])[:2]:
        if f not in {o["path"] for o in order}:
            order.append({"priority": 3, "role": "reward_function",
                           "path": f, "reason": "Reward design is critical"})

    # Priority 4: Environment
    for f in by_role.get("environment", [])[:1]:
        if f not in {o["path"] for o in order}:
            order.append({"priority": 4, "role": "environment",
                           "path": f, "reason": "Environment dynamics"})

    # Priority 5: Config
    for f in by_role.get("config", [])[:1]:
        if f not in {o["path"] for o in order}:
            order.append({"priority": 5, "role": "config",
                           "path": f, "reason": "Hyperparameter configuration"})

    return order


def main():
    parser = argparse.ArgumentParser(
        description="Scan an RL project and output a structured summary for LLM analysis"
    )
    parser.add_argument("--dir", default=".", help="Project directory to scan (default: current dir)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Additional directories to exclude")
    parser.add_argument("--output", help="Save JSON to file instead of stdout")
    parser.add_argument("--max-files", type=int, default=80,
                        help="Max Python files to scan (default: 80, prevents huge projects from being slow)")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(json.dumps({"error": f"Directory not found: {args.dir}"}))
        sys.exit(1)

    # Find Python files
    py_files = find_python_files(root, set(args.exclude))

    if len(py_files) > args.max_files:
        # Prioritize: files in root or named suggestively first
        priority_names = {"train", "main", "run", "model", "agent", "env", "buffer",
                          "network", "policy", "reward", "config", "ppo", "sac", "dqn", "td3"}
        priority_files = []
        other_files = []
        for f in py_files:
            stem = f.stem.lower()
            if any(pn in stem for pn in priority_names):
                priority_files.append(f)
            else:
                other_files.append(f)
        py_files = priority_files + other_files[:args.max_files - len(priority_files)]

    # Scan each file
    scan_results = [scan_file(f) for f in py_files]
    scan_results = [r for r in scan_results if r.get("roles") or r.get("frameworks") or r.get("algorithms")]
    non_rl_files = len(py_files) - len(scan_results)

    if not scan_results:
        print(json.dumps({
            "error": "No RL-related Python files found in the project.",
            "total_python_files_scanned": len(py_files),
            "hint": "Is this an RL project? Check if the directory is correct.",
        }))
        sys.exit(1)

    # Aggregate results
    framework = detect_framework(scan_results)
    algorithm = detect_algorithm(scan_results)
    patterns = collect_project_patterns(scan_results)
    file_classification = classify_files(scan_results)
    review_order = generate_review_order(file_classification, algorithm)

    # Project-level issues
    issues = []
    for check in PROJECT_ISSUE_CHECKS:
        try:
            if check["check"](patterns, algorithm):
                issues.append({"id": check["id"], "severity": check["severity"],
                                "message": check["message"]})
        except Exception:
            continue

    # File count summary
    role_counts = {role: len(files) for role, files in file_classification["by_role"].items()}

    # Assemble output
    result = {
        "project_root": str(root),
        "total_python_files": len(py_files),
        "rl_related_files": len(scan_results),
        "non_rl_files_skipped": non_rl_files,
        "framework": framework,
        "algorithm": algorithm,
        "file_roles": role_counts,
        "primary_training_entry": file_classification.get("primary_training_entry"),
        "rl_patterns_found": sorted(patterns),
        "potential_issues": issues,
        "suggested_review_order": review_order,
        "files": {
            "by_role": {role: [{"path": p, "lines": next(
                (r["lines"] for r in scan_results if r["path"] == p), None
            )} for p in paths[:5]]  # top 5 per role
                for role, paths in file_classification["by_role"].items()},
        },
        "action_required": "Review the suggested_review_order above. Read the files that are most relevant to the user's question, then proceed with diagnosis.",
    }

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"Saved to {args.output}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()
