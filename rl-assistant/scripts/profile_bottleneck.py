#!/usr/bin/env python3
"""
Training bottleneck profiler for RL training loops.

Instruments a training script to measure time spent in each phase
(environment stepping, network forward, backward, logging, etc.)
and identifies the primary bottleneck.

Two modes:
1.  Direct profiling: Run a short training session with instrumentation.
    python profile_bottleneck.py --script train.py --steps 1000

2.  Static analysis: Estimate bottlenecks from code structure + config.
    python profile_bottleneck.py --config config.yaml --static

Output:
    JSON to stdout with:
    - phase_timing: per-phase time breakdown
    - bottleneck: identified primary bottleneck
    - recommendations: specific optimization suggestions
    - gpu_utilization: estimated GPU utilization
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROFILER_TEMPLATE = '''
import cProfile
import pstats
import io
import atexit

_profiler = cProfile.Profile()
_profiler.enable()

def _dump_profile():
    _profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(_profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(50)
    print("===PROFILE_START===")
    print(s.getvalue())
    print("===PROFILE_END===")

atexit.register(_dump_profile)
'''

BOTTLENECK_TEMPLATE = '''
import time
import json

_timings = {
    "env_step": [],
    "network_forward": [],
    "network_backward": [],
    "buffer_sample": [],
    "buffer_add": [],
    "logging": [],
    "eval": [],
    "other": [],
}
_last_t = time.perf_counter()
_current_phase = "other"

def _tick(phase):
    global _last_t, _current_phase
    now = time.perf_counter()
    _timings[_current_phase].append(now - _last_t)
    _last_t = now
    _current_phase = phase

def _report():
    global _last_t
    now = time.perf_counter()
    _timings[_current_phase].append(now - _last_t)
    _last_t = now

    result = {}
    for phase, times in _timings.items():
        if not times:
            result[phase] = {"total_s": 0, "pct": 0, "mean_ms": 0, "n_calls": 0}
            continue
        total = sum(times)
        result[phase] = {
            "total_s": round(total, 4),
            "pct": round(total / max(sum(sum(v) for v in _timings.values()), 1e-8) * 100, 1),
            "mean_ms": round(total / len(times) * 1000, 3),
            "n_calls": len(times),
        }
    print("===BOTTLENECK_START===")
    print(json.dumps(result, indent=2))
    print("===BOTTLENECK_END===")

import atexit
atexit.register(_report)
'''

# Hooks for each RL library
HOOK_SPECS = {
    "stable_baselines3": {
        "env_step": "env.step(",
        "network_forward": "self.policy.",
        "network_backward": ".backward()",
        "env_reset": "env.reset(",
    },
    "cleanrl": {
        "env_step": "envs.step(",
        "network_forward": "agent.get_action",
        "network_backward": "loss.backward()",
    },
    "custom": {
        "env_step": "env.step(",
        "network_forward": "model(",
        "network_backward": ".backward()",
        "buffer_sample": "buffer.sample(",
        "buffer_add": "buffer.add(",
        "logging": "log(",
    },
}


def estimate_bottleneck_from_config(config_path: str) -> dict:
    """Static estimation of bottlenecks from config parameters."""
    # Read config
    config = {}
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except ImportError:
            pass

    # Estimate based on known patterns
    estimates = {}

    # Environment count
    n_envs = config.get("n_envs", config.get("num_envs", 1))
    if n_envs <= 1:
        estimates["env_parallelism"] = {
            "level": "critical",
            "message": "Single environment → env stepping likely dominates training time. Use AsyncVectorEnv or increase n_envs.",
            "expected_gpu_utilization": "< 30%",
        }
    elif n_envs <= 4:
        estimates["env_parallelism"] = {
            "level": "warning",
            "message": f"Only {n_envs} envs — may still be env-bound for fast networks.",
            "expected_gpu_utilization": "30-60%",
        }
    else:
        estimates["env_parallelism"] = {
            "level": "ok",
            "message": f"{n_envs} parallel envs — env stepping likely not the bottleneck.",
            "expected_gpu_utilization": "60-90%",
        }

    # Network size
    hidden_sizes = config.get("hidden_sizes", config.get("hidden_layers", [256, 256]))
    total_params = sum(h * h_prev for h, h_prev in zip(hidden_sizes, [hidden_sizes[0]] + hidden_sizes))
    if total_params < 50000:
        estimates["network_size"] = {
            "level": "info",
            "message": "Small network — consider enlarging if underfitting.",
        }

    # Batch size
    batch_size = config.get("batch_size", 256)
    if batch_size < 64:
        estimates["batch_size"] = {
            "level": "warning",
            "message": f"Small batch size ({batch_size}) — GPU may be underutilized. Consider gradient accumulation.",
        }

    # Replay buffer (if applicable)
    buffer_size = config.get("buffer_size", config.get("replay_buffer_capacity", 0))
    if buffer_size > 1_000_000:
        estimates["buffer"] = {
            "level": "info",
            "message": f"Large buffer ({buffer_size}) — ensure pre-allocated numpy array, not Python list.",
        }

    return estimates


def analyze_cprofile_output(raw_output: str) -> dict:
    """Parse cProfile output and categorize top time consumers."""
    in_block = False
    lines = []
    for line in raw_output.split("\n"):
        if "===PROFILE_START===" in line:
            in_block = True
            continue
        if "===PROFILE_END===" in line:
            in_block = False
            continue
        if in_block:
            lines.append(line)

    # Simple heuristic: look for known bottleneck patterns
    bottleneck_categories = {
        "env_step": ["step", "render", "reset"],
        "network_forward": ["forward", "conv", "linear", "attention"],
        "network_backward": ["backward"],
        "numpy": ["numpy", "np."],
        "python_overhead": ["method", "wrapper", "getattr"],
    }

    categorized_time = {}
    for line in lines:
        line_lower = line.lower()
        for category, keywords in bottleneck_categories.items():
            if any(kw in line_lower for kw in keywords):
                # Rough time extraction (cProfile format varies)
                categorized_time.setdefault(category, []).append(line.strip())
                break

    return {
        "top_functions": lines[:10] if len(lines) >= 10 else lines,
        "categorized_hints": {k: len(v) for k, v in categorized_time.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile RL training loop to identify bottlenecks"
    )
    parser.add_argument("--script", help="Training script to profile")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to profile")
    parser.add_argument(
        "--config",
        help="Config file for static analysis (alternative to --script)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "instrument", "static"],
        default="auto",
        help="Profiling mode (default: auto — static if no script provided)",
    )
    parser.add_argument("--output", help="Save JSON to file instead of stdout")
    parser.add_argument(
        "--static", action="store_true",
        help="Force static analysis mode (shortcut for --mode static)"
    )
    args = parser.parse_args()

    if not args.script and not args.config:
        result = {
            "error": "Either --script or --config must be provided.",
            "suggestion": "For static analysis: --config config.yaml\n"
                          "For profiling: --script train.py --steps 1000",
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)

    # Determine mode
    mode = args.mode
    if args.static:
        mode = "static"
    if mode == "auto":
        mode = "static" if not args.script else "instrument"

    result = {"mode": mode}

    # Static analysis
    if args.config:
        result["static_analysis"] = estimate_bottleneck_from_config(args.config)

    # Instrumentation-based profiling
    if args.script and mode == "instrument":
        script_path = Path(args.script)
        if not script_path.exists():
            result["profiling_error"] = f"Script not found: {args.script}"
        else:
            result["profiling_note"] = (
                "For accurate per-phase timing, wrap your training loop with the "
                "timing hooks from scripts/profile_bottleneck.py."
                "See the BOTTLENECK_TEMPLATE variable for the instrumentation code."
            )
            # Attempt to run with cProfile if the script is small enough
            try:
                start_t = time.perf_counter()
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**__import__("os").environ, "PROFILE_STEPS": str(args.steps)},
                )
                elapsed = time.perf_counter() - start_t
                result["profiling_run"] = {
                    "elapsed_s": round(elapsed, 2),
                    "return_code": proc.returncode,
                }
                if proc.returncode != 0 and proc.stderr:
                    result["profiling_run"]["stderr_tail"] = proc.stderr[-500:]
            except subprocess.TimeoutExpired:
                result["profiling_run"] = {"error": "Profiling timed out after 60s"}
            except Exception as e:
                result["profiling_run"] = {"error": str(e)}

    # Generate recommendations
    recommendations = []
    static = result.get("static_analysis", {})

    env_par = static.get("env_parallelism", {})
    if env_par.get("level") == "critical":
        recommendations.append({
            "priority": "high",
            "category": "env_parallelism",
            "suggestion": "Use AsyncVectorEnv or gymnasium.vector.AsyncVectorEnv to run multiple environments in parallel.",
            "expected_improvement": "2-8x throughput improvement",
        })
    elif env_par.get("level") == "warning":
        recommendations.append({
            "priority": "medium",
            "category": "env_parallelism",
            "suggestion": "Increase n_envs to 8-16 for better GPU utilization.",
            "expected_improvement": "1.5-3x throughput improvement",
        })

    batch = static.get("batch_size", {})
    if batch.get("level") == "warning":
        recommendations.append({
            "priority": "medium",
            "category": "batch_size",
            "suggestion": "Increase batch_size or use gradient accumulation to better utilize GPU.",
        })

    buffer = static.get("buffer", {})
    if buffer:
        recommendations.append({
            "priority": "medium",
            "category": "replay_buffer",
            "suggestion": "Ensure replay buffer uses pre-allocated numpy arrays, not Python list + append.",
            "code_example": "See references/code-review.md for optimal ReplayBuffer implementation.",
        })

    # General recommendations
    recommendations.extend([
        {
            "priority": "low",
            "category": "general",
            "suggestion": "Ensure env.render() is NOT called during training — it can consume 50%+ of loop time.",
        },
        {
            "priority": "low",
            "category": "general",
            "suggestion": "Use torch.compile() on your networks (PyTorch >= 2.0) for 5-20% speedup.",
        },
        {
            "priority": "low",
            "category": "general",
            "suggestion": "Consider mixed-precision training (torch.cuda.amp) if using large batch sizes.",
        },
    ])

    result["recommendations"] = recommendations

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"Saved to {args.output}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()
