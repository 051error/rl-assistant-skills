#!/usr/bin/env python3
"""
Reward statistical profile generator for RL experiments.

Reads training log files (CSV / WandB CSV export / TensorBoard scalar export),
computes a compact statistical summary of each reward component,
and prints JSON to stdout — designed to be consumed by an LLM for diagnosis.

Usage:
    python analyze_reward.py --log runs/exp1/progress.csv
    python analyze_reward.py --log wandb_export.csv --components reward_goal reward_bonus
    python analyze_reward.py --log runs/exp1/progress.csv --window 100 --convergence-pct 0.2

Output:
    JSON to stdout with keys:
    - data_quality: overall data sufficiency assessment
    - components: per-component statistics
    - cross_seed: cross-seed stability (if multiple seed columns detected)
    - trends: trend detection for total reward
    - diagnostics: automated diagnostic flags
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    """Load CSV, returns (column_names, data_array)."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                continue  # skip non-numeric rows
    return headers, np.array(rows)


def find_reward_columns(headers: List[str]) -> List[str]:
    """Find all columns that are reward components (starting with 'reward_')."""
    reward_cols = [h for h in headers if h.startswith("reward_")]
    if not reward_cols:
        # Fallback: look for 'episode_reward' or 'total_reward'
        for alias in ["episode_reward", "total_reward", "return", "cumulative_reward"]:
            if alias in headers:
                reward_cols = [alias]
                break
    return reward_cols


def compute_component_stats(
    data: np.ndarray,
    col_idx: int,
    col_name: str,
    window: int,
    convergence_pct: float,
) -> dict:
    """Compute statistical profile for a single reward component."""
    series = data[:, col_idx]
    n = len(series)

    # Remove NaN/Inf
    series = series[np.isfinite(series)]
    if len(series) < 2:
        return {
            "name": col_name,
            "n_episodes": len(series),
            "error": "Insufficient data (need >= 2 episodes)",
        }

    # Full-series statistics
    mean = float(np.mean(series))
    std = float(np.std(series))
    cv = float(std / abs(mean)) if abs(mean) > 1e-8 else float("inf")
    min_val = float(np.min(series))
    max_val = float(np.max(series))
    median = float(np.median(series))

    # Convergence window statistics (last N% of episodes)
    con_size = max(int(n * convergence_pct), min(window, n // 4))
    con_start = max(0, n - con_size)
    if con_start >= n:
        con_start = max(0, n // 2)  # fallback: use second half
    con_series = series[con_start:]
    con_mean = float(np.mean(con_series))
    con_std = float(np.std(con_series))
    con_cv = float(con_std / abs(con_mean)) if abs(con_mean) > 1e-8 else float("inf")

    # Rolling window
    window_size = min(window, n // 2)
    if window_size >= 2:
        rolling_mean = np.array(
            [np.mean(series[i : i + window_size]) for i in range(n - window_size + 1)]
        )
        rolling_trend = "increasing" if rolling_mean[-1] > rolling_mean[0] * 1.05 else \
                        "decreasing" if rolling_mean[-1] < rolling_mean[0] * 0.95 else \
                        "flat"
    else:
        rolling_trend = "insufficient_data"

    # Signal quality assessment
    if cv < 0.3:
        signal_quality = "clear"  # single observation is trustworthy
    elif cv < 1.0:
        signal_quality = "moderate_noise"  # need rolling average
    else:
        signal_quality = "high_noise"  # single observation is meaningless

    return {
        "name": col_name,
        "n_episodes": n,
        "full_series": {
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": min_val,
            "max": max_val,
            "median": median,
        },
        "convergence_region": {
            "start_episode": con_start,
            "mean": con_mean,
            "std": con_std,
            "cv": con_cv,
        },
        "rolling_trend": rolling_trend,
        "signal_quality": signal_quality,
    }


def compute_cross_seed_stats(
    data: np.ndarray,
    headers: List[str],
    reward_cols: List[str],
) -> Optional[dict]:
    """Detect and compute cross-seed statistics if multiple seed columns exist."""
    seed_cols = [h for h in headers if "seed" in h.lower()]
    if not seed_cols:
        # Check if there's a 'run_id' or 'group' column that might indicate seeds
        for alias in ["run_id", "group", "experiment"]:
            if alias in headers:
                seed_cols = [alias]
                break
    if not seed_cols:
        return None

    seed_idx = headers.index(seed_cols[0])
    unique_seeds = np.unique(data[:, seed_idx])
    n_seeds = len(unique_seeds)

    if n_seeds < 2:
        return {"n_seeds": n_seeds, "note": "Only 1 seed detected, cross-seed analysis not possible"}

    # For each reward component, compute per-seed convergence mean
    per_seed_means = {col: [] for col in reward_cols}
    for seed in unique_seeds:
        seed_mask = data[:, seed_idx] == seed
        seed_data = data[seed_mask]
        if len(seed_data) < 10:
            continue
        for col in reward_cols:
            col_idx = headers.index(col)
            series = seed_data[:, col_idx]
            series = series[np.isfinite(series)]
            # Use last 20% as convergence
            con_start = max(len(series) * 4 // 5, 1)
            per_seed_means[col].append(float(np.mean(series[int(con_start):])))

    result = {"n_seeds": n_seeds}
    for col in reward_cols:
        means = per_seed_means[col]
        if len(means) < 2:
            result[col] = {"error": "Not enough seeds with sufficient data"}
            continue
        cross_std = float(np.std(means))
        within_std = float(np.mean([
            np.std(data[(data[:, seed_idx] == seed), headers.index(col)])
            for seed in unique_seeds[: len(means)]
        ]))
        result[col] = {
            "cross_seed_std_of_mean": cross_std,
            "per_seed_means": means,
            "stability": "stable" if cross_std < within_std else
                         "moderate" if cross_std < 2 * within_std else
                         "unstable",
        }
    return result


def compute_data_quality(stats: List[dict], cross_seed: Optional[dict]) -> dict:
    """Assess overall data quality for diagnosis."""
    n_components = len(stats)
    n_episodes_min = min((s.get("n_episodes", 0) for s in stats), default=0)
    n_episodes_max = max((s.get("n_episodes", 0) for s in stats), default=0)
    high_noise = [s["name"] for s in stats if s.get("signal_quality") == "high_noise"]

    issues = []
    if n_episodes_min < 10:
        issues.append("Episode count too low (< 10), conclusions may be unreliable")
    if high_noise:
        issues.append(f"High-noise components: {', '.join(high_noise)} — single observations are not trustworthy")

    if cross_seed:
        n_seeds = cross_seed.get("n_seeds", 1)
        if n_seeds < 3:
            issues.append(f"Only {n_seeds} seed(s) — need ≥ 3 for reliable cross-seed comparison")
        unstable = [k for k, v in cross_seed.items()
                    if isinstance(v, dict) and v.get("stability") == "unstable"]
        if unstable:
            issues.append(f"Unstable across seeds: {', '.join(unstable)} — seed variance > within-seed variance")

    return {
        "total_episodes": n_episodes_max,
        "n_components": n_components,
        "sufficient_for_diagnosis": len(issues) == 0,
        "issues": issues,
    }


def compute_correlation_with_total(
    data: np.ndarray,
    headers: List[str],
    reward_cols: List[str],
) -> dict:
    """Compute correlation of each reward component with total reward."""
    # Find total reward column
    total_col = None
    for alias in ["reward_total", "episode_reward", "total_reward", "return"]:
        if alias in headers:
            total_col = alias
            break
    if total_col is None:
        # Use first reward column as total if only one exists
        if len(reward_cols) == 1:
            return {}
        # Try to infer: the column without "reward_" prefix but containing "reward"
        candidates = [h for h in headers if "reward" in h.lower() and not h.startswith("reward_")]
        if candidates:
            total_col = candidates[0]
        else:
            return {}

    total_idx = headers.index(total_col)
    total_series = data[:, total_idx]
    total_series = total_series[np.isfinite(total_series)]

    correlations = {}
    for col in reward_cols:
        if col == total_col:
            continue
        col_idx = headers.index(col)
        series = data[:, col_idx]
        series = series[np.isfinite(series)]
        min_len = min(len(series), len(total_series))
        if min_len < 5:
            correlations[col] = None
            continue
        corr = float(np.corrcoef(series[:min_len], total_series[:min_len])[0, 1])
        if np.isnan(corr):
            corr = 0.0
        # Interpretation
        if abs(corr) > 0.5:
            interpretation = "strong_driver"
        elif abs(corr) > 0.2:
            interpretation = "weak_driver"
        else:
            interpretation = "likely_decorative"  # little impact on optimization
        correlations[col] = {
            "r": corr,
            "interpretation": interpretation,
        }
    return correlations


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistical profile of RL reward components"
    )
    parser.add_argument("--log", required=True, help="Path to training log CSV file")
    parser.add_argument(
        "--components",
        nargs="*",
        help="Specific reward columns to analyze (default: auto-detect columns starting with 'reward_')",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling window size for trend detection (default: 100)",
    )
    parser.add_argument(
        "--convergence-pct",
        type=float,
        default=0.2,
        help="Fraction of final episodes to treat as convergence region (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        help="Save JSON to file instead of stdout",
    )
    args = parser.parse_args()

    # Validate input
    log_path = Path(args.log)
    if not log_path.exists():
        print(json.dumps({"error": f"File not found: {args.log}"}))
        sys.exit(1)

    # Load data
    try:
        headers, data = load_csv(str(log_path))
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse CSV: {e}"}))
        sys.exit(1)

    if data.shape[0] < 2:
        print(json.dumps({"error": "CSV contains fewer than 2 rows of data"}))
        sys.exit(1)

    # Identify reward columns
    reward_cols = args.components if args.components else find_reward_columns(headers)
    if not reward_cols:
        print(json.dumps({
            "error": "No reward columns found. Columns must start with 'reward_' or use --components to specify.",
            "available_columns": headers,
        }))
        sys.exit(1)

    # Compute per-component statistics
    stats = []
    for col in reward_cols:
        try:
            col_idx = headers.index(col)
        except ValueError:
            stats.append({"name": col, "error": f"Column not found in CSV"})
            continue
        comp_stats = compute_component_stats(
            data, col_idx, col, args.window, args.convergence_pct
        )
        stats.append(comp_stats)

    # Cross-seed analysis
    cross_seed = compute_cross_seed_stats(data, headers, reward_cols)

    # Data quality
    data_quality = compute_data_quality(stats, cross_seed)

    # Correlations with total reward
    correlations = compute_correlation_with_total(data, headers, reward_cols)

    # Assemble output
    result = {
        "source": str(log_path),
        "n_episodes_total": data.shape[0],
        "reward_columns_analyzed": reward_cols,
        "data_quality": data_quality,
        "components": stats,
        "correlations_with_total": correlations,
        "cross_seed_analysis": cross_seed,
    }

    # Diagnostics (auto-generated flags for the LLM)
    diagnostics = []
    for s in stats:
        if s.get("signal_quality") == "high_noise":
            diagnostics.append({
                "severity": "warning",
                "component": s["name"],
                "message": f"CV={s['full_series']['cv']:.2f} — high noise, single observations are unreliable. Use rolling average.",
            })
        if s.get("rolling_trend") == "decreasing":
            diagnostics.append({
                "severity": "warning",
                "component": s["name"],
                "message": "Decreasing trend detected — possible overfitting or reward hacking.",
            })
        if s.get("full_series", {}).get("cv", 0) > 0.3 and s.get("rolling_trend") == "flat":
            diagnostics.append({
                "severity": "info",
                "component": s["name"],
                "message": "Flat trend with moderate noise — training may have stalled.",
            })

    for col, corr_info in correlations.items():
        if isinstance(corr_info, dict) and corr_info.get("interpretation") == "likely_decorative":
            diagnostics.append({
                "severity": "info",
                "component": col,
                "message": f"Correlation with total reward is only r={corr_info['r']:.3f} — this term may not be driving policy improvement.",
            })

    if cross_seed:
        unstable = [k for k, v in cross_seed.items()
                    if isinstance(v, dict) and v.get("stability") == "unstable"]
        if unstable:
            diagnostics.append({
                "severity": "warning",
                "component": ", ".join(unstable),
                "message": "Cross-seed instability detected — seed choice matters more than algorithm improvement. Increase seed count.",
            })

    result["diagnostics"] = diagnostics

    # Output
    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"Saved to {args.output}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()
