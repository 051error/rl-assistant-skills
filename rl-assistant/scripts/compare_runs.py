#!/usr/bin/env python3
"""
Multi-experiment comparison tool for RL training runs.

Compares reward metrics across multiple experiment directories,
performs basic statistical tests, and outputs a compact JSON summary.

Usage:
    python compare_runs.py --runs logs/exp1 logs/exp2 logs/exp3
    python compare_runs.py --runs logs/exp1 logs/exp2 --metric reward_total
    python compare_runs.py --runs logs/exp1 logs/exp2 --last-n 50

Output:
    JSON to stdout with keys:
    - runs: per-run summary statistics
    - pairwise: pairwise comparisons with effect size and significance
    - ranking: runs ranked by convergence performance
    - summary: human-readable comparison summary
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


def find_log_file(run_dir: Path) -> Optional[Path]:
    """Find a CSV log file in the run directory."""
    # Common names
    candidates = ["progress.csv", "log.csv", "metrics.csv", "train.csv"]
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    # Fallback: any .csv file
    csv_files = list(run_dir.glob("*.csv"))
    return csv_files[0] if csv_files else None


def load_run_data(log_path: Path, metric: str) -> np.ndarray:
    """Load a specific metric column from a run's log file."""
    with open(log_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)

        # Find the metric column
        if metric in headers:
            col_idx = headers.index(metric)
        else:
            # Fuzzy match
            matches = [h for h in headers if metric.lower() in h.lower()]
            if not matches:
                raise ValueError(
                    f"Metric '{metric}' not found in {log_path}. "
                    f"Available columns: {headers}"
                )
            col_idx = headers.index(matches[0])

        values = []
        for row in reader:
            try:
                values.append(float(row[col_idx]))
            except (ValueError, IndexError):
                continue
        return np.array(values)


def compute_run_stats(series: np.ndarray, last_n: int) -> dict:
    """Compute summary statistics for a single run."""
    series = series[np.isfinite(series)]
    full_mean = float(np.mean(series))
    full_std = float(np.std(series))

    # Convergence region (last N points)
    con_series = series[-min(last_n, len(series)):]
    con_mean = float(np.mean(con_series))
    con_std = float(np.std(con_series))

    # Best value (max for reward, min for loss — we assume reward)
    best = float(np.max(series))
    best_episode = int(np.argmax(series))

    return {
        "n_episodes": len(series),
        "full_mean": full_mean,
        "full_std": full_std,
        "convergence_mean": con_mean,
        "convergence_std": con_std,
        "best": best,
        "best_episode": best_episode,
    }


def pairwise_comparison(
    run_a: np.ndarray,
    run_b: np.ndarray,
    name_a: str,
    name_b: str,
    last_n: int,
) -> dict:
    """Compare two runs with effect size and statistical test."""
    series_a = run_a[np.isfinite(run_a)]
    series_b = run_b[np.isfinite(run_b)]

    # Use last N for comparison
    con_a = series_a[-min(last_n, len(series_a)):]
    con_b = series_b[-min(last_n, len(series_b)):]

    mean_a, mean_b = float(np.mean(con_a)), float(np.mean(con_b))
    std_a, std_b = float(np.std(con_a)), float(np.std(con_b))

    delta = mean_a - mean_b
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

    # Cohen's d effect size
    d = delta / pooled_std if pooled_std > 1e-8 else 0.0

    # Welch's t-test
    try:
        t_stat, p_value = sp_stats.ttest_ind(con_a, con_b, equal_var=False)
        p_value = float(p_value)
    except Exception:
        t_stat, p_value = None, None

    # Practical significance
    if abs(d) < 0.2:
        significance = "negligible"
    elif abs(d) < 0.5:
        significance = "small"
    elif abs(d) < 0.8:
        significance = "medium"
    else:
        significance = "large"

    # Is the difference > 2 * pooled_std?
    exceeds_2sigma = bool(abs(delta) > 2 * pooled_std) if pooled_std > 1e-8 else False

    winner = name_a if delta > 0 and exceeds_2sigma else \
             name_b if delta < 0 and exceeds_2sigma else \
             "no_clear_winner"

    return {
        "comparison": f"{name_a} vs {name_b}",
        "delta": float(delta),
        "cohens_d": float(d),
        "p_value_Welch_ttest": float(p_value) if p_value is not None else None,
        "practically_significant": bool(abs(d) >= 0.5),
        "exceeds_2sigma": exceeds_2sigma,
        "winner": winner,
        "pooled_std": float(pooled_std),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL experiment runs statistically"
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Experiment log directories or CSV files")
    parser.add_argument(
        "--metric",
        default="reward_total",
        help="Metric column to compare (default: reward_total)",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=100,
        help="Number of final episodes for convergence comparison (default: 100)",
    )
    parser.add_argument("--output", help="Save JSON to file instead of stdout")
    args = parser.parse_args()

    # Resolve log files
    run_data = {}
    for run_path_str in args.runs:
        run_path = Path(run_path_str)
        if run_path.is_file():
            log_file = run_path
            run_name = run_path.stem
        elif run_path.is_dir():
            log_file = find_log_file(run_path)
            run_name = run_path.name
        else:
            print(json.dumps({"error": f"Path not found: {run_path_str}"}))
            sys.exit(1)

        if log_file is None:
            print(json.dumps({"error": f"No CSV log file found in {run_path_str}"}))
            sys.exit(1)

        try:
            series = load_run_data(log_file, args.metric)
            run_data[run_name] = {"path": str(log_file), "series": series}
        except ValueError as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)

    if len(run_data) < 2:
        print(json.dumps({"error": "Need at least 2 runs to compare"}))
        sys.exit(1)

    # Compute per-run stats
    runs_summary = {}
    for name, rd in run_data.items():
        runs_summary[name] = compute_run_stats(rd["series"], args.last_n)
        runs_summary[name]["source"] = rd["path"]

    # Ranking by convergence mean
    ranking = sorted(
        runs_summary.items(),
        key=lambda x: x[1]["convergence_mean"],
        reverse=True,
    )
    ranking_list = [
        {"rank": i + 1, "run": name, "convergence_mean": stats["convergence_mean"],
         "convergence_std": stats["convergence_std"]}
        for i, (name, stats) in enumerate(ranking)
    ]

    # Pairwise comparisons
    run_names = list(run_data.keys())
    pairwise = []
    for i in range(len(run_names)):
        for j in range(i + 1, len(run_names)):
            comp = pairwise_comparison(
                run_data[run_names[i]]["series"],
                run_data[run_names[j]]["series"],
                run_names[i],
                run_names[j],
                args.last_n,
            )
            pairwise.append(comp)

    # Summary
    best_run = ranking[0][0]
    best_mean = ranking[0][1]["convergence_mean"]
    worst_run = ranking[-1][0]
    worst_mean = ranking[-1][1]["convergence_mean"]
    gap_pct = abs((best_mean - worst_mean) / abs(worst_mean)) * 100 if abs(worst_mean) > 1e-8 else 0

    significant_pairs = [p for p in pairwise if p.get("practically_significant")]
    n_significant = len(significant_pairs)
    total_pairs = len(pairwise)

    summary_lines = [
        f"Best run: {best_run} (convergence mean = {best_mean:.3f})",
        f"Worst run: {worst_run} (convergence mean = {worst_mean:.3f})",
        f"Gap: {gap_pct:.1f}%",
        f"Practically significant pairwise differences: {n_significant}/{total_pairs}",
    ]

    result = {
        "metric": args.metric,
        "last_n_episodes_for_convergence": args.last_n,
        "n_runs": len(run_data),
        "runs": runs_summary,
        "ranking": ranking_list,
        "pairwise_comparisons": pairwise,
        "summary": summary_lines,
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
