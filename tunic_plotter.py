#!/usr/bin/env python3
"""Plot hyperparameter search results from a tunic results JSON file."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot tunic hyperparameter search results")
    p.add_argument("results_file", type=Path, help="Path to results JSON file")
    p.add_argument("--metric", default="val_auroc", help="Metric to plot (default: val_auroc)")
    p.add_argument("--trial_sort", action="store_true", help="Plot trials in original order instead of sorted by metric value")
    return p.parse_args()


def plot_metric(data, trials, metric, results_file, trial_sort):
    values = []
    for t in trials:
        if metric not in t:
            print(f"Warning: metric '{metric}' not found in trial data, skipping.", file=sys.stderr)
            return
        values.append(t[metric] if t[metric] is not None else float("nan"))

    values = np.array(values)
    best_val = float(np.nanmax(values))

    if not trial_sort:
        order = np.argsort(values)
        values = values[order]
        best_idx = len(values) - 1 - int(np.sum(np.isnan(values)))
    else:
        best_idx = int(np.nanargmax(values))

    running_best = np.maximum.accumulate(np.where(np.isnan(values), -np.inf, values))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(range(len(values)), values, color="steelblue", s=40, zorder=3, label="Trial")
    ax.scatter([best_idx], [best_val], color="crimson", s=80, zorder=4, label=f"Best ({best_val:.4f})")
    if trial_sort:
        ax.step(range(len(running_best)), running_best,
                color="orange", linewidth=1.5, where="post", label="Running best")
    ax.axhline(best_val, color="crimson", linewidth=0.8, linestyle="--", alpha=0.5)

    if "auroc" in metric:
        ax.set_ylim(0.5, 1.0)
    elif "acc" in metric:
        ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Trial rank" if not trial_sort else "Trial")
    ax.set_ylabel(metric)
    ax.set_title(
        f"{data.get('model', '')}  |  {metric}  |  "
        f"{data.get('completed_trials', len(trials))}/{data.get('n_trials', len(trials))} trials  |  "
        f"{data.get('epochs', '?')} epochs"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = results_file.with_suffix("").with_name(results_file.stem + f"_{metric}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    args = parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    trials = data["all_trials"]

    if args.metric == "val_auroc":
        # default: plot both
        plot_metric(data, trials, "val_auroc", args.results_file, args.trial_sort)
        plot_metric(data, trials, "val_acc",   args.results_file, args.trial_sort)
    else:
        plot_metric(data, trials, args.metric, args.results_file, args.trial_sort)


if __name__ == "__main__":
    main()
