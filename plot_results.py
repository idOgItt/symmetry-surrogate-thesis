"""Render a comparison across regularization methods and sample budgets."""

from itertools import cycle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_overview(summary, config, path):
    metrics = ("relative_l2", "roc_auc", "f1")
    labels = ("Relative L2 error ↓", "ROC AUC ↑", "F1 at fixed threshold ↑")
    fig, axes = plt.subplots(
        len(config.train_sizes), len(metrics), figsize=(10.8, 5.8),
        layout="constrained", squeeze=False,
    )
    colors = {"none": "#52606d", "ridge": "#2166ac", "smooth": "#b44b21"}
    noise_styles = dict(zip(config.noise_levels, cycle(("-", "--", ":", "-."))))
    for count, row_axes in zip(config.train_sizes, axes):
        for metric, label, ax in zip(metrics, labels, row_axes):
            for sigma, linestyle in noise_styles.items():
                regime = [r for r in summary if r["train_size"] == count and r["noise"] == sigma]
                baseline = next(r for r in regime if r["method"] == "none")
                ax.axhline(baseline[f"{metric}_mean"], color=colors["none"], linestyle=linestyle, linewidth=1)
                for method in ("ridge", "smooth"):
                    series = sorted(
                        (r for r in regime if r["method"] == method), key=lambda r: r["strength"]
                    )
                    values = [r[f"{metric}_mean"] for r in series]
                    strengths = [r["strength"] for r in series]
                    ax.plot(
                        strengths, values, color=colors[method], linestyle=linestyle,
                        marker="o", markersize=4,
                    )
            if metric == "roc_auc":
                ax.set_ylim(0.48, 1.02)
                ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            elif metric == "f1":
                ax.set_ylim(-0.04, 1.04)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xscale("log")
            ax.set_xticks(config.strengths, [f"{v:g}" for v in config.strengths])
            ax.set_xlabel("Regularization strength")
            ax.set_title(f"{label}   ·   n = {count}")
            ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    handles = [Line2D([], [], color=color, label=method) for method, color in colors.items()]
    handles += [
        Line2D([], [], color="#202830", linestyle=style, label=f"noise {sigma:.0%}")
        for sigma, style in noise_styles.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    fig.suptitle("Surrogate operators and symmetry detection", fontsize=15)
    fig.savefig(path, dpi=160)
    plt.close(fig)
