#!/usr/bin/env python3
"""Plot the saved solutions and precision comparison."""

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#8B96A2", "#2568A2", "#CE6A2B"]
MODELS = {"schrodinger": "Schrödinger", "allen_cahn": "Allen–Cahn"}
PHASES = ("baseline", "float32", "float64")


def load_metrics(folder):
    return json.loads((folder / "metrics.json").read_text())


def save_figure(fig, output, name):
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_schrodinger(folder, output):
    with np.load(folder / "predictions.npz") as saved:
        x, t = saved["x"], saved["t"]
        reference, prediction = saved["reference"], saved["prediction"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.7), layout="constrained")
    vmax = max(abs(reference).max(), abs(prediction).max())
    panels = [
        (abs(reference), "Reference |h|", "viridis", vmax),
        (abs(prediction), "PyTorch baseline |h|", "viridis", vmax),
        (abs(prediction - reference), "Complex error |h - h_ref|", "magma", None),
    ]
    for ax, (values, title, cmap, upper) in zip(axes[0], panels):
        image = ax.pcolormesh(
            t, x, values.T, shading="nearest", cmap=cmap, vmin=0, vmax=upper, rasterized=True
        )
        ax.set(xlabel="t", ylabel="x", title=title)
        fig.colorbar(image, ax=ax, pad=0.015, shrink=0.86)
    for ax, index in zip(axes[1], (75, 100, 125)):
        ax.plot(x, abs(reference[index]), color="#202B36", lw=1.6, label="Reference")
        ax.plot(x, abs(prediction[index]), color=COLORS[1], lw=1.7, ls="--", label="Baseline")
        ax.set(xlabel="x", ylabel="|h|", title=f"t = {t[index]:.3f}")
        ax.grid(alpha=0.16)
    axes[1, 0].legend(frameon=False, fontsize=11)
    save_figure(fig, output, "nls_baseline")


def plot_allen_cahn(folder, output):
    with np.load(folder / "predictions.npz") as saved:
        x = saved["x"]
        reference, prediction = saved["reference"], saved["prediction"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.5), layout="constrained")
    axes[0].plot(x, reference, color="#202B36", lw=1.6, label="Reference")
    axes[0].plot(x, prediction, color=COLORS[1], lw=1.7, ls="--", label="Baseline")
    axes[0].set(xlabel="x", ylabel="u", title="Target snapshot, t = 0.9")
    axes[0].legend(frameon=False, fontsize=11, loc="lower right")
    axes[1].plot(x, abs(prediction - reference), color=COLORS[2], lw=1.6)
    axes[1].set(xlabel="x", ylabel="Absolute error", title="Pointwise prediction error")
    for ax in axes:
        ax.grid(alpha=0.16)
    save_figure(fig, output, "ac_baseline")


def plot_precision_errors(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1))
    labels = ("Baseline", "float32 continuation", "float64 continuation")
    for ax, (model, title) in zip(axes, MODELS.items()):
        records = [load_metrics(results / model / phase) for phase in PHASES]
        if model == "schrodinger":
            keys = ("relative_l2_amplitude", "relative_l2_complex")
            for index, record in enumerate(records):
                errors = [100 * record[key] for key in keys]
                bars = ax.bar(
                    np.arange(2) + (index - 1) * 0.25,
                    errors,
                    width=0.23,
                    color=COLORS[index],
                    label=labels[index],
                )
                ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=10)
            ax.set_xticks([0, 1], ["Amplitude |h|", "Complex h"])
            ax.set_ylim(0, 0.18)
        else:
            errors = [100 * record["relative_l2_target_snapshot"] for record in records]
            bars = ax.bar(np.arange(3), errors, color=COLORS, width=0.58)
            ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=11)
            ax.set_xticks([0, 1, 2], ["Baseline", "float32", "float64"])
            ax.set_ylim(0, 0.48)
        ax.set(title=title, ylabel="Relative L2 error (%)")
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03)
    )
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.075, right=0.985, wspace=0.26)
    save_figure(fig, output, "precision_errors")


def plot_precision_loss(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), layout="constrained")
    for ax, (model, title) in zip(axes, MODELS.items()):
        iterations = {}
        for index, dtype in enumerate(("float32", "float64"), 1):
            folder = results / model / dtype
            metrics = load_metrics(folder)
            iterations[dtype] = metrics["optimizer"]["nit_this_run"]
            with (folder / "history.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            initial_loss = float(rows[0]["loss"])
            points = [(0, initial_loss)]
            points += [
                (int(row["step"]), float(row["loss"]))
                for row in rows
                if row["phase"] == "lbfgs_eval"
            ]
            points.append((metrics["optimizer"]["nfev_this_run"], float(rows[-1]["loss"])))
            evaluations, losses = np.array(points).T
            ax.plot(
                evaluations,
                losses / initial_loss,
                color=COLORS[index],
                lw=1.8,
                ls="--" if dtype == "float32" else "-",
                label=dtype,
            )
            ax.scatter(
                evaluations[-1],
                losses[-1] / initial_loss,
                facecolors="white" if dtype == "float32" else COLORS[index],
                edgecolors=COLORS[index],
                marker="s" if dtype == "float32" else "o",
                s=36,
                zorder=4,
            )
        annotation = (
            f"float32: {iterations['float32']:,} iterations\n"
            f"float64: {iterations['float64']:,} iterations"
        )
        ax.text(
            0.04,
            0.1,
            annotation,
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
        )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 10, 100, 1000, 5000], ["0", "1", "10", "100", "1000", "5000"])
        ax.tick_params(axis="x", labelsize=10)
        ax.set(
            xlim=(-0.15, 7000),
            ylim=(0.7, 1.04),
            title=title,
            xlabel="Additional objective evaluations",
            ylabel="Loss / starting loss",
        )
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, loc="upper right", fontsize=11)
    save_figure(fig, output, "precision_loss")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/reference")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )
    plot_schrodinger(args.results / "schrodinger/baseline", args.output)
    plot_allen_cahn(args.results / "allen_cahn/baseline", args.output)
    plot_precision_errors(args.results, args.output)
    plot_precision_loss(args.results, args.output)
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
