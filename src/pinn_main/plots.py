import csv
import os
import numpy as np


def make_plots(problem, saved, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    if problem.config["model"] == "schrodinger":
        ref, pred = abs(saved["reference"]), abs(saved["prediction"])
        extent = [saved["t"][0], saved["t"][-1], -5, 5]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for ax, values, title in zip(
            axes,
            [ref, pred, abs(pred - ref)],
            ["Reference |h|", "Predicted |h|", "Amplitude error"],
        ):
            options = {} if title == "Amplitude error" else dict(vmin=ref.min(), vmax=ref.max())
            im = ax.imshow(
                values.T, origin="lower", aspect="auto", extent=extent, cmap="magma", **options
            )
            ax.set(xlabel="Time t", ylabel="Position x", title=title)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(output / "solution.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
        for ax, fraction in zip(axes, [0.25, 0.5, 0.75]):
            i = int(round(fraction * (len(saved["t"]) - 1)))
            ax.plot(saved["x"], ref[i], label="Reference")
            ax.plot(saved["x"], pred[i], "--", label="PINN")
            ax.set(xlabel="Position x", ylabel="|h|", title=f"t = {saved['t'][i]:.3f}")
            ax.legend(frameon=False)
        fig.savefig(output / "slices.png", dpi=160)
        plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        i0 = problem.config["initial_time_index"]
        axes[0].plot(saved["x"], problem.reference.solution[i0].real, label="Reference")
        axes[0].scatter(
            problem.arrays["initial_points"],
            problem.arrays["initial_values"],
            s=10,
            c="tab:orange",
            label="Training data",
        )
        axes[1].plot(saved["x"], saved["reference"], label="Reference")
        axes[1].plot(saved["x"], saved["prediction"], "--", label="PINN")
        for ax, t in zip(axes, [saved["initial_time"], saved["target_time"]]):
            ax.set(xlabel="Position x", ylabel="u", title=f"t = {t:.2f}")
            ax.legend(frameon=False)
        fig.savefig(output / "solution.png", dpi=160)
        plt.close(fig)
    with (output / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for key in ["loss", *problem.loss_keys]:
        values = np.array([float(r[key]) for r in rows])
        if np.any(values > 0):
            ax.semilogy(
                [float(r["elapsed_seconds"]) for r in rows], np.maximum(values, 1e-30), label=key
            )
    ax.set(
        xlabel="Training time (s)",
        ylabel="Loss",
        title="Training history (includes L-BFGS line-search trials)",
    )
    ax.legend(frameon=False)
    fig.savefig(output / "loss.png", dpi=160)
    plt.close(fig)
