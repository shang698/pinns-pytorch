"""Train the Schrodinger or Allen-Cahn PINN."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import scipy
from scipy.optimize import minimize
import torch

from .config import load_config, write_json
from .data import ROOT
from .problems import Problem
from .optim import TF1Adam, ScipyObjective
from .plots import make_plots


def run(config, output, device="auto", warm_start=None):
    config = dict(config)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.set_num_threads(config["threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    problem = Problem(config, device)
    model = problem.model
    if warm_start:
        saved = torch.load(warm_start, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
    adam = TF1Adam(model.parameters(), lr=config["learning_rate"])

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": device,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
        "warm_start_sha256": hashlib.sha256(Path(warm_start).read_bytes()).hexdigest()
        if warm_start
        else None,
    }
    write_json(output / "environment.json", environment)
    np.savez_compressed(output / "training_points.npz", **problem.arrays)
    print(f"{config['model']} | {config['dtype']} | {device}", flush=True)
    print(f"Output: {output}", flush=True)
    start = time.perf_counter()

    def checkpoint(name):
        torch.save({"model": model.state_dict(), "config": config}, output / name)

    def gradient_max():
        return float(torch.stack([p.grad.detach().abs().max() for p in model.parameters()]).max())

    with (output / "history.csv").open("w", newline="") as history:
        writer = csv.DictWriter(
            history,
            fieldnames=[
                "phase",
                "step",
                "elapsed_seconds",
                "loss",
                *problem.loss_keys,
                "gradient_max",
            ],
        )
        writer.writeheader()

        def log(phase, step, losses, gradmax):
            elapsed = time.perf_counter() - start
            writer.writerow(
                dict(
                    phase=phase, step=step, elapsed_seconds=elapsed, gradient_max=gradmax, **losses
                )
            )
            history.flush()
            print(
                f"{phase:12s} {step:6d}  loss={losses['loss']:.6e}  "
                f"grad={gradmax:.3e}  time={elapsed:.1f}s",
                flush=True,
            )

        losses = problem.backward()
        log("initial", 0, losses, gradient_max())
        checkpoint("initial.pt")
        for step in range(1, config["adam_steps"] + 1):
            losses = problem.backward()
            gradmax = gradient_max()
            adam.step()
            if step == 1 or step % config["log_every"] == 0 or step == config["adam_steps"]:
                # Adam losses are recorded before the parameter update.
                log("adam_pre", step, losses, gradmax)
            if step % config["checkpoint_every"] == 0:
                checkpoint("checkpoint.pt")
        checkpoint("before_lbfgs.pt")

        def on_eval(nfev, losses, gradmax):
            if nfev == 1 or nfev % config["log_every"] == 0:
                log("lbfgs_eval", nfev, losses, gradmax)

        objective = ScipyObjective(problem, on_eval)
        iterations = 0

        def on_iteration(values):
            nonlocal iterations
            objective.unpack(values)
            iterations += 1
            if iterations % config["checkpoint_every"] == 0:
                checkpoint("checkpoint.pt")

        optimizer_info = {"message": "L-BFGS not requested"}
        if config["lbfgs_maxiter"] > 0 and config["lbfgs_maxfun"] > 0:
            result = minimize(
                objective,
                objective.pack(),
                method="L-BFGS-B",
                jac=True,
                callback=on_iteration,
                options=dict(
                    maxiter=config["lbfgs_maxiter"],
                    maxfun=config["lbfgs_maxfun"],
                    maxcor=config["lbfgs_history"],
                    maxls=config["lbfgs_maxls"],
                    ftol=config["lbfgs_ftol"],
                    gtol=config["lbfgs_gtol"],
                ),
            )
            objective.unpack(result.x)
            optimizer_info = {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nit_this_run": int(result.nit),
                "nfev_this_run": int(result.nfev),
            }
        final_losses = problem.backward()
        log("final", iterations, final_losses, gradient_max())
    training_seconds = time.perf_counter() - start
    checkpoint("model.pt")
    metrics, predictions = problem.evaluate()
    metrics.update(
        status="completed",
        model=config["model"],
        protocol=config["protocol"],
        optimizer=optimizer_info,
        training_seconds=training_seconds,
        adam_steps_completed=config["adam_steps"],
        lbfgs_iterations_completed=iterations,
        lbfgs_function_evaluations=objective.nfev,
    )
    np.savez_compressed(output / "predictions.npz", **predictions)
    make_plots(problem, predictions, output)
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warm-start", help="Load weights into a fresh optimizer")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("dtype", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    run(config, args.output, args.device, args.warm_start)


if __name__ == "__main__":
    main()
