#!/usr/bin/env python3
"""Recompute errors from saved model weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from pinn_main.problems import Problem


def evaluate(folder):
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    config = saved["config"]
    torch.set_num_threads(config["threads"])
    problem = Problem(config, "cpu")
    problem.model.load_state_dict(saved["model"])
    metrics, prediction = problem.evaluate()
    original = json.loads((folder / "metrics.json").read_text())
    with np.load(folder / "predictions.npz") as arrays:
        np.testing.assert_array_equal(prediction["reference"], arrays["reference"])
        np.testing.assert_allclose(
            prediction["prediction"], arrays["prediction"], rtol=1e-5, atol=3e-6
        )
    with np.load(folder / "training_points.npz") as arrays:
        for key, values in problem.arrays.items():
            np.testing.assert_array_equal(values, arrays[key])
    for key, value in metrics.items():
        np.testing.assert_allclose(value, original[key], rtol=1e-4, atol=1e-8)
    return {"run": folder.name, "model": config["model"], "dtype": config["dtype"], **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--reference", action="store_true", help="Check all six included runs")
    args = parser.parse_args()
    if bool(args.run) == args.reference:
        parser.error("Give one run folder or --reference")
    if not args.reference:
        print(json.dumps(evaluate(args.run), indent=2))
        return
    for model in ("schrodinger", "allen_cahn"):
        folder = ROOT / "results/reference" / model
        baseline = torch.load(folder / "baseline/model.pt", map_location="cpu", weights_only=True)
        for phase in ("baseline", "float32", "float64"):
            if phase != "baseline":
                initial = torch.load(
                    folder / phase / "initial.pt", map_location="cpu", weights_only=True
                )
                for key, value in initial["model"].items():
                    torch.testing.assert_close(
                        value, baseline["model"][key].to(value.dtype), rtol=0, atol=0
                    )
            print(json.dumps(evaluate(folder / phase), indent=2))


if __name__ == "__main__":
    main()
