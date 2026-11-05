#!/usr/bin/env python3
"""Continue one checkpoint in float32 and float64."""

import argparse
import json
from pathlib import Path

import torch
from pinn_main.data import ROOT
from pinn_main.train import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--maxiter", type=int)
    args = parser.parse_args()
    if args.maxiter is not None and args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if any((args.output / dtype).exists() for dtype in ("float32", "float64")):
        parser.error("Choose an output directory without existing continuation runs")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(saved["config"])
    settings = json.loads((ROOT / f"configs/{config['model']}_precision_restart.json").read_text())
    settings.pop("extends")
    config.update(settings)
    if args.maxiter is not None:
        config.update(lbfgs_maxiter=args.maxiter, lbfgs_maxfun=max(7500, int(1.5 * args.maxiter)))
    for dtype in ("float32", "float64"):
        run(dict(config, dtype=dtype), args.output / dtype, args.device, warm_start=args.checkpoint)


if __name__ == "__main__":
    main()
