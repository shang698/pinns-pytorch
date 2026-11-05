#!/usr/bin/env python3
"""Check the bundled data and IRK coefficients."""

import hashlib
import json

import numpy as np
from pinn_main.data import ROOT, load_reference, load_irk


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
    for model, nx, bounds, end in [
        ("schrodinger", 256, (-5, 5), np.pi / 2),
        ("allen_cahn", 512, (-1, 1), 1.0),
    ]:
        ref = load_reference(model)
        np.testing.assert_allclose(ref.x, np.linspace(*bounds, nx, endpoint=False), atol=1e-12)
        np.testing.assert_allclose(ref.t, np.linspace(0, end, 201), atol=1e-12)
        print(f"{model}: {len(ref.x)} spatial points, {len(ref.t)} snapshots")
    weights, nodes = load_irk()
    np.testing.assert_allclose(weights[:-1].sum(axis=1), nodes, atol=1e-12, rtol=0)
    np.testing.assert_allclose(weights[-1].sum(), 1, atol=1e-12, rtol=0)
    print(f"IRK: {len(nodes)} stages; {len(manifest['files'])} file checksums matched")


if __name__ == "__main__":
    main()
