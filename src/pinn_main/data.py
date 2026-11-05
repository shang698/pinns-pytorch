from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .sampling import latin_hypercube

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "third_party/raissi"


@dataclass
class Reference:
    x: np.ndarray
    t: np.ndarray
    solution: np.ndarray


def load_reference(model):
    filename = {"schrodinger": "NLS.mat", "allen_cahn": "AC.mat"}[model]
    mat = loadmat(OFFICIAL / "main/Data" / filename)
    x, t, solution = mat["x"].ravel(), mat["tt"].ravel(), mat["uu"].T
    if solution.shape != (len(t), len(x)) or not np.isfinite(solution).all():
        raise ValueError("Invalid reference solution")
    return Reference(x, t, solution)


def load_irk(q=100):
    values = np.loadtxt(OFFICIAL / f"Utilities/IRK_weights/Butcher_IRK{q}.txt").ravel()
    if values.size != q * q + 2 * q:
        raise ValueError("Unexpected IRK coefficient count")
    weights = values[: q * (q + 1)].reshape(q + 1, q)
    nodes = values[q * (q + 1) :]
    return weights, nodes


def sample_training(reference, config):
    rng = np.random.RandomState(config["seed"])
    ids = rng.choice(len(reference.x), config["n_initial"], replace=False)
    if config["model"] == "schrodinger":
        # Draw x, then t, then residual points to match the reference seed.
        time_ids = rng.choice(len(reference.t), config["n_boundary"], replace=False)
        initial = reference.solution[0, ids]
        lower, upper = np.array([-5.0, 0.0]), np.array([5.0, np.pi / 2])
        residual = lower + (upper - lower) * latin_hypercube(config["n_residual"], rng)
        return {
            "initial_points": np.column_stack((reference.x[ids], np.zeros(len(ids)))),
            "initial_values": np.column_stack((initial.real, initial.imag)),
            "boundary_times": reference.t[time_ids, None],
            "residual_points": residual,
            "initial_indices": ids,
            "boundary_time_indices": time_ids,
        }
    i0, i1 = config["initial_time_index"], config["target_time_index"]
    if not 0 <= i0 < i1 < len(reference.t):
        raise ValueError("Require initial time < target time")
    return {
        "initial_points": reference.x[ids, None],
        "initial_values": reference.solution[i0, ids, None].real,
        "boundary_points": np.array([[-1.0], [1.0]]),
        "initial_indices": ids,
        "dt": np.array(reference.t[i1] - reference.t[i0]),
        "initial_time": np.array(reference.t[i0]),
        "target_time": np.array(reference.t[i1]),
    }
