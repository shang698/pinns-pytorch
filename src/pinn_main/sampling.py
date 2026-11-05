import numpy as np


def latin_hypercube(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Two-dimensional Latin hypercube with pyDOE sampling order."""
    cut = np.linspace(0, 1, n + 1)
    uniforms = rng.rand(n, 2)
    stratified = cut[:-1, None] + uniforms * np.diff(cut)[:, None]
    result = np.empty_like(stratified)
    for column in range(2):
        result[:, column] = stratified[rng.permutation(n), column]
    return result
