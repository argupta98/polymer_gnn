from rdkit.SimDivFilters import rdSimDivPickers
from typing import Optional
import time
import numpy as np


def get_max_min_picks(
    vectors: np.ndarray,
    N_picks: Optional[int] = None,
    threshold: Optional[float] = None,
    picker: str = "MaxMin",
) -> list:
    """
    Picks the most diverse points from a set of vectors

    :param vectors: set of vectors
    :param N_picks: number of points to pick
    :param threshold: threshold for leader picker
    :param picker: maxmin or leader
    :return: list of indices of points picked
    """

    start = time.time()
    # Sphere exclusion picker
    # picks: cluster centroids that are a minimum of threshold apart
    if picker == "Leader":
        P = rdSimDivPickers.LeaderPicker()
        if threshold is None:
            raise ValueError("threshold must be specified for leader picker")
    elif picker == "MaxMin":
        P = rdSimDivPickers.MaxMinPicker()
        if N_picks is None:
            raise ValueError("N_picks must be specified for maxmin picker")
    else:
        raise ValueError(f"picker must be `MaxMin` or `Leader`, got {picker}")

    def fn(i, j, fps=vectors):
        return float(np.linalg.norm(fps[i] - fps[j]))

    print(f"Picker: {picker}")
    print(f"Shape: {vectors.shape}")
    print(f"Pool size: {len(vectors)}")

    # For MaxMin picking cases we pass, distFunc, poolSize, and pickSize
    # For Leader picking cases we pass, distFunc, poolSize, and threshold
    if picker == "MaxMin":
        picks = P.LazyPick(fn, len(vectors), N_picks)
    elif picker == "Leader":
        picks = P.LazyPick(fn, len(vectors), threshold)

    end = time.time()

    elapsed = end - start
    print(f"Elapsed: {elapsed / 60}")
    print(f"Picks count: {len(picks)}")

    return picks
