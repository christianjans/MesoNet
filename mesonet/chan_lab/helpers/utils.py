import argparse
import yaml

import numpy as np
from typing import Tuple
import scipy


def config_to_namespace(config: str) -> argparse.Namespace:
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


# A translation to Python of the MATLAB code function, reorderMAT, available in
# https://sites.google.com/site/bctnet/home?authuser=0.
def reorder_matrix(
    matrix: np.array,
    h: int = 10000,
    cost: str = "line"
) -> Tuple[np.array, np.array]:
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]

    n = matrix.shape[0]
    diag = np.diag(np.diag(matrix))
    matrix = matrix - diag

    if cost == "line":
        profile = scipy.stats.norm.pdf(range(n), 0, n / 2)[::-1]
    elif cost == "circ":
        profile = scipy.stats.norm.pdf(range(n), n / 2, n / 4)[::-1]
    else:
        raise ValueError(f"Unrecognized cost: {cost}")

    cost = scipy.linalg.toeplitz(profile, profile)

    low_matrix_cost = np.sum(cost * matrix)

    starting_matrix = matrix
    start_a = np.array([i for i in range(n)])

    for _ in range(h):
        a = np.array([i for i in range(n)])
        r = np.random.permutation(n)
        a[r[0]] = r[1]
        a[r[1]] = r[0]
        new_matrix_cost = np.sum(matrix[a, :][:, a] * cost)
        if (new_matrix_cost < low_matrix_cost):
            matrix = matrix[a, :][:, a]
            r0, r1 = start_a[r[0]], start_a[r[1]]
            start_a[r[0]], start_a[r[1]] = r1, r0
            low_matrix_cost = new_matrix_cost

    matrix_reordered = \
        starting_matrix[start_a, :][:, start_a] + diag[start_a, :][:, start_a]

    return matrix_reordered, start_a
