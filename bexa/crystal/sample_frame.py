"""Sample-to-grain transformation (port of ``Sample2Grain.m``).

``sample_to_grain`` returns the matrix that maps a vector expressed in the
sample (lab) frame onto the grain's Cartesian frame, for a crystal whose
surface normal and in-plane directions are given as direct-lattice indices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bexa.crystal.lattice import LatticeParameters

__all__ = ["lattice_vectors", "sample_to_grain"]

LEVI_CIVITA_SIGN = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])


def lattice_vectors(lattice: LatticeParameters, decimals: int = 12) -> np.ndarray:
    """Columns ``a, b, c`` of the cell in a Cartesian frame with ``a`` along x, ``b`` in xy."""
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = np.radians([lattice.alpha, lattice.beta, lattice.gamma])
    volume = (
        a
        * b
        * c
        * np.sqrt(
            1
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
    )
    vectors = np.array(
        [
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * np.sin(gamma), 0.0],
            [
                c * np.cos(beta),
                c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                volume / (a * b * np.sin(gamma)),
            ],
        ]
    ).T
    return np.round(vectors, decimals)


def sample_to_grain(lattice: LatticeParameters, surface_vectors: Any) -> np.ndarray:
    """Matrix converting sample-frame vectors into grain-frame Cartesian vectors.

    Parameters
    ----------
    lattice
        Cell parameters of the grain.
    surface_vectors
        ``3 x 3`` array whose columns are the lab x, y and z directions written
        as direct-lattice indices ``[u, v, w]``. Exactly one column may be all
        zeros; it is completed as the cross product of the other two (with the
        sign that keeps the frame right-handed). Example: a (110) surface with
        the y axis along [1 -1 1] is ``[[0, 1, 1], [0, -1, 1], [0, 1, 0]]``.
    """
    L = lattice_vectors(lattice)
    S = np.asarray(surface_vectors, dtype=float)
    if S.shape != (3, 3):
        raise ValueError("surface_vectors must be a 3 x 3 array of column vectors")
    given = [j for j in range(3) if np.any(S[:, j] != 0)]
    missing = [j for j in range(3) if j not in given]
    if len(given) < 2:
        raise ValueError("at least two lab directions must be given")
    R = np.linalg.inv(L) @ S
    if missing:
        i1, i2 = given[0], given[1]
        R[:, missing[0]] = L.T @ np.cross(R[:, i1], R[:, i2]) * LEVI_CIVITA_SIGN[i1, i2]
    R = R / np.linalg.norm(R, axis=0)
    return L @ R
