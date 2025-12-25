import numpy as np
from scipy import sparse
from typing import Tuple, Optional


def build_toy_ray_path_matrix(M: int = 5, N: int = 5,
                               delta_x: float = 1.0,
                               delta_y: float = 1.0) -> np.ndarray:
    """Build ray-path matrix A for toy problem based on Figure 1 geometry.

    Returns 8x25 matrix representing 8 ray paths through a 5x5 grid.
    Each row represents one ray, each column represents one grid cell.
    """
    SQRT2 = np.sqrt(2)

    A = np.array([
        [0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, 0]
    ])

    if delta_x != 1.0 or delta_y != 1.0:
        delta_diag = np.sqrt(delta_x**2 + delta_y**2)
        A = A * (delta_diag / SQRT2) * (A == SQRT2).astype(float) + A * delta_x * (A == 1).astype(float)

    return A


def build_derivative_matrix_2d(M: int, N: int,
                               direction: str = 'x') -> sparse.csr_matrix:
    """Build finite difference derivative matrix for 2D grid (forward differences)."""
    n = M * N

    if direction == 'y':
        main_diag = -np.ones(n)
        upper_diag = np.zeros(n - 1)

        for j in range(N):
            for i in range(M - 1):
                idx = i + j * M
                upper_diag[idx] = 1.0

        for j in range(N):
            row_idx = (M - 1) + j * M
            main_diag[row_idx] = 0.0

        D = sparse.diags([main_diag, upper_diag], [0, 1], shape=(n, n), format='csr')

    elif direction == 'x':
        main_diag = -np.ones(n)
        upper_diag = np.zeros(n - M)
        upper_diag[:] = 1.0

        for i in range(M):
            row_idx = i + (N - 1) * M
            main_diag[row_idx] = 0.0

        D = sparse.diags([main_diag, upper_diag], [0, M], shape=(n, n), format='csr')

    else:
        raise ValueError(f"Invalid direction: {direction}")

    return D


def build_derivative_matrix_3d(n: int, direction: str) -> sparse.csr_matrix:
    """Build finite difference derivative matrix for 3D grid (n x n x n)."""
    N = n ** 3

    if direction == 'x':
        main_diag = -np.ones(N)
        upper_diag = np.ones(N - n)

        for k in range(n):
            for i in range(n):
                row_idx = i + (n - 1) * n + k * n * n
                main_diag[row_idx] = 0.0

        D = sparse.diags([main_diag, upper_diag], [0, n], shape=(N, N), format='csr')

    elif direction == 'y':
        main_diag = -np.ones(N)
        upper_diag = np.zeros(N - 1)

        for k in range(n):
            for j in range(n):
                for i in range(n - 1):
                    idx = i + j * n + k * n * n
                    upper_diag[idx] = 1.0

        for k in range(n):
            for j in range(n):
                row_idx = (n - 1) + j * n + k * n * n
                main_diag[row_idx] = 0.0

        D = sparse.diags([main_diag, upper_diag], [0, 1], shape=(N, N), format='csr')

    elif direction == 'z':
        main_diag = -np.ones(N)
        upper_diag = np.ones(N - n * n)

        for j in range(n):
            for i in range(n):
                row_idx = i + j * n + (n - 1) * n * n
                main_diag[row_idx] = 0.0

        D = sparse.diags([main_diag, upper_diag], [0, n * n], shape=(N, N), format='csr')

    else:
        raise ValueError(f"Invalid direction: {direction}")

    return D


def compute_gradient_magnitude(Dx_x: np.ndarray, Dy_x: np.ndarray,
                               M: int, N: int) -> np.ndarray:
    """Compute gradient magnitude: G = sqrt((Dx)^2 + (Dy)^2)."""
    grad_x = Dx_x.reshape(M, N)
    grad_y = Dy_x.reshape(M, N)
    G = np.sqrt(grad_x**2 + grad_y**2)
    return G


def build_combined_derivative_matrix(M: int, N: int,
                                    dimensions: int = 2) -> sparse.csr_matrix:
    """Build combined derivative matrix L = [Dx; Dy] or L = [Dx; Dy; Dz]."""
    if dimensions == 2:
        Dx = build_derivative_matrix_2d(M, N, 'x')
        Dy = build_derivative_matrix_2d(M, N, 'y')
        L = sparse.vstack([Dx, Dy], format='csr')
    elif dimensions == 3:
        n = M
        Dx = build_derivative_matrix_3d(n, 'x')
        Dy = build_derivative_matrix_3d(n, 'y')
        Dz = build_derivative_matrix_3d(n, 'z')
        L = sparse.vstack([Dx, Dy, Dz], format='csr')
    else:
        raise ValueError("dimensions must be 2 or 3")

    return L
