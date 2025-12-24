"""
Matrix construction utilities for X-ray tomography reconstruction.
Implements ray-path matrix A and derivative operators (Dx, Dy, Dz).
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional


def build_toy_ray_path_matrix(M: int = 5, N: int = 5, 
                               delta_x: float = 1.0, 
                               delta_y: float = 1.0) -> np.ndarray:
    """
    Build the ray-path matrix A for the toy problem (5x5 grid).
    
    According to Figure 1:
    - 7 sources (numbered counter-clockwise)
    - 5 receivers (numbered counter-clockwise)
    - Each source-receiver pair creates one measurement
    
    Args:
        M: Number of rows in grid
        N: Number of columns in grid
        delta_x: Grid spacing in x direction
        delta_y: Grid spacing in y direction
    
    Returns:
        A: Ray-path matrix (num_measurements × M*N)
    """
    # For the toy problem: 7 sources × 5 receivers = 35 measurements
    num_sources = 7
    num_receivers = 5
    num_measurements = num_sources * num_receivers
    num_unknowns = M * N
    
    A = np.zeros((num_measurements, num_unknowns))
    
    # Define source and receiver positions
    # Sources (counter-clockwise): positions 1,2 on left, 3 on bottom-left, 4 on bottom,
    # 5 on bottom-right, 6,7 on top
    # Receivers: 1 on left-middle, 2 on bottom-middle, 3,4,5 on right side
    
    # This needs to be filled based on Figure 1 geometry
    # Each row of A represents one ray path
    # Element A[i,j] = distance traversed in cell j by ray i
    
    # Calculate diagonal distance
    delta_diag = np.sqrt(delta_x**2 + delta_y**2)
    
    # TODO: Implement based on Figure 1 geometry
    # For each source-receiver pair:
    #   - Determine which cells the ray passes through
    #   - Calculate distance in each cell (delta_x, delta_y, or delta_diag)
    #   - Fill corresponding row of A
    
    print("WARNING: Ray-path matrix needs manual implementation based on Figure 1")
    print(f"Matrix shape: {A.shape}")
    
    return A


def build_derivative_matrix_2d(M: int, N: int, 
                               direction: str = 'x') -> sparse.csr_matrix:
    """
    Build finite difference derivative matrix for 2D grid using forward differences.
    
    For vertical direction (y):
        (∂_y X)_{i,j} = X_{i+1,j} - X_{i,j}  for i ∈ {1,...,M-1}
        Derivative at i=M is assumed zero
    
    For horizontal direction (x):
        (∂_x X)_{i,j} = X_{i,j+1} - X_{i,j}  for j ∈ {1,...,N-1}
        Derivative at j=N is assumed zero
    
    Args:
        M: Number of rows in grid
        N: Number of columns in grid
        direction: 'x' for horizontal derivative, 'y' for vertical derivative
    
    Returns:
        D: Sparse derivative matrix (MN × MN)
    """
    n = M * N
    
    if direction == 'y':
        # Vertical derivative: differentiate along rows
        # For column-stacked representation: x = [X11, X21, X31, ..., X12, X22, ...]
        # X_{i+1,j} is at position (i+1) + j*M  (0-indexed: i + j*M)
        # X_{i,j} is at position i + j*M
        
        diagonals = []
        offsets = []
        
        for j in range(N):  # For each column
            for i in range(M - 1):  # For each row except last
                row_idx = i + j * M
                # D[row_idx, row_idx] = -1
                # D[row_idx, row_idx + 1] = 1
                diagonals.append(np.ones(1))
                offsets.append((row_idx, row_idx))
                diagonals.append(np.ones(1))
                offsets.append((row_idx, row_idx + 1))
        
        # More efficient: use diags
        main_diag = -np.ones(n)
        upper_diag = np.zeros(n - 1)
        
        # Set upper diagonal to 1 where appropriate
        for j in range(N):
            for i in range(M - 1):
                idx = i + j * M
                upper_diag[idx] = 1.0
        
        D = sparse.diags([main_diag, upper_diag], [0, 1], shape=(n, n), format='csr')
        
        # Zero out rows corresponding to i=M-1 (last row of each column)
        for j in range(N):
            row_idx = (M - 1) + j * M
            D[row_idx, :] = 0
    
    elif direction == 'x':
        # Horizontal derivative: differentiate along columns
        # X_{i,j+1} is at position i + (j+1)*M
        # X_{i,j} is at position i + j*M
        
        main_diag = -np.ones(n)
        upper_diag = np.zeros(n - M)
        upper_diag[:] = 1.0
        
        D = sparse.diags([main_diag, upper_diag], [0, M], shape=(n, n), format='csr')
        
        # Zero out rows corresponding to j=N-1 (last column)
        for i in range(M):
            row_idx = i + (N - 1) * M
            D[row_idx, :] = 0
    
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'x' or 'y'")
    
    return D


def build_derivative_matrix_3d(n: int, direction: str) -> sparse.csr_matrix:
    """
    Build finite difference derivative matrix for 3D grid (n×n×n voxels).
    
    Column-stack convention: stack slices along z direction
    x = [X111, X211, ..., Xn11, X121, ..., Xnn1, X112, X212, ..., Xnnn]
    
    For a voxel at (i,j,k):
        - Linear index: i + j*n + k*n²  (0-indexed: i-1 + (j-1)*n + (k-1)*n²)
    
    Args:
        n: Grid dimension (n×n×n)
        direction: 'x', 'y', or 'z'
    
    Returns:
        D: Sparse derivative matrix (n³ × n³)
    """
    N = n ** 3
    
    if direction == 'x':
        # ∂_x: X_{i,j+1,k} - X_{i,j,k}
        # Indices: (i + (j+1)*n + k*n²) - (i + j*n + k*n²) = n apart
        main_diag = -np.ones(N)
        upper_diag = np.ones(N - n)
        
        D = sparse.diags([main_diag, upper_diag], [0, n], shape=(N, N), format='csr')
        
        # Zero out boundary (j = n-1)
        for k in range(n):
            for i in range(n):
                row_idx = i + (n - 1) * n + k * n * n
                D[row_idx, :] = 0
    
    elif direction == 'y':
        # ∂_y: X_{i+1,j,k} - X_{i,j,k}
        # Indices: (i+1 + j*n + k*n²) - (i + j*n + k*n²) = 1 apart
        main_diag = -np.ones(N)
        upper_diag = np.zeros(N - 1)
        
        # Set upper diagonal, but skip boundaries
        for k in range(n):
            for j in range(n):
                for i in range(n - 1):
                    idx = i + j * n + k * n * n
                    upper_diag[idx] = 1.0
        
        D = sparse.diags([main_diag, upper_diag], [0, 1], shape=(N, N), format='csr')
        
        # Zero out boundary rows (i = n-1)
        for k in range(n):
            for j in range(n):
                row_idx = (n - 1) + j * n + k * n * n
                D[row_idx, :] = 0
    
    elif direction == 'z':
        # ∂_z: X_{i,j,k+1} - X_{i,j,k}
        # Indices: (i + j*n + (k+1)*n²) - (i + j*n + k*n²) = n² apart
        main_diag = -np.ones(N)
        upper_diag = np.ones(N - n * n)
        
        D = sparse.diags([main_diag, upper_diag], [0, n * n], shape=(N, N), format='csr')
        
        # Zero out boundary (k = n-1)
        for j in range(n):
            for i in range(n):
                row_idx = i + j * n + (n - 1) * n * n
                D[row_idx, :] = 0
    
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'x', 'y', or 'z'")
    
    return D


def compute_gradient_magnitude(Dx_x: np.ndarray, Dy_x: np.ndarray, 
                               M: int, N: int) -> np.ndarray:
    """
    Compute gradient magnitude: G_ij = sqrt((∂_x X)²_ij + (∂_y X)²_ij)
    
    Args:
        Dx_x: Result of Dx @ x (column-stacked)
        Dy_x: Result of Dy @ x (column-stacked)
        M: Number of rows
        N: Number of columns
    
    Returns:
        G: Gradient magnitude as 2D array (M×N)
    """
    # Reshape to 2D
    grad_x = Dx_x.reshape(M, N)
    grad_y = Dy_x.reshape(M, N)
    
    # Compute magnitude
    G = np.sqrt(grad_x**2 + grad_y**2)
    
    return G


def build_combined_derivative_matrix(M: int, N: int, 
                                    dimensions: int = 2) -> sparse.csr_matrix:
    """
    Build combined derivative matrix L = [Dx; Dy] or L = [Dx; Dy; Dz].
    
    Args:
        M: Grid dimension (for 2D: M×N, for 3D: M=N=n)
        N: Grid dimension (for 2D: M×N, for 3D: ignored, uses M)
        dimensions: 2 or 3
    
    Returns:
        L: Stacked derivative matrix
    """
    if dimensions == 2:
        Dx = build_derivative_matrix_2d(M, N, 'x')
        Dy = build_derivative_matrix_2d(M, N, 'y')
        L = sparse.vstack([Dx, Dy], format='csr')
    
    elif dimensions == 3:
        n = M  # Assume cubic grid
        Dx = build_derivative_matrix_3d(n, 'x')
        Dy = build_derivative_matrix_3d(n, 'y')
        Dz = build_derivative_matrix_3d(n, 'z')
        L = sparse.vstack([Dx, Dy, Dz], format='csr')
    
    else:
        raise ValueError("dimensions must be 2 or 3")
    
    return L
