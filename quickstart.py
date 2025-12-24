#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.matrix_construction import build_derivative_matrix_2d, compute_gradient_magnitude
from src.solvers import cgls
from src.data_utils import load_density_example, grid_to_vector, vector_to_grid
from src.visualization import visualize_2d_field, visualize_gradients


def example_derivatives():
    """Example 1: Derivative Matrices and Gradient Visualization"""
    print("="*60)
    print("Example 1: Derivatives and Gradients")
    print("="*60)

    try:
        X = load_density_example('data/X1.mat')
        print(f"\nLoaded X1.mat: {X.shape}")
    except FileNotFoundError:
        print("\nWarning: X1.mat not found. Using synthetic data.")
        X = np.random.randn(50, 50)

    M, N = X.shape

    print("Building derivative matrices...")
    Dx = build_derivative_matrix_2d(M, N, 'x')
    Dy = build_derivative_matrix_2d(M, N, 'y')

    x = grid_to_vector(X)
    Dx_x = Dx @ x
    Dy_x = Dy @ x
    G = compute_gradient_magnitude(Dx_x, Dy_x, M, N)

    print(f"Gradient magnitude range: [{G.min():.4f}, {G.max():.4f}]")
    visualize_gradients(X, Dx_x, Dy_x, G)


def example_toy_problem():
    """Example 2: Toy Problem Setup"""
    print("\n" + "="*60)
    print("Example 2: Toy Problem (5x5 grid)")
    print("="*60)

    M, N = 5, 5
    num_measurements = 35

    A = np.random.randn(num_measurements, M*N)
    x_true = np.random.rand(M*N)
    y = A @ x_true + 0.01 * np.random.randn(num_measurements)

    print(f"\nA shape: {A.shape}")
    print(f"Solving with CGLS...")

    from src.matrix_construction import build_combined_derivative_matrix
    L = build_combined_derivative_matrix(M, N, dimensions=2)
    result = cgls(A, y, L, lam=1e-5, tol=1e-6, max_iter=50)

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final objective: {result.objective_values[-1]:.6e}")

    X_recon = vector_to_grid(result.x, (M, N))
    X_true = vector_to_grid(x_true, (M, N))
    print(f"Reconstruction error: {np.linalg.norm(X_recon - X_true):.6f}")


def example_3d_reconstruction():
    """Example 3: 3D Reconstruction"""
    print("\n" + "="*60)
    print("Example 3: 3D Reconstruction")
    print("="*60)

    n = 10
    num_measurements = 300

    print(f"\nVolume: {n}^3 = {n**3} voxels")
    print(f"Measurements: {num_measurements}")

    A = np.random.randn(num_measurements, n**3)
    x_true = np.random.rand(n**3)
    y = A @ x_true + 0.01 * np.random.randn(num_measurements)

    from src.matrix_construction import build_combined_derivative_matrix
    L = build_combined_derivative_matrix(n, n, dimensions=3)

    print("Solving with CGLS...")
    result = cgls(A, y, L, lam=1e-5, tol=1e-6, max_iter=100)

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")

    X_recon = vector_to_grid(result.x, (n, n, n))
    print(f"Reconstruction shape: {X_recon.shape}")


def main():
    print("\n" + "="*60)
    print("X-RAY TOMOGRAPHY - QUICK START")
    print("="*60)

    example_derivatives()
    example_toy_problem()
    example_3d_reconstruction()

    print("\n" + "="*60)
    print("Complete. Check notebooks/ for detailed examples.")
    print("="*60)


if __name__ == '__main__':
    main()
