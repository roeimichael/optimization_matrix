#!/usr/bin/env python3
"""
Quick Start Example for X-ray Tomography Project

This script demonstrates basic usage of the reconstruction tools.
"""

import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from matrix_construction import build_derivative_matrix_2d, compute_gradient_magnitude
from solvers import cgls
from data_utils import load_density_example, grid_to_vector, vector_to_grid
from visualization import visualize_2d_field, visualize_derivatives


def example_derivatives():
    """Example: Compute and visualize derivatives of a density field."""
    print("="*60)
    print("Example 1: Derivative Matrices and Gradient Visualization")
    print("="*60)
    
    # Load example density field
    try:
        X = load_density_example('data/X1.mat')
        print(f"\nLoaded X1.mat with shape: {X.shape}")
    except FileNotFoundError:
        print("\nWarning: data/X1.mat not found. Creating synthetic data.")
        X = np.random.randn(50, 50)
    
    M, N = X.shape
    
    # Build derivative matrices
    print("\nBuilding derivative matrices...")
    Dx = build_derivative_matrix_2d(M, N, 'x')
    Dy = build_derivative_matrix_2d(M, N, 'y')
    
    print(f"Dx shape: {Dx.shape}")
    print(f"Dy shape: {Dy.shape}")
    
    # Apply to density field
    x = grid_to_vector(X)
    Dx_x = Dx @ x
    Dy_x = Dy @ x
    
    # Compute gradient magnitude
    G = compute_gradient_magnitude(Dx_x, Dy_x, M, N)
    
    # Convert back to 2D for visualization
    Dx_X = vector_to_grid(Dx_x, (M, N))
    Dy_X = vector_to_grid(Dy_x, (M, N))
    
    print(f"\nGradient magnitude range: [{G.min():.4f}, {G.max():.4f}]")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_derivatives(X, Dx_X, Dy_X, G)


def example_toy_problem():
    """Example: Set up and solve toy problem."""
    print("\n" + "="*60)
    print("Example 2: Toy Problem Setup (5x5 grid)")
    print("="*60)
    
    # Create synthetic toy problem
    M, N = 5, 5
    num_measurements = 35  # 7 sources × 5 receivers
    
    # Random ray-path matrix (replace with actual construction)
    print("\nCreating synthetic ray-path matrix A...")
    A = np.random.randn(num_measurements, M*N)
    
    # True density and measurements
    x_true = np.random.rand(M*N)
    y = A @ x_true + 0.01 * np.random.randn(num_measurements)
    
    print(f"A shape: {A.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of unknowns: {M*N}")
    print(f"Number of measurements: {num_measurements}")
    
    # Build regularization matrix
    from matrix_construction import build_combined_derivative_matrix
    L = build_combined_derivative_matrix(M, N, dimensions=2)
    
    print(f"L shape: {L.shape}")
    
    # Solve with CGLS
    print("\nSolving with CGLS (λ=1e-5)...")
    result = cgls(A, y, L, lam=1e-5, tol=1e-6, max_iter=50)
    
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final objective: {result.objective_values[-1]:.6e}")
    
    # Visualize solution
    X_recon = vector_to_grid(result.x, (M, N))
    X_true = vector_to_grid(x_true, (M, N))
    
    print("\nTrue density field:")
    print(X_true)
    print("\nReconstructed density field:")
    print(X_recon)
    print(f"\nReconstruction error: {np.linalg.norm(X_recon - X_true):.6f}")


def example_3d_reconstruction():
    """Example: 3D reconstruction workflow."""
    print("\n" + "="*60)
    print("Example 3: 3D Reconstruction Workflow")
    print("="*60)
    
    # Create synthetic 3D problem
    n = 10  # Small 10×10×10 volume
    num_measurements = 300
    
    print(f"\nCreating synthetic 3D problem...")
    print(f"Volume size: {n}³ = {n**3} voxels")
    print(f"Number of measurements: {num_measurements}")
    
    A = np.random.randn(num_measurements, n**3)
    x_true = np.random.rand(n**3)
    y = A @ x_true + 0.01 * np.random.randn(num_measurements)
    
    # Build 3D derivative operator
    from matrix_construction import build_combined_derivative_matrix
    L = build_combined_derivative_matrix(n, n, dimensions=3)
    
    print(f"L shape: {L.shape}")
    
    # Solve with CGLS
    print("\nSolving with CGLS...")
    result = cgls(A, y, L, lam=1e-5, tol=1e-6, max_iter=100)
    
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    
    # Convert to 3D grid
    X_recon = vector_to_grid(result.x, (n, n, n))
    
    print(f"\nReconstruction shape: {X_recon.shape}")
    print(f"Density range: [{X_recon.min():.4f}, {X_recon.max():.4f}]")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" X-RAY TOMOGRAPHY RECONSTRUCTION - QUICK START EXAMPLES")
    print("="*70)
    
    # Run examples
    example_derivatives()
    example_toy_problem()
    example_3d_reconstruction()
    
    print("\n" + "="*70)
    print("Examples completed! Check the notebooks/ directory for detailed usage.")
    print("="*70)


if __name__ == '__main__':
    main()
