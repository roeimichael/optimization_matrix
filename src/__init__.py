"""
X-ray Tomography Reconstruction Package

This package implements algorithms for X-ray tomography reconstruction:
- Tikhonov regularization (smooth solutions)
- Total Variation regularization (edge-preserving solutions)
"""

from .matrix_construction import (
    build_toy_ray_path_matrix,
    build_derivative_matrix_2d,
    build_derivative_matrix_3d,
    build_combined_derivative_matrix,
    compute_gradient_magnitude
)

from .solvers import (
    cgls,
    irls,
    gradient_descent,
    SolverResult
)

from .data_utils import (
    load_mat_file,
    load_density_example,
    load_toy_measurements,
    load_3d_data,
    measurements_to_vector,
    grid_to_vector,
    vector_to_grid,
    save_reconstruction
)

__all__ = [
    # Matrix construction
    'build_toy_ray_path_matrix',
    'build_derivative_matrix_2d',
    'build_derivative_matrix_3d',
    'build_combined_derivative_matrix',
    'compute_gradient_magnitude',

    # Solvers
    'cgls',
    'irls',
    'gradient_descent',
    'SolverResult',

    # Data utilities
    'load_mat_file',
    'load_density_example',
    'load_toy_measurements',
    'load_3d_data',
    'measurements_to_vector',
    'grid_to_vector',
    'vector_to_grid',
    'save_reconstruction',
]

__version__ = '1.0.0'
