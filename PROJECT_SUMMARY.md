# X-ray Tomography Project Initialization Summary

## Project Overview

This is a complete implementation framework for the Bar-Ilan University Optimization Course Project 2: X-ray Tomography Reconstruction.

The project implements computational tomography reconstruction using:
1. **Tikhonov Regularization** (ℓ² norm) via CGLS
2. **Total Variation Regularization** (ℓ¹ norm) via IRLS

## Project Structure

```
xray_tomography/
├── README.md                      # Main documentation
├── THEORETICAL_QUESTIONS.md       # Solutions guide for theory questions
├── requirements.txt               # Python dependencies
├── quickstart.py                  # Quick start examples
├── .gitignore                     # Git ignore rules
│
├── src/                           # Core implementation modules
│   ├── __init__.py               # Package initialization
│   ├── matrix_construction.py    # Ray-path and derivative matrices
│   ├── solvers.py                # CGLS and IRLS algorithms
│   ├── data_utils.py             # Data loading and preprocessing
│   └── visualization.py          # Plotting and visualization
│
├── notebooks/                     # Jupyter notebooks for each task
│   ├── Q3_Q4_derivatives.ipynb   # Derivative matrices (Q3-Q4)
│   ├── Q10_Q11_cgls.ipynb        # CGLS implementation (Q10-Q11)
│   └── Q15_Q16_irls.ipynb        # IRLS implementation (Q15-Q16)
│
├── data/                          # Data directory (add .mat files here)
│   ├── X1.mat                    # Example density fields
│   ├── X2.mat
│   ├── X3.mat
│   ├── Y.mat                     # Toy problem measurements
│   ├── Small/                    # Unzip Small.zip here
│   └── Large/                    # Unzip Large.zip here
│
└── results/                       # Output reconstructions
```

## Core Components

### 1. Matrix Construction (`src/matrix_construction.py`)
- **`build_derivative_matrix_2d(M, N, direction)`**: Finite difference operators for 2D
- **`build_derivative_matrix_3d(n, direction)`**: Finite difference operators for 3D
- **`build_combined_derivative_matrix(M, N, dimensions)`**: Stacked L = [Dx; Dy; Dz]
- **`build_toy_ray_path_matrix()`**: Ray-path matrix A for 5×5 toy problem
- **`compute_gradient_magnitude()`**: Gradient magnitude from derivatives

### 2. Solvers (`src/solvers.py`)
- **`cgls(A, y, L, lam, ...)`**: Conjugate Gradient Least Squares
  - Solves: min (1/2)||Ax - y||² + (λ/2)||Lx||²
  - Memory efficient: avoids A^T A computation
  - Returns: SolverResult with solution, convergence info
  
- **`irls(A, y, L, alpha, ...)`**: Iteratively Reweighted Least Squares
  - Solves: min (1/2)||Ax - y||² + α||Lx||₁
  - Iteratively updates weights W from gradient magnitude
  - Uses CGLS as inner solver
  
- **`gradient_descent(Q, b, step_size, ...)`**: Basic GD for quadratic problems

### 3. Data Utilities (`src/data_utils.py`)
- **`load_mat_file(filepath)`**: Load MATLAB .mat files
- **`load_density_example(filepath)`**: Load X1, X2, X3 examples
- **`load_3d_data(data_dir)`**: Load y.mat and A.mat from Small/Large
- **`grid_to_vector(X)`**: Convert grid to column-stack representation
- **`vector_to_grid(x, shape)`**: Convert column-stack back to grid
- **`save_reconstruction(x, filepath)`**: Save results as .npz

### 4. Visualization (`src/visualization.py`)
- **`visualize_2d_field(X)`**: Display 2D density field
- **`visualize_derivatives(X, Dx_X, Dy_X, G)`**: Show derivatives and gradients
- **`visualize_3d_slices(X, slice_indices, axis)`**: Display 3D volume slices
- **`plot_convergence(objectives, residuals)`**: Plot solver convergence
- **`plot_histogram(x)`**: Histogram of density values
- **`compare_reconstructions(X_tikh, X_tv)`**: Side-by-side comparison

## Getting Started

### 1. Installation
```bash
cd xray_tomography
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Data Files
Place the following files in the `data/` directory:
- X1.mat, X2.mat, X3.mat (example density fields)
- Y.mat (toy problem measurements)
- Extract Small.zip → data/Small/
- Extract Large.zip → data/Large/

### 3. Run Quick Start
```bash
python quickstart.py
```

### 4. Work Through Notebooks
Open Jupyter and work through notebooks in order:
```bash
jupyter notebook notebooks/
```

## Question Mapping

### Theoretical Questions (see THEORETICAL_QUESTIONS.md)
- **Q1-Q2**: Midpoint rule, ray-path matrix construction
- **Q5**: Ill-posed problem analysis
- **Q6**: Normal equations derivation
- **Q7-Q9**: Gradient descent analysis
- **Q12-Q14**: Total Variation theory

### Code Questions (see notebooks/)
- **Q3-Q4**: `Q3_Q4_derivatives.ipynb` - Derivative matrices
- **Q10-Q11**: `Q10_Q11_cgls.ipynb` - CGLS solver and reconstruction
- **Q15-Q16**: `Q15_Q16_irls.ipynb` - IRLS solver and TV reconstruction

## Key Algorithms

### CGLS Algorithm
```python
Initialize: x = 0, s = -y, r = A^T s, p = -r
For k = 0, 1, 2, ...:
    q = Ap
    α = γ / ||q||²  (with regularization adjustment)
    x ← x + αp
    s ← s + αq  (recursive update)
    r ← A^T s + λL^T Lx
    β = ||r||² / γ
    p ← -r + βp
    γ ← ||r||²
```

### IRLS Algorithm
```python
Initialize: x = x_tikhonov
For outer iterations:
    γ = Lx
    W_ii = 1/max(|γ_i|, ε)
    Solve: min (1/2)||Ax-y||² + (α/2)||W^(1/2)Lx||² with CGLS
    Check convergence
```

## Mathematical Background

### Beer-Lambert Law
```
y = -log(I/I₀) = ∫_γ ρ(s) ds
```

### Discretization
```
y ≈ Σⱼ ρ(x̃ⱼ) Δl
y = Ax (linear system)
```

### Tikhonov Regularization
```
min_x (1/2)||Ax - y||² + (λ/2)||Lx||²
```
- Closed form: x = (A^T A + λL^T L)^(-1) A^T y
- Iterative: Use CGLS (avoids matrix inversion)
- Favors smooth solutions

### Total Variation Regularization
```
min_x (1/2)||Ax - y||² + α||Lx||₁
```
- Non-smooth optimization
- Iterative: Use IRLS (reweighted ℓ² approximation)
- Favors piecewise constant solutions (preserves edges)

## Important Implementation Details

### Column-Stack Convention
```
X = [X11  X12  X13]     →  x = [X11, X21, X31, X12, X22, X32, X13, X23, X33]
    [X21  X22  X23]
    [X31  X32  X33]
```
Use `order='F'` (Fortran/column-major) when flattening.

### Sparse Matrices
- A is sparse (ray paths through limited cells)
- L is sparse (finite differences)
- Use `scipy.sparse` for efficiency
- Never compute A^T A explicitly

### Boundary Conditions
- Forward differences: derivative at boundary = 0
- Last row/column/slice has zero derivative

## Performance Tips

1. **Use sparse matrices** for A and L
2. **Avoid dense operations** like A^T A
3. **Set appropriate tolerances**: tol ~ 1e-6 for CGLS
4. **Limit iterations**: max_iter ~ few hundred for CGLS
5. **Choose λ carefully**: try 1e-3, 1e-4, 1e-5, 1e-6
6. **Initialize IRLS with Tikhonov solution**
7. **Monitor convergence** with objective/residual plots

## Expected Results

### Toy Problem (5×5)
- 25 unknowns, 35 measurements
- CGLS converges in < 50 iterations
- Condition number κ ≈ 10² - 10³

### Small Bag (19×19×19)
- 6859 unknowns
- CGLS: 100-200 iterations
- IRLS: 10-20 outer iterations
- Should see distinct objects in reconstruction

### Large Bag (49×49×49)
- 117,649 unknowns
- CGLS: 200-300 iterations
- IRLS: 15-25 outer iterations
- More computation time but clearer reconstruction

## Next Steps

1. **Complete theoretical questions** using THEORETICAL_QUESTIONS.md
2. **Implement ray-path matrix A** for toy problem (Q1-Q2)
3. **Run derivative tests** using Q3_Q4_derivatives.ipynb
4. **Test CGLS** on toy problem first
5. **Reconstruct small bag** with different λ values
6. **Implement IRLS** for Total Variation
7. **Reconstruct large bag** and identify contents
8. **Compare Tikhonov vs TV** qualitatively

## Debugging Tips

- **Matrix dimensions**: Check A.shape, L.shape, x.shape
- **Column-stack order**: Use `order='F'` consistently
- **Convergence issues**: Reduce tolerance, increase max_iter
- **Numerical stability**: Check condition number, use regularization
- **Visualization**: Plot slices at different depths to debug reconstruction

## Contact & Support

For questions about:
- **Theory**: See THEORETICAL_QUESTIONS.md
- **Implementation**: Check docstrings in src/ modules
- **Usage**: Run quickstart.py and notebooks

Good luck with your project! 🚀
