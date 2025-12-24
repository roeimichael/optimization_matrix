# X-ray Tomography Reconstruction Project

Optimization course project implementing computational tomography reconstruction methods.

## Project Structure

```
xray_tomography/
├── src/
│   ├── matrix_construction.py   # Ray-path matrix A and derivative operators
│   ├── solvers.py                # CGLS and IRLS implementations
│   ├── visualization.py          # Plotting utilities
│   └── data_utils.py             # Data loading and preprocessing
├── data/                         # Place .mat files here
│   ├── X1.mat, X2.mat, X3.mat   # Example density fields
│   ├── Y.mat                     # Toy problem measurements
│   ├── Small/                    # Small bag data (unzip Small.zip here)
│   └── Large/                    # Large bag data (unzip Large.zip here)
├── notebooks/                    # Jupyter notebooks for analysis
├── results/                      # Output reconstructions
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Tasks

### Part 1: Toy Problem (Q1-Q6)
- **Q1-Q2**: Discretize Beer-Lambert integral, build ray-path matrix A
- **Q3-Q4**: Construct derivative matrices Dx, Dy and visualize gradients
- **Q5**: Analyze ill-posed nature of the inverse problem
- **Q6**: Derive normal equations and closed-form solution

### Part 2: Large Scale Optimization (Q7-Q11)
- **Q7-Q9**: Analyze gradient descent convergence
- **Q10**: Implement CGLS (Conjugate Gradient Least Squares)
- **Q11**: Reconstruct small bag using Tikhonov regularization

### Part 3: Total Variation Regularization (Q12-Q16)
- **Q12-Q14**: Analyze TV regularization properties
- **Q15**: Implement IRLS (Iteratively Reweighted Least Squares)
- **Q16**: Reconstruct large bag contents

## Key Algorithms

### CGLS (Conjugate Gradient Least Squares)
Solves: `min_x (1/2)||Ax - y||² + (λ/2)||Lx||²`

Features:
- Avoids computing A^T A explicitly
- Memory efficient for large sparse matrices
- O(nnz(A)) per iteration

### IRLS (Iteratively Reweighted Least Squares)
Solves: `min_x (1/2)||Ax - y||² + α||Lx||₁`

Approximates ℓ¹ norm via iteratively solving weighted ℓ² problems:
1. Compute weights W from current solution
2. Solve weighted least squares with CGLS
3. Repeat until convergence

## Usage Examples

### Load and visualize example density fields
```python
from src.data_utils import load_density_example
from src.visualization import visualize_2d_field

X = load_density_example('data/X1.mat')
visualize_2d_field(X, title='Example Density Field')
```

### Build derivative matrices and compute gradients
```python
from src.matrix_construction import build_derivative_matrix_2d, compute_gradient_magnitude

M, N = X.shape
Dx = build_derivative_matrix_2d(M, N, 'x')
Dy = build_derivative_matrix_2d(M, N, 'y')

x = X.flatten(order='F')  # Column-stack
Dx_x = Dx @ x
Dy_x = Dy @ x
G = compute_gradient_magnitude(Dx_x, Dy_x, M, N)
```

### Solve with CGLS (Tikhonov regularization)
```python
from src.solvers import cgls
from src.matrix_construction import build_combined_derivative_matrix

L = build_combined_derivative_matrix(M, N, dimensions=2)
result = cgls(A, y, L, lam=1e-5, tol=1e-6, max_iter=100)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final objective: {result.objective_values[-1]}")
```

### Solve with IRLS (Total Variation)
```python
from src.solvers import irls

result = irls(A, y, L, alpha=0.5, x0=None, 
              epsilon=1e-8, tol=1e-4, max_outer_iter=20)

# Visualize result
X_recon = result.x.reshape((n, n, n), order='F')
```

## Mathematical Background

### Beer-Lambert Law
```
y = -log(I/I₀) = ∫_γ ρ(s) ds
```
where ρ is object density and γ is ray path.

### Discretization (Midpoint Rule)
```
∫ᵇₐ f(x)dx ≈ Σⱼ f(xⱼ)Δx
```

### Tikhonov Regularization
```
min_x (1/2)||Ax - y||² + (λ/2)||Lx||²
```
Favors smooth solutions (L = gradient operator).

### Total Variation Regularization
```
min_x (1/2)||Ax - y||² + α||Lx||₁
```
Favors piecewise smooth solutions (preserves edges).

## Notes

- **Column-stack convention**: `x = [X₁₁, X₂₁, X₃₁, ..., X₁₂, X₂₂, ...]`
- **3D stacking**: Stack slices along z direction
- **Derivative boundary**: Assumed zero at boundaries
- **Sparse matrices**: Use scipy.sparse for efficiency

## References

- Project specification: `Project_2_-_X-ray_tomography.pdf`
- Optimization course, Bar-Ilan University, 2025/6
