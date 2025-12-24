# X-ray Tomography Reconstruction

X-ray tomography reconstruction using Tikhonov and Total Variation regularization.

## Structure

```
├── src/                    # Source code
│   ├── matrix_construction.py
│   ├── solvers.py
│   ├── data_utils.py
│   └── visualization.py
├── data/                   # Data files
│   ├── X1.mat, X2.mat, X3.mat
│   ├── Y.mat
│   ├── Small/
│   └── Large/
├── notebooks/              # Jupyter notebooks
└── results/               # Outputs
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
python quickstart.py
```

## Algorithms

### CGLS (Tikhonov Regularization)
Solves: `min (1/2)||Ax - y||^2 + (λ/2)||Lx||^2`

```python
from src.solvers import cgls
from src.matrix_construction import build_combined_derivative_matrix

L = build_combined_derivative_matrix(M, N, dimensions=2)
result = cgls(A, y, L, lam=1e-5, tol=1e-6)
```

### IRLS (Total Variation)
Solves: `min (1/2)||Ax - y||^2 + α||Lx||_1`

```python
from src.solvers import irls

result = irls(A, y, L, alpha=0.5, tol=1e-4)
```

## Notebooks

- `Q3_Q4_derivatives.ipynb` - Derivative matrices and gradients
- `Q8_Q9_gradient_descent.ipynb` - Gradient descent analysis
- `Q10_Q11_cgls.ipynb` - CGLS and small bag reconstruction
- `Q12_curve_fitting.ipynb` - Regularization comparison
- `Q15_Q16_irls.ipynb` - Total Variation and large bag

## Notes

- Column-stack convention: `x = [X11, X21, X31, ..., X12, X22, ...]`
- Use `order='F'` for grid conversions
- Sparse matrices for efficiency
