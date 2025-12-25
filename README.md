# X-ray Tomography Reconstruction

Computational reconstruction of 3D objects from X-ray measurements using Tikhonov and Total Variation regularization.

## Structure

```
├── src/                    # Core implementations
│   ├── matrix_construction.py  # Ray-path and derivative matrices
│   ├── solvers.py              # CGLS, IRLS, gradient descent
│   ├── data_utils.py           # Data loading and conversions
│   └── visualization.py        # Plotting functions
├── data/                   # .mat data files
├── xray_tomography.ipynb   # Main notebook
└── results/                # Generated outputs
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open `xray_tomography.ipynb` in Jupyter for the complete workflow.

## Algorithms

**Tikhonov Regularization (CGLS)**
```python
from src.solvers import cgls
result = cgls(A, y, L, lam=1e-5)
```
Minimizes: `||Ax - y||² + λ||Lx||²`

**Total Variation (IRLS)**
```python
from src.solvers import irls
result = irls(A, y, L, alpha=0.5)
```
Minimizes: `||Ax - y||² + α||Lx||₁`

## Results

### Derivative Matrices
- Forward finite differences on 2D/3D grids
- Optimized sparse matrix construction (O(n) complexity)

### Gradient Descent vs CGLS
- CGLS converges significantly faster than gradient descent
- Condition number analysis for step size selection

### 3D Reconstruction
- Small bag: 19³ = 6,859 voxels
- Large bag: 49³ = 117,649 voxels
- Total Variation preserves edges better than Tikhonov smoothing

## Implementation Notes

- Column-stack convention (Fortran order)
- Augmented system formulation for CGLS: `B = [A; √λ·L]`
- Sparse matrices for memory efficiency
