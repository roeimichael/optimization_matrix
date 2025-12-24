import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

try:
    import scipy.io as sio
except ImportError:
    warnings.warn("scipy.io not available")
    sio = None


def load_mat_file(filepath: str) -> Dict:
    """Load MATLAB .mat file."""
    if sio is None:
        raise ImportError("scipy is required to load .mat files")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        data = sio.loadmat(str(filepath))
    except NotImplementedError:
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])

    data = {k: v for k, v in data.items() if not k.startswith('__')}
    return data


def load_density_example(filepath: str) -> np.ndarray:
    """Load example density field (X1.mat, X2.mat, X3.mat)."""
    from scipy import sparse
    data = load_mat_file(filepath)

    for key in ['X', 'X1', 'X2', 'X3']:
        if key in data:
            X = data[key]
            if sparse.issparse(X):
                X = X.toarray()
            if X.ndim == 2:
                return X

    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value
        elif sparse.issparse(value) and hasattr(value, 'toarray'):
            arr = value.toarray()
            if arr.ndim == 2:
                return arr

    raise ValueError(f"No 2D density field found in {filepath}")


def load_toy_measurements(filepath: str) -> np.ndarray:
    """Load toy problem measurements (Y.mat)."""
    from scipy import sparse
    data = load_mat_file(filepath)

    if 'Y' in data:
        Y = data['Y']
        if sparse.issparse(Y):
            Y = Y.toarray()
        return Y

    for value in data.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value
        elif sparse.issparse(value) and hasattr(value, 'toarray'):
            arr = value.toarray()
            if arr.ndim == 2:
                return arr

    raise ValueError(f"No measurement matrix found in {filepath}")


def load_3d_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 3D tomography data (y.mat and A.mat)."""
    data_dir = Path(data_dir)

    y_file = data_dir / 'y.mat'
    if not y_file.exists():
        raise FileNotFoundError(f"y.mat not found in {data_dir}")

    y_data = load_mat_file(y_file)
    y = None
    for key in ['y', 'Y']:
        if key in y_data:
            y = y_data[key].flatten()
            break

    if y is None:
        y = list(y_data.values())[0].flatten()

    A_file = data_dir / 'A.mat'
    if not A_file.exists():
        raise FileNotFoundError(f"A.mat not found in {data_dir}")

    A_data = load_mat_file(A_file)
    A = None
    for key in ['A']:
        if key in A_data:
            A = A_data[key]
            break

    if A is None:
        for value in A_data.values():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                A = value
                break

    if A is None:
        raise ValueError(f"No matrix found in {A_file}")

    print(f"Loaded {data_dir.name}: y.shape={y.shape}, A.shape={A.shape}, volume={int(round(A.shape[1] ** (1/3)))}^3")

    return y, A


def measurements_to_vector(Y: np.ndarray,
                          source_order: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert measurement matrix Y to vector y."""
    from scipy import sparse

    if sparse.issparse(Y):
        Y = Y.toarray()

    if source_order is None:
        return Y.flatten(order='C')
    else:
        y = []
        for src_idx in source_order:
            y.extend(Y[src_idx, :])
        return np.array(y)


def grid_to_vector(X: np.ndarray) -> np.ndarray:
    """Convert grid to column-stack vector."""
    if X.ndim == 2:
        return X.flatten(order='F')
    elif X.ndim == 3:
        n = X.shape[0]
        x = np.zeros(n ** 3)
        for k in range(n):
            slice_start = k * n * n
            slice_end = (k + 1) * n * n
            x[slice_start:slice_end] = X[:, :, k].flatten(order='F')
        return x
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def vector_to_grid(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Convert column-stack vector to grid."""
    if len(shape) == 2:
        M, N = shape
        return x.reshape((M, N), order='F')
    elif len(shape) == 3:
        n = shape[0]
        X = np.zeros(shape)
        for k in range(n):
            slice_start = k * n * n
            slice_end = (k + 1) * n * n
            X[:, :, k] = x[slice_start:slice_end].reshape((n, n), order='F')
        return X
    else:
        raise ValueError(f"Invalid shape: {shape}")


def save_reconstruction(x: np.ndarray,
                       filepath: str,
                       metadata: Optional[Dict] = None) -> None:
    """Save reconstruction to .npz file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {'x': x}
    if metadata:
        save_dict.update(metadata)

    np.savez_compressed(filepath, **save_dict)
    print(f"Saved to {filepath}")
