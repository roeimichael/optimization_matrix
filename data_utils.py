import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings


try:
    import scipy.io as sio
except ImportError:
    warnings.warn("scipy.io not available, .mat loading will fail")
    sio = None


def load_mat_file(filepath: str) -> Dict:
    """
    Load a MATLAB .mat file.
    
    Args:
        filepath: Path to .mat file
    
    Returns:
        Dictionary with variables from .mat file
    """
    if sio is None:
        raise ImportError("scipy is required to load .mat files")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        data = sio.loadmat(str(filepath))
    except NotImplementedError:
        # Try with newer format (HDF5-based .mat files v7.3+)
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
    
    # Remove metadata keys
    data = {k: v for k, v in data.items() if not k.startswith('__')}
    
    return data


def load_density_example(filepath: str) -> np.ndarray:
    """
    Load example density field (X1.mat, X2.mat, X3.mat).
    
    Args:
        filepath: Path to .mat file
    
    Returns:
        2D density array
    """
    data = load_mat_file(filepath)
    
    # The variable name might be 'X', 'X1', 'X2', 'X3', etc.
    for key in ['X', 'X1', 'X2', 'X3']:
        if key in data:
            X = data[key]
            if X.ndim == 2:
                return X
    
    # If not found, return the first 2D array
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value
    
    raise ValueError(f"No 2D density field found in {filepath}")


def load_toy_measurements(filepath: str) -> np.ndarray:
    """
    Load toy problem measurements (Y.mat).
    
    Args:
        filepath: Path to Y.mat
    
    Returns:
        Measurement matrix Y (num_sources × num_receivers)
    """
    data = load_mat_file(filepath)
    
    if 'Y' in data:
        return data['Y']
    
    # Return first 2D array found
    for value in data.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value
    
    raise ValueError(f"No measurement matrix found in {filepath}")


def load_3d_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 3D tomography data (y.mat and A.mat from Small.zip or Large.zip).
    
    Args:
        data_dir: Directory containing y.mat and A.mat
    
    Returns:
        Tuple of (y, A) where:
            y: Observation vector (num_measurements,)
            A: Ray path matrix (num_measurements × n³)
    """
    data_dir = Path(data_dir)
    
    # Load observations
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
        # Take first array
        y = list(y_data.values())[0].flatten()
    
    # Load ray path matrix
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
        # Take first 2D array
        for value in A_data.values():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                A = value
                break
    
    if A is None:
        raise ValueError(f"No matrix found in {A_file}")
    
    print(f"Loaded data from {data_dir}:")
    print(f"  y shape: {y.shape}")
    print(f"  A shape: {A.shape}")
    print(f"  Volume size: {int(round(A.shape[1] ** (1/3)))}³")
    
    return y, A


def measurements_to_vector(Y: np.ndarray, 
                          source_order: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert measurement matrix Y to vector y according to ray numbering.
    
    For toy problem: Y[i,j] = measurement from source i to receiver j
    Need to stack according to how matrix A was built.
    
    Args:
        Y: Measurement matrix (num_sources × num_receivers)
        source_order: Order to stack sources (if None, uses row-major)
    
    Returns:
        y: Measurement vector
    """
    # Default: stack row-by-row (flatten in row-major order)
    if source_order is None:
        return Y.flatten(order='C')
    else:
        # Custom ordering
        y = []
        for src_idx in source_order:
            y.extend(Y[src_idx, :])
        return np.array(y)


def grid_to_vector(X: np.ndarray) -> np.ndarray:
    """
    Convert grid representation to column-stack vector.
    
    Convention: x = [X11, X21, X31, ..., X12, X22, X32, ..., X1N, X2N, ..., XNN]
    Stack columns: first column, then second column, etc.
    
    Args:
        X: Grid array (M × N for 2D, or n × n × n for 3D)
    
    Returns:
        x: Column-stacked vector
    """
    if X.ndim == 2:
        # 2D: stack columns (column-major order, Fortran-style)
        return X.flatten(order='F')
    elif X.ndim == 3:
        # 3D: stack slices along z, then within each slice use column-major
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
    """
    Convert column-stack vector back to grid representation.
    
    Args:
        x: Column-stacked vector
        shape: Target shape (M, N) for 2D or (n, n, n) for 3D
    
    Returns:
        X: Grid array
    """
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
    """
    Save reconstruction to .npz file.
    
    Args:
        x: Reconstructed density (vector or grid)
        filepath: Output path
        metadata: Optional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {'x': x}
    if metadata:
        save_dict.update(metadata)
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Saved reconstruction to {filepath}")
