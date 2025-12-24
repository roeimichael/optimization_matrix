import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path


def visualize_2d_field(X: np.ndarray,
                       title: str = "Density Field",
                       cmap: str = 'gray',
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (8, 6)) -> None:
    """Visualize 2D density field."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(X, cmap=cmap, origin='lower', interpolation='nearest')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Density')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_gradients(X: np.ndarray,
                       Dx_x: np.ndarray,
                       Dy_x: np.ndarray,
                       G: np.ndarray,
                       title: str = "Gradient Visualization",
                       save_path: Optional[str] = None) -> None:
    """Visualize field and its gradients."""
    M, N = X.shape
    Dx_2d = Dx_x.reshape(M, N)
    Dy_2d = Dy_x.reshape(M, N)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(X, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original Field')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(Dx_2d, cmap='RdBu_r', origin='lower')
    axes[0, 1].set_title('Horizontal Derivative')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(Dy_2d, cmap='RdBu_r', origin='lower')
    axes[1, 0].set_title('Vertical Derivative')
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(G, cmap='hot', origin='lower')
    axes[1, 1].set_title('Gradient Magnitude')
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_convergence(residuals: list,
                    objective_values: list,
                    title: str = "Convergence",
                    save_path: Optional[str] = None) -> None:
    """Plot convergence curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(residuals, 'b-o', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Residual Norm')
    ax1.set_title('Residual')
    ax1.grid(True, alpha=0.3)

    ax2.plot(objective_values, 'r-o', markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Objective Value')
    ax2.set_title('Objective')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_3d_slices(X: np.ndarray,
                       axis: str = 'z',
                       num_slices: int = 9,
                       cmap: str = 'gray',
                       title: Optional[str] = None,
                       save_path: Optional[str] = None) -> None:
    """Visualize slices from 3D volume."""
    n = X.shape[0]
    slice_indices = np.linspace(0, n-1, num_slices, dtype=int)

    ncols = 3
    nrows = int(np.ceil(num_slices / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(slice_indices):
        if axis == 'z':
            slice_data = X[:, :, slice_idx]
            slice_label = f'z = {slice_idx}'
        elif axis == 'y':
            slice_data = X[:, slice_idx, :]
            slice_label = f'y = {slice_idx}'
        else:
            slice_data = X[slice_idx, :, :]
            slice_label = f'x = {slice_idx}'

        im = axes[idx].imshow(slice_data, cmap=cmap, origin='lower')
        axes[idx].set_title(slice_label)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')

    if title is None:
        title = f"3D Slices ({axis}-axis)"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def compare_reconstructions(X1: np.ndarray,
                           X2: np.ndarray,
                           labels: Tuple[str, str] = ("Method 1", "Method 2"),
                           slice_idx: Optional[int] = None,
                           axis: str = 'z',
                           cmap: str = 'gray',
                           save_path: Optional[str] = None) -> None:
    """Compare two reconstructions side by side."""
    if X1.ndim == 3:
        if slice_idx is None:
            slice_idx = X1.shape[0] // 2

        if axis == 'z':
            X1_slice = X1[:, :, slice_idx]
            X2_slice = X2[:, :, slice_idx]
        elif axis == 'y':
            X1_slice = X1[:, slice_idx, :]
            X2_slice = X2[:, slice_idx, :]
        else:
            X1_slice = X1[slice_idx, :, :]
            X2_slice = X2[slice_idx, :, :]
    else:
        X1_slice = X1
        X2_slice = X2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(X1_slice, cmap=cmap, origin='lower')
    axes[0].set_title(labels[0])
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(X2_slice, cmap=cmap, origin='lower')
    axes[1].set_title(labels[1])
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    diff = X2_slice - X1_slice
    im3 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_histogram(data: np.ndarray,
                   bins: int = 50,
                   title: str = "Density Distribution",
                   xlabel: str = "Density Value",
                   save_path: Optional[str] = None) -> None:
    """Plot histogram of values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
