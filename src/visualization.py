"""
Visualization utilities for X-ray tomography reconstruction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple, List
from pathlib import Path


def visualize_2d_field(X: np.ndarray,
                       title: str = "Density Field",
                       cmap: str = 'gray',
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualize 2D density field.

    Args:
        X: 2D density array (M × N)
        title: Plot title
        cmap: Colormap name
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(X, cmap=cmap, origin='lower', interpolation='nearest')
    ax.set_xlabel('x (column index)')
    ax.set_ylabel('y (row index)')
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def visualize_gradients(X: np.ndarray,
                       Dx_x: np.ndarray,
                       Dy_x: np.ndarray,
                       G: np.ndarray,
                       title: str = "Gradient Visualization",
                       save_path: Optional[str] = None) -> None:
    """
    Visualize original field and its gradients.

    Args:
        X: Original 2D field (M × N)
        Dx_x: Horizontal derivative (flattened, length M*N)
        Dy_x: Vertical derivative (flattened, length M*N)
        G: Gradient magnitude (M × N)
        title: Main title
        save_path: Optional path to save figure
    """
    M, N = X.shape
    Dx_2d = Dx_x.reshape(M, N)
    Dy_2d = Dy_x.reshape(M, N)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original field
    im0 = axes[0, 0].imshow(X, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original Field X')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])

    # Horizontal derivative
    im1 = axes[0, 1].imshow(Dx_2d, cmap='RdBu_r', origin='lower')
    axes[0, 1].set_title('Horizontal Derivative (∂X/∂x)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])

    # Vertical derivative
    im2 = axes[1, 0].imshow(Dy_2d, cmap='RdBu_r', origin='lower')
    axes[1, 0].set_title('Vertical Derivative (∂X/∂y)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])

    # Gradient magnitude
    im3 = axes[1, 1].imshow(G, cmap='hot', origin='lower')
    axes[1, 1].set_title('Gradient Magnitude |∇X|')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_convergence(residuals: List[float],
                    objective_values: List[float],
                    title: str = "Convergence",
                    save_path: Optional[str] = None) -> None:
    """
    Plot convergence curves for iterative solvers.

    Args:
        residuals: List of residual norms
        objective_values: List of objective function values
        title: Main title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Residual plot
    ax1.semilogy(residuals, 'b-o', markersize=4, linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Residual Norm')
    ax1.set_title('Residual Convergence')
    ax1.grid(True, alpha=0.3)

    # Objective plot
    ax2.plot(objective_values, 'r-o', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Objective Value')
    ax2.set_title('Objective Function')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def visualize_3d_slices(X: np.ndarray,
                       axis: str = 'z',
                       num_slices: int = 9,
                       cmap: str = 'gray',
                       title: Optional[str] = None,
                       save_path: Optional[str] = None) -> None:
    """
    Visualize slices from 3D volume.

    Args:
        X: 3D array (n × n × n)
        axis: Slicing axis ('x', 'y', or 'z')
        num_slices: Number of slices to show
        cmap: Colormap
        title: Main title
        save_path: Optional path to save figure
    """
    n = X.shape[0]

    # Select slice indices evenly spaced
    slice_indices = np.linspace(0, n-1, num_slices, dtype=int)

    # Determine grid layout
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
        elif axis == 'x':
            slice_data = X[slice_idx, :, :]
            slice_label = f'x = {slice_idx}'
        else:
            raise ValueError(f"Invalid axis: {axis}")

        im = axes[idx].imshow(slice_data, cmap=cmap, origin='lower',
                             interpolation='nearest')
        axes[idx].set_title(slice_label)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    # Hide unused subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')

    if title is None:
        title = f"3D Volume Slices (along {axis}-axis)"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def compare_reconstructions(X1: np.ndarray,
                           X2: np.ndarray,
                           labels: Tuple[str, str] = ("Tikhonov", "Total Variation"),
                           slice_idx: Optional[int] = None,
                           axis: str = 'z',
                           cmap: str = 'gray',
                           save_path: Optional[str] = None) -> None:
    """
    Compare two reconstruction methods side by side.

    Args:
        X1: First reconstruction (2D or 3D)
        X2: Second reconstruction (2D or 3D)
        labels: Labels for the two methods
        slice_idx: Slice index (for 3D, if None uses middle)
        axis: Slicing axis for 3D
        cmap: Colormap
        save_path: Optional path to save figure
    """
    if X1.ndim == 3:
        # Extract slice from 3D volume
        if slice_idx is None:
            slice_idx = X1.shape[0] // 2

        if axis == 'z':
            X1_slice = X1[:, :, slice_idx]
            X2_slice = X2[:, :, slice_idx]
        elif axis == 'y':
            X1_slice = X1[:, slice_idx, :]
            X2_slice = X2[:, slice_idx, :]
        else:  # axis == 'x'
            X1_slice = X1[slice_idx, :, :]
            X2_slice = X2[slice_idx, :, :]
    else:
        X1_slice = X1
        X2_slice = X2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # First reconstruction
    im1 = axes[0].imshow(X1_slice, cmap=cmap, origin='lower')
    axes[0].set_title(labels[0])
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Second reconstruction
    im2 = axes[1].imshow(X2_slice, cmap=cmap, origin='lower')
    axes[1].set_title(labels[1])
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Difference
    diff = X2_slice - X1_slice
    im3 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_histogram(data: np.ndarray,
                   bins: int = 50,
                   title: str = "Density Distribution",
                   xlabel: str = "Density Value",
                   save_path: Optional[str] = None) -> None:
    """
    Plot histogram of density values.

    Args:
        data: 1D array of values
        bins: Number of histogram bins
        title: Plot title
        xlabel: X-axis label
        save_path: Optional path to save figure
    """
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
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_ray_geometry(M: int = 5, N: int = 5,
                      source_positions: Optional[List[Tuple[float, float]]] = None,
                      receiver_positions: Optional[List[Tuple[float, float]]] = None,
                      ray_paths: Optional[List[List[int]]] = None,
                      save_path: Optional[str] = None) -> None:
    """
    Visualize toy problem geometry with sources, receivers, and ray paths.

    Args:
        M: Number of grid rows
        N: Number of grid columns
        source_positions: List of (x, y) coordinates for sources
        receiver_positions: List of (x, y) coordinates for receivers
        ray_paths: List of cell indices for each ray
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid
    for i in range(M + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
    for j in range(N + 1):
        ax.axvline(j, color='gray', linewidth=0.5)

    # Label cells
    for i in range(M):
        for j in range(N):
            cell_idx = i + j * M + 1  # 1-indexed
            ax.text(j + 0.5, i + 0.5, str(cell_idx),
                   ha='center', va='center', fontsize=10, color='blue')

    # Plot sources
    if source_positions:
        for idx, (x, y) in enumerate(source_positions, 1):
            ax.plot(x, y, 'rx', markersize=15, markeredgewidth=3)
            ax.text(x + 0.2, y + 0.2, f'S{idx}', fontsize=12, color='red')

    # Plot receivers
    if receiver_positions:
        for idx, (x, y) in enumerate(receiver_positions, 1):
            ax.plot(x, y, 'go', markersize=12, markeredgewidth=2, fillstyle='none')
            ax.text(x + 0.2, y + 0.2, f'R{idx}', fontsize=12, color='green')

    ax.set_xlim(-0.5, N + 0.5)
    ax.set_ylim(-0.5, M + 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Toy Problem Geometry (5×5 Grid)')
    ax.set_aspect('equal')
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()
