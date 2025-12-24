"""
Solvers for X-ray tomography reconstruction.
Implements CGLS (Conjugate Gradient Least Squares) and IRLS (Iteratively Reweighted Least Squares).
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class SolverResult:
    """Container for solver results."""
    x: np.ndarray
    iterations: int
    residuals: list
    objective_values: list
    converged: bool


def cgls(A: np.ndarray, 
         y: np.ndarray, 
         L: Optional[sparse.csr_matrix] = None,
         lam: float = 0.0,
         x0: Optional[np.ndarray] = None,
         tol: float = 1e-6,
         max_iter: Optional[int] = None) -> SolverResult:
    """
    Conjugate Gradient Least Squares (CGLS) solver.
    
    Solves: min_x (1/2)||Ax - y||² + (λ/2)||Lx||²
    
    Equivalent form: min_x (1/2)||[A; √λL]x - [y; 0]||²
    
    Algorithm avoids computing A^T A explicitly and uses:
        - Only one multiplication with A per iteration
        - Only one multiplication with A^T per iteration
        - Recursive update: s_{k+1} = s_k + α_k A d_k
    
    Args:
        A: Measurement matrix (m × n) - can be sparse
        y: Observations (m,)
        L: Regularization matrix (p × n) - if None, no regularization
        lam: Regularization parameter λ ≥ 0
        x0: Initial guess (n,) - if None, starts with zeros
        tol: Convergence tolerance
        max_iter: Maximum iterations - if None, uses n
    
    Returns:
        SolverResult containing solution and convergence info
    """
    if sparse.issparse(A):
        A = A.tocsr()
    if L is not None and sparse.issparse(L):
        L = L.tocsr()
    
    m, n = A.shape
    
    # Initialize
    if x0 is None:
        x = np.zeros(n)
        s = -y.copy()  # s = Ax - y, initially Ax = 0
    else:
        x = x0.copy()
        s = A @ x - y
    
    # Compute initial residual for normal equations: A^T(Ax - y) + λL^TLx
    if L is not None and lam > 0:
        r = A.T @ s + lam * (L.T @ (L @ x))
    else:
        r = A.T @ s
    
    p = -r.copy()
    gamma = np.dot(r, r)
    
    if max_iter is None:
        max_iter = n
    
    # Storage for convergence monitoring
    residuals = [np.linalg.norm(r)]
    objective_values = [compute_objective(x, A, y, L, lam)]
    
    # CGLS iterations
    for k in range(max_iter):
        # Check convergence
        if np.sqrt(gamma) < tol:
            return SolverResult(x, k, residuals, objective_values, True)
        
        # Compute Ap
        q = A @ p
        
        # Add regularization term
        if L is not None and lam > 0:
            Lp = L @ p
            q_full = np.concatenate([q, np.sqrt(lam) * Lp])
            s_full = np.concatenate([s, np.sqrt(lam) * (L @ x)])
            
            # Step size: α = γ_k / ||Ap||² where A includes regularization
            alpha = gamma / np.dot(q_full, q_full)
        else:
            alpha = gamma / np.dot(q, q)
        
        # Update solution
        x = x + alpha * p
        
        # Recursive update of s = Ax - y
        s = s + alpha * q
        
        # Update residual for normal equations
        if L is not None and lam > 0:
            r = A.T @ s + lam * (L.T @ (L @ x))
        else:
            r = A.T @ s
        
        gamma_new = np.dot(r, r)
        beta = gamma_new / gamma
        
        # Update search direction
        p = -r + beta * p
        
        gamma = gamma_new
        
        # Store convergence metrics
        residuals.append(np.sqrt(gamma))
        objective_values.append(compute_objective(x, A, y, L, lam))
    
    return SolverResult(x, max_iter, residuals, objective_values, False)


def irls(A: np.ndarray,
         y: np.ndarray,
         L: sparse.csr_matrix,
         alpha: float,
         x0: Optional[np.ndarray] = None,
         epsilon: float = 1e-8,
         tol: float = 1e-4,
         max_outer_iter: int = 20,
         max_inner_iter: int = 100,
         verbose: bool = True) -> SolverResult:
    """
    Iteratively Reweighted Least Squares (IRLS) for Total Variation regularization.
    
    Solves: min_x (1/2)||Ax - y||² + α||Lx||₁
    
    IRLS approximates the ℓ¹ norm by iteratively solving weighted ℓ² problems:
        ∇||Lx||₁ = L^T W L x
        where W is diagonal with W_ii = 1/max(|γ_i|, ε), γ = Lx
    
    At each iteration k:
        1. Compute W^(k) from x^(k)
        2. Solve: min_x (1/2)||Ax - y||² + (α/2)||W^(k)^(1/2) Lx||²
    
    Args:
        A: Measurement matrix (m × n)
        y: Observations (m,)
        L: Gradient operator (3n³ × n³ for 3D)
        alpha: TV regularization parameter
        x0: Initial guess - if None, uses Tikhonov solution
        epsilon: Smoothing parameter for W to handle γ_i ≈ 0
        tol: Convergence tolerance on relative change ||x^(k+1) - x^k||/||x^k||
        max_outer_iter: Maximum IRLS iterations
        max_inner_iter: Maximum CGLS iterations per IRLS step
        verbose: Print progress
    
    Returns:
        SolverResult with final reconstruction
    """
    if x0 is None:
        # Initialize with Tikhonov solution (small λ)
        print("Initializing with Tikhonov solution...")
        result = cgls(A, y, L, lam=1e-5, max_iter=max_inner_iter)
        x = result.x
    else:
        x = x0.copy()
    
    objective_values = []
    residuals = []
    
    for outer_iter in range(max_outer_iter):
        # Compute γ = Lx
        gamma = L @ x
        
        # Build weight matrix W: W_ii = 1/max(|γ_i|, ε)
        weights = 1.0 / np.maximum(np.abs(gamma), epsilon)
        W = sparse.diags(weights, format='csr')
        W_sqrt = sparse.diags(np.sqrt(weights), format='csr')
        
        # Solve weighted least squares: min (1/2)||Ax-y||² + (α/2)||W^(1/2)Lx||²
        # Equivalent to: min (1/2)||[A; √α W^(1/2)L]x - [y; 0]||²
        # Which is standard form with effective lambda = α and L_eff = W^(1/2)L
        L_weighted = W_sqrt @ L
        
        # Solve with CGLS
        result = cgls(A, y, L_weighted, lam=alpha, x0=x, 
                     tol=tol/10, max_iter=max_inner_iter)
        x_new = result.x
        
        # Compute objective: f(x) = (1/2)||Ax-y||² + α||Lx||₁
        obj = compute_tv_objective(x_new, A, y, L, alpha)
        objective_values.append(obj)
        
        # Check convergence
        rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
        residuals.append(rel_change)
        
        if verbose:
            print(f"IRLS iter {outer_iter + 1}: objective = {obj:.6e}, "
                  f"rel_change = {rel_change:.6e}, inner_iters = {result.iterations}")
        
        if rel_change < tol:
            if verbose:
                print(f"Converged after {outer_iter + 1} iterations")
            return SolverResult(x_new, outer_iter + 1, residuals, 
                              objective_values, True)
        
        x = x_new
    
    if verbose:
        print(f"Maximum iterations ({max_outer_iter}) reached")
    
    return SolverResult(x, max_outer_iter, residuals, objective_values, False)


def compute_objective(x: np.ndarray, 
                     A: np.ndarray, 
                     y: np.ndarray,
                     L: Optional[sparse.csr_matrix],
                     lam: float) -> float:
    """
    Compute Tikhonov objective: (1/2)||Ax - y||² + (λ/2)||Lx||²
    """
    residual = A @ x - y
    obj = 0.5 * np.dot(residual, residual)
    
    if L is not None and lam > 0:
        Lx = L @ x
        obj += 0.5 * lam * np.dot(Lx, Lx)
    
    return obj


def compute_tv_objective(x: np.ndarray,
                        A: np.ndarray,
                        y: np.ndarray,
                        L: sparse.csr_matrix,
                        alpha: float) -> float:
    """
    Compute Total Variation objective: (1/2)||Ax - y||² + α||Lx||₁
    """
    residual = A @ x - y
    data_term = 0.5 * np.dot(residual, residual)
    
    Lx = L @ x
    tv_term = alpha * np.sum(np.abs(Lx))
    
    return data_term + tv_term


def gradient_descent(Q: np.ndarray,
                    b: np.ndarray,
                    step_size: float,
                    x0: Optional[np.ndarray] = None,
                    max_iter: int = 1000,
                    tol: float = 1e-6) -> SolverResult:
    """
    Gradient Descent for quadratic problem: min_x (1/2)x^T Q x + b^T x
    
    Gradient: ∇f(x) = Qx + b
    Update: x^(k+1) = x^(k) - α(Qx^(k) + b)
    
    Args:
        Q: Positive definite matrix (n × n)
        b: Linear term (n,)
        step_size: Step size α (must satisfy α < 2/λ_max(Q))
        x0: Initial guess
        max_iter: Maximum iterations
        tol: Convergence tolerance on ||∇f(x)||
    
    Returns:
        SolverResult with solution
    """
    n = Q.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    objective_values = []
    residuals = []
    
    for k in range(max_iter):
        # Compute gradient
        grad = Q @ x + b
        grad_norm = np.linalg.norm(grad)
        
        # Compute objective
        obj = 0.5 * x.T @ Q @ x + b.T @ x
        objective_values.append(obj)
        residuals.append(grad_norm)
        
        if grad_norm < tol:
            return SolverResult(x, k, residuals, objective_values, True)
        
        # Update
        x = x - step_size * grad
    
    return SolverResult(x, max_iter, residuals, objective_values, False)
