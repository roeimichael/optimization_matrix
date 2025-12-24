import numpy as np
from scipy import sparse
from typing import Optional
from dataclasses import dataclass


@dataclass
class SolverResult:
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
    """Conjugate Gradient Least Squares: min (1/2)||Ax - y||^2 + (lam/2)||Lx||^2"""
    if sparse.issparse(A):
        A = A.tocsr()
    if L is not None and sparse.issparse(L):
        L = L.tocsr()

    m, n = A.shape

    if x0 is None:
        x = np.zeros(n)
        s = -y.copy()
    else:
        x = x0.copy()
        s = A @ x - y

    if L is not None and lam > 0:
        r = A.T @ s + lam * (L.T @ (L @ x))
    else:
        r = A.T @ s

    p = -r.copy()
    gamma = np.dot(r, r)

    if max_iter is None:
        max_iter = n

    residuals = [np.linalg.norm(r)]
    objective_values = [compute_objective(x, A, y, L, lam)]

    for k in range(max_iter):
        if np.sqrt(gamma) < tol:
            return SolverResult(x, k, residuals, objective_values, True)

        q = A @ p

        if L is not None and lam > 0:
            Lp = L @ p
            q_full = np.concatenate([q, np.sqrt(lam) * Lp])
            s_full = np.concatenate([s, np.sqrt(lam) * (L @ x)])
            alpha = gamma / np.dot(q_full, q_full)
        else:
            alpha = gamma / np.dot(q, q)

        x = x + alpha * p
        s = s + alpha * q

        if L is not None and lam > 0:
            r = A.T @ s + lam * (L.T @ (L @ x))
        else:
            r = A.T @ s

        gamma_new = np.dot(r, r)
        beta = gamma_new / gamma
        p = -r + beta * p
        gamma = gamma_new

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
    """Iteratively Reweighted Least Squares: min (1/2)||Ax - y||^2 + alpha||Lx||_1"""
    if x0 is None:
        result = cgls(A, y, L, lam=1e-5, max_iter=max_inner_iter)
        x = result.x
    else:
        x = x0.copy()

    objective_values = []
    residuals = []

    for outer_iter in range(max_outer_iter):
        gamma = L @ x
        weights = 1.0 / np.maximum(np.abs(gamma), epsilon)
        W = sparse.diags(weights, format='csr')
        W_sqrt = sparse.diags(np.sqrt(weights), format='csr')
        L_weighted = W_sqrt @ L

        result = cgls(A, y, L_weighted, lam=alpha, x0=x,
                     tol=tol/10, max_iter=max_inner_iter)
        x_new = result.x

        obj = compute_tv_objective(x_new, A, y, L, alpha)
        objective_values.append(obj)

        rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
        residuals.append(rel_change)

        if verbose:
            print(f"IRLS iter {outer_iter + 1}: obj = {obj:.6e}, "
                  f"rel_change = {rel_change:.6e}, inner_iters = {result.iterations}")

        if rel_change < tol:
            return SolverResult(x_new, outer_iter + 1, residuals,
                              objective_values, True)

        x = x_new

    return SolverResult(x, max_outer_iter, residuals, objective_values, False)


def gradient_descent(Q: np.ndarray,
                    b: np.ndarray,
                    step_size: float,
                    x0: Optional[np.ndarray] = None,
                    max_iter: int = 1000,
                    tol: float = 1e-6) -> SolverResult:
    """Gradient Descent for quadratic: min (1/2)x^T Q x + b^T x"""
    n = Q.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    objective_values = []
    residuals = []

    for k in range(max_iter):
        grad = Q @ x + b
        grad_norm = np.linalg.norm(grad)

        obj = 0.5 * x.T @ Q @ x + b.T @ x
        objective_values.append(obj)
        residuals.append(grad_norm)

        if grad_norm < tol:
            return SolverResult(x, k, residuals, objective_values, True)

        x = x - step_size * grad

    return SolverResult(x, max_iter, residuals, objective_values, False)


def compute_objective(x: np.ndarray,
                     A: np.ndarray,
                     y: np.ndarray,
                     L: Optional[sparse.csr_matrix],
                     lam: float) -> float:
    """Compute Tikhonov objective: (1/2)||Ax - y||^2 + (lam/2)||Lx||^2"""
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
    """Compute Total Variation objective: (1/2)||Ax - y||^2 + alpha||Lx||_1"""
    residual = A @ x - y
    data_term = 0.5 * np.dot(residual, residual)

    Lx = L @ x
    tv_term = alpha * np.sum(np.abs(Lx))

    return data_term + tv_term
