# Theoretical Questions - Guidance and Solutions Template

This document provides guidance for answering the theoretical questions in the project.

## Part 1: Toy Problem Setup

### Q1: Apply Midpoint Rule to Discretize the Integral

**Question:** Apply the midpoint rule to discretize the integral in equation (1):
```
y = ∫_γ(l) ρ(γ(l)) dl
```

**Approach:**
1. The ray path γ can be horizontal, vertical, or diagonal
2. For each type, determine Δl:
   - Horizontal rays: Δl = Δx̃
   - Vertical rays: Δl = Δỹ  
   - Diagonal rays: Δl = √(Δx̃² + Δỹ²)

3. Apply midpoint rule:
```
∫ f(s) ds ≈ Σⱼ f(s̃ⱼ) Δs
```

**Solution Template:**
For a ray passing through cells with indices j₁, j₂, ..., jₖ:
```
y ≈ Σᵢ ρ(x̃ⱼᵢ) · Δlᵢ
```
where Δlᵢ depends on ray direction through cell jᵢ.

---

### Q2: Write Relation Between y, A, and x

**Question:** Using the approximation from Q1, write y = Ax where:
- y: measurement vector
- A: ray-path matrix
- x: column-stack representation of density

**Answer:**
```
y = Ax
```

where:
- **A[i,j]**: Distance traversed by ray i in cell j
- If ray i doesn't pass through cell j: A[i,j] = 0
- If ray i passes through cell j: A[i,j] = Δl (horizontal/vertical/diagonal)

**Type of relation:** Linear system

**Matrix construction based on Figure 1:**
- 7 sources × 5 receivers = 35 measurements (35 rows)
- 5×5 grid = 25 unknowns (25 columns)
- Each row represents one ray path
- Non-zero entries indicate which cells the ray passes through

---

### Q5: Analysis of the Ill-Posed Inverse Problem

**Question:** 
1. Number of unknowns?
2. Number of observations?
3. Set of solutions for min_x (1/2)||Ax - y||²?

**Answers:**

**Number of unknowns:** 25 (5×5 grid)

**Number of observations:** 35 (7 sources × 5 receivers)

**Solution set analysis:**
- Since m=35 > n=25, the system is overdetermined
- However, A may not have full rank (rank < 25)
- If rank(A) < n: infinite solutions (non-unique)
- Solution set is **convex** (intersection of affine space with ℝⁿ)
- Not necessarily a singleton (may have multiple solutions)

**Key insight:** Even though m > n, the specific geometry of ray paths may result in rank deficiency, making the problem ill-posed.

---

## Part 2: Large Scale Optimization

### Q6: Derive Normal Equations

**Question:** Apply first-order optimality condition to:
```
min_x (1/2)||Ax - y||² + (λ/2)||Lx||²
```

**Solution:**

Objective function:
```
f(x) = (1/2)(Ax - y)ᵀ(Ax - y) + (λ/2)(Lx)ᵀ(Lx)
```

Gradient:
```
∇f(x) = Aᵀ(Ax - y) + λLᵀLx = 0
```

**Normal equations:**
```
(AᵀA + λLᵀL)x = Aᵀy
```

**Closed-form solution:**
```
x* = (AᵀA + λLᵀL)⁻¹ Aᵀy
```

**Note:** Computing this directly is infeasible for large problems due to:
- Dense matrix multiplication: AᵀA
- Matrix inversion: O(n³) complexity
- Need iterative methods like CGLS instead

---

### Q7: Reformulate as Quadratic Problem

**Question:** Bring problem to form f(x) = (1/2)xᵀQx + bᵀx + c

**Solution:**

Starting from:
```
f(x) = (1/2)||Ax - y||² + (λ/2)||Lx||²
```

Expand:
```
f(x) = (1/2)(xᵀAᵀAx - 2yᵀAx + yᵀy) + (λ/2)(xᵀLᵀLx)
     = (1/2)xᵀ(AᵀA + λLᵀL)x - (Aᵀy)ᵀx + (1/2)yᵀy
```

Therefore:
```
Q = AᵀA + λLᵀL
b = -Aᵀy
c = (1/2)yᵀy
```

**Why Q ≻ 0 (positive definite)?**

For any x ≠ 0:
```
xᵀQx = xᵀAᵀAx + λxᵀLᵀLx
     = ||Ax||² + λ||Lx||²
     > 0
```

This is strictly positive because:
1. ||Ax||² ≥ 0 (always)
2. λ||Lx||² > 0 when λ > 0 and x ≠ 0 (since L has full column rank)

**Implication:** Unique minimizer exists (solution is a singleton)

---

### Q8: Gradient Descent Algorithm

**Question:** State GD algorithm for minimizing f(x) = (1/2)xᵀQx + bᵀx + c

**Algorithm:**

Initialize x⁽⁰⁾
For k = 0, 1, 2, ...:
1. Compute gradient: g⁽ᵏ⁾ = Qx⁽ᵏ⁾ + b
2. Update: x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ - α g⁽ᵏ⁾
3. Check convergence: ||g⁽ᵏ⁾|| < ε

**Gradient in terms of A, L, λ:**
```
g = Qx + b = (AᵀA + λLᵀL)x - Aᵀy
          = Aᵀ(Ax - y) + λLᵀ(Lx)
```

**Step size α:**
- Must satisfy: 0 < α < 2/λ_max(Q)
- For fastest convergence: α_opt = 2/(λ_min + λ_max)

---

### Q9: Convergence Analysis

**Question:** 
1. Values of step-size ensuring convergence?
2. Condition number of Q?
3. Iterations to reduce objective by factor 10?

**Answers:**

**Step-size for convergence:**
```
0 < α < 2/λ_max(Q)
```
where λ_max(Q) is the largest eigenvalue of Q.

**Condition number:**
```
κ(Q) = λ_max(Q) / λ_min(Q)
```

Compute using:
```python
eigenvalues = np.linalg.eigvalsh(Q)
kappa = eigenvalues[-1] / eigenvalues[0]
```

**Iterations for 10× reduction:**

GD convergence rate: 
```
f(x⁽ᵏ⁾) - f* ≤ ((κ-1)/(κ+1))²ᵏ (f(x⁽⁰⁾) - f*)
```

For 10× reduction:
```
((κ-1)/(κ+1))²ᵏ ≤ 0.1
k ≥ log(0.1) / (2 log((κ-1)/(κ+1)))
k ≥ 1.15 / log(1 + 2/(κ-1))
```

For large κ: k ≈ O(κ)

---

## Part 3: Total Variation Regularization

### Q12: Compare ℓ¹ vs ℓ² for Curve Fitting

**Question:** Compute ||Dx f||₁ and ||Dx f||₂ for both f₁(x) and f₂(x)

**Approach:**

For f₁(x) (piecewise constant):
- Sharp jump at x=0
- ||Dx f₁||₁ = sum of absolute derivative values
- ||Dx f₁||₂ = sqrt(sum of squared derivative values)

For f₂(x) (smooth sigmoid):
- Gradual transition
- Same ||Dx f₂||₁ as f₁ (approximately)
- Much larger ||Dx f₂||₂ due to non-zero derivatives everywhere

**Key insight:**
- ℓ¹ norm treats both equally (total variation)
- ℓ² norm penalizes smooth functions more
- For piecewise constant data: ℓ¹ is preferable

---

### Q13: Properties of TV Problem

**Question:** Is problem (6) convex? Strictly convex? Smooth?

```
min_x (1/2)||Ax - y||² + α||Lx||₁
```

**Answers:**

**Convex?** YES
- (1/2)||Ax - y||² is convex (quadratic)
- ||Lx||₁ is convex (composition of convex ||·||₁ with linear L)
- Sum of convex functions is convex

**Strictly convex?** NO
- The ℓ¹ term is not strictly convex
- However, if A has full column rank, the quadratic term ensures uniqueness

**Smooth?** NO
- ||Lx||₁ is not differentiable at points where (Lx)ᵢ = 0
- This is why we use IRLS (smoothing approximation)

---

### Q14: IRLS Derivation

**Question:**
1. Show ∂||Lx||₁/∂xₖ = Σᵢ Lᵢ,ₖ γᵢ/|γᵢ| where γ = Lx
2. Show ∇||Lx||₁ = LᵀWLx with diagonal W
3. Write first-order optimality condition
4. Interpret as weighted least squares

**Solutions:**

**Part 1:** Derivative of ℓ¹ norm
```
||Lx||₁ = Σᵢ |γᵢ| where γᵢ = (Lx)ᵢ = Σₖ Lᵢ,ₖ xₖ

∂||Lx||₁/∂xₖ = Σᵢ sign(γᵢ) · ∂γᵢ/∂xₖ
              = Σᵢ (γᵢ/|γᵢ|) · Lᵢ,ₖ
```

**Part 2:** Matrix form with W
```
∂||Lx||₁/∂xₖ = Σᵢ Lᵢ,ₖ · Wᵢ,ᵢ · γᵢ

where Wᵢ,ᵢ = 1/|γᵢ| (or 1/ε if |γᵢ| < ε)

In matrix form:
∇||Lx||₁ = LᵀW(Lx) = LᵀWγ
```

Since γ = Lx:
```
∇||Lx||₁ = LᵀWLx
```

**Part 3:** First-order optimality
```
∇f(x) = Aᵀ(Ax - y) + α LᵀWLx = 0
(AᵀA + α LᵀWL)x = Aᵀy
```

**Part 4:** Weighted least squares (with fixed W)
```
min_x (1/2)||Ax - y||² + (α/2)||W^(1/2)Lx||²
```

This is equivalent to:
```
min_x (1/2) ||[A; √α W^(1/2)L]x - [y; 0]||²
```

---

## Summary Checklist

Theoretical questions completed:
- [ ] Q1: Midpoint rule discretization
- [ ] Q2: Linear system y = Ax
- [ ] Q5: Ill-posed problem analysis
- [ ] Q6: Normal equations derivation
- [ ] Q7: Quadratic reformulation, Q ≻ 0 proof
- [ ] Q8: Gradient descent algorithm
- [ ] Q9: Step-size, condition number, iterations
- [ ] Q12: ℓ¹ vs ℓ² comparison
- [ ] Q13: Convexity properties
- [ ] Q14: IRLS derivation

Code-based questions (see notebooks):
- [ ] Q3: Derivative matrices (notebook: Q3_Q4_derivatives.ipynb)
- [ ] Q4: Gradient visualization (notebook: Q3_Q4_derivatives.ipynb)
- [ ] Q10: CGLS implementation (notebook: Q10_Q11_cgls.ipynb)
- [ ] Q11: Small bag reconstruction (notebook: Q10_Q11_cgls.ipynb)
- [ ] Q15: IRLS implementation (notebook: Q15_Q16_irls.ipynb)
- [ ] Q16: Large bag reconstruction (notebook: Q15_Q16_irls.ipynb)
