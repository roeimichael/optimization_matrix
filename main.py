import scipy.io
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import math
from scipy.sparse import vstack
from scipy.sparse.linalg import lsqr

#Compute the derivative along the rows (dy) for a given matrix X1.
def compute_dy(X1, M, N):

    # Initialize an empty matrix for the row derivatives
    dy = lil_matrix((M, N))

    # Compute the derivatives for all rows except the last one
    for i in range(M - 1):  # The last row will be zero
        for j in range(N):
            dy[i, j] = X1[i + 1, j] - X1[i, j]

    return dy


#Compute the derivative along the columns (dx) for a given matrix X1.
def compute_dx(X1, M, N):

    # Initialize an empty matrix for the column derivatives
    dx = lil_matrix((M, N))

    # Compute the derivatives for all columns except the last one
    for i in range(M):  # For each row
        for j in range(N - 1):  # The last column will be zero
            dx[i, j] = X1[i, j + 1] - X1[i, j]

    return dx


# Compute the derivative along the x-axis (dx) for a given 3D matrix X1.
def compute_dx2(X1, M, N, P):
    # Initialize an empty matrix for the x-axis derivatives (dx).
    dx = lil_matrix((M, N, P))

    # Compute the derivatives for all columns except the last one
    for i in range(M):  # For each row
        for j in range(N - 1):  # For each column except the last
            for k in range(P):  # For each z-layer
                dx[i, j, k] = X1[i, j + 1, k] - X1[i, j, k]

    return dx

# Compute the derivative along the y-axis (dy) for a given 3D matrix X1.
def compute_dy2(X1, M, N, P):
    # Initialize an empty matrix for the y-axis derivatives (dy).
    dy = lil_matrix((M, N, P))

    # Compute the derivatives for all rows except the last one
    for i in range(M - 1):  # The last row will be zero
        for j in range(N):  # For each column
            for k in range(P):  # For each z-layer
                dy[i, j, k] = X1[i + 1, j, k] - X1[i, j, k]

    return dy

# Compute the derivative along the z-axis (dz) for a given 3D matrix X1.
def compute_dz2(X1, M, N, P):
    # Initialize an empty matrix for the z-axis derivatives (dz).
    dz = lil_matrix((M, N, P))

    # Compute the derivatives for all z-layers except the last one
    for i in range(M):  # For each row
        for j in range(N):  # For each column
            for k in range(P - 1):  # The last z-layer will be zero
                dz[i, j, k] = X1[i, j, k + 1] - X1[i, j, k]

    return dz



#Compute the sparse matrix Dx for a given matrix dimensions (M, N).
def compute_Dx(M, N):

    Dx = lil_matrix((M * N, M * N))
    
    # Fill the matrix Dx
    for i in range(N * M - M):
        Dx[i, i] = -1
        Dx[i, i + M] = 1

    return Dx

#Compute the sparse matrix Dy for a given matrix dimensions (M, N).
def compute_Dy(M, N):

    Dy = lil_matrix((M * N, M * N))

    # Fill the matrix Dy
    for i in range(N * M):
        if (i + 1) % M != 0:
            Dy[i, i + 1] = 1
            Dy[i, i] = -1

    return Dy


def compute_Dx2(M, N, P):

    Dx = lil_matrix((M * N * P, M * N * P))
    for k in range(P):
    # Fill the matrix Dx
        for i in range(N * M - M):
            Dx[i+ (k * M * N), i+ (k * M * N)] = -1
            Dx[i+ (k * M * N), i + M + (k * M * N)] = 1

    return Dx

def compute_Dy2(M, N, P):

    Dy = lil_matrix((M * N * P, M * N * P))

    # Fill the matrix Dy
    for k in range(P):
        for i in range(N * M):
            if (i + 1) % M != 0:
                Dy[i+ (k * M * N), i + 1+ (k * M * N)] = 1
                Dy[i+ (k * M * N), i+ (k * M * N)] = -1

    return Dy

def compute_Dz2(M, N, P):

    Dz = lil_matrix((M * N * P, M * N * P))
    # Fill the matrix Dy
    for i in range(N * M * P - M * N):
        Dz[i, i] = -1
        Dz[i, i + N * M] = 1

    return Dz



def cgls(A, y, L, lambda_, x0, tol=1e-6, imax=800):
    """
    Conjugate Gradient Least Squares (CGLS) solver.
    Solves min_x ||Ax - y||^2 + lambda * ||Lx||^2.
    """
    # Initialize variables
    xk = x0
    B=scipy.sparse.vstack([A, np.sqrt(lambda_) * L])
    y_padded = np.concatenate([y, np.zeros((L.shape[0], 1))])

    y = y.reshape(-1, 1)
    xk = x0.reshape(-1, 1)

    sk = B @ xk - y_padded  # Initial residual
    #print(y.shape)
    #print(A.shape)
    #print(xk.shape)
    #print(sk.shape)
    #print(np.dot(A,xk).shape)
    #print(np.dot(A.T , sk).shape)
    #print(np.dot(L.T , np.dot(L , xk)).shape)
    #gk = A.T @ sk + lambda_ * (L.T @ (L @ xk))  # Initial gradient
    gk = B.T @ sk
    #print(gk.shape)
    dk = -gk  # Initial direction
    err = []
    for k in range(imax):
        #Adk = A @ dk
        #Ldk = L @ dk
        Bdk= B @ dk

        # Compute alpha_k
        #alpha_k = (gk.T @ gk)[0, 0] / ((Adk.T @ Adk)[0, 0] + lambda_ * (Ldk.T @ Ldk)[0, 0])
        #alpha_k = (gk.T @ gk)[0, 0] / ((Bdk.T @ Bdk)[0, 0])
        alpha_k = np.dot(gk.ravel(), gk.ravel()) / np.dot(Bdk.ravel(), Bdk.ravel())
        # Update xk
        xk = xk + alpha_k * dk

        # Update sk
        sk = sk + alpha_k * Bdk
        #sk = sk + alpha_k * (Adk + lambda_* (L.T @ Ldk))


        # Compute new gradient
        #gk_new = A.T @ sk + lambda_ * (L.T @ (L @ xk))
        gk_new = B.T @ sk

        #print(f"Iteration {k + 1}, norm(gk_new): {np.linalg.norm(gk) / np.linalg.norm(y_padded)}")

        # Check for convergence
        #if np.linalg.norm(gk_new) < tol:
            #break
        err.append(np.linalg.norm(gk) / np.linalg.norm(y_padded))
        if np.linalg.norm(gk) / np.linalg.norm(y_padded) < tol:
            break

        # Compute beta_k
        beta_k = np.dot(gk_new.ravel(), gk_new.ravel()) / np.dot(gk.ravel(), gk.ravel())

        # Update direction
        dk = -gk_new + beta_k * dk


        # Update gradient
        gk = gk_new

    return xk, np.linalg.norm(gk) / np.linalg.norm(y_padded), err




'''
def solve_with_lsqr(A, y, L, lambda_, tol=1e-6, imax=100):

    # Construct the augmented system
    B = scipy.sparse.vstack([A, np.sqrt(lambda_) * L])
    y_padded = np.concatenate([y, np.zeros((L.shape[0], 1))])

    # Solve using lsqr
    result = scipy.sparse.linalg.lsqr(B, y_padded, damp=np.sqrt(lambda_), atol=1e-6, btol=1e-6)

    # Extract solution and iteration info
    x_solution = result[0]  # Solution vector
    num_iterations = result[2]  # Number of iterations performed
    print(num_iterations)

    return x_solution, num_iterations
'''

def q3():
    print("===================")
    print("Question 3:")
    print()

    print("Dx =", compute_Dx(5, 5))
    print()
    print("Dy =", compute_Dy(5, 5))

    print()
    print("===================")




def q4(i=1):
    # Load X1.mat
    
    data = scipy.io.loadmat('X1.mat')
    X1 = data['X1']
    X2 = scipy.io.loadmat("X2.mat")["X2"]
    X3 = scipy.io.loadmat("X3.mat")["X3"]
    
    #M, N = X1.shape
    if i==1:
        X=X1
    elif i==2:
        X=X2
    else:
        X=X3
    print(f"X{i} shape: {X.shape}")
    m, n = X.shape
    Dx = compute_Dx(m, n)
    Dy = compute_Dy(m, n)

    #dx = compute_dx(X1, M, N)
    #dy = compute_dy(X1, M, N)
    dx = Dx @ X.reshape([-1, 1])
    dy = Dy @ X.reshape([-1, 1])

    # Convert sparse matrices to dense arrays for visualization
    #dx_dense = dx.toarray()
    #dy_dense = dy.toarray()

    # Calculate g(i,j) = sqrt(dx(i,j)^2 + dy(i,j)^2)
    #g = np.sqrt(dx_dense**2 + dy_dense**2)
    g=np.sqrt(dx**2 + dy**2)
    # Plotting 4 graphs in one figure
    plt.figure(figsize=(15, 10))

    # Plot X1 (original matrix)
    plt.subplot(2, 2, 1)  # First graph
    plt.imshow(X, cmap='viridis', aspect='auto')
    plt.title('Original Matrix (X)' + str(i))
    plt.colorbar()

    # Plot the x-derivative (dx)
    plt.subplot(2, 2, 2)
    plt.imshow(dx, cmap='viridis', aspect='auto')
    plt.title('Derivative in X (dx)')
    plt.colorbar()  # Show color bar

    # Plot the y-derivative (dy)
    plt.subplot(2, 2, 3)
    plt.imshow(dy, cmap='viridis', aspect='auto')
    plt.title('Derivative in Y (dy)')
    plt.colorbar()  # Show color bar

    # Plot g (gradient magnitude)
    plt.subplot(2, 2, 4)
    plt.imshow(g, cmap='viridis', aspect='auto')
    plt.title('Gradient Magnitude (g)')
    plt.colorbar()  # Show color bar

    plt.tight_layout()  # Adjust the layout to prevent overlap

    # Save the figure to a file
    plt.savefig('figure1'+str(i)+'.png')

    # Optionally, you can close the figure to free up memory
    plt.close()
    
    
def q9():
    SQRT2=np.sqrt(2)
    A = np.array([
        [0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, 0]
    ])
    Dx_L=compute_Dx(5,5)
    Dy_L=compute_Dy(5,5)

    # Now, L should be a sparse matrix
    L = lil_matrix((50, 25))  # Create L as sparse matrix

    # Fill L with Dx_L and Dy_L
    #L[:Dx_L.shape[0], :] = Dx_L
    #L[Dx_L.shape[0]:, :] = Dy_L
    L = vstack([Dx_L, Dy_L])
    L_dense = L.toarray()
    lambdaa=10**-5
    Q = np.dot(A.T, A) + lambdaa * np.dot(L_dense.T, L_dense)
    print(Q.shape)
    eigenvalues = np.linalg.eigvals(Q)
    max_eigenvalue = max(eigenvalues)
    min_eigenvalue = min(eigenvalues)
    print(max_eigenvalue)
    print(min_eigenvalue)
    k=max_eigenvalue/min_eigenvalue
    print(k)
    print(1-(1/k))
    print(math.log(0.1)/math.log(1-(1/k)))
    
    
def q10():
    data = scipy.io.loadmat("Y.mat")["Y"]
    y = np.array([[0.2322],
                [1.9477],
                [1.5065],
                [1.5065],
                [1.9477],
                [1.5065],
                [0.3361],
                [2.5065]])
    lambdaa=10**-5
    SQRT2=np.sqrt(2)
    A = np.array([
            [0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0, 0, 0, 0, 0, SQRT2, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, SQRT2, 0, 0, 0, 0]
        ])
    Dx_L=compute_Dx(5,5)
    Dy_L=compute_Dy(5,5)

    # Now, L should be a sparse matrix
    L = lil_matrix((50, 25))  # Create L as sparse matrix

    # Fill L with Dx_L and Dy_L
    #L[:Dx_L.shape[0], :] = Dx_L
    #L[Dx_L.shape[0]:, :] = Dy_L
    L = vstack([Dx_L, Dy_L])
    L_dense = L.toarray()
    x0 = np.zeros((25, 1))  # Initial guess for x (25 x 1)
    # Solve the problem
    x_solution, error, err_list = cgls(A, y, L, lambdaa, x0)
    #x_superb=solve_with_lsqr(A, y, L, lambdaa, tol=1e-6, imax=100)
    print("Solution x:", x_solution)

    total_cost = np.linalg.norm(A @ x_solution - y)**2 + lambdaa * np.linalg.norm(L @ x_solution)**2
    print(f"Total cost (||Ax - y||^2 + λ ||Lx||^2): {total_cost}")
    plt.figure()
    plt.plot(err_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Norm of the Gradient")
    plt.title("Algorithm Convergence per Iteration")
    # Save the figure to a file
    plt.savefig('convergence_plot.png')
    # Optionally, you can close the figure to free up memory
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(x_solution.reshape([5, 5]), cmap="Grays")
    ax.set_title('CGLS output')
    plt.tight_layout()
    # Save the figure to a file
    plt.savefig('cgls_output.png')
    # Optionally, you can close the figure to free up memory
    plt.close()

def eq_4(A, lamda, L, x, y):
    first_term = scipy.sparse.vstack([A, np.sqrt(lamda) * L]) @ x
    
    zeros_dim = first_term.shape[0] - y.shape[0]

    second_term = np.hstack((y, np.zeros(zeros_dim)))
    
    vec = first_term - second_term
    return vec.T @ vec

def qus11(lambdaa=10**-5, Large=False):
    if Large:
        y = scipy.io.loadmat("Large/y.mat")["y"]
        A = scipy.io.loadmat("Large/A.mat")["A"]
        x0 = np.zeros((49*49*49, 1))  # Initial guess for x (19*19*19 x 1)
        Dx_L=compute_Dx2(49 ,49 , 49)
        Dy_L=compute_Dy2(49 ,49 , 49)
        Dz_L=compute_Dz2(49 ,49 , 49)
        size=49
    else: # Small
        y = scipy.io.loadmat("Small/y.mat")["y"]
        A = scipy.io.loadmat("Small/A.mat")["A"]
        x0 = np.zeros((19*19*19, 1))  # Initial guess for x (19*19*19 x 1)
        Dx_L=compute_Dx2(19 ,19 , 19)
        Dy_L=compute_Dy2(19 ,19 , 19)
        Dz_L=compute_Dz2(19 ,19 , 19)
        size=19

    #y = scipy.io.loadmat("Large/y.mat")["y"]
    #A = scipy.io.loadmat("Large/A.mat")["A"]
    #print(A.shape)
    
    L = vstack([Dx_L, Dy_L, Dz_L])
    #print(Dx_L.shape)
    #print(Dy_L.shape)
    #print(Dz_L.shape)
    
    x_opt,err,errlist=cgls(A, y, L, lambdaa, x0)
    x_opt1 = x_opt.reshape([size, size, size])
    for _ in range(size):
       show_image(x_opt1[:, :, _], title=f"q10_image_{_+1}", save=True)
'''
    obj_value = eq_4(
            A=A,
            lamda=lambdaa,
            L=L,
            x=np.squeeze(x_opt), 
            y=np.squeeze(y)
        )
    print(f"For lambda={lambdaa}, objective_value = {obj_value}")
'''

def qus15(lambdaa=10**-5, Large=False):
    if Large:
        y_data = scipy.io.loadmat("Large/y.mat")["y"]
        A_matrix = scipy.io.loadmat("Large/A.mat")["A"]
        initial_guess = np.zeros((49*49*49, 1))  # Initial guess for x (19*19*19 x 1)
        Dx_operator=compute_Dx2(49 ,49 , 49)
        Dy_operator=compute_Dy2(49 ,49 , 49)
        Dz_operator=compute_Dz2(49 ,49 , 49)
    else: # Small
        y_data = scipy.io.loadmat("Small/y.mat")["y"]
        A_matrix = scipy.io.loadmat("Small/A.mat")["A"]
        initial_guess = np.zeros((19*19*19, 1))  # Initial guess for x (19*19*19 x 1)
        Dx_operator = compute_Dx2(19, 19, 19)
        Dy_operator = compute_Dy2(19, 19, 19)
        Dz_operator = compute_Dz2(19, 19, 19)
    # Load data from .mat files
    #y_data = scipy.io.loadmat("Small/y.mat")["y"]
    #A_matrix = scipy.io.loadmat("Small/A.mat")["A"]
    y_vector = np.squeeze(y_data)
    x_dim = A_matrix.shape[1]

    # Compute differential operators
    #Dx_operator = compute_Dx2(19, 19, 19)
    #Dy_operator = compute_Dy2(19, 19, 19)
    #Dz_operator = compute_Dz2(19, 19, 19)
    L_matrix = scipy.sparse.vstack([Dx_operator, Dy_operator, Dz_operator])
    regularization_param = 10**-5

    # Initial guess for x
    #initial_guess = np.zeros((19 * 19 * 19, 1))
    X_solution, initial_error,err_list = cgls(A_matrix, y_data, L_matrix, regularization_param, initial_guess)
    alpha = 1 / 2
    max_iterations = 10000
    epsilon = 1e-12
    tolerance = 1e-6
    row_indices = [_ for _ in range(L_matrix.shape[0])]  # number of rows of L_matrix
    IRLS_num_iter_list = []
    X_solution = np.squeeze(X_solution.reshape([-1, 1]))
    IRLS_max_iterations = 10000
    IRLS_tolerance = 1e-6
    i=0
    for iteration in range(max_iterations):
        i+=1
        # Compute weights based on the current solution
        weight_values = 1 / (np.sqrt(np.absolute(L_matrix @ X_solution)) + np.sqrt(epsilon))
        
        # Create a lil_matrix and assign weights to the diagonal
        weight_matrix = lil_matrix((L_matrix.shape[0], L_matrix.shape[0]))
        weight_matrix[row_indices, row_indices] = weight_values
        
        # Perform CGLS optimization with the weighted L_matrix
        IRLS_x_opt, IRLS_error, err_list = cgls(
            A_matrix,
            L=(weight_matrix @ L_matrix),
            y=y_data,
            x0=initial_guess,
            lambda_=np.sqrt(alpha),
            imax=IRLS_max_iterations,
            tol=IRLS_tolerance,
        )
        
        # Compute the objective function value
        objective_value = (1 / 2) * (IRLS_x_opt.T @ A_matrix.T @ A_matrix @ IRLS_x_opt +
                                     alpha * IRLS_x_opt.T @ L_matrix.T @ weight_matrix @ weight_matrix @ L_matrix @ IRLS_x_opt -
                                     IRLS_x_opt.T @ A_matrix.T @ y_data -
                                     y_data.T @ A_matrix @ IRLS_x_opt +
                                     y_vector.T @ y_vector)
        print(f"Objective value in iteration {iteration} is: {objective_value}")
        
        # Update the solution
        X_solution = IRLS_x_opt
        
        # Check for convergence
        if IRLS_error < tolerance:
            break

    # Reshape the optimal solution to the original dimensions
    
    if Large:
        x_optimal = X_solution.reshape([49, 49, 49])
        # Plot the optimal solution
        plt.figure()
        plt.imshow(x_optimal[:, :, 9], cmap="gray")
        plt.title("Optimal Solution")
        plt.show()
        for _ in range(49):
            show_image(x_optimal[:, :, _], title=f"q16_image_{_+1}", save=True)

    else:   # Small
        x_optimal = X_solution.reshape([19, 19, 19])
    print(f"Optimal solution: {x_optimal} with number of iterations: {i}")
    return X_solution



def show_image(x, title="", save=False):
    plt.figure()
    plt.imshow(x, cmap="Grays")
    if title:
        plt.title(title)
    if save:
        plt.savefig(title + ".png")
    #plt.show()
    
