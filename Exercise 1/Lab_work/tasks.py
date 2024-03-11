import numpy as np
import matplotlib.pyplot as plt



def plot_states(x, title_prefix="State"):
    n, d = x.shape  # n: number of states, d: number of plot dots
    
    if n == 1:
        plt.figure(figsize=(8, 6))
        plt.plot(range(d), x[0])
        plt.title(f'{title_prefix} 1')
        plt.ylabel(f'{title_prefix} 1 Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(n, 1, figsize=(8, 6 * n))
    
        for i in range(n):
            axs[i].plot(range(d), x[i])
            axs[i].set_title(f'{title_prefix} {i+1}')
            axs[i].set_ylabel(f'{title_prefix} {i+1} Value')
            axs[i].grid(True)  # Add grid to the plot
    
        plt.tight_layout()
        plt.show()

def checkDimensions(excRows, excCols, matrix):
    """Function that checks if the dimensions of the matrix
    are equal to the wanted ones

    Args:
        excRows (_type_): wanted row dimension
        excCols (_type_): wanted  column dimension
        matrix (_type_): matrix whoes dimensions we are checking

    Returns:
        boolean: true - if dimensions match, 
                 false - if dimensions do not match
    """
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return True
    return rows == excRows \
        and cols == excCols


def LTI(e, f, c, d, x0, u):
    """Function that solves the diskrete state-space system interpretation
    x[k +  1] = e * X[k] + f * u[k]
    y[k] = c * c[k] + d * u[k]

    Args:
        e (matrix_): n x n matrix, where n represents the number of states
        f (matrix): n x m matrix, where m is the number of inputs
        c (matrix): r x n matrix, where r representt the number of outputs
        d (matrix): r x m matrix
        x0 (matrix): initial conditions n x 1 matrix
        u (matrix): control inputs m x N matrix, where N is the number of samples

    Raises:
        AttributeError: if the system matrix are not valid

    Returns:
        Plots the states
    """
    n, n1 = x0.shape
    m, n2 = u.shape
    r, n3 = c.shape
    
    print(n)
    
    if not checkDimensions(n, n, e) or not checkDimensions(n, m, f)\
        or not checkDimensions(r, n, c) or not checkDimensions(r, m, d) or \
            n1 != 1 or n3 != n:
            raise AttributeError("LTI system dimensions are not correct")
    
    X = np.zeros((n, n2)) # State vector
    Y = np.zeros((r, n2)) # Output vector
    
    X[:, 0] = x0[:, 0]

    
    for i in range(0, n2 - 1):
        print(i)
        X[:, i + 1] = np.dot(e, X[:, i]) + np.dot(f, u[:, i])
        Y[:, i] = np.dot(c, X[:, i]) + np.dot(d, u[:, i])
        
    Y [:, n2 - 1] = np.dot(c, X[:, n2 - 1]) + np.dot(d,  u[:, n2 - 1])
        
    return X, Y
    
    

def task4():
    
    e = np.array([
    [1.1269, -0.4940, 0.1129],
    [1., 0., 0.],
    [1., 0., 0.]])
    
    f = np.array([
    [-0.3832, -0.3832],
    [0.5919, 0.8577],
    [0.5191, 0.4546]])
    
    c = np.array([
    [1., 1., 0.]])
    
    d = np.array([[0, 0]])
    
    u = np.zeros((2, 100))
    
    x0 = np.array([[3.], [5.],[-4.]])

    
    X, Y = LTI(e, f, c, d, x0, u)
    
    plot_states(X, "X")
    plot_states(Y, "Y")