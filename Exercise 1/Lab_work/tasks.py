import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp



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



def f123(x):
    return x**3 - np.cos(5*x)

def task1():
    segments = [(-1, -0.2), (-0.2, 0.4), (0.4, 1)]
    colors = ['blue', 'green', 'black']
    markers = ['x', 'o']

    # a) Nacrtati funkciju f na istom grafiku na segmentima [-1,-0.2], [-0.2,0.4], i [0.4,1]
    x_values = np.linspace(-1, 1, 400)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f123(x_values), color='gray', linestyle='-', label='f(x) = x^3 - cos(5x)')

    for i, (start, end) in enumerate(segments):
        x_values_segment = np.linspace(start, end, 100)
        plt.plot(x_values_segment, f123(x_values_segment), color=colors[i], linestyle='-', label=f'Segment {i+1}')

    # b) Pronaći minimalnu i maksimalnu vrijednost funkcije f za sve tri zadane segmente
    for i, (start, end) in enumerate(segments):
        segment_values = f123(np.linspace(start, end, 100))
        argmax_idx = np.argmax(segment_values)
        argmin_idx = np.argmin(segment_values)
        argmax_x = np.linspace(start, end, 100)[argmax_idx]
        argmin_x = np.linspace(start, end, 100)[argmin_idx]
        plt.plot(argmax_x, segment_values[argmax_idx], color='red', marker='o')
        plt.plot(argmin_x, segment_values[argmin_idx], color='blue', marker='x')
        plt.text(argmax_x, segment_values[argmax_idx], f'Max: ({argmax_x:.2f}, {segment_values[argmax_idx]:.2f})', fontsize=8, ha='right')
        plt.text(argmin_x, segment_values[argmin_idx], f'Min: ({argmin_x:.2f}, {segment_values[argmin_idx]:.2f})', fontsize=8, ha='right')

    # c) Analiza
    plt.title('Graf funkcije f(x) = x^3 - cos(5x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()


def f(x1, x2):
    return 4*x1**2 + x2**2 + 16*x1**2*x2**2

def task2():
    # Definiranje ograničenja
    x1_values = np.linspace(-0.5, 0.5, 100)
    x2_values = np.linspace(-1, 1, 100)
    X1, X2 = np.meshgrid(x1_values, x2_values)
    Z = f(X1, X2)

    # a) Nacrtati površ koju funkcija f predstavlja u 3D prostoru sa ograničenjima x1 ∈ [-0.5,0.5] i x2 ∈ [-1,1]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('3D Površina funkcije f(x1, x2)')
    plt.show()

    # b) Nacrtati contour plot funkcije f sa zadatim ograničenjima po varijablama
    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot funkcije f(x1, x2)')
    plt.grid(True)
    plt.show()

    # c) Nacrtati contour plotove prvih parcijalnih izvoda funkcije f po svim njenim varijablama
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contour(X1, X2, 8*X1 + 32*X1*X2**2, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot ∂f/∂x1')

    plt.subplot(1, 2, 2)
    plt.contour(X1, X2, 2*X2 + 32*X1**2*X2, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot ∂f/∂x2')

    plt.tight_layout()
    plt.show()


def task3():
    x1, x2 = sp.symbols('x1 x2')
     # Lets define the fucntion
    f = x1**2 + 2*x2**2

    # Lets calculate the partial derivatives
    df_dx1 = sp.diff(f, x1)
    df_dx2 = sp.diff(f, x2)

    f_func = sp.lambdify((x1, x2), f, 'numpy')
    df_dx1_func = sp.lambdify((x1, x2), df_dx1, 'numpy')
    df_dx2_func = sp.lambdify((x1, x2), df_dx2, 'numpy')

    x1_values = np.linspace(-2, 2, 100)
    x2_values = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1_values, x2_values)
    
    # Evaluate function and partial derivatives on grid
    Z = f_func(X1, X2)
    df_dx1_values = df_dx1_func(X1, X2)
    df_dx2_values = df_dx2_func(X1, X2)


    # Plot surface plot of the function
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('3D Surface Plot of f(x1, x2)')
    plt.show()

    # Plot contour plot of the function
    plt.figure(figsize=(8, 6))
    plt.imshow(((X1**2+X2**2<=2) & (X1-X2<=1) & (X1>=0)).astype(int), extent=(X1.min(),X1.max(),X2.min(),X2.max()),origin="lower", cmap="Greys", alpha=0.7)
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of f(x1, x2)')
    plt.grid(True)
    plt.show()

    # Plot contour plots of the partial derivatives
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contour(X1, X2, df_dx1_values, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of ∂f/∂x1')

    plt.subplot(1, 2, 2)
    plt.contour(X1, X2, df_dx2_values, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of ∂f/∂x2')

    plt.tight_layout()
    plt.show()
    # plt.imshow(((X1**2+X2**2<=2) & (X1-X2<=1) & (X1>=0)).astype(int) , extent=(X1.min(),X1.max(),X2.min(),X2.max()),origin="lower", cmap="Greys", alpha = 0.7)
    # plt.show()

def task5():
    # Define symbolic variables
    x1, x2 = sp.symbols('x1 x2')

    # Define the function
    f = 4*x1**2 + x2**2 + 16*x1**2*x2**2

    # Calculate partial derivatives
    df_dx1 = sp.diff(f, x1)
    df_dx2 = sp.diff(f, x2)

    # Convert symbolic expressions to numpy functions
    f_func = sp.lambdify((x1, x2), f, 'numpy')
    df_dx1_func = sp.lambdify((x1, x2), df_dx1, 'numpy')
    df_dx2_func = sp.lambdify((x1, x2), df_dx2, 'numpy')

    # Define grid for plotting
    x1_values = np.linspace(-0.5, 0.5, 100)
    x2_values = np.linspace(-1, 1, 100)
    X1, X2 = np.meshgrid(x1_values, x2_values)

    # Evaluate function and partial derivatives on grid
    Z = f_func(X1, X2)
    df_dx1_values = df_dx1_func(X1, X2)
    df_dx2_values = df_dx2_func(X1, X2)

    # Plot surface plot of the function
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('3D Surface Plot of f(x1, x2)')
    plt.show()

    # Plot contour plot of the function
    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of f(x1, x2)')
    plt.grid(True)
    plt.show()

    # Plot contour plots of the partial derivatives
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contour(X1, X2, df_dx1_values, levels=50, cmap='viridis')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of ∂f/∂x1')

    plt.subplot(1, 2, 2)
    plt.contour(X1, X2, df_dx2_values, levels=50, cmap='viridis')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot of ∂f/∂x2')

    plt.tight_layout()
    plt.show()
    # plt.imshow( ((X1**2+X2**2<=2) & (X1-X2<=1) & (X1>=0)).astype(int) , extent=(X1.min(),X1.max(),X2.min(),X2.max()),origin="lower", cmap="Greys", alpha = 0.7);
    




