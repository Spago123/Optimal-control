import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime

def y(x):
    return 1 - np.exp(np.sin(x)/x)

def z(x, y):
    return np.sin(np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2)

def isEven(number):
    return number % 2 == 0

def my_factoriel(n):
    if n == 0:
        return 1
    
    return n * my_factoriel(n - 1)


def sin_approximation(N = 5):
    def taylor_series_sin(x):
        result = 0
        for n in range(N):
            coeffs = (-1) ** n / my_factoriel(2 * n + 1)
            result += coeffs * (x ** (2 * n + 1))
        return result
    return taylor_series_sin

def task1_a():
    x_values = np.linspace(-6*np.pi, 6*np.pi, 50)
    y_values = y(x_values)

    plt.plot(x_values, y_values, 'g--')  
    plt.grid(True) 
    plt.xlabel('x')  
    plt.ylabel('y(x)')
    plt.title('y(x) = 1 - e^sin(x) za x ∈ [-6π, 6π]')  
    plt.show()
    

def task1_b():
    x_values = np.arange(-6*np.pi, 6*np.pi, np.pi/4)
    y_values = y(x_values)

    plt.stem(x_values, y_values, 'X')  
    plt.grid(True) 
    plt.xlabel('x') 
    plt.ylabel('y(x)')  
    plt.title('y(x) = 1 - e^sin(x) za x ∈ [-6π, 6π]')  
    plt.show()
    
def task2():
    x = np.arange(-10, 10.1, 0.1)
    y = np.arange(-10, 10.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = z(X, Y)

    
    plt.subplot(1, 2, 1)  
    surface_plot = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(surface_plot)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Surface prikaz funkcije z(x, y) = sin(sqrt(x^2 + y^2))')
    plt.grid(True)

    plt.subplot(1, 2, 2)  
    contour_plot = plt.contour(X, Y, Z, cmap='viridis')
    plt.colorbar(contour_plot)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour prikaz funkcije z(x, y) = sin(sqrt(x^2 + y^2))')
    plt.grid(True)

 
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.tight_layout()  
    plt.show()
    
    
def task3(vector):
    if len(vector) == 0:
        raise AttributeError("Proslijeđeni vektor je neispravan")
    
    odd_count = 0
    even_count = 0
    
    for elem in vector:
        if(isEven(elem)):
            even_count = even_count + 1
        else:
            odd_count = odd_count + 1
    
    return {
        "odd": odd_count,
        "even": even_count
    }
            
        
def task4():
    x_values = [-1., -0.5, 0., 0.5, 1.]
    sin_approx = sin_approximation(2)
    print(" x   sin(x)      Approx      Diff ")
    print("-----------------------------------")
    for x in x_values:
        approx = sin_approx(x)
        exact = np.sin(x)
        diff = abs(approx - exact)
        print(f"{x:.2f}   {exact:.6g}      {approx:.2f}      {diff:.2}")


def task5(A):
    rows, cols = A.shape
    B = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            print(isprime(A[i, j]))
            if A[i, j] == 1:
                B[i, j] = 0.5
            elif isprime(A[i, j]):
                B[i, j] = 1
            else:
                B[i, j] = 0
    return B
                
        
def Funkcija(x, y, opseg, k, x0):
    # Određivanje podopsega za crtanje
    start_idx = max(opseg[0], 0)
    end_idx = min(opseg[1], len(x)-1)

    # Podaci za crtanje
    x_plot = x[start_idx:end_idx+1:k]
    y_plot = y[start_idx:end_idx+1:k]

    # Crtanje funkcije
    plt.plot(x_plot, y_plot)

    # Označavanje minimalne i maksimalne vrijednosti funkcije
    min_idx = np.argmin(y_plot)
    max_idx = np.argmax(y_plot)
    plt.plot(x_plot[min_idx], y_plot[min_idx], 'go')  # Zeleni krug za minimum
    plt.plot(x_plot[max_idx], y_plot[max_idx], 'go')  # Zeleni krug za maksimum

    # Podešavanje osa i oznaka
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafik funkcije y = f(x)')
    plt.grid(True)
    plt.show()

def task6():
    # Primjer upotrebe
    x = np.linspace(0, 10, 100)
    y = np.exp(-x)
    opseg = [20, 80]
    k = 2
    x0 = x[opseg[0]]

    Funkcija(x, y, opseg, k, x0)
    
def ProstiFaktori(n):
    startFact = 2
    factorList = []
    while startFact <= n:
        if isprime(startFact)\
            and n % startFact == 0:
                n = n / startFact
                factorList.append(startFact)
        else:
            startFact = startFact + 1
            
    return factorList

def task7(n):
    print(ProstiFaktori(n))

