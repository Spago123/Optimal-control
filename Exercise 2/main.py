import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import signal

def solve_with_timeout(eqns, vars, timeout=60):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout)
    try:
        solutions = sp.solve(eqns, vars, dict=True)
        return solutions
    except TimeoutError:
        print("Solver timed out")
        return None
    finally:
        signal.alarm(0)  # Reset the alarm

def check_if_has_been(solutions, find_var, find_value):
    for sol in solutions:
        for var, value in sol.items():
            if var == find_var and abs(value) == abs(find_value):
                return  True
            
    return False

def remove_duplicate_solutions(solutions):
    unique_solutions = []
    
    for sol in solutions:
        unique = True
        for var, value in sol.items():
            if str(var).startswith('s') \
                and check_if_has_been(unique_solutions, var, value):
                    unique = False
                    break
        if unique:
            unique_solutions.append(sol)
            
    
    return unique_solutions


def is_complex(symbolic_expr):
    return sp.im(symbolic_expr) != 0

def check_the_solution(solution):
    for var, value in solution.items():
        if str(var).startswith('s') and \
            is_complex(value):
                return False
    
    return True

# Define custom sorting function
def custom_sort(variable):
    var_str = str(variable)
    if var_str[0] not in ('h', 's'):
        return (0, var_str)
    elif var_str[0] == 'h':
        return (1, var_str)
    else:
        return (2, var_str)

def get_modified_inequality(h, s, ineq):
    if ineq.rel_op == "<=":
        return h * (ineq.lhs - ineq.rhs + s**2)
    elif  ineq.rel_op == ">=":
         return h * (ineq.lhs - ineq.rhs - s**2)
     
    raise  ValueError("Invalid inequality type")

def eval_function(f, var, sol):
    result = {"value": f}
    for v in var:
        result["value"] = result["value"].subs({v : sol[v]})
        result[v] = sol[v]
    
    return result

def solver(f, ineq_const = None):
    """Function that solves optimization problem using langrnage multiplicatos and slack variables

    Args:
        f (): criterium function
        ineq_const (list): list of boundraries

    Returns:
        optimization problem solutions
    """
    copy_f = f
    vars = f.free_symbols
    vars = list(vars)
    
    order = len(vars)
        
    if ineq_const == None:
        num_of_ineq = 0
    else:
        num_of_ineq = len(ineq_const)
    
    
    
    for i, ineq in enumerate(ineq_const):
        h_ = 'h{}'.format(i)
        s_ = 's{}'.format(i)
        vars.append(sp.symbols(h_))
        vars.append(sp.symbols(s_))
        f = f + get_modified_inequality(sp.symbols(h_), sp.symbols(s_), ineq)
        
    vars = sorted(vars, key=custom_sort)
    diffs = []
    for var in vars:
        diffs.append(sp.diff(f, var))
        
    ### Lets solve the last num_of_ineq equations

    sols = sp.solve(diffs[-num_of_ineq:], vars[-2*num_of_ineq:], dict=True)
    
    
    diffs = diffs[:-num_of_ineq]

    solutions = []
    possible_sols = []
    for sol in sols:
        new_diff = []
        for diff in diffs:
            diff_copy = diff
            for k, v in sol.items():
                try:
                    diff_copy = diff_copy.subs({k:v})
                except Exception as e:
                    pass
            new_diff.append(diff_copy)
        
        symbolic_list = vars
        for k, v in sol.items():
            symbolic_list = [symbolic_var for symbolic_var in symbolic_list if symbolic_var != k]
        
        try:
            my_sol = solve_with_timeout(new_diff, symbolic_list, 10)
            # my_sol = sp.solve(new_diff, symbolic_list, dict=True)
            # my_sol = sp.nsolve(new_diff, symbolic_list, [0, 1, 2, -1, -2], prec=15, tol=1e-4, maxsteps=50)
            possible_sols.append(my_sol)
        
            for s in my_sol:
                if check_the_solution(s):
                    solutions.append(s)
        except Exception as e:
            pass

        
        
    # for po in possible_sols:
    #     for p in po:
    #         print(p)
    #     print("------------------------")
        
    for po in solutions:
        print(po)
        print("..........................")
        
    solutions = remove_duplicate_solutions(solutions)
    
    for po in solutions:
        print(po)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    
    for var in vars[:order]:
        print(var)
        
    result = []
    for po in solutions:
        result.append(eval_function(copy_f, vars[:order], po))
    
    print("Results:")  
    for po in result:
        print(po)
        
    return result
        
def plot_boundries(f, sols, ineq_const=None):
    (x, y) = sp.symbols('x y')
    f_func = sp.lambdify((x, y), f, 'numpy') 
    
    x_points = []
    y_points = []
    f_points = []
    for s in sols:
        x_points.append(float(s[x].evalf()))
        y_points.append(float(s[y].evalf()))
        f_points.append(float(s['value'].evalf()))
        
    min_value_data = sols[0]
    max_value_data = sols[0]

    # Iterate over the list to find min and max values
    for d in sols:
        value = d['value']
        if value < min_value_data['value']:
            min_value_data = d
        if value > max_value_data['value']:
            max_value_data = d
    print("Max: ")
    print(max_value_data)
    print("Min: ")
    print(min_value_data)
    
    x_max = float(max_value_data[x].evalf())
    y_max = float(max_value_data[y].evalf())
    
    x_min = float(min_value_data[x].evalf())
    y_min = float(min_value_data[y].evalf())
        
    if len(sols) != 0:
        x_ = np.linspace(float(min(x_points)) - 1, float(max(x_points)) + 1, 1000)
        y_ = np.linspace(float(min(y_points)) - 1, float(max(y_points)) + 1, 1000)
    else:
        x_ = np.linspace(-5, 5, 1000)
        y_ = np.linspace(-5, 5, 1000)
        
    X1, X2 = np.meshgrid(x_, y_)
    Z = f_func(X1, X2)
    
    # Plot surface plot of the function
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_points, y_points, f_points, color="red", s=20, alpha=1)
    ax.plot_surface(X1, X2, Z, color='skyblue', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('3D Surface Plot of f(x, y)')
    plt.show()
    
    # Convert symbolic expression to a numpy function
    constraint_func = sp.lambdify((x, y), sp.And(*ineq_const), 'numpy')

    # Evaluate the constraint function on the mesh grid
    constraint = constraint_func(X1, X2)
    
    # Plot contour plot of the function
    plt.figure(figsize=(8, 6))
    plt.imshow(constraint, extent=(X1.min(),X1.max(),X2.min(),X2.max()),origin="lower", cmap="Greys", alpha=0.7)
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.plot(x_points, y_points, 'o', color="red")
    plt.plot(x_max, y_max, 'o', color="green")
    plt.plot(x_min, y_min, 'o', color="blue")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour Plot of f(x, y)')
    plt.grid(True)
    plt.show()


(x, y) = sp.symbols('x y')

f = x**2 + 3*y**2
ineq_const=[(x**2 + y**2 <= 3), (x + y >= 1), (x >= 0)]

# f = x**2 + 3*y**2
# ineq_const=[(x**2 + y**2 <= 4), (x >= 0), (y <= 1)]

# f = (x + 1)**2 + y**2
# ineq_const=[(x**2 + y**2 <= 0.75), (-x + y <= 0.5), (y >= 0)]

# f = (x + 4)**2 + (y + 4)**2
# ineq_const=[(3*x**2 + 0.5*(y - 1)**2 <= 1), (y - x <= 1.5), (y >= 0)]

sols = solver(f, ineq_const)
plot_boundries(f, sols, ineq_const=ineq_const)

