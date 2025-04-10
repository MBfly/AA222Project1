import numpy as np
import matplotlib.pyplot as plt


class EvalCounter:
    """
    Class for tracking the cost of function
    """
    def __init__(self):
        self.count = 0

    def inc_f(self):
        self.count += 1

    def inc_g(self):
        self.count += 2

    def get_count(self):
        return self.count

def optimize_path(f, g, x0, n, count, prob):
    """
    Optimize f using gradient descent with backtracking line search and record the path.
    
    Args:
        f (function): The function to be optimized.
        g (function): Gradient function for f.
        x0 (np.array): Initial starting point.
        n (int): Maximum allowed evaluation cost (each call to g costs 2, each call to f costs 1).
        count (function): Callable that returns the current evaluation count.
        prob (str): Name of the problem. So you can use a different strategy
                    for each problem. `prob` can be 'simple1', 'simple2', 'simple3',
                    'secret1' or 'secret2'.
    Returns:
        x_best (np.array): Best selection of variables found.
    """
    x_best = np.copy(x0)
    path = [x_best.copy()]  # Initialize the list to record the iterates.
    
    tol = 1e-6             # Stop tolerance
    alpha_init = 1.4       # Initial step length
    c = 1e-4               # Armijo condition.
    
    while count() < n:
        if count() + 2 > n:
            break

        # Compute the gradient
        grad = g(x_best)
        grad_norm = np.linalg.norm(grad)
        
        # Optimization check
        if grad_norm < tol:
            break

        if count() + 1 > n:
            break
        fx = f(x_best)
        
        # Backtracking line search.
        t = alpha_init
        while True:
            if count() + 1 > n:
                return x_best, np.array(path)
            x_candidate = x_best - t * grad
            f_candidate = f(x_candidate)

            # Check the Armijo condition.
            if f_candidate <= fx - c * t * np.dot(grad, grad):
                break
            t *= 0.23  # Reduce the step length
        
        # Update current point.
        x_best = x_best - t * grad
        path.append(x_best.copy())
    
    # Return the final point and the path of iterates as a NumPy array.
    return x_best, np.array(path)


def rosenbrock_f(x):
#   f(x) = 100 * (x2 - x1^2)^2 + (1 - x1)^2.
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    grad0 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad1 = 200 * (x[1] - x[0]**2)
    return np.array([grad0, grad1])

def plot_rosenbrock_paths():
    starting_points = [
        np.array([-1.5, 2.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0])
    ]
    
    n = 20  
    paths = []  
    final_points = []  
    
    # Run the optimization
    for x0 in starting_points:
        counter = EvalCounter()  
        x_best, path = optimize_path(rosenbrock_f, rosenbrock_grad, x0, n, counter.get_count, 'simple1')
        paths.append(path)
        final_points.append(x_best)
    
    # Create a grid to plot the contours of the Rosenbrock function.
    x_vals = np.linspace(-2, 3, 400)
    y_vals = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    # Compute Rosenbrock function values over the grid.
    Z = 100*(Y - X**2)**2 + (1 - X)**2
    
    plt.figure(figsize=(8, 6))
    contour_levels = np.logspace(-1, 3, 20)
    cp = plt.contour(X, Y, Z, levels=contour_levels, cmap='viridis')
    plt.clabel(cp, inline=1, fontsize=10)
    
    colors = ['red', 'green', 'blue']
    # Plot path
    for i, path in enumerate(paths):
        plt.plot(path[:, 0], path[:, 1], marker='o', color=colors[i],
                 label=f"Start: {starting_points[i]}")
    
    plt.plot(1, 1, 'kx', markersize=10, label="Optimum (1,1)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Paths on the Rosenbrock Function')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_rosenbrock_paths()
