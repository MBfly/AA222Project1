#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np


def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized.
        g (function): Gradient function for `f`.
        x0 (np.array): Initial position to start from.
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`.
        count (function): Takes no arguments and returns current count.
        prob (str): Name of the problem. So you can use a different strategy
                    for each problem. `prob` can be 'simple1', 'simple2', 'simple3',
                    'secret1' or 'secret2'.
    Returns:
        x_best (np.array): Best selection of variables found.
    """

    x_best = np.copy(x0)

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
                return x_best
            x_candidate = x_best - t * grad
            f_candidate = f(x_candidate)
            
            # Check Armijo condition.
            if f_candidate <= fx - c * t * np.dot(grad, grad):
                break  
            
            t *= 0.23  # Reduce the step length

        # Update current point.
        x_best = x_best - t * grad

    return x_best