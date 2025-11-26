import numpy as np

# ==========================================
# 1. OPTIMIZATION FUNCTIONS & DERIVATIVES
# ==========================================


# first function given
def func1(x):
    x1, x2 = x[0], x[1]
    return (5*x1 - x2)**4 + (x1 - 2)**2 + x1 - 2*x2 + 12

def grad_func1(x):
    x1, x2 = x[0], x[1]
    df_dx1 = 20 * (5*x1 - x2)**3 + 2 * (x1 - 2) + 1
    df_dx2 = -4 * (5*x1 - x2)**3 - 2
    return np.array([df_dx1, df_dx2])

def hess_func1(x):
    x1, x2 = x[0], x[1]
    d2_dx1_dx1 = 300 * (5*x1 - x2)**2 + 2
    d2_dx1_dx2 = -60 * (5*x1 - x2)**2
    d2_dx2_dx2 = 12 * (5*x1 - x2)**2
    return np.array([[d2_dx1_dx1, d2_dx1_dx2], 
                     [d2_dx1_dx2, d2_dx2_dx2]])

# --- Rosenbrock Function ---
def func2(x):
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def grad_func2(x):
    x1, x2 = x[0], x[1]
    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

def hess_func2(x):
    x1, x2 = x[0], x[1]
    d2_dx1_dx1 = 1200 * x1**2 - 400 * x2 + 2
    d2_dx1_dx2 = -400 * x1
    d2_dx2_dx2 = 200
    return np.array([[d2_dx1_dx1, d2_dx1_dx2], 
                     [d2_dx1_dx2, d2_dx2_dx2]])

# ==========================================
# 2. EXACT LINE SEARCH (Golden Section)
# ==========================================

def golden_section_search(f, x_k, d_k, eps2=0.005, a=-100, b=100):
    """
    Finds the step size alpha that minimizes phi(alpha) = f(x_k + alpha * d_k)
    using the Golden Section method.
    """
    gr = (np.sqrt(5) + 1) / 2  # Golden Ratio
    
    # Local wrapper for the 1D function phi(alpha)
    def phi(alpha):
        return f(x_k + alpha * d_k)

    c = b - (b - a) / gr
    d = a + (b - a) / gr
    
    while abs(c - d) > eps2:
        if phi(c) < phi(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

# ==========================================
# 3. MAIN SOLVER (Handles all 4 Methods)
# ==========================================

def solve_optimization(method_name, func, grad, hess, x0, epsilon1, max_iter=1000):
    """
    Solves the minimization problem using the specified method.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)  # Inverse Hessian approximation (Identity for Start)
    
    print("-" * 100)
    print(f"METHOD: {method_name} | Start: {x0} | Epsilon: {epsilon1}")
    print("-" * 100)
    print(f"{'k':<4} | {'x^(k)':<25} | {'f(x^(k))':<12}  | {'d^(k)':<25} | {'alpha':<10} | {'x^(k+1)':<25}")
    print("-" * 100)
    
    for k in range(max_iter):
        g = grad(x)
        f_val = func(x)
        
        # --- Check Convergence ---
        if np.linalg.norm(g) < epsilon1:
            print("-" * 100)
            print(f"Converged at iteration {k}")
            print(f"Optimal x*: {x}")
            print(f"Optimal f(x*): {f_val:.6f}")
            return

        # --- Determine Direction d_k ---
        if method_name == "Steepest Descent":
            #  Steepest Descent: d = -gradient
            d = -g
            
        elif method_name == "Newton":
            # d = - Hessian_inv * gradient
            # Using solve is more numerically stable than inv
            H_mat = hess(x)
            try:
                d = -np.linalg.solve(H_mat, g)
            except np.linalg.LinAlgError:
                # Fallback if Hessian is singular 
                d = -g 

        elif method_name in ["DFP", "BFGS"]:
            d = -np.dot(H, g)

        # --- Exact Line Search ---
        alpha = golden_section_search(func, x, d)
        
        # --- Update x ---
        x_new = x + alpha * d
        
        # --- Logging ---
        # Formatting arrays for clean table output
        x_str = np.array2string(x, precision=4, separator=',')
        d_str = np.array2string(d, precision=4, separator=',')
        x_next_str = np.array2string(x_new, precision=4, separator=',')
        
        print(f"{k:<4} | {x_str:<25} | {f_val:<12.6f} | {d_str:<25} | {alpha:<10.5f} | {x_next_str:<25}")

        # --- Quasi-Newton Updates (DFP/BFGS) ---
        if method_name in ["DFP", "BFGS"]:
            s = x_new - x      # Change in position (s_k)
            y = grad(x_new) - g # Change in gradient (y_k)
            
            # Avoid division by zero
            if np.dot(y, s) > 1e-10:
                if method_name == "DFP":
                    # DFP Update Formula
                    term1 = np.outer(s, s) / np.dot(s, y)
                    term2 = np.dot(np.dot(H, np.outer(y, y)), H) / np.dot(np.dot(y, H), y)
                    H = H + term1 - term2
                    
                elif method_name == "BFGS":
                    # BFGS Update Formula
                    rho = 1.0 / np.dot(y, s)
                    I = np.eye(n)
                    term1 = I - rho * np.outer(s, y)
                    term2 = I - rho * np.outer(y, s)
                    H = np.dot(term1, np.dot(H, term2)) + rho * np.outer(s, s)
        
        x = x_new
        
        # Special check for Rosenbrock Steepest Descent slowness
        if method_name == "Steepest Descent" and "Rosenbrock" in str(func.__name__) and k == max_iter - 1:
            print("\n[Observation]: Steepest Descent is very slow on Rosenbrock (curved valley).")
            print("Max iterations reached before epsilon convergence.")

    print(f"Stopped after {max_iter} iterations.")
    print(f"Current x: {x}")

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================

if __name__ == "__main__":
    
    # --- Define the two test sets (Epsilon1, Start Point) ---
    # Set 1
    eps1_a = 1e-3
    x0_a = [0.0, 0.0]
    
    # Set 2
    eps1_b = 1e-4
    x0_b = [1.2, 1.2] # Close to solution for faster check, or pick something else like [-1, 1]

    # List of methods to run
    methods = ["Steepest Descent", "Newton", "DFP", "BFGS"]

    print("============================================================")
    print("RUNNING FUNCTION 1: f(x) = (5x1 - x2)^4 + ...")
    print("============================================================")
    
    # Run for Set 1
    for m in methods:
        solve_optimization(m, func1, grad_func1, hess_func1, x0_a, eps1_a)
        print("\n")

    # Run for Set 2
    for m in methods:
        solve_optimization(m, func1, grad_func1, hess_func1, x0_b, eps1_b)
        print("\n")

    print("============================================================")
    print("RUNNING FUNCTION 2: Rosenbrock")
    print("============================================================")
    
    # Run for Set 1
    for m in methods:
        solve_optimization(m, func2, grad_func2, hess_func2, x0_a, eps1_a, max_iter=2000) # Increased limit for Rosenbrock
        print("\n")

    # Run for Set 2
    for m in methods:
        solve_optimization(m, func2, grad_func2, hess_func2, x0_b, eps1_b, max_iter=2000)
        print("\n")