import numpy as np
import matplotlib.pyplot as plt

# --- 1. Objective Function --------------------------------------------------

def f(x):
    """
    The objective function to be minimized.
    f(x1, x2) = (5*x1 - x2)^4 + (x1 - 2)^2 + x1 - 2*x2 + 12
    """
    x1 = x[0]
    x2 = x[1]
    
    term1 = (5*x1 - x2)**4
    term2 = (x1 - 2)**2
    term3 = x1 - 2*x2 + 12
    
    return term1 + term2 + term3

# --- 2. Exact Line Search (Golden Section) ----------------------------------

def golden_section_search(x_start, direction, a, b, eps2):
    """
    Performs an exact line search for f(x_start + alpha * direction)
    in the interval [a, b] until the interval width is < eps2.
    
    Returns the optimal steplength 'alpha'.
    """
    
    # Define the 1D function g(alpha)
    g = lambda alpha: f(x_start + alpha * direction)
    
    # Golden ratio conjugate
    tau = (np.sqrt(5) - 1) / 2
    
    # Initialize interior points
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    
    f1 = g(x1)
    f2 = g(x2)
    
    while (b - a) > eps2:
        if f1 < f2:
            # The minimum is in [a, x2]
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - tau) * (b - a)
            f1 = g(x1)
        else:
            # The minimum is in [x1, b]
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + tau * (b - a)
            f2 = g(x2)
            
    # Return the midpoint of the final interval
    return (a + b) / 2

# --- 3. Optimization Algorithm (Hooke & Jeeves) -----------------------------

def hooke_jeeves(x0, eps1, a, b, eps2, max_iter=500):
    """
    Performs the Hook & Jeeves Method with exact line search.
    
    Returns:
    - x_base: The final optimal point
    - f(x_base): The function value at the optimal point
    - path_history: A list of all base points visited
    """
    print("--- Solution for Hook & Jeeves Method ---")
    
    x_base = np.array(x0, dtype=float)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    
    # Store the path
    path_history = [x_base.copy()]

    # Header for the table
    header = f"| {'k':<3} | {'x^(k)':<20} | {'f(x^(k))':<15} | {'x_temp':<20} | {'d^(k)':<20} | {'alpha^(k)':<10} | {'r^(k+1)':<20} |"
    print(header)
    print("-" * len(header))

    def exploratory_move(x_start):
        """Helper function to perform one cycle of coordinate search."""
        x = x_start.copy()
        for d in directions:
            # Use Golden Section for line search
            alpha = golden_section_search(x, d, a, b, eps2)
            x = x + alpha * d
        return x

    k = 0
    f_prev = f(x_base) # For checking function value change
    
    while k < max_iter:
        f_base = f(x_base)
        
        # 1. Exploratory Move from the base point
        x_temp = exploratory_move(x_base)
        
        # 2. Pattern Move
        d_k = x_temp - x_base
        
        # Check stopping criterion 1: Norm of pattern move
        if np.linalg.norm(d_k) < eps1:
            print(f"Stopping criterion (d_k norm < {eps1}) met.")
            break
            
        # Line search along the pattern direction using Golden Section
        alpha_k = golden_section_search(x_temp, d_k, a, b, eps2)
        
        r_k_plus_1 = x_temp + alpha_k * d_k
        f_r = f(r_k_plus_1)
        
        # 3. Update Base Point - determine x^(k+1)
        if f_r < f_base:
            # Pattern move was successful
            x_k_plus_1 = r_k_plus_1
        else:
            # Pattern move failed, restart from x_temp
            x_k_plus_1 = x_temp
            
        # Format for table row
        row = (
            f"| {k:<3} | "
            f"[{x_base[0]:.6f}, {x_base[1]:.6f}] | "
            f"{f_base:<15.6f} | "
            f"[{x_temp[0]:.6f}, {x_temp[1]:.6f}] | "
            f"[{d_k[0]:.6f}, {d_k[1]:.6f}] | "
            f"{alpha_k:<10.6f} | "
            f"[{r_k_plus_1[0]:.6f}, {r_k_plus_1[1]:.6f}] |" 
        )
        print(row)
        
        # Update the base point for the next iteration
        x_base = x_k_plus_1
        path_history.append(x_base.copy()) # Add new point to path
        f_new = f(x_base)
        
        # Check stopping criterion 2: Change in function value
        if abs(f_new - f_prev) < eps1 and k > 0:
             print(f"Stopping criterion (f_change < {eps1}) met.")
             break
        f_prev = f_new
            
        k += 1

    if k == max_iter:
        print("Max iterations reached.")
        
    # The final solution is the last best base point
    print("-" * len(header))
    print(f"x* = [{x_base[0]:.6f}, {x_base[1]:.6f}]")
    print(f"f(x*) = {f(x_base):.6f}\n")
    return x_base, f(x_base), path_history

# --- 4. Visualization Function ----------------------------------------------

def plot_optimization_path(path, title, x_range, y_range):
    """
    Plots the optimization path on a contour plot of the function f.
    """
    
    # Create a grid for the contour plot
    x1_vals = np.linspace(x_range[0], x_range[1], 400)
    x2_vals = np.linspace(y_range[0], y_range[1], 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f([X1, X2])
    
    # Convert path list to a numpy array for easier slicing
    path = np.array(path)
    
    plt.figure(figsize=(10, 8))
    
    # Plot the contour lines.
    levels = np.logspace(0, 5, 30) 
    plt.contour(X1, X2, Z, levels=levels, cmap='viridis_r', alpha=0.7)
    
    # Plot the optimization path
    plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=3, label='Optimization Path')
    
    # Mark the start and end points
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label=f'Start ({path[0,0]:.2f}, {path[0,1]:.2f})')
    plt.plot(path[-1, 0], path[-1, 1], 'bo', markersize=10, label=f'End ({path[-1,0]:.2f}, {path[-1,1]:.2f})')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# --- 5. Main Execution ------------------------------------------------------

if __name__ == "__main__":
    
    # --- Parameters for Line Search (fixed) ---
    LS_a = -100.0
    LS_b = 100.0
    LS_eps2 = 0.005
    
    # Define plot ranges
    plot_x_range = (-1, 8)
    plot_y_range = (-1, 35)
    
    # --- Parameter Set 1 ---
    print("="*40 + "\n          PARAMETER SET 1 (Hooke-Jeeves)\n" + "="*40 + "\n")
    set1_eps1 = 1e-3
    set1_x0 = [0.0, 0.0]
    
    x_star1, f_star1, path1 = hooke_jeeves(set1_x0, set1_eps1, LS_a, LS_b, LS_eps2)
    plot_optimization_path(path1, "Hooke-Jeeves Path (Set 1)", plot_x_range, plot_y_range)
    print("\n") 

    # --- Parameter Set 2 ---
    print("="*40 + "\n          PARAMETER SET 2 (Hooke-Jeeves)\n" + "="*40 + "\n")
    set2_eps1 = 1e-5
    set2_x0 = [5.0, 5.0] 
    
    x_star2, f_star2, path2 = hooke_jeeves(set2_x0, set2_eps1, LS_a, LS_b, LS_eps2)
    plot_optimization_path(path2, "Hooke-Jeeves Path (Set 2)", plot_x_range, plot_y_range)