import numpy as np

def f(x):
    
    #The objective function to be minimized.
    #f(x1, x2) = (5*x1 - x2)^4 + (x1 - 2)^2 + x1 - 2*x2 + 12
    
    x1 = x[0]
    x2 = x[1]
    
    term1 = (5*x1 - x2)**4
    term2 = (x1 - 2)**2
    term3 = x1 - 2*x2 + 12
    
    return term1 + term2 + term3

# exact line search

def golden_section_search(x_start, direction, a, b, eps2):
    
    #Performs an exact line search for f(x_start + alpha * direction) in the interval [a, b] until the interval width is < eps2.
    
    #Returns the optimal steplength 'alpha'.
    
    
    # Define the 1D function g(alpha)
    g = lambda alpha: f(x_start + alpha * direction)
    
    # Golden ratio conjugate
    tau = (np.sqrt(5) - 1) / 2
    
    # Initialize interior points
    x1_interval = a + (1 - tau) * (b - a)
    x2_interval = a + tau * (b - a)
    
    f1 = g(x1_interval)
    f2 = g(x2_interval)
    
    while (b - a) > eps2:
        if f1 < f2:
            # The minimum is in [a, x2_interval]
            b = x2_interval
            x2_interval = x1_interval
            f2 = f1
            x1_interval = a + (1 - tau) * (b - a)
            f1 = g(x1_interval)
        else:
            # The minimum is in [x1_interval, b]
            a = x1_interval
            x1_interval = x2_interval
            f1 = f2
            x2_interval = a + tau * (b - a)
            f2 = g(x2_interval)
            
    # Return the midpoint of the final interval
    return (a + b) / 2


def hooke_jeeves(x0, eps1, a, b, eps2, max_iter=500):
   
    #Performs the Hook & Jeeves Method with exact line search.
    
    #x0: Initial point [x1, x2]
    #eps1: Stopping criterion (norm of the pattern move or change in f_value)
    #a, b, eps2: Parameters for the exact line search
   
    print("--- Solution for Hook & Jeeves Method ---")
    
    x_base = np.array(x0, dtype=float)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    # Header for the table 
    header = f"| {'k':<3} | {'x^(k)':<20} | {'f(x^(k))':<15} | {'x_temp':<20} | {'d^(k)':<20} | {'alpha^(k)':<10} | {'r^(k+1)':<20} |"
    print(header)
    print("-" * len(header))

    def exploratory_move(x_start):
        #Helper function to perform one cycle of coordinate search
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
            f"[{r_k_plus_1[0]:.6f}, {r_k_plus_1[1]:.6f}] |"  # Print r^(k+1) as requested in PDF
        )
        print(row)
        
        # Update the base point for the next iteration
        x_base = x_k_plus_1
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
    return x_base, f(x_base)





if __name__ == "__main__":
    
    # --- Parameters for Line Search (fixed) ---
    LS_a = -100.0
    LS_b = 100.0
    LS_eps2 = 0.005
    
    # --- Parameter Set 1 ---
    print("="*40 + "\n          PARAMETER SET 1 (Hooke-Jeeves)\n" + "="*40 + "\n")
    set1_eps1 = 1e-3
    set1_x0 = [0.0, 0.0]
    
    hooke_jeeves(set1_x0, set1_eps1, LS_a, LS_b, LS_eps2)
    print("\n") # Add spacing

    # --- Parameter Set 2 ---
    print("="*40 + "\n          PARAMETER SET 2 (Hooke-Jeeves)\n" + "="*40 + "\n")
    set2_eps1 = 1e-5
    set2_x0 = [5.0, 5.0]  # A different starting point
    
    hooke_jeeves(set2_x0, set2_eps1, LS_a, LS_b, LS_eps2)