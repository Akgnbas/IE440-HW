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

def cyclic_coordinate_search_cycle_based(x0, eps1, a, b, eps2, max_cycles=250):
    """
    k counts FULL CYCLES (as in the textbook), and j indicates coordinate index.

    Inputs:
        x0        : initial point (list or np.array)
        eps1      : stopping tolerance on ||x^{k+1} - x^{k}|| after a full cycle
        a, b      : line search interval
        eps2      : line search tolerance for golden section
        max_cycles: maximum number of full coordinate cycles

    Output:
        x_star, f(x_star)
    """

    x = np.array(x0, dtype=float)
    n = len(x)

    print("--- Solution for Cyclic Coordinate Search ---")
    header = (
        f"| {'k':<3} | {'j':<2} | {'x (before)':<22} | {'f(x)':<15} | "
        f"{'d^(k,j)':<18} | {'alpha^(k,j)':<12} | {'x (after)':<22} |"
    )
    print(header)
    print("-" * len(header))

    k = 0  # it counts the cycle as algorithm progress. A cycle means two steps are taken. 
    # First one is along e1, second one is along e2

    while k < max_cycles:
        x_cycle_start = x.copy()

        # One FULL CYCLE over all coordinates j = 1..n
        for j in range(n):
            # these are directions for each cycle (ej terms in the lectures)
            d = np.zeros(n)
            d[j] = 1.0

            # this alpha is the best step length along this direction ej from current location x
            alpha = golden_section_search(x, d, a, b, eps2)

            x_new = x + alpha * d

            # gives rows in the table
            row = (
                f"| {k:<3} | {j+1:<2} | "
                f"[{x[0]:.6f}, {x[1]:.6f}] | "
                f"{f(x):<15.6f} | "
                f"[{d[0]:.1f}, {d[1]:.1f}]{'':<8} | "
                f"{alpha:<12.6f} | "
                f"[{x_new[0]:.6f}, {x_new[1]:.6f}] |"
            )
            print(row)

            x = x_new  # location is updated

        # after a full cycle is completed( steps are taken in both e1 and e2), we need to check the improvement from the previous point as a stopping condition
        if np.linalg.norm(x - x_cycle_start) < eps1:
            print(f"Stopping criterion (||x^(k+1) - x^(k)|| < {eps1}) met after cycle {k}.")
            break

        k += 1

    if k >= max_cycles:
        print("Max number of cycles reached.")

    print("-" * len(header))
    print(f"x* = [{x[0]:.6f}, {x[1]:.6f}]")
    print(f"f(x*) = {f(x):.6f}\n")

    return x, f(x)






# def hooke_jeeves(x0, eps1, a, b, eps2, max_iter=500):
   
#     #Performs the Hook & Jeeves Method with exact line search.
    
#     #x0: Initial point [x1, x2]
#     #eps1: Stopping criterion (norm of the pattern move or change in f_value)
#     #a, b, eps2: Parameters for the exact line search
   
#     print("--- Solution for Hook & Jeeves Method ---")
    
#     x_base = np.array(x0, dtype=float)
#     directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

#     # Header for the table 
#     header = f"| {'k':<3} | {'x^(k)':<20} | {'f(x^(k))':<15} | {'x_temp':<20} | {'d^(k)':<20} | {'alpha^(k)':<10} | {'r^(k+1)':<20} |"
#     print(header)
#     print("-" * len(header))

#     def exploratory_move(x_start):
#         #Helper function to perform one cycle of coordinate search
#         x = x_start.copy()
#         for d in directions:
#             # Use Golden Section for line search
#             alpha = golden_section_search(x, d, a, b, eps2)
#             x = x + alpha * d
#         return x

#     k = 0
#     f_prev = f(x_base) # For checking function value change
    
#     while k < max_iter:
#         f_base = f(x_base)
        
#         # 1. Exploratory Move from the base point
#         x_temp = exploratory_move(x_base)
        
#         # 2. Pattern Move
#         d_k = x_temp - x_base
        
#         # Check stopping criterion 1: Norm of pattern move
#         if np.linalg.norm(d_k) < eps1:
#             print(f"Stopping criterion (d_k norm < {eps1}) met.")
#             break
            
#         # Line search along the pattern direction using Golden Section
#         alpha_k = golden_section_search(x_temp, d_k, a, b, eps2)
        
#         r_k_plus_1 = x_temp + alpha_k * d_k
#         f_r = f(r_k_plus_1)
        
#         # 3. Update Base Point - determine x^(k+1)
#         if f_r < f_base:
#             # Pattern move was successful
#             x_k_plus_1 = r_k_plus_1
#         else:
#             # Pattern move failed, restart from x_temp
#             x_k_plus_1 = x_temp
            
#         # Format for table row
#         row = (
#             f"| {k:<3} | "
#             f"[{x_base[0]:.6f}, {x_base[1]:.6f}] | "
#             f"{f_base:<15.6f} | "
#             f"[{x_temp[0]:.6f}, {x_temp[1]:.6f}] | "
#             f"[{d_k[0]:.6f}, {d_k[1]:.6f}] | "
#             f"{alpha_k:<10.6f} | "
#             f"[{r_k_plus_1[0]:.6f}, {r_k_plus_1[1]:.6f}] |"  # Print r^(k+1) as requested in PDF
#         )
#         print(row)
        
#         # Update the base point for the next iteration
#         x_base = x_k_plus_1
#         f_new = f(x_base)
        
#         # Check stopping criterion 2: Change in function value
#         if abs(f_new - f_prev) < eps1 and k > 0:
#              print(f"Stopping criterion (f_change < {eps1}) met.")
#              break
#         f_prev = f_new
            
#         k += 1

#     if k == max_iter:
#         print("Max iterations reached.")
        
#     # The final solution is the last best base point
#     print("-" * len(header))
#     print(f"x* = [{x_base[0]:.6f}, {x_base[1]:.6f}]")
#     print(f"f(x*) = {f(x_base):.6f}\n")
#     return x_base, f(x_base)


# def simplex_search(initial_points, alpha=1.0, beta=0.5, gamma=2.0, max_iter=500, tol=1e-5):
#     """
#     Performs the Nelder-Mead Simplex Search
    
#     initial_points: A list of n+1 starting points (e.g., [[0,0], [1,0], [0,1]])
#     alpha, beta, gamma: Reflection, Contraction, Expansion coefficients
#     tol: Stopping criterion 
#     shrink operation is now added as well
#     """
    
#     print("--- Solution for Simplex Search ---")
    
#     # n = number of dimensions
#     n = len(initial_points[0])
    
#     # Initializing Simplex Search
#     simplex = []
#     for pt in initial_points:
#         pt_array = np.array(pt, dtype=float)
#         simplex.append((f(pt_array), pt_array))

#     # Printing Table Header
#     header = f"| {'Iter':<5} | {'x_bar':<22} | {'x_h':<22} | {'x_l':<22} | {'x_new':<22} | {'f(x_new)':<12} | {'Type':<4} |"
    
#     print(header)
#     print("-" * len(header))

#     k = 0
#     #number of max iterations are decided as 500, may be experimented differently
#     while k < max_iter:
        
#         # Sorting Vertices
#         simplex.sort(key=lambda x: x[0])
        
#         f_b = simplex[0][0]  # Best f(x)
#         x_b = simplex[0][1]  # Best point (x_l)
#         f_s = simplex[1][0]  # Second-worst f(x)
#         x_s = simplex[1][1]  # Second-worst point
#         f_h = simplex[-1][0] # Worst f(x)
#         x_h = simplex[-1][1] # Worst point (x_h)
        
#         # Stopping Criterion
#         f_values = [s[0] for s in simplex]
#         if np.std(f_values) < tol:
#             print("Stopping criterion (std dev of f_values < tol) met.")
#             break
            
#         # Calculate Centroid (excluding the worst point)
#         x_c = (x_b + x_s) / n  # This is x_bar
        
#         # Store data for the table row
#         xh_str = f"[{x_h[0]:.6f}, {x_h[1]:.6f}]" # x_h
#         xb_str = f"[{x_b[0]:.6f}, {x_b[1]:.6f}]" # x_l
#         xc_str = f"[{x_c[0]:.6f}, {x_c[1]:.6f}]" # x_bar
        
        
#         new_point = None
#         f_new = None
#         op_type = ""

#         # Reflection 
#         x_r = x_c + alpha * (x_c - x_h)
#         f_r = f(x_r)
        
#         if f_r < f_s:
#             if f_r < f_b:
#                 # Expansion
#                 x_e = x_c + gamma * (x_r - x_c)
#                 f_e = f(x_e)
                
#                 if f_e < f_r:
#                     new_point = x_e
#                     f_new = f_e
#                     op_type = "E"
#                 else:
#                     new_point = x_r
#                     f_new = f_r
#                     op_type = "R"
#             else:
#                 # f_b <= f_r < f_s
#                 new_point = x_r
#                 f_new = f_r
#                 op_type = "R"
#         else:
#             # f_r >= f_s
#             #  Contraction 
#             if f_r < f_h:
#                 # Outside Contraction
#                 x_con = x_c + beta * (x_r - x_c)
#             else:
#                 # Inside Contraction
#                 x_con = x_c - beta * (x_c - x_h)
                
#             f_con = f(x_con)
            
#             if f_con < min(f_r, f_h):
#                 # Contraction was successful
#                 new_point = x_con
#                 f_new = f_con
#                 op_type = "C"
#             else:
#                 # Contraction failed, perform Shrink
#                 op_type = "S"
                
#                 s1_new = x_b + 0.5 * (x_s - x_b) 
#                 s2_new = x_b + 0.5 * (x_h - x_b) 
                
#                 simplex[1] = (f(s1_new), s1_new)
#                 simplex[2] = (f(s2_new), s2_new)
                
#                 new_point = x_b
#                 f_new = f_b
            
        
#         # Update Simplex and Print Row
#         if op_type != "S":
#             simplex[-1] = (f_new, new_point)

#         # Format for table row
#         x_new_str = f"[{new_point[0]:.6f}, {new_point[1]:.6f}]" # x_new
        
#         row = (
#             f"| {k:<5} | "
#             f"{xc_str:<22} | "   # x_bar
#             f"{xh_str:<22} | "   # x_h
#             f"{xb_str:<22} | "   # x_l
#             f"{x_new_str:<22} | " # x_new
#             f"{f_new:<12.6f} | "   # f(x_new)
#             f"{op_type:<4} |"
#         )
        
#         print(row)
        
#         k += 1

#     if k == max_iter:
#         print("Max iterations reached.")
        
#     # Final Solution
#     simplex.sort(key=lambda x: x[0])
#     final_best_f = simplex[0][0]
#     final_best_x = simplex[0][1]
    
#     print("-" * len(header))
#     print(f"x* = [{final_best_x[0]:.6f}, {final_best_x[1]:.6f}]")
#     print(f"f(x*) = {final_best_f:.6f}\n")
#     return final_best_x, final_best_f


if __name__ == "__main__":
    
    # --- Parameters for Line Search (fixed) ---
    LS_a = -100.0
    LS_b = 100.0
    LS_eps2 = 0.005

    # Parameter Set 1
    print("="*40 + "\n   PARAMETER SET 1 (Cyclic Coordinate)\n" + "="*40 + "\n")
    cc_x0_1 = [0.0, 0.0]
    eps1_set1 = 1e-3  # 0.02
    cyclic_coordinate_search_cycle_based(cc_x0_1, eps1_set1, LS_a, LS_b, LS_eps2)

    # Parameter Set 2
    print("="*40 + "\n   PARAMETER SET 2 (Cyclic Coordinate)\n" + "="*40 + "\n")
    cc_x0_2 = [5.0, 5.0]
    eps1_set2 = 1e-5  # veya 3e-2 de olur
    cyclic_coordinate_search_cycle_based(cc_x0_2, eps1_set2, LS_a, LS_b, LS_eps2)
    
    # # --- Parameter Set 1 ---
    # print("="*40 + "\n          PARAMETER SET 1 (Hooke-Jeeves)\n" + "="*40 + "\n")
    # set1_eps1 = 1e-3
    # set1_x0 = [0.0, 0.0]
    
    # hooke_jeeves(set1_x0, set1_eps1, LS_a, LS_b, LS_eps2)
    # print("\n") # Add spacing

    # # --- Parameter Set 2 ---
    # print("="*40 + "\n          PARAMETER SET 2 (Hooke-Jeeves)\n" + "="*40 + "\n")
    # set2_eps1 = 1e-5
    # set2_x0 = [5.0, 5.0]  # A different starting point
    
    # hooke_jeeves(set2_x0, set2_eps1, LS_a, LS_b, LS_eps2)

    # # --- Simplex Coefficients ---
    # alpha = 1.0
    # beta = 0.5
    # gamma = 2.0
    
    # # --- Parameter Set 1 (Simplex) ---
    # print("="*40 + "\n           PARAMETER SET 1 (Simplex)\n" + "="*40 + "\n")
    # # a simple triangle around the origin
    # set1_initial_points = [
    #     [0.0, 0.0],
    #     [1.0, 0.0],
    #     [0.0, 1.0]
    # ]
    # simplex_search(set1_initial_points, alpha, beta, gamma)
    # print("\n") 
    
    # # --- Parameter Set 2 (Simplex) ---
    # print("="*40 + "\n           PARAMETER SET 2 (Simplex)\n" + "="*40 + "\n")
    # # a different triangle, starting further away
    # set1_initial_points_2 = [
    #     [5.0, 5.0],
    #     [6.0, 5.0],
    #     [5.0, 6.0]
    # ]
    # simplex_search(set1_initial_points_2, alpha, beta, gamma)
    # print("\n") 