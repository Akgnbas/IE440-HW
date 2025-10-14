import math

#Creating the function and its derivatives 
def f(x):
    #f(x) = x^3 * cos(x) * sin(x) + 3x^2 * sin(x) - 3x
    return x**3 * math.cos(x) * math.sin(x) + 3 * x**2 * math.sin(x) - 3 * x

def f_prime(x):
    #first derivative f'(x)
    return (3 * x**2 * math.sin(2*x) / 2) + x**3 * math.cos(2*x) + 6 * x * math.sin(x) + 3 * x**2 * math.cos(x) - 3

def f_double_prime(x):
    #second derivative f"(x)
    return (-2*x**3*math.sin(2*x)
            + 6*x**2*math.cos(2*x)
            + 3*x*math.sin(2*x)
            - 3*x**2*math.sin(x)
            + 12*x*math.cos(x)
            + 6*math.sin(x))


# Intervals for bisection and golden section methods
common_intervals = [
    # Set 1: Multi-modal interval. f(x) is not unimodal here. (For test)
    (1e-4, -10.0, 10.0),
    # Set 2: Unimodal interval
    (1e-6, 5.0, 6.0),
    # Set 3: Unimodal interval
    (1e-3, -6.0, -5.0),
]

# The function that calculates the Convergence Ratio (R) and Logarithmic Difference (L):
def calculate_ratios(current_x, prev_x, prev2_x, order=1):
    if prev_x is None or prev2_x is None:
        return "N/A", "N/A"

    abs_diff_k = abs(current_x - prev_x)
    abs_diff_prev = abs(prev_x - prev2_x)

    if abs_diff_prev == 0:
        return "N/A", "N/A"

    # Ratio R: |x(k+1) - x(k)| / |x(k) - x(k-1)|^p
    ratio_R = abs_diff_k / (abs_diff_prev ** order)

    # Logarithmic Difference L: -log|x(k+1) - x(k)| + log|x(k) - x(k-1)|
    log_diff_L = "N/A"
    if abs_diff_k > 0 and abs_diff_prev > 0:
        # L = log(|x(k) - x(k-1)| / |x(k+1) - x(k)|)
        log_diff_L = math.log10(abs_diff_prev / abs_diff_k)

    return ratio_R, log_diff_L

# Bisection Method
def bisection_method(obj_func, a_init, b_init, epsilon, run_id):
    # Title
    print(f"\n--- Bisection Method Run {run_id} (e={epsilon}, a={a_init}, b={b_init}) ---")
    a, b = a_init, b_init

    # Check if interval is valid
    if a >= b:
        print(f"Error: Invalid interval a={a}, b={b}. Must have a < b.")
        return

    # Columns: Iteration, a, b, T (x_k), f(T), Ratio R, Log L
    print(f"{'Iter':<5}{'a':<12}{'b':<12}{'T (x_k)':<12}{'f(T)':<15}{'Ratio R (p=1)':<12}{'Log L':<10}")
    print("-" * 76)
    
    x_prev = None
    x_prev2 = None
    iteration = 0
    CONVERGENCE_ORDER = 1  # Convergence order for Bisection is p=1

    while abs(b - a) > epsilon:
        T = (a + b) / 2
        fT = obj_func(T)
        delta = (b - a) / 1000  # Small step for numerical slope estimation
        x_right = T + delta
        f_right = obj_func(x_right)

        # If f(x_right) > f(T), the function is increasing, so minimum is likely to the left
        if f_right > fT:
            b = T  # Narrow interval to [a, T]
        else:
            a = T  # Narrow interval to [T, b]

        # Calculate convergence ratios
        ratio_R, log_diff_L = calculate_ratios(T, x_prev, x_prev2, order=CONVERGENCE_ORDER)

        # Print iteration results
        print(
            f"{iteration:<5}{a:<12.6f}{b:<12.6f}{T:<12.6f}{fT:<15.6f}"
            f"{(f'{ratio_R:.6f}' if isinstance(ratio_R, (int, float)) else 'N/A'):<12}"
            f"{(f'{log_diff_L:.6f}' if isinstance(log_diff_L, (int, float)) else 'N/A'):<10}"
        )

        x_prev2 = x_prev
        x_prev = T
        iteration += 1

        # Prevent infinite loops
        if iteration > 1000:
            print("Warning: Maximum iterations reached.")
            break

    final_x = (a + b) / 2
    final_fx = obj_func(final_x)

    print("-" * 76)
    print(f"Solution for Bisection Method:")
    print(f"x* = {final_x:.8f}")
    print(f"f(x*) = {final_fx:.8f}")
    print("========================================================\n")
    
    return final_x, final_fx  

# Tests
def run_bisection_tests():
    """Bisection tests: (epsilon, a, b)"""
    print("========================================================")
    print("TEST GROUP: BISECTION METHOD")
    for i, (eps, a, b) in enumerate(common_intervals):
        bisection_method(f, a, b, eps, i + 1)

def golden_section_algorithm(a, b, ε, max_iteration=100):
    # Before the first iteration, we should determine x, y and their values
    γ = (math.sqrt(5) - 1) / 2
    x = b - (γ * (b - a))
    y = a + (γ * (b - a))
    fx = f(x)
    fy = f(y)

    # Each iteration will create a dictionary in the list, which contains necessary information for outputs
    results = []
    results.append({
        "k": 0,
        "a": a,
        'b': b,
        'x': x,
        'y': y,
        'fx': fx,
        'fy': fy,
        'ratio': "N/A",
        'log_ratio': "N/A"
    })

    interval_prev = b - a  # Track previous interval length
    interval_prev2 = None  # Track second-previous interval length
    k = 1

    # This block is about how the iterations are made
    while (b - a) >= ε and k < max_iteration:
        new_a, new_b = a, b
        if fx < fy:
            new_b = y
            y = x
            fy = fx
            x = new_b - (γ * (new_b - new_a))
            fx = f(x)
        else:
            new_a = x
            x = y
            fx = fy
            y = new_a + (γ * (new_b - new_a))
            fy = f(y)

        # Select x_k as the point with the lower function value
        x_k = x if fx < fy else y

        # Calculate convergence ratio based on interval length
        interval_current = new_b - new_a
        ratio, log_ratio = calculate_ratios(interval_current, interval_prev, interval_prev2, order=1)

        # Store results
        results.append({
            'k': k,
            'a': new_a,
            'b': new_b,
            'x': x,
            'y': y,
            'fx': fx,
            'fy': fy,
            'ratio': ratio,
            'log_ratio': log_ratio
        })

        interval_prev2 = interval_prev
        interval_prev = interval_current
        a, b = new_a, new_b
        k = k + 1

    # Optimal points at the end
    if fx < fy:
        x_star = x
        f_star = fx
    else:
        x_star = y
        f_star = fy
    return x_star, f_star, results

# Printing the information related to each iteration
def showing_results(results, x_star, f_star, a, b, ε):
    print("\n" + "-"*120)
    print(f"Golden Section Algorithm - Parameters: a={a}, b={b}, ε={ε}")
    print("-"*120)
    print(f"{'Iteration':<10} {'a':<14} {'b':<14} {'x':<14} {'y':<14} {'f(x)':<14} {'f(y)':<14} {'Ratio':<12} {'Log Ratio':<12}")
    print("-"*120)

    for i in results:
        if i["ratio"] is not None and isinstance(i["ratio"], (int, float)):
            ratio_str = f"{i['ratio']:.6f}"
        else:
            ratio_str = "---"

        if i['log_ratio'] is not None and isinstance(i["log_ratio"], (int, float)):
            log_ratio_str = f"{i['log_ratio']:.6f}"
        else:
            log_ratio_str = "---"

        print(f"{i['k']:<10} {i['a']:<14.8f} {i['b']:<14.8f} {i['x']:<14.8f} {i['y']:<14.8f} "
              f"{i['fx']:<14.8f} {i['fy']:<14.8f} {ratio_str:<12} {log_ratio_str:<12}")

    print("-"*120)
    print(f"x* = {x_star:.10f}")
    print(f"f(x*) = {f_star:.10f}")
    print("="*120 + "\n")

# Newton's Method
def newtons_method_algorithm(obj_func, d1_func, d2_func, x0, tol=1e-8, max_iter=100, run_id=1):
    # Prints a table as:
    # Iteration | x^(k) | f(x^(k)) | f'(x^(k)) | f''(x^(k)) | |x^{k+1}-x^{k}| / |x^{k}-x^{k-1}|^2
    print(f"\nSolution for Newton’s Method Run {run_id}:")
    print(f"x0 = {x0:.10f}")
    print(f"{'Iteration':<10}{'x^(k)':<18}{'f(x^(k))':<18}{'f\'(x^(k))':<18}{'f\"(x^(k))':<18}"
          f"{'|Δx_k|/|Δx_{k-1}|^2':<22}")
    print("-" * 100)
    # initialize
    x_prev2 = None
    x_prev  = None
    x       = float(x0)
    # classical newton step
    for k in range(max_iter):
        fx  = obj_func(x)
        g   = d1_func(x)
        H   = d2_func(x)

        # A simple safeguard here for near-zero curvature
        if abs(H) < 1e-14:
            step = -g * 1e-3
        else:
            step = -g / H

        x_new = x + step

        # Convergence ratio with p=2 (quadratic for Newton)
        R, L = calculate_ratios(x_new, x, x_prev, order=2)

        # Print the row for iteration k
        print(f"{k:<10}{x:<18.10f}{fx:<18.10f}{g:<18.10f}{H:<18.10f}"
              f"{(f'{R:.6e}' if isinstance(R,(int,float)) else '---'):<22}")

        # Stop tests: gradient or step small
        if abs(g) <= tol or abs(x_new - x) <= tol:
            x_star = x_new
            f_star = obj_func(x_star)
            print("-" * 100)
            print(f"x* = {x_star:.10f}")
            print(f"f(x*) = {f_star:.10f}")
            return x_star, f_star

        # updating
        x_prev2, x_prev, x = x_prev, x, x_new

    # If max iteration hits
    x_star = x
    f_star = obj_func(x_star)
    print("-" * 100)
    print("Reached max_iter.")
    print(f"x* = {x_star:.10f}")
    print(f"f(x*) = {f_star:.10f}")
    return x_star, f_star

# Secant Method
def secant_method_algorithm(obj_func, d1_func, x0, x1, tol=1e-8, max_iter=200, run_id=1):
    # Prints a table as:
    # Iteration | x^(k) | f(x^(k)) | f'(x^(k)) | |x^{k+1}-x^{k}| / |x^{k}-x^{k-1}|^{φ}
    PHI = 1.6180339887498948  # golden ratio (order of convergence)
    print(f"\nSolution for Secant Method Run {run_id}:")
    print(f"x0 = {x0:.10f}")
    print(f"x1 = {x1:.10f}")
    print(f"{'Iteration':<10}{'x^(k)':<18}{'f(x^(k))':<18}{'f\'(x^(k))':<18}"
          f"{'|Δx_k|/|Δx_{k-1}|^{φ}':<24}")
    print("-" * 90)
    # initialize
    x_prev = float(x0)
    x      = float(x1)
    g_prev = d1_func(x_prev)
    # secant step
    for k in range(max_iter):
        fx = obj_func(x)
        g  = d1_func(x)
        denom = (g - g_prev)
        if abs(denom) < 1e-14:
            dx = -g * 1e-3  # fallback step
        else:
            dx = -g * (x - x_prev) / denom
        x_new = x + dx
        # Convergence ratio with p = φ
        R, L = calculate_ratios(x_new, x, x_prev, order=PHI)
        print(f"{k:<10}{x:<18.10f}{fx:<18.10f}{g:<18.10f}"
              f"{(f'{R:.6e}' if isinstance(R,(int,float)) else '---'):<24}")
        if abs(g) <= tol or abs(x_new - x) <= tol:
            x_star = x_new
            f_star = obj_func(x_star)
            print("-" * 90)
            print(f"x* = {x_star:.10f}")
            print(f"f(x*) = {f_star:.10f}")
            return x_star, f_star
        # updating
        x_prev, g_prev, x = x, g, x_new
    # if max iteration hits
    x_star = x
    f_star = obj_func(x_star)
    print("-" * 90)
    print("Reached max_iter.")
    print(f"x* = {x_star:.10f}")
    print(f"f(x*) = {f_star:.10f}")
    return x_star, f_star

# Modified main execution block to control order
if __name__ == "__main__":
    print("--- Execution Started ---")
    # Bisection tests
    run_bisection_tests()
    # Golden Section tests
    print("*****" + " "*20 + "Common Interval 1" + " "*20 + "*****")
    x_star, f_star, results = golden_section_algorithm(-10, 10, 1e-4, 100)
    showing_results(results, x_star, f_star, -10, 10, 1e-4)
    print("*****" + " "*20 + "Common Interval 2" + " "*20 + "*****")
    x_star, f_star, results = golden_section_algorithm(5, 6, 1e-6, 100)
    showing_results(results, x_star, f_star, 5, 6, 1e-6)
    print("*****" + " "*20 + "Common Interval 3" + " "*20 + "*****")
    x_star, f_star, results = golden_section_algorithm(-6, -5, 1e-3, 100)
    showing_results(results, x_star, f_star, -6, -5, 1e-3)
    # Newton's method tests
    print("***** Newton's Method Tests *****")
    newtons_method_algorithm(f, f_prime, f_double_prime, x0=0.0, tol=1e-6, max_iter=100, run_id=1)
    newtons_method_algorithm(f, f_prime, f_double_prime, x0=5.5, tol=1e-10, max_iter=100, run_id=2)
    newtons_method_algorithm(f, f_prime, f_double_prime, x0=-5.5, tol=1e-6, max_iter=100, run_id=3)
    # Secant method tests
    print("***** Secant Method Tests *****")
    secant_method_algorithm(f, f_prime, x0=-1.0, x1=1.0, tol=1e-6, max_iter=100, run_id=1)
    secant_method_algorithm(f, f_prime, x0=5.0, x1=6.0, tol=1e-10, max_iter=100, run_id=2)
    secant_method_algorithm(f, f_prime, x0=-6.0, x1=-5.0, tol=1e-6, max_iter=100, run_id=3)
    print("--- Execution Completed ---")