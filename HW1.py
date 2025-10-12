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


#Intervals for bisection and golden section methods

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
    #Title
    print(f"\n--- Bisection Method Run {run_id} (e={epsilon}, a={a_init}, b={b_init}) ---")
    a, b = a_init, b_init

    # Bisection requires f'(a) and f'(b) to have opposite signs
    if f_prime(a) * f_prime(b) > 0:
        print(f"Error: f'({a})={f_prime(a):.2e} and f'({b})={f_prime(b):.2e}. f'(a) * f'(b) > 0. Root not guaranteed.")

    # Columns: Iteration, a, b, T (x_k), f(T), Ratio R, Log L
    print(f"{'Iter':<5}{'a':<12}{'b':<12}{'T (x_k)':<12}{'f(T)':<15}{'Ratio R (p=1)':<12}{'Log L':<10}")
    print("-" * 76)

    x_prev = None
    x_prev2 = None
    iteration = 0
    
    # Convergence order for Bisection is p=1 
    CONVERGENCE_ORDER = 1 

    while abs(b - a) / 2 > epsilon: 
        T = (a + b) / 2
        fT = f(T)
        delta = (b - a) / 1000  # Small step for comparison
        x_plus_e = T + delta 
        if f(x_plus_e) <= fT:
            # If moving slightly right decreases f(x), the minimum is to the right.
            a = T 
        else:
            # If moving slightly right increases f(x), the minimum is to the left.
            b = T


        ratio_R, log_diff_L = calculate_ratios(T, x_prev, x_prev2, order=CONVERGENCE_ORDER)

    
    # Print iteration results in the specified format
        print(
            f"{iteration:<5}{a:<12.6f}{b:<12.6f}{T:<12.6f}{fT:<15.6f}"
            f"{str(ratio_R):<12}{str(log_diff_L):<10}"
        )
        
        x_prev2 = x_prev
        x_prev = T

        iteration += 1

    final_x = (a + b) / 2
    final_fx = f(final_x)


    print("-" * 76)
    print(f"Solution for Bisection Method:")
    print(f"x*={final_x:.8f}")
    print(f"f(x*)={final_fx:.8f}")
    print("========================================================\n")



# Tests 

def run_bisection_tests():
    """Bisection tests: (epsilon, a, b)"""
    print("========================================================")
    print("TEST GROUP: BISECTION METHOD")

    for i, (eps, a, b) in enumerate(common_intervals):
        bisection_method(f, a, b, eps, i + 1)




if __name__ == "__main__":
    print("--- Execution Started ---")
    run_bisection_tests()


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
        'ratio': None,
        'log_ratio': None})
    
    x_previous = x
    k = 1

    # This block is about how the iterations are made
    while (b - a) >= ε and k < max_iteration:
        if fx < fy:
            b = y
            y = x
            fy = fx
            x = b - ((γ) * (b - a))
            fx = f(x)
        else:
            a = x
            x = y
            fx = fy
            y = a + ((γ) * (b - a))
            fy = f(y)

        # Calculating how fast it converges (pace of the convergence)
        if fx<fy:
            x_k = x 
        else:
            x_k = y

        if k > 1:
            ratio, log_ratio = calculate_ratios(current_x=x_k,prev_x=x_previous,prev2_x=x_2previous,order=1)
        else:
            ratio, log_ratio = "N/A", "N/A"

        # Necessary information for the iteration is added to the list as a dictionary    
        results.append({
            'k': k,
            'a': a,
            'b': b,
            'x': x,
            'y': y,
            'fx': fx,
            'fy': fy,
            'ratio': ratio,
            'log_ratio': log_ratio})

        x_2previous = x_previous
        x_previous = x_k
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


# Common Interval 1
print("*****" + " "*20 + "Common Interval 1" + " "*20 + "*****")
x_star, f_star, results= golden_section_algorithm(-10,10,1e-4,100)
showing_results(results, x_star,f_star,-10,10,1e-4)

# Common Interval 2
print("*****" + " "*20 + "Common Interval 2" + " "*20 + "*****")
x_star, f_star, results= golden_section_algorithm(5, 6, 1e-6,100)
showing_results(results, x_star,f_star,5, 6, 1e-6)

# Common Interval 3
print("*****" + " "*20 + "Common Interval 3" + " "*20 + "*****")
x_star, f_star, results= golden_section_algorithm(-6, -5, 1e-3,100)
showing_results(results, x_star,f_star,-6, -5, 1e-3)





