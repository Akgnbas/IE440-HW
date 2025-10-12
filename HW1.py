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
    
    # Logarithmic Difference L: -log|x(k+1) - x(k)| + log|x(k) - log(k-1)|
    log_diff_L = "N/A"
    if abs_diff_k > 0 and abs_diff_prev > 0:
        # L = log(|x(k) - x(k-1)| / |x(k+1) - x(k)|)
        log_diff_L = math.log10(abs_diff_prev / abs_diff_k)

    return ratio_R, log_diff_L