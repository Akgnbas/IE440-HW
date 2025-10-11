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

