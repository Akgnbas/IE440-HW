import csv 
import numpy as np
import math


coordinates="coordinates_10.csv"
demands="demand_10.csv"
costs="costs_10.csv"

epsilon=1e-8 
weizfeld_max_iter=1000
weizfeld_tolerance=1e-6


def load_data():

    # Load customer coordinates (a_j)
    customer_coords = []
    with open(coordinates, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            customer_coords.append([float(row[0]), float(row[1])])
    customer_coords = np.array(customer_coords)
    n = customer_coords.shape[0]

    # Load customer demand (h_j)
    demand = []
    with open(demands, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            demand.append(float(row[0]))
    demand = np.array(demand)

     # Load unit transport costs (c_ij)
    unit_costs = []
    with open(costs, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            unit_costs.append([float(c) for c in row])
    unit_costs = np.array(unit_costs)
    m = unit_costs.shape[0]


    # Calculate total transport cost C_ij = h_j * c_ij
    transport_costs = unit_costs * demand

    print(f"Number of customers (n): {n}")
    print(f"Number of facilities (m): {m}")
    print("-" * 30)
    
    return n, m, customer_coords, demand, unit_costs, transport_costs


# Distance Functions 

def squared_euclidean_distance(p1, p2):
    return np.sum((p1 - p2)**2)

def euclidean_distance(p1, p2, epsilon=epsilon):
    return math.sqrt(np.sum((p1 - p2)**2)) + epsilon  # To prevent division by zero 


# Q1

def solve_part1(facility_index, customer_coords, transport_costs):
    print(f"Solving for Facility {facility_index + 1}")

    # Get the transport costs (C_ij) for the chosen facility
    C_j = transport_costs[facility_index, :]

    # Calculate the optimal location by formulas
    # x_1* = (sum(C_j * a_1j)) / sum(C_j)
    # x_2* = (sum(C_j * a_2j)) / sum(C_j)
   
    sum_C_j = np.sum(C_j) 

    # This calculates [sum(C_j * a_1j), sum(C_j * a_2j)]
    weighted_coords_sum = np.dot(C_j, customer_coords)
    # Division in formula
    optimal_location = weighted_coords_sum / sum_C_j

    # Calculate the total objective cost
    total_cost = 0.0
    for j in range(customer_coords.shape[0]):
        dist_sq = squared_euclidean_distance(optimal_location, customer_coords[j])
        total_cost += C_j[j] * dist_sq
    
    # Outputs

    print(f"Optimal Location: ({optimal_location[0]:.2f}, {optimal_location[1]:.2f})")
    print(f"Minimum Total Cost: {total_cost:,.2f}")
    
    return optimal_location, total_cost



# Q2

def solve_part2(facility_index, customer_coords, transport_costs, initial_location):
    print(f"Solving for Facility {facility_index + 1} with Weiszfeld's Algorithm")
    print(f"Initial Location: ({initial_location[0]:.2f}, {initial_location[1]:.2f})")

    # Get the number of customers
    n = customer_coords.shape[0]
    # Get the transport cost vector (C_j) for this specific facility (i)
    C_j = transport_costs[facility_index, :]
    # Set the starting location for the iterative algorithm
    current_location = np.copy(initial_location)

    # Start the iterative Weiszfeld's algorithm
    for k in range(weizfeld_max_iter):
        # Initialize the numerator and denominator for the update formula
        numerator = np.zeros(2)   # [0.0, 0.0]
        denominator = 0.0

        # For all customers j=1 to n
        for j in range(n):
            a_j = customer_coords[j] # Location of customer j
            # Calculate Euclidean distance
            dist = euclidean_distance(current_location, a_j, epsilon)
            
            # Weiszfeld's update formula:
            # x^(k+1) = [ sum(C_j * a_j / ||x - a_j||) ] / [ sum(C_j / ||x - a_j||) ]
            
            # Add this customer's contribution to the numerator
            numerator += (C_j[j] * a_j) / dist
            # Add this customer's contribution to the denominator
            denominator += C_j[j] / dist

        # Calculate the new location (x^(k+1))
        new_location = numerator / denominator

        # If the change in location is very small, we've found our answer
        if euclidean_distance(new_location, current_location) < weizfeld_tolerance:
            print(f"Converged after {k + 1} iterations.")
            current_location = new_location # Set the final location
            break # Exit the loop

        current_location = new_location
        
    # If the loop finished 1000 iterations and never converged:
    else:
        print(f"Reached max iterations ({weizfeld_max_iter}) without converging.")

    # Calculate final total cost
    total_cost = 0.0
    for j in range(n):
        dist = euclidean_distance(current_location, customer_coords[j])
        total_cost += C_j[j] * dist
        
    print(f"Final Location: ({current_location[0]:.2f}, {current_location[1]:.2f})")
    print(f"Minimum Total Cost: {total_cost:,.2f}")
    
    return current_location, total_cost



            


    
    


# --- Main Execution ---

if __name__ == "__main__":
        # Load all data
        n, m, customer_coords, demand, unit_costs, transport_costs = load_data()
        
        # Part 1 
        print("\n=== Q 1: Single Facility (Squared Euclidean) ===")
        # We chose Facility 1 (index 0)
        FACILITY_TO_TEST = 0
        part1_location, part1_cost = solve_part1(FACILITY_TO_TEST, customer_coords, transport_costs)
    
        # Part 2 
        print("\n=== PART 2: Single Facility (Euclidean - Weiszfeld) ===")
        # Use the result from Part 1 as the initial location
        solve_part2(FACILITY_TO_TEST, customer_coords, transport_costs, part1_location)
        








    



    