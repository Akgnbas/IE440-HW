import csv 
import numpy as np
import math
import sys


coordinates="coordinates_10.csv"
demands="demand_10.csv"
costs="costs_10.csv"

epsilon=1e-8 
weizfeld_max_iter=1000
weizfeld_tolerance=1e-6

N_TRIALS = 1000
MAX_ALA_ITERATIONS = 100 # Max iterations for the inner ALA loop

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
    transport_costs = unit_costs * demand[np.newaxis, :]
    #small error corrected here above

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

def solve_part3(n, m, customer_coords, transport_costs):
    """
    Solves the Multi-Facility Problem using the
    Alternative Location-Allocation (ALA) Heuristic.
    """
    print(f"Running ALA Heuristic for {N_TRIALS} trials...")
    
    # Store the final cost of each trial
    all_trial_costs = []
    
    # --- NEW: Store best results ---
    best_cost = np.inf
    best_locations = np.zeros((m, 2))
    best_assignments = np.zeros(n, dtype=int)
    # -------------------------------
    
    # 3.3: Outer loop for 1000 trials
    for trial in range(N_TRIALS):
        
        # --- 3.1: Randomly allocate customers to facilities ---
        assignments = np.random.randint(0, m, size=n)
        facility_locations = np.zeros((m, 2))
        
        # Inner loop for ALA convergence
        for ala_iter in range(MAX_ALA_ITERATIONS):
            
            # --- A: Location Step ---
            new_locations = np.zeros((m, 2))
            for i in range(m):
                customer_indices = np.where(assignments == i)[0]
                
                if len(customer_indices) > 0:
                    assigned_coords = customer_coords[customer_indices]
                    C_j_weights = transport_costs[i, customer_indices]
                    sum_C_j = np.sum(C_j_weights)
                    weighted_coords_sum = np.dot(C_j_weights, assigned_coords)
                    
                    if sum_C_j > 0:
                        new_locations[i] = weighted_coords_sum / sum_C_j
                    else:
                        new_locations[i] = np.mean(assigned_coords, axis=0)
                else:
                    rand_customer_idx = np.random.randint(0, n)
                    new_locations[i] = customer_coords[rand_customer_idx]
            
            facility_locations = new_locations
            
            # --- B: Allocation Step ---
            new_assignments = np.zeros(n, dtype=int)
            for j in range(n):
                customer_loc = customer_coords[j]
                costs_to_all_facilities = np.zeros(m)
                for i in range(m):
                    C_ij = transport_costs[i, j]
                    dist_sq = squared_euclidean_distance(facility_locations[i], customer_loc)
                    costs_to_all_facilities[i] = C_ij * dist_sq
                    
                new_assignments[j] = np.argmin(costs_to_all_facilities)

            # --- C: Check Convergence ---
            if np.all(assignments == new_assignments):
                break
            
            assignments = new_assignments
            
        # --- End of inner ALA loop ---
        
        # Calculate the final total cost for this converged trial
        trial_total_cost = 0.0
        for j in range(n):
            i = assignments[j]
            C_ij = transport_costs[i, j]
            dist_sq = squared_euclidean_distance(facility_locations[i], customer_coords[j])
            trial_total_cost += C_ij * dist_sq
            
        all_trial_costs.append(trial_total_cost)
        
        # --- NEW: Check if this is the best result so far ---
        if trial_total_cost < best_cost:
            best_cost = trial_total_cost
            best_locations = np.copy(facility_locations)
            best_assignments = np.copy(assignments)
        # ----------------------------------------------------

        # Simple progress bar
        if (trial + 1) % 100 == 0:
            print(f"  Completed trial {trial + 1}/{N_TRIALS}")

    # --- End of 1000 trials ---
    
    # 3.3: Report the average and best results
    # best_cost = np.min(all_trial_costs) # <--- We already have this
    avg_cost = np.mean(all_trial_costs)
    
    print("\n--- ALA Heuristic Results (Part 3) ---")
    print(f"Best (Minimum) Cost found: {best_cost:,.2f}")
    print(f"Average Cost over {N_TRIALS} trials: {avg_cost:,.2f}")
    
    # --- NEW: Print the best results ---
    print("\n--- Details for Best Result ---")
    print("Optimal Facility Locations (x1, x2):")
    for i in range(m):
        print(f"  Facility {i + 1}: ({best_locations[i, 0]:.2f}, {best_locations[i, 1]:.2f})")
        
    print("\nCustomer Assignments (Customer -> Facility):")
    for j in range(n):
        print(f"  Customer {j + 1} -> Facility {best_assignments[j] + 1}")
    # -------------------------------------

    return best_cost, avg_cost

            
# Q4

def pure_euclidian_distance(p, q):
    # Euclidian distance to find actual distance without any epsilon
    return math.sqrt(np.sum((p - q) ** 2))

def weiszfeld(a_coords, weights, x0, tol=weizfeld_tolerance, eps=epsilon, max_it=weizfeld_max_iter):
    # This function is intended to perform weiszfeld algorithm for only one facility.
    # I thought that it would ease my work for the main part of the function called "solve_part4".
    # Let me introduce the parameters first.
    # a_coords = coordinates of customers assigned to facility
    # weights = C_ij = h_j * c_ij
    # x0 = starting point of facility
    # tol = for convergence, eps = for checking 1 / 0 conditions, max_it = maximum iteration

    x = np.array(x0, dtype=float)
    for _ in range(max_it):
        d = np.linalg.norm(a_coords - x, axis=1) # vector gives the distance values of facility i to customer j's

        # If a facility is closer to a customer than tol, take the facility to the customer's location
        near = np.where(d < tol)[0]
        if near.size > 0 and np.any(weights[near] > 0):
            x_new = a_coords[near[0]]
            if pure_euclidian_distance(x_new, x) < tol:
                return x_new
            x = x_new
            continue

        # 1 / d (x,dj) in the formula. To avoid 1/0, I take 1/max(d,eps)
        invd = 1.0 / np.maximum(d, eps)

        # Sum of [Cij*customer coordinates* 1/d]
        num = (weights[:, None] * a_coords * invd[:, None]).sum(axis=0)
        # Sum of [Cij*1/d]
        den = (weights * invd).sum()

        # New location of facility
        x_new = a_coords.mean(axis=0) if den <= 0 else num / den

        # if the location change is lower than tol, it is enough
        if pure_euclidian_distance(x_new, x) < tol:
            return x_new
        x = x_new

        # If it reaches max iteration
    return x  

def total_cost(x_fac, assignments, customer_coords, transport_costs):
    # x_fac = coordinates of facilities
    # assignments = which customer assigned to which facility. It is a vector
    # This function is intended for calculating the total cost.
    # It spans over customers for each facility.
    # sum_j C_{i_j, j} * ||x_{i_j} - a_j|| ]

    total = 0.0
    for j in range(customer_coords.shape[0]):
        i = assignments[j]
        total += transport_costs[i, j] * pure_euclidian_distance(x_fac[i], customer_coords[j])
    return total

def assigning_customers(customer_coords, x_fac, transport_costs):
    # This function is intended for assigning customers to nearest weighted facilities.
    # For each customer j, i = argmin_i (C_ij * ||x_i - a_j||)
    n, m = customer_coords.shape[0], x_fac.shape[0]
    assignments = np.zeros(n, dtype=int)
    for j in range(n):
        vals = np.empty(m)
        for i in range(m):
            vals[i] = transport_costs[i, j] * pure_euclidian_distance(x_fac[i], customer_coords[j])
        assignments[j] = int(np.argmin(vals))
    return assignments

def solve_part4(n, m, customer_coords, transport_costs, restarts=1000, max_outer=100, seed=123):
    rng = np.random.default_rng(seed)
    best_cost = float('inf')
    best_locations = None
    best_assignments = None
    best_iters = None
    best_restart = None
    all_costs= []

    for r in range(restarts):
        # random assigning as a starting point
        assignments = rng.integers(low=0, high=m, size=n)

        # starting facilities locations
        x_fac = np.zeros((m, 2))
        for i in range(m):
            idx = np.where(assignments == i)[0] # gives the indexes of customers assigned to ith facility
            if idx.size > 0:
                # take a random index from assigned customers
                # and set the location of a facility there to start from a closer area.
                x_fac[i] = customer_coords[idx[rng.integers(0, idx.size)]] 
            else:
                # if there is no assigned customer to that facility, locate it randomly
                x_fac[i] = customer_coords[rng.integers(0, n)] 

        prev_obj = float('inf')
        for t in range(1, max_outer + 1):
            # This is intended for weiszfeld algorithm for each facility.
            for i in range(m):
                idx = np.where(assignments == i)[0]
                if idx.size == 0:
                    # facility having no assigned customer, we locate it to a random customers location
                    x_fac[i] = customer_coords[rng.integers(0, n)]
                    continue
                a_i = customer_coords[idx]
                w_i = transport_costs[i, idx]
                x_fac[i] = weiszfeld(a_i, w_i, x_fac[i])

            # After weiszfeld algorithm, calculation of total cost
            cur_obj = total_cost(x_fac, assignments, customer_coords, transport_costs)

            # Assigning the customers to the nearest weighted facilities
            new_assignments = assigning_customers(customer_coords, x_fac, transport_costs)

            # If there is no improvement and assigned customers are the same
            if prev_obj - cur_obj <= 1e-8:
                break

            assignments = new_assignments
            prev_obj = cur_obj

        # For each restart, we calculate the final obj.
        final_obj = total_cost(x_fac, assignments, customer_coords, transport_costs)
        all_costs.append(final_obj)

        # If this restart's result is better than others, we take it
        if final_obj < best_cost:
            best_cost = final_obj
            best_locations = x_fac.copy()
            best_assignments = assignments.copy()
            best_iters = t
            best_restart = r

        # To see how many trial is completed
        if (r + 1) % 100 == 0:
            print(f"  Completed {r+1}/{restarts} restarts")

    avg_cost = float(np.mean(all_costs))


    print("\n=== RESULTS ===")
    print(f"Average objective over {restarts} restarts: {avg_cost:.6f}")
    print(f"Best objective: {best_cost:.6f}")
    # it shows how many ALA iteration is done in the best restart case giving the best case
    # print(f"Outer iterations (best run): {best_iters}") 
    # it shows at which restart, the best cost is achieved
    # print(f"Restart index (best run): {best_restart}")
    # Since they are not asked, I dont include them
    print("\nFacility locations (x1, x2):")
    for i, (x1, x2) in enumerate(best_locations, start=1):
        print(f"  Facility {i}: ({x1:.4f}, {x2:.4f})")
    print("\nCustomer assignments (customer -> facility):")
    for j, i in enumerate(best_assignments, start=1):
        print(f"Customer {j} -> Facility {i+1}")

    return best_cost, best_locations, best_assignments, avg_cost

    
    


# --- Main Execution ---

if __name__ == "__main__":
        # Load all data
        n, m, customer_coords, demand, unit_costs, transport_costs = load_data()
        
        # Part 1 
        print("\n=== PART 1: Single Facility (Squared Euclidean) ===")
        # We chose Facility 1 (index 0)
        FACILITY_TO_TEST = 0
        part1_location, part1_cost = solve_part1(FACILITY_TO_TEST, customer_coords, transport_costs)
    
        # Part 2 
        print("\n=== PART 2: Single Facility (Euclidean - Weiszfeld) ===")
        # Use the result from Part 1 as the initial location
        solve_part2(FACILITY_TO_TEST, customer_coords, transport_costs, part1_location)
        
        # Part 3
        print("\n=== PART 3: Multi-Facility (ALA Heuristic) ===")
        solve_part3(n, m, customer_coords, transport_costs)
        
        # Part 4
        print("\n=== PART 4: Multi-Facility (ALA + Weiszfeld, Euclidean) ===")
        solve_part4(n, m, customer_coords, transport_costs)








    



    