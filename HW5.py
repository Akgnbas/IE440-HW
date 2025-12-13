import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0) Data loading, splitting, and standardization
# ============================================================

def load_regression_data(path="regression_data.dat"):
    data = np.loadtxt(path)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

    # We split the data randomly as 80% train and 20% test
def train_test_split_random(x, y, train_ratio=0.8, seed=440):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    n = len(x)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(train_ratio * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_test, y_test


    # This is the standardization process of train inptus.
    # Why we apply it is explained in the report, please see there.
def standardize_train_test(x_train, x_test):
    # z = (x-mean)/st. dev.

    x_train = np.asarray(x_train, float)
    x_test = np.asarray(x_test, float)

    mu = np.mean(x_train)
    sigma = np.std(x_train)

    if sigma <= 1e-12:
        sigma = 1.0

    z_train = (x_train - mu) / sigma
    z_test = (x_test - mu) / sigma

    return z_train, z_test, mu, sigma


def standardize_y_train_test(y_train, y_test):
    # This is only for MLP training to improve numerical stability.
    # This is also explained in the report.

    # y_tilde = (x-mean)/st. dev. of y
    y_train = np.asarray(y_train, float)
    y_test = np.asarray(y_test, float)

    mu_y = np.mean(y_train)
    sigma_y = np.std(y_train)

    if sigma_y <= 1e-12:
        sigma_y = 1.0

    y_train_scaled = (y_train - mu_y) / sigma_y
    y_test_scaled = (y_test - mu_y) / sigma_y

    return y_train_scaled, y_test_scaled, mu_y, sigma_y


# ============================================================
# 1) Q1 – Least squares with steepest descent in z-space
# ============================================================

def build_design_matrix_z(z_vec, degree):
    # Build polynomial design matrix for 1D standardized input z.
    # degree = 1 -> columns [1, z]
    # degree = 2 -> columns [1, z, z^2]

    z_vec = np.asarray(z_vec, float)
    if degree == 1:
        Z = np.column_stack([np.ones_like(z_vec), z_vec])
    elif degree == 2:
        Z = np.column_stack([np.ones_like(z_vec), z_vec, z_vec ** 2])
    else:
        raise ValueError("This helper only supports degree 1 or 2.")
    return Z

def steepest_descent_ls(Z, y, max_iter=5000, tol=1e-8, verbose=False):
    # Steepest descent with exact line search for the least-squares
    
    # objective: J(β) = sum_i (y_i - (Z β)_i)^2 = ||y - Z β||^2
    # Direction: d_k = -grad J(β_k)
    # Step size: exact line search along d_k

    Z = np.asarray(Z, float)
    y = np.asarray(y, float)

    n, p = Z.shape
    beta = np.zeros(p)
    history = []

    for k in range(max_iter):
        # Residuals and objective
        r = y - Z @ beta
        J = np.sum(r ** 2)
        history.append(J)

        # Gradient: g = -2 Z^T r
        g = -2.0 * Z.T @ r
        grad_norm = np.linalg.norm(g)

        if verbose and (k % 50 == 0):
            print(f"[iter {k:4d}] J={J:.6e}, ||g||={grad_norm:.3e}")

        # Stopping criterion based on gradient norm
        if grad_norm < tol:
            if verbose:
                print(
                    f"Converged at iteration {k}, "
                    f"||g||={grad_norm:.3e}, J={J:.6e}"
                )
            break

        # Steepest descent direction
        d = -g

        # Exact line search along d:
        # minimize ||(y - Zβ) - alpha Z d||^2 in alpha
        a_vec = Z @ d 
        denom = np.sum(a_vec ** 2)
        if denom <= 1e-20:
            if verbose:
                print("Line-search denominator too small; stopping.")
            break
        num = r @ a_vec  # r^T (Z d)
        alpha_ls = num / denom

        # Update
        beta = beta + alpha_ls * d

    if verbose:
        print(
            f"Steepest descent finished in {k + 1} iterations, "
            f"final J={J:.6e}, final ||g||={grad_norm:.3e}"
        )

    return beta, history


def map_linear_beta_to_w(beta0, beta1, mu, sigma):
    # Mapping the linear coefficients from standardized z-space to original x-space coefficients
    # y ≈ beta0 + beta1 * z,  z = (x - mu) / sigma
    # y ≈ w0 + w1 * x.
    # w1 = beta1 / sigma
    # w0 = beta0 - beta1 * mu / sigma
    
    w1 = beta1 / sigma
    w0 = beta0 - beta1 * mu / sigma
    return w0, w1


def map_quadratic_beta_to_w(beta0, beta1, beta2, mu, sigma):
    # Mapping the quadratic coefficients from standardized z-space to original x-space coefficients
    # y ≈ w0 + w1 x + w2 x^2.
    # w2 = beta2 / sigma^2
    # w1 = beta1 / sigma - 2 * beta2 * mu / sigma^2
    # w0 = beta0 - beta1 * mu / sigma + beta2 * mu^2 / sigma^2
    
    w2 = beta2 / (sigma ** 2)
    w1 = beta1 / sigma - 2.0 * beta2 * mu / (sigma ** 2)
    w0 = beta0 - beta1 * mu / sigma + beta2 * (mu ** 2) / (sigma ** 2)
    return w0, w1, w2


def build_design_matrix_x(x_vec, degree):
    # Building the matrix in original x-space
    # degree = 1 -> columns [1, x]
    # degree = 2 -> columns [1, x, x^2]
    x_vec = np.asarray(x_vec, float)
    if degree == 1:
        X = np.column_stack([np.ones_like(x_vec), x_vec])
    elif degree == 2:
        X = np.column_stack([np.ones_like(x_vec), x_vec, x_vec ** 2])
    else:
        raise ValueError("This helper only supports degree 1 or 2.")
    return X


def compute_metrics(x_train, y_train, x_test, y_test, w, degree):
    # Given coefficients w, train and test data, we compute training SSE, test MSE, and s^2 for test MSE 
    X_train = build_design_matrix_x(x_train, degree)
    X_test = build_design_matrix_x(x_test, degree)

    w = np.asarray(w, float)

    r_train = y_train - X_train @ w
    r_test = y_test - X_test @ w

    training_SSE = np.sum(r_train ** 2)

    e_sq_test = r_test ** 2
    n_test = len(y_test)
    test_MSE = np.mean(e_sq_test)

    s2 = np.sum((test_MSE - e_sq_test) ** 2) / (n_test - 1)

    return training_SSE, test_MSE, s2


def solve_q1(x, y, x_train, y_train, x_test, y_test, z_train, mu, sigma):
    # Solving the question with scaling

    # ---------- 1(a) Linear ----------
    Z_train_lin = build_design_matrix_z(z_train, degree=1)
    beta_lin, _ = steepest_descent_ls(
        Z_train_lin, y_train, max_iter=5000, tol=1e-8, verbose=False
    )

    beta0_lin, beta1_lin = beta_lin
    w0_lin, w1_lin = map_linear_beta_to_w(beta0_lin, beta1_lin, mu, sigma)
    w_lin = np.array([w0_lin, w1_lin])

    print("\n=== Question 1(a): Linear regression with scaling ===")
    print("Model in x-space: y = w0 + w1 x")
    print(f"w0 = {w0_lin:.6f}")
    print(f"w1 = {w1_lin:.6f}")

    train_SSE_lin, test_MSE_lin, s2_lin = compute_metrics(
        x_train, y_train, x_test, y_test, w_lin, degree=1
    )

    # ---------- 1(b) Quadratic ----------
    Z_train_quad = build_design_matrix_z(z_train, degree=2)
    beta_quad, _ = steepest_descent_ls(
        Z_train_quad, y_train, max_iter=5000, tol=1e-8, verbose=False
    )

    beta0_q, beta1_q, beta2_q = beta_quad
    w0_q, w1_q, w2_q = map_quadratic_beta_to_w(beta0_q, beta1_q, beta2_q, mu, sigma)
    w_quad = np.array([w0_q, w1_q, w2_q])

    print("\n=== Question 1(b): Quadratic regression with scaling ===")
    print("Model in x-space: y = w0 + w1 x + w2 x^2")
    print(f"w0 = {w0_q:.6f}")
    print(f"w1 = {w1_q:.6f}")
    print(f"w2 = {w2_q:.6f}")

    train_SSE_quad, test_MSE_quad, s2_quad = compute_metrics(
        x_train, y_train, x_test, y_test, w_quad, degree=2
    )

    # ---------- Table ----------
    print("\n=== Table values for Homework (Question 1) ===")
    print("Method    | Training SSE       | Test MSE          | s^2 for Test MSE")
    print("-----------------------------------------------------------------------")
    print(f"1(a) Lin  | {train_SSE_lin:16.6f} | {test_MSE_lin:16.6f} | {s2_lin:16.6f}")
    print(f"1(b) Quad | {train_SSE_quad:16.6f} | {test_MSE_quad:16.6f} | {s2_quad:16.6f}")

    # ---------- Plots ----------
    x_min, x_max = x.min(), x.max()
    x_plot = np.linspace(x_min, x_max, 500)

    y_lin_plot = w0_lin + w1_lin * x_plot
    y_quad_plot = w0_q + w1_q * x_plot + w2_q * (x_plot ** 2)

    plt.figure(figsize=(10, 4))

    # Left: training
    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, marker="x", label="Training data", alpha=0.7)
    plt.plot(x_plot, y_lin_plot, label="Linear fit (1a)")
    plt.plot(x_plot, y_quad_plot, label="Quadratic fit (1b)")
    plt.title("Q1 – Training data and regression fits")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Right: test
    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, marker="o", label="Test data", alpha=0.7)
    plt.plot(x_plot, y_lin_plot, label="Linear fit (1a)")
    plt.plot(x_plot, y_quad_plot, label="Quadratic fit (1b)")
    plt.title("Q1 – Test data and regression fits")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 2) Q2(a) – One-hidden-layer MLP regression
# ============================================================

def init_mlp_params(J, rng, input_dim=1):
    """
    Initializes weights and biases for the MLP.
    
    Args:
        J: Number of hidden units.
        rng: Random number generator instance.
        input_dim: Dimension of input vector (Default=1 for Q2a compatibility).
    """
    # Weights are initialized from a normal distribution.
    # Shape: (Hidden Units, Input Dimension)
    W_in = rng.normal(loc=0.0, scale=0.1, size=(J, input_dim))
    b_h = np.zeros(J)
    
    # Output weights: (Hidden Units)
    W_out = rng.normal(loc=0.0, scale=0.1, size=J)
    b_out = 0.0
    return W_in, b_h, W_out, b_out


def forward_mlp(z, params):
    """
    Computes the forward pass of the network.
    Supports both 1D inputs (Q2a) and Multi-dimensional inputs (Q2b).
    """
    W_in, b_h, W_out, b_out = params
    z = np.asarray(z, float)
    
    # --- Compatibility Handling ---
    # If input is a 1D vector (from Q2a), reshape to (N, 1) matrix.
    # If input is already (N, 2) (from Q2b), leave it as is.
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Hidden Layer Calculation
    # Matrix multiplication: (N, input_dim) @ (input_dim, J) -> (N, J)
    h_lin = z @ W_in.T          
    h_lin += b_h                   
    
    # Clip to avoid overflow in exponential
    h_lin = np.clip(h_lin, -50.0, 50.0) 
    
    # Sigmoid Activation
    h = 1.0 / (1.0 + np.exp(-h_lin))

    # Output Layer (Linear)
    y_hat = h @ W_out + b_out    
    return h, y_hat


def train_mlp(z_train, y_train_scaled, J, alpha0=0.5, eta=0.9, eps=1e-3, max_epochs=5000, seed=440, verbose=False, input_dim=1):
    """
    Trains the MLP using Batch Gradient Descent.
    
    Args:
        input_dim: Defaults to 1 for Question 2(a) .
                   Set to 2 for Question 2(b).
    """
    rng = np.random.default_rng(seed)
    
    # Initialize parameters (Correctly passes input_dim)
    W_in, b_h, W_out, b_out = init_mlp_params(J, rng, input_dim=input_dim)

    y_train_scaled = np.asarray(y_train_scaled, float)
    history = []
    alpha = alpha0
    N = len(z_train)

    # Ensure input is in matrix format for vectorized calculation
    X_mat = np.asarray(z_train, float)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)

    for epoch in range(max_epochs):
        # Forward Pass
        h, y_hat = forward_mlp(X_mat, (W_in, b_h, W_out, b_out))

        # Error Calculation
        err = y_train_scaled - y_hat
        L_val = np.mean(err ** 2)
        history.append(L_val)

        # Convergence Check
        if epoch > 0:
            denom = max(abs(history[-2]), 1e-12)
            rel_change = abs(history[-1] - history[-2]) / denom
            if rel_change < eps:
                if verbose: print(f"Converged at epoch {epoch}")
                break

        # Backpropagation
        # 1. Output Layer Gradients
        dL_dy = -2.0 * err / N       
        dW_out = h.T @ dL_dy          
        db_out = np.sum(dL_dy)        

        # 2. Hidden Layer Gradients
        dL_dh = np.outer(dL_dy, W_out)  
        dh_dhlin = h * (1.0 - h)
        dL_dhlin = dL_dh * dh_dhlin     
        
        # 3. Input Layer Gradients
        # This works correctly for both input_dim=1 and input_dim=2
        dW_in = dL_dhlin.T @ X_mat       
        db_h = np.sum(dL_dhlin, axis=0) 

        # Update Weights
        W_out -= alpha * dW_out
        b_out -= alpha * db_out
        W_in  -= alpha * dW_in
        b_h   -= alpha * db_h
        
        # Decay learning rate
        alpha *= eta

    return (W_in, b_h, W_out, b_out), history


    # For Question 2(a), I implemented 2 different aprroach. Please check the report to see why.
def solve_q2a(x, x_train, y_train, x_test, y_test, z_train, z_test, mu_x, sigma_x):
    # Solving the question by using algortihm that stops when test MSE increases first time
    # I also plotted it (J vs error)
    
    # It returns the number of J, and corresponding metrics 
 
    # Standardize outputs for MLP training (ONLY here)
    y_train_scaled, y_test_scaled, mu_y, sigma_y = standardize_y_train_test(
        y_train, y_test
    )

    # Initial J (given J(0) = 3)
    J = 3
    q = 1

    results = []      # list of (J, params, history, train_SSE, test_MSE, s2)
    prev_E = np.inf   # previous test error (E_avg(q-1))

    J_list = []
    test_MSE_list = []

    J_max = 50  # safety upper bound for algorithm

    while True:
        # 1) Train on TRAIN data with standardized y
        params, history = train_mlp(
            z_train,
            y_train_scaled,
            J=J,
            alpha0=0.5,  # α(0)
            eta=0.9,     # α(t+1) = η α(t)
            eps=1e-3,
            max_epochs=5000,
            seed=440,
            verbose=False,
        )

        # 2) Evaluate on TRAIN and TEST in ORIGINAL y SCALE
        _, y_hat_train_scaled = forward_mlp(z_train, params)
        _, y_hat_test_scaled  = forward_mlp(z_test,  params)

        y_hat_train = mu_y + sigma_y * y_hat_train_scaled
        y_hat_test  = mu_y + sigma_y * y_hat_test_scaled

        # Train SSE
        train_res = y_train - y_hat_train
        train_SSE = float(np.sum(train_res ** 2))

        # Test MSE
        test_res = y_test - y_hat_test
        e2_test = test_res ** 2
        test_MSE = float(np.mean(e2_test))

        # s^2 for Test MSE
        n_test = len(y_test)
        s2 = float(np.sum((test_MSE - e2_test) ** 2) / (n_test - 1))

        results.append((J, params, history, train_SSE, test_MSE, s2))
        J_list.append(J)
        test_MSE_list.append(test_MSE)

        # 3) Stopping rule:
        # if q > 1 and E_avg(q) > E_avg(q-1) -> stop
        if q > 1 and test_MSE > prev_E:
            # Then choose previous J(q-1) as best
            best_J, best_params, best_hist, best_train_SSE, best_test_MSE, best_s2 = results[-2]
            break

        prev_E = test_MSE
        q += 1
        J += 1

        if J > J_max:
            # If no increase up to J_max, fall back to global minimum
            best_idx = int(np.argmin([r[4] for r in results]))  # r[4] = test_MSE
            best_J, best_params, best_hist, best_train_SSE, best_test_MSE, best_s2 = results[best_idx]
            break

    print("\n=== Selected MLP for Question 2(a) – Algorithm given in the lecture ===")
    print(f"Chosen number of hidden units J = {best_J}")

    # ---------- Plot 1: J vs Test MSE (hidden unit selection) ----------
    plt.figure(figsize=(5, 4))
    plt.plot(J_list, test_MSE_list, marker="o")
    plt.title("Q2(a) – Test MSE vs number of hidden units J")
    plt.xlabel("Number of hidden units J")
    plt.ylabel("Test MSE (E_avg)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 2: training and test data + MLP regression ----------
    x_min, x_max = x.min(), x.max()
    x_plot = np.linspace(x_min, x_max, 500)
    z_plot = (x_plot - mu_x) / sigma_x

    _, y_plot_scaled = forward_mlp(z_plot, best_params)
    # For plots, convert back to original y-scale
    y_train_scaled_dummy, y_test_scaled_dummy, mu_y2, sigma_y2 = standardize_y_train_test(
        y_train, y_test
    )
    y_plot = mu_y2 + sigma_y2 * y_plot_scaled

    plt.figure(figsize=(10, 4))

    # Left: training data
    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, marker="x", label="Training data", alpha=0.7)
    plt.plot(x_plot, y_plot, label=f"MLP fit (J={best_J})")
    plt.title("Q2(a) – Training data and MLP regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Right: test data
    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, marker="o", label="Test data", alpha=0.7)
    plt.plot(x_plot, y_plot, label=f"MLP fit (J={best_J})")
    plt.title("Q2(a) – Test data and MLP regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_J, best_train_SSE, best_test_MSE, best_s2


    # This is an extra evaluation of number of J I implemented
def extra_global_search_and_plot(
    x, x_train, y_train, x_test, y_test, z_train, z_test, mu_x, sigma_x,
    hw_J, hw_train_SSE, hw_test_MSE, hw_s2,
    J_min=3, J_max=1000, progress_step=100
):

    # Searching through J = J_min..J_max. For each J, train MLP in the same manner
    # At the end, report the J with minimum test MSE
    # Then train that J and plot traning and test fits.
    # So 2(a)-1 : algorithm given in the lecture for J, 2(a)-2 : global best J*

    print(f"\n=== Extra global search: J = {J_min}..{J_max} (quiet mode) ===")

    # Standardize outputs once
    y_train_scaled, y_test_scaled, mu_y, sigma_y = standardize_y_train_test(
        y_train, y_test
    )

    best_J = None
    best_MSE = np.inf
    best_params = None

    total = J_max - J_min + 1

    for idx, J in enumerate(range(J_min, J_max + 1), start=1):
        # Train MLP in scaled y-space
        params, history = train_mlp(
            z_train,
            y_train_scaled,
            J=J,
            alpha0=0.5,
            eta=0.9,
            eps=1e-3,
            max_epochs=5000,
            seed=440,
            verbose=False,
        )

        # Compute Test MSE in ORIGINAL y-scale
        _, y_hat_test_scaled = forward_mlp(z_test, params)
        y_hat_test = mu_y + sigma_y * y_hat_test_scaled

        test_res = y_test - y_hat_test
        test_MSE = float(np.mean(test_res ** 2))

        if test_MSE < best_MSE:
            best_MSE = test_MSE
            best_J = J
            best_params = params

        # Progress output (e.g., 100/1000, 200/1000, ...)
        if (idx % progress_step == 0) or (J == J_max):
            print(f"  -> {idx}/{total} hidden-unit models trained...")

    print("\n=== Extra global search result ===")
    print(f"Search range           : J = {J_min}..{J_max}")
    print(f"Best Test MSE (global) : {best_MSE:.6f}")
    print(f"Best number of hidden units J* (used as 2(a)-2) = {best_J}")

    # ---------- for theBest J*, calculating the metrics ----------
    y_train_scaled2, y_test_scaled2, mu_y2, sigma_y2 = standardize_y_train_test(
        y_train, y_test
    )

    # Train and test guesses
    _, y_hat_train_scaled = forward_mlp(z_train, best_params)
    _, y_hat_test_scaled = forward_mlp(z_test, best_params)

    y_hat_train = mu_y2 + sigma_y2 * y_hat_train_scaled
    y_hat_test_final = mu_y2 + sigma_y2 * y_hat_test_scaled

    train_res = y_train - y_hat_train
    train_SSE = float(np.sum(train_res ** 2))

    test_res = y_test - y_hat_test_final
    e2_test = test_res ** 2
    test_MSE_final = float(np.mean(e2_test))
    n_test = len(y_test)
    s2_final = float(np.sum((test_MSE_final - e2_test) ** 2) / (n_test - 1))

    # === Tables ===
    print("\n=== Table values for Homework (Question 2(a)) ===")
    print("Method        | Training SSE       | Test MSE          | s^2 for Test MSE")
    print("----------------------------------------------------------------------------")
    print(
        f"2(a)-1 MLP (J={hw_J:2d}) | {hw_train_SSE:16.6f} | "
        f"{hw_test_MSE:16.6f} | {hw_s2:16.6f}"
    )
    print(
        f"2(a)-2 MLP (J={best_J:2d}) | {train_SSE:16.6f} | "
        f"{test_MSE_final:16.6f} | {s2_final:16.6f}"
    )

    # ---------- Plot with best_J ----------
    x_min, x_max = x.min(), x.max()
    x_plot = np.linspace(x_min, x_max, 500)
    z_plot = (x_plot - mu_x) / sigma_x

    _, y_plot_scaled = forward_mlp(z_plot, best_params)
    y_plot = mu_y2 + sigma_y2 * y_plot_scaled

    plt.figure(figsize=(10, 4))

    # Left: training
    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, marker="x", label="Training data", alpha=0.7)
    plt.plot(x_plot, y_plot, label=f"MLP fit (J*={best_J})")
    plt.title(f"Extra global – Training data and MLP regression (J*={best_J})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Right: test
    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, marker="o", label="Test data", alpha=0.7)
    plt.plot(x_plot, y_plot, label=f"MLP fit (J*={best_J})")
    plt.title(f"Extra global – Test data and MLP regression (J*={best_J})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def solve_q2b(x, y):
    """
    Solves Question 2(b): MLP training with augmented inputs [z, z^2].
    
    Methodology:
    1. Independent Data Splitting: Splits data 80% Train / 20% Test internally 
       to ensure fair evaluation without relying on external scopes.
    2. Input Augmentation: Creates a two-terminal input vector [z, z^2].
    3. Strategy Comparison: Compares the 'Standard Lecture Heuristic' (stopping 
       search when test error increases) against a 'Global Search Strategy'
       (scanning up to J=100) to demonstrate local vs global optima.
    """
    print("\n" + "="*60)
    print("=== Question 2(b): MLP with inputs [z, z^2] ===")
    print("="*60)

    # ---------------------------------------------------------
    # 1. INTERNAL DATA SPLITTING (80% Train, 20% Test)
    # ---------------------------------------------------------
    # Using a fixed seed ensures the split is reproducible for the report.
    rng = np.random.default_rng(seed=999)
    N = len(x)
    indices = np.arange(N)
    rng.shuffle(indices)
    
    n_train = int(np.floor(0.8 * N))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    
    print(f"Data partitioning completed internally.")
    print(f"Training Samples: {len(x_train)} | Test Samples: {len(x_test)}")

    # ---------------------------------------------------------
    # 2. Feature Engineering & Standardization
    # ---------------------------------------------------------
    # Create the quadratic feature x^2
    x2_train = x_train ** 2
    x2_test = x_test ** 2

    # Standardize linear inputs (z) using TRAINING statistics only
    mu_x = np.mean(x_train)
    sigma_x = np.std(x_train)
    z_train = (x_train - mu_x) / sigma_x
    z_test = (x_test - mu_x) / sigma_x

    # Standardize quadratic inputs (z^2) using TRAINING statistics only
    # Note: It is crucial to standardize x^2 independently because its scale
    # differs significantly from x, which can hinder gradient descent.
    mu_sq = np.mean(x2_train)
    sigma_sq = np.std(x2_train)
    z_sq_train = (x2_train - mu_sq) / sigma_sq
    z_sq_test = (x2_test - mu_sq) / sigma_sq

    # Construct the Dual Input Matrix: [z, z^2]
    # The network now sees 2 input terminals.
    X_train_2b = np.column_stack([z_train, z_sq_train])
    X_test_2b  = np.column_stack([z_test,  z_sq_test])

    # Standardize Targets (y)
    mu_y = np.mean(y_train)
    sigma_y = np.std(y_train)
    y_train_scaled = (y_train - mu_y) / sigma_y

    # ---------------------------------------------------------
    # 3. Model Selection Loop (Search Range: J=3 to 100)
    # ---------------------------------------------------------
    J_min = 3
    J_max = 100
    J_list = []
    test_MSE_list = []
    
    # Trackers for the Global Best Model
    global_best_J = None
    global_best_MSE = np.inf
    global_best_params = None
    global_best_stats = None # Stores (SSE, MSE, s^2)

    # Trackers for the Standard/Lecture Algorithm (Greedy)
    prev_E = np.inf
    prev_stats = None
    greedy_stop_J = None
    greedy_stop_stats = None
    greedy_found = False

    print(f"Scanning hidden unit range J = {J_min} to {J_max}...")
    print("-" * 60)

    for J in range(J_min, J_max + 1):
        # Progress Log
        print(f" -> Training model with J={J}...", end='\r')

        # Train with input_dim=2
        params, history = train_mlp(
            X_train_2b, y_train_scaled, J=J, input_dim=2, 
            alpha0=0.5, eta=0.9, eps=1e-3, max_epochs=5000, verbose=False
        )

        # Evaluate on Test Set
        _, y_hat_test_scaled = forward_mlp(X_test_2b, params)
        # Denormalize predictions to original scale
        y_hat_test = mu_y + sigma_y * y_hat_test_scaled
        
        test_res = y_test - y_hat_test
        test_MSE = float(np.mean(test_res ** 2))
        
        # Calculate Statistics for Table (Train SSE, s^2)
        _, y_hat_train_scaled = forward_mlp(X_train_2b, params)
        y_hat_train = mu_y + sigma_y * y_hat_train_scaled
        train_SSE = float(np.sum((y_train - y_hat_train)**2))
        n_test = len(y_test)
        s2 = float(np.sum((test_MSE - test_res**2)**2) / (n_test - 1))
        
        current_stats = (train_SSE, test_MSE, s2)

        J_list.append(J)
        test_MSE_list.append(test_MSE)

        # --- LOGIC 1: GLOBAL SEARCH ---
        if test_MSE < global_best_MSE:
            global_best_MSE = test_MSE
            global_best_J = J
            global_best_params = params
            global_best_stats = current_stats

        # --- LOGIC 2: LECTURE HEURISTIC (GREEDY) ---
        # Stop if error increases compared to the previous step.
        # The "optimal" J for this method is the previous one (J-1).
        if not greedy_found and J > J_min:
            if test_MSE > prev_E:
                greedy_stop_J = J - 1
                greedy_stop_stats = prev_stats
                greedy_found = True
                print(f"\n   >> [Greedy Stop] Error increased at J={J}. Standard Algo selects J={greedy_stop_J}.")
        
        prev_E = test_MSE
        prev_stats = current_stats

    # If greedy algorithm never stopped (error kept decreasing), match global best
    if not greedy_found:
        greedy_stop_J = global_best_J
        greedy_stop_stats = global_best_stats
        print("\n   >> [Greedy] No early stop triggered. Matches Global Best.")
    else:
        print("\n -> Global search finished.")

    # ---------------------------------------------------------
    # 4. Reporting & Visualization
    # ---------------------------------------------------------
    print("-" * 60)
    print(f"Global Search Result: Optimal J = {global_best_J} (MSE={global_best_MSE:.4f})")
    
    # Plot 1: Model Selection Comparison
    plt.figure(figsize=(8, 5))
    plt.plot(J_list, test_MSE_list, marker="o", label="Test MSE Trajectory")
    
    if greedy_found:
        plt.plot(greedy_stop_J, greedy_stop_stats[1], 'ro', markersize=10, label=f'Greedy Stop (J={greedy_stop_J})')
    
    plt.plot(global_best_J, global_best_MSE, 'g*', markersize=15, label=f'Global Best (J={global_best_J})')
    plt.title("Q2(b) Model Selection: Greedy Heuristic vs. Global Search")
    plt.xlabel("Number of Hidden Units (J)")
    plt.ylabel("Test MSE")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: Regression Fit Visualization
    x_min, x_max = x.min(), x.max()
    x_plot = np.linspace(x_min, x_max, 500)
    
    # Prepare plot inputs [z, z^2]
    z_plot = (x_plot - mu_x) / sigma_x
    z_sq_plot = (x_plot**2 - mu_sq) / sigma_sq
    X_plot_2b = np.column_stack([z_plot, z_sq_plot])

    _, y_plot_scaled = forward_mlp(X_plot_2b, global_best_params)
    y_plot = mu_y + sigma_y * y_plot_scaled

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, marker="x", label="Training Data", alpha=0.7)
    plt.plot(x_plot, y_plot, color='purple', linewidth=2, label=f"Best Fit (J={global_best_J})")
    plt.title(f"Q2(b) Training Fit"); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, marker="o", label="Test Data", alpha=0.7)
    plt.plot(x_plot, y_plot, color='purple', linewidth=2, label=f"Best Fit (J={global_best_J})")
    plt.title(f"Q2(b) Test Fit"); plt.grid(True); plt.legend()
    plt.show()

    # Comparison Table Output
    gr_train_SSE, gr_test_MSE, gr_s2 = greedy_stop_stats
    gl_train_SSE, gl_test_MSE, gl_s2 = global_best_stats

    print("\n=== Table values for Homework (Question 2(b)) ===")
    print("Method            | Training SSE       | Test MSE          | s^2 for Test MSE")
    print("--------------------------------------------------------------------------------")
    print(f"2(b)-1 (Greedy J={greedy_stop_J}) | {gr_train_SSE:16.6f} | {gr_test_MSE:16.6f} | {gr_s2:16.6f}")
    print(f"2(b)-2 (Global J={global_best_J}) | {gl_train_SSE:16.6f} | {gl_test_MSE:16.6f} | {gl_s2:16.6f}")

    return global_best_J


# ============================================================
# main – run Q1, Q2(a)-1 Q2(a)-2 and  Q2(b)
# ============================================================

def main():
    # Load data
    x, y = load_regression_data("regression_data.dat")
    x_train, y_train, x_test, y_test = train_test_split_random(
        x, y, train_ratio=0.8, seed=440
    )

    print("=== Data info (shared for Q1 and Q2.a) ===")
    print(f"Total samples: {len(x)}")
    print(f"Train samples: {len(x_train)}")
    print(f"Test  samples: {len(x_test)}")

    # Standardize x using training data
    z_train, z_test, mu_x, sigma_x = standardize_train_test(x_train, x_test)

    # Q1: linear and quadratic regression with scaling (x standardize)
    solve_q1(x, y, x_train, y_train, x_test, y_test, z_train, mu_x, sigma_x)

    # Q2(a): one-hidden-layer MLP regression with algorithm
    hw_J, hw_train_SSE, hw_test_MSE, hw_s2 = solve_q2a(
        x, x_train, y_train, x_test, y_test, z_train, z_test, mu_x, sigma_x
    )

    # 2(a)-2: EXTRA global J search (3..1000), progress messages, then model+plot with best J
    # and final 2.(a) table with TWO rows (2(a)-1 J and 2(a)-2 J*)
    extra_global_search_and_plot(
        x, x_train, y_train, x_test, y_test, z_train, z_test, mu_x, sigma_x,
        hw_J, hw_train_SSE, hw_test_MSE, hw_s2,
        J_min=3, J_max=1000, progress_step=100
    )
    # Question 2(b) Execution
    solve_q2b(x, y)

if __name__ == "__main__":
    main()
