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

def init_mlp_params(J, rng):
    # 1 input (z), 1 linear output neuron, J hidden sigmoidal neurons

    # Weights from input to hidden (J x 1) and hidden biases (J,)
    W_in = rng.normal(loc=0.0, scale=0.1, size=(J, 1))
    b_h = np.zeros(J)

    # Weights from hidden to output (J,) and output bias (scalar)
    W_out = rng.normal(loc=0.0, scale=0.1, size=J)
    b_out = 0.0

    return W_in, b_h, W_out, b_out


def forward_mlp(z, params):
    # z: standardized input values

    # return h: hidden layer activations
    # return y_hat: network outputs (in the scale of y we used for training)

    W_in, b_h, W_out, b_out = params

    z = np.asarray(z, float).reshape(-1, 1)  # (N, 1)

    # Hidden layer: h = sigmoid(h_lin)
    h_lin = z @ W_in.T          
    h_lin += b_h                   
    h_lin = np.clip(h_lin, -50.0, 50.0) 

    h = 1.0 / (1.0 + np.exp(-h_lin))

    # Output neuron: linear activation g(h) = h
    y_hat = h @ W_out + b_out    

    return h, y_hat


def train_mlp(
    z_train,
    y_train_scaled,
    J,
    alpha0=0.5, 
    eta=0.9,     
    eps=1e-3,
    max_epochs=5000,
    seed=440,
    verbose=False,
):
    
    # update rule: w^{(t+1)} = w^{(t)} - α^{(t)} * ∇L
    # α^{(t+1)} = η * α^{(t)}
    rng = np.random.default_rng(seed)
    W_in, b_h, W_out, b_out = init_mlp_params(J, rng)

    y_train_scaled = np.asarray(y_train_scaled, float)
    history = []

    alpha = alpha0  # current learning rate α(t)
    N = len(z_train)

    for epoch in range(max_epochs):
        # Forward pass
        h, y_hat = forward_mlp(z_train, (W_in, b_h, W_out, b_out))

        # MSE on training set (IN SCALED Y SPACE)
        err = y_train_scaled - y_hat
        L_val = np.mean(err ** 2)
        history.append(L_val)

        if verbose and (epoch % 500 == 0):
            print(f"[epoch {epoch:4d}] MSE_scaled = {L_val:.6e}, alpha={alpha:.4f}")

        # Stopping rule: small relative change in MSE (scaled space)
        if epoch > 0:
            denom = max(abs(history[-2]), 1e-12)
            rel_change = abs(history[-1] - history[-2]) / denom
            if rel_change < eps:
                if verbose:
                    print(
                        f"Stopping at epoch {epoch}, "
                        f"relative change in MSE_scaled = {rel_change:.3e} < eps."
                    )
                break

        # ----------------------------------------------------
        # Backpropagation (batch gradients)
        # Loss: L = (1/N) * sum_i (y_tilde_i - y_hat_i)^2
        # dL/dy_hat = -(2/N) * (y_tilde - y_hat)
        # ----------------------------------------------------
        dL_dy = -2.0 * err / N       

        # Output layer
        dW_out = h.T @ dL_dy          
        db_out = np.sum(dL_dy)        

        # Hidden layer
        dL_dh = np.outer(dL_dy, W_out)  
        dh_dhlin = h * (1.0 - h)        # derivative of sigmoid
        dL_dhlin = dL_dh * dh_dhlin     

        zcol = z_train.reshape(-1, 1)   
        dW_in = dL_dhlin.T @ zcol       
        db_h = np.sum(dL_dhlin, axis=0) 

        W_out -= alpha * dW_out
        b_out -= alpha * db_out
        W_in  -= alpha * dW_in
        b_h   -= alpha * db_h

        # Update learning rate
        alpha *= eta

    params = (W_in, b_h, W_out, b_out)
    return params, history


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
    print(f"Chosen number of hidden units from homework rule J = {best_J}")

    # ---------- Plot 1: J vs Test MSE (hidden unit selection) ----------
    plt.figure(figsize=(5, 4))
    plt.plot(J_list, test_MSE_list, marker="o")
    plt.title("Q2(a) – Test MSE vs number of hidden units J (homework algo)")
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
    plt.title("Q2(a) – Training data and MLP regression (homework J)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Right: test data
    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, marker="o", label="Test data", alpha=0.7)
    plt.plot(x_plot, y_plot, label=f"MLP fit (J={best_J})")
    plt.title("Q2(a) – Test data and MLP regression (homework J)")
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

    # Train ve test tahminleri
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


# ============================================================
# main – run Q1, Q2(a)-1 and Q2(a)-2
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

    # Q2(a): one-hidden-layer MLP regression with homework algorithm
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


if __name__ == "__main__":
    main()
