##2D
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import t

def J_matrix(L):
    n = L * L  # total number of spins same as d
    J = np.zeros((n, n), dtype=int)
    
    def index(r, c):
        return r * L + c  

    for r in range(L):
        for c in range(L):
            i = index(r, c)

            if c + 1 < L:
                j = index(r, c + 1)
                J[i, j] = J[j, i] = 1

            if r + 1 < L:
                j = index(r + 1, c)
                J[i, j] = J[j, i] = 1
                
    return J


def f(X, J, d, beta):
    return np.array([np.exp(beta * (x @ J @ x)) for x in X])

def tanimoto_kernel(x, y):
    dot = np.dot(x, y)
    denom = np.sum(x) + np.sum(y) - dot
    return dot / denom if denom != 0 else 0.0

def gram_matrix(X):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = tanimoto_kernel(X[i], X[j])
            K[i, j] = K[j, i] = val
    return K

def kernel_mean_embedding(X):
    y = X[0]  
    return np.mean([tanimoto_kernel(x, y) for x in X])

def double_integral(X):
    n = len(X)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += tanimoto_kernel(X[i], X[j])
    return total / (n * n)

# def gaussian_kernel(x, y, sigma):
#     return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))

# def gram_matrix(X, sigma):
#     n = len(X)
#     K = np.zeros((n, n))

#     for i in range(n):
#         for j in range(i, n):  
#             K[i, j] = K[j, i] = gaussian_kernel(X[i], X[j], sigma)

#     return K

# def kernel_mean_embedding(X, sigma):
#     y = X[0]  
#     return np.mean([gaussian_kernel(x, y, sigma) for x in X])

# def double_integral(sigma, d):
#     X = np.array(list(product([0, 1], repeat=d)))  # All x ∈ {0,1}^d
#     total = 0.0
#     n = len(X)

#     for i in range(n):
#         for j in range(n):
#             total += gaussian_kernel(X[i], X[j], sigma)

#     return total / (n * n)

def bayesian_cubature(X, f_vals, d):
    n = len(X)
    K = gram_matrix(X)
    z_scalar = kernel_mean_embedding(X)
    z = np.full((n, 1), z_scalar)  
    f_vals = f_vals.reshape(n, 1)
    K += 1e-8 * np.eye(n)  # regularization
    K_inv = np.linalg.inv(K)
    mean = (z.T @ K_inv @ f_vals).item()
    PiPi_k = double_integral(X)
    var = PiPi_k - (z.T @ K_inv @ z).item()
    return mean, var

def generate_unique_X(n, d, seed=0):
    np.random.seed(seed)
    all_states = np.array(list(product([0, 1], repeat=d)))
    indices = np.random.choice(len(all_states), size=n, replace=False)
    return all_states[indices]

def run_experiment(f, n_vals, lambda_, d, L, seed=0):
    np.random.seed(seed)
    J = J_matrix(L) 
    all_states = np.array(list(product([0, 1], repeat=d)))
    true_expectation = np.mean(f(all_states, J, d, beta))

    bmc_means, bmc_lows, bmc_highs = [], [], []
    mc_means, mc_stds = [], []

    for n in n_vals:
        # X = generate_unique_X(n, d)
        X = np.random.choice([0, 1], size=(n, d))
        X = np.unique(X, axis=0)
        f_vals = f(X, J, d, beta)

        mu_bmc, var_bmc = bayesian_cubature(X, f_vals, d)
        df = len(X)
        scale = np.sqrt(max(var_bmc, 0))
        ci_low, ci_high = t.interval(0.95, df=df, loc=mu_bmc, scale=scale)
        bmc_means.append(mu_bmc)
        bmc_lows.append(ci_low)
        bmc_highs.append(ci_high)

        mu_mc = np.mean(f_vals)
        std_mc = np.std(f_vals) / np.sqrt(len(X))
        mc_means.append(mu_mc)
        mc_stds.append(std_mc)


    return {
        "true_val": true_expectation,
        "bmc_means": np.array(bmc_means),
        "bmc_lows": np.array(bmc_lows),
        "bmc_highs": np.array(bmc_highs),
        "mc_means": np.array(mc_means),
        "mc_stds": np.array(mc_stds)
    }



k_b = 1
T_c = 2.269
beta = 1 / (k_b * T_c)
lambda_ = 3
d = 9
L = 3

# Run and plot
n_vals = np.arange(5, 2000, 50)
results = run_experiment(f, n_vals, lambda_, d, L)

plt.figure(figsize=(10, 6))

# Clip values to avoid log-scale issues
bmc_means = np.clip(results["bmc_means"], 1e-10, None)
bmc_lows = np.clip(results["bmc_lows"], 1e-10, None)
bmc_highs = np.clip(results["bmc_highs"], 1e-10, None)
mc_means = np.clip(results["mc_means"], 1e-10, None)
true_val = max(results["true_val"], 1e-10)

plt.plot(n_vals, mc_means, 'ko', label="MC")
plt.plot(n_vals, bmc_means, 'ro', label="BQ")
plt.fill_between(n_vals, bmc_lows, bmc_highs, color='red', alpha=0.2)
plt.axhline(true_val, color='green', linestyle='--', label="True")

# plt.xscale('log')  # optional: also log on x-axis
# plt.yscale('log')

plt.title("Bayesian Quadrature vs Monte Carlo - 3x3 lattice, {0,1} domain")
plt.xlabel("n (number of points)")
plt.ylabel("Estimated Expectation (log scale)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()