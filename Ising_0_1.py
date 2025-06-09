import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import t
import os
import time

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
                
    return 0.1 * J 

def f(X, J, d, beta):
    return np.array([np.exp(-beta * (x @ J @ x)) for x in X])

def kernel_embedding(lambda_, d):
    embedding = np.exp(-lambda_ * d / 2) * (np.cosh(lambda_ / 2) ** d)
    return embedding

def double_integral(lambda_, d):
    double_sum = np.exp(-lambda_ * d / 2) * (np.cosh(lambda_ / 2) ** d)
    return double_sum

def gram_matrix(X, lambda_, d):
    X_rescaled = 2 * X - 1.  # Rescale {0, 1} to {-1, 1}
    inner_products = X_rescaled @ X_rescaled.T  # shape (n_samples, n_samples)
    K = np.exp(-lambda_ * 0.5 * (d - inner_products))
    return K

def bayesian_cubature(X, f_vals, lambda_, d):
    n = len(X)
    K = gram_matrix(X, lambda_, d)
    z_scalar = kernel_embedding(lambda_, d)
    z = np.full((n, 1), z_scalar)  
    f_vals = f_vals.reshape(n, 1)
    K += 1e-8 * np.eye(n)  # regularization
    K_inv = np.linalg.inv(K)
    mean = (z.T @ K_inv @ f_vals).item()
    PiPi_k = double_integral(lambda_, d)
    var = PiPi_k - (z.T @ K_inv @ z).item()
    return mean, var

def generate_unique_X(n, d, seed):
    np.random.seed(seed)
    all_states = np.array(list(product([0., 1.], repeat=d)))
    indices = np.random.choice(len(all_states), size=n, replace=False)
    return all_states[indices]

def run_experiment(f, n_vals, lambda_, d, L, seed):
    np.random.seed(seed)
    J = J_matrix(L) 
    all_states = np.array(list(product([0., 1.], repeat=d)))
    true_expectation = np.mean(f(all_states, J, d, beta))

    bmc_means, bmc_lows, bmc_highs = [], [], []
    mc_means, mc_stds = [], []

    for n in n_vals:
        unique_X = generate_unique_X(n, d, seed)
        X = np.random.choice([0., 1.], size=(n, d))
        # X = np.unique(X, axis=0)
        f_vals = f(X, J, d, beta)
        f_unqiue_vals = f(unique_X, J, d, beta)

        mu_bmc, var_bmc = bayesian_cubature(unique_X, f_unqiue_vals, lambda_, d)
        df = len(unique_X)
        scale = np.sqrt(max(var_bmc, 0))
        ci_low, ci_high = t.interval(0.95, df=df, loc=mu_bmc, scale=scale)
        bmc_means.append(mu_bmc)
        bmc_lows.append(ci_low)
        bmc_highs.append(ci_high)
        print(f"n={n}, BMC err={np.abs(mu_bmc-true_expectation):.4f}")
        mu_mc = np.mean(f_vals)
        std_mc = np.std(f_vals) / np.sqrt(len(X))
        print(f"n={n}, MC err={np.abs(mu_mc-true_expectation):.4f}")
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

def run_multiple_seeds(f, n_vals, lambda_, d, L, seeds):
    bmc_means_all = []
    bmc_lows_all = []
    bmc_highs_all = []
    mc_means_all = []
    mc_stds_all = []

    for seed in range(seeds):
        result = run_experiment(f, n_vals, lambda_, d, L, seed)
        
        bmc_means_all.append(result["bmc_means"])
        bmc_lows_all.append(result["bmc_lows"])
        bmc_highs_all.append(result["bmc_highs"])
        mc_means_all.append(result["mc_means"])
        mc_stds_all.append(result["mc_stds"])

    # Average over seeds
    bmc_means_avg = np.mean(bmc_means_all, axis=0)
    bmc_lows_avg = np.mean(bmc_lows_all, axis=0)
    bmc_highs_avg = np.mean(bmc_highs_all, axis=0)
    mc_means_avg = np.mean(mc_means_all, axis=0)
    mc_stds_avg = np.mean(mc_stds_all, axis=0)
    
    # All seeds use the same true expectation
    true_val = result["true_val"]

    return {
        "true_val": true_val,
        "bmc_means": bmc_means_avg,
        "bmc_lows": bmc_lows_avg,
        "bmc_highs": bmc_highs_avg,
        "mc_means": mc_means_avg,
        "mc_stds": mc_stds_avg
    }


def plot_results(n_vals, results, save_path="results"):

    os.makedirs(save_path, exist_ok=True)

    true_val = results["true_val"]
    bmc_errors = np.abs(results["bmc_means"] - true_val)
    mc_errors = np.abs(results["mc_means"] - true_val)

    # Clip to avoid log(0)
    bmc_errors = np.clip(bmc_errors, 1e-10, None)
    mc_errors = np.clip(mc_errors, 1e-10, None)

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, mc_errors, 'ko-', label="MC Absolute Error")
    plt.plot(n_vals, bmc_errors, 'ro-', label="BQ Absolute Error")

    plt.title("Absolute Error: Bayesian Quadrature vs Monte Carlo\n(Pointwise Comparison kernel, modified Ising model input)")
    plt.xlabel("n (number of points)")
    plt.ylabel("Absolute Error (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abs_error_ising_0_1.png"), dpi=300)
    plt.close()

def main():
    global beta  # so it's accessible in f()
    k_b = 1
    T_c = 2.269
    beta = 1 / (k_b * T_c)
    lambda_val = 0.1
    L = 4
    d = L * L
    # n_vals = np.arange(5, 100, 5)
    n_vals = np.array([10, 50, 100, 200])
    seeds = 10

    # seed = int(time.time() % 1000)
    results = run_multiple_seeds(f, n_vals, lambda_val, d, L, seeds)
    plot_results(n_vals, results, save_path="results")

if __name__ == "__main__":
    main()