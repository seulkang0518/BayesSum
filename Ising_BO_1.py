import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import t
import os
import time
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from multiprocessing import Pool, cpu_count

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

def f_single(x, J, beta):
    energy = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            energy += J[i, j] * (x[i] == x[j])
    return np.exp(-beta * energy)


def f_parallel(X, J, beta, n_processes=None):
    if n_processes is None:
        n_processes = cpu_count()
    args = [(x, J, beta) for x in X]
    with Pool(processes=n_processes) as pool:
        result = pool.starmap(f_single, args)
    return np.array(result)

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
    K = gram_matrix(X, lambda_, d) + 1e-8 * np.eye(n)
    z = np.full((n, 1), kernel_embedding(lambda_, d))
    f_vals = f_vals.reshape(n, 1)
    L, lower = cho_factor(K)
    K_inv_z = cho_solve((L, lower), z)
    K_inv_f = cho_solve((L, lower), f_vals)
    mean = (z.T @ K_inv_f).item()
    PiPi_k = double_integral(lambda_, d)
    var = PiPi_k - (z.T @ K_inv_z).item()
    return mean, var

def generate_unique_X(n, d, seed):
    np.random.seed(seed)
    all_states = np.array(list(product([0., 1.], repeat=d)))
    indices = np.random.choice(len(all_states), size=n, replace=False)
    return all_states[indices]


########### Bayesian Optimisation ###########

def kernel_vec(x, X, lambda_, d):
    x_rescaled = 2 * x - 1
    X_rescaled = 2 * X - 1
    inner_products = X_rescaled @ x_rescaled
    return np.exp(-lambda_ * 0.5 * (d - inner_products)).reshape(-1, 1)

def compute_integral_variance(X, lambda_, d):
    K = gram_matrix(X, lambda_, d) + 1e-8 * np.eye(len(X))
    z = np.full((len(X), 1), kernel_embedding(lambda_, d))
    L, lower = cho_factor(K)
    K_inv = cho_solve((L, lower), z)

    if K_inv.ndim == 1:
        K_inv = K_inv[:, None]  # force to shape (n, 1)
    return double_integral(lambda_, d) - (z.T @ K_inv).item()

def compute_augmented_variance(X_obs, y_obs, x, f_sample, lambda_, d):
    X_aug = np.vstack([X_obs, x])
    return compute_integral_variance(X_aug, lambda_, d)


def compute_kg(X_obs, y_obs, candidates, lambda_, d, num_samples):
    K = gram_matrix(X_obs, lambda_, d) + 1e-8 * np.eye(len(X_obs))
    z = np.full((len(X_obs), 1), kernel_embedding(lambda_, d))
    L, lower = cho_factor(K)

    K_inv_y = cho_solve((L, lower), y_obs)
    K_inv_z = cho_solve((L, lower), z)
    if K_inv_z.ndim == 1:
        K_inv_z = K_inv_z[:, None]

    PiPi_k = double_integral(lambda_, d)
    current_Z_var = PiPi_k - (z.T @ K_inv_z).item()

    def mu(x): return kernel_vec(x, X_obs, lambda_, d).T @ K_inv_y
    def sigma2(x):
        z_k = kernel_vec(x, X_obs, lambda_, d)
        K_inv_zk = cho_solve((L, lower), z_k)
        if K_inv_zk.ndim == 1:
            K_inv_zk = K_inv_zk[:, None]
        return 1. - z_k.T @ K_inv_zk

    if len(candidates) > 200:
        candidates = candidates[np.random.choice(len(candidates), 200, replace=False)]

    KG_values = []
    for x in candidates:
        mu_x = mu(x).item()
        sigma_x = np.sqrt(max(sigma2(x).item(), 1e-10))
        f_samples = np.random.normal(mu_x, sigma_x, size=num_samples)
        updated_vars = Parallel(n_jobs=-1)(
            delayed(compute_augmented_variance)(X_obs, y_obs, x, y_j, lambda_, d)
            for y_j in f_samples
        )
        expected_reduction = current_Z_var - np.mean(updated_vars)
        KG_values.append(expected_reduction)

    return candidates[np.argmax(KG_values)]

###############################################

def run_experiment(f, n_vals, lambda_, d, L, seed, use_bo, num_mc):
    np.random.seed(seed)
    J = J_matrix(L)
    all_states = list(product([0., 1.], repeat=d))
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(f_single, [(x, J, beta) for x in all_states])
    true_expectation = np.mean(results)

    bmc_means, bmc_lows, bmc_highs = [], [], []
    mc_means, mc_stds = [], []

    for n in n_vals:
        if use_bo:
            X, y = [], []
            candidates = np.array(list(product([0., 1.], repeat=d)))
            np.random.shuffle(candidates)
            x0 = candidates[0]
            X.append(x0)
            y.append(f(np.array([x0]), J, beta)[0])

            for _ in range(1, n):
                x_next = compute_kg(np.array(X), np.array(y).reshape(-1, 1),candidates, lambda_, d, num_mc)
                y_next = f_parallel(np.array([x_next]), J, beta)[0]
                X.append(x_next)
                y.append(y_next)

            X = np.array(X)
            f_vals = np.array(y)

        else:
            X = generate_unique_X(n, d, seed)
            f_vals = f_parallel(X, J, beta)

        mu_bmc, var_bmc = bayesian_cubature(X, f_vals, lambda_, d)
        df = len(X)
        scale = np.sqrt(max(var_bmc, 0))
        ci_low, ci_high = t.interval(0.95, df=df, loc=mu_bmc, scale=scale)
        bmc_means.append(mu_bmc)
        bmc_lows.append(ci_low)
        bmc_highs.append(ci_high)
        print(f"n={n}, BMC err={np.abs(mu_bmc - true_expectation):.4f}")

        mu_mc = np.mean(f_vals)
        std_mc = np.std(f_vals) / np.sqrt(len(X))
        print(f"n={n}, MC err={np.abs(mu_mc - true_expectation):.4f}")
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

    n_vals = np.array([1, 2, 10, 50, 100, 200, 300])
    # seeds = 0
    seed = 0 
    use_bo = True
    num_mc = 10

    result = run_experiment(f_single, n_vals, lambda_val, d, L, seed, use_bo, num_mc)
    print("Results with Bayesian Optimization (KG):")
    print("BMC means:", result["bmc_means"])
    print("True expectation:", result["true_val"])
    # seed = int(time.time() % 1000)
    # results = run_multiple_seeds(f, n_vals, lambda_val, d, L, seeds)
    # plot_results(n_vals, results, save_path="results")

if __name__ == "__main__":
    main()