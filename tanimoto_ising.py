import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import t
import os
import time
from scipy.special import comb

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

def tanimoto_kernel(x, y):
    dot = np.dot(x, y)
    denom = np.sum(x) + np.sum(y) - dot
    return dot / denom if denom != 0 else 0.0

def kernel_embedding(y):
    d = len(y)
    t = int(np.sum(y))  # ensure integer
    embedding = 0.0

    for s in range(d + 1):
        a_min = int(max(0, s + t - d))
        a_max = int(min(s, t))
        for a in range(a_min, a_max + 1):
            denom = s + t - a
            if denom == 0:
                continue
            term = (a / denom) * comb(t, a) * comb(d - t, s - a)
            embedding += term

    return embedding / (2 ** d)


def double_integral(d):
    double_integral = 0.0

    for t in range(d + 1):
        for s in range(d + 1):
            a_min = max(0, s + t - d)
            a_max = min(s, t)
            for a in range(a_min, a_max + 1):
                denominator = s + t - a
                if denominator == 0:
                    continue  # avoid division by zero
                term = (a / denominator) * comb(t, a) * comb(d - t, s - a)
                double_integral += term

    return double_integral / (2 ** (2 * d))


def gram_matrix(X):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            kij = tanimoto_kernel(X[i], X[j])
            K[i, j] = kij
            K[j, i] = kij 
    return K

def bayesian_cubature(X, f_vals, d):

    n = len(X)
    K = gram_matrix(X)
    K += 1e-8 * np.eye(n)  
    z = np.array([kernel_embedding(x) for x in X]).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)
    K_inv = np.linalg.inv(K)
    mean = (z.T @ K_inv @ f_vals).item()
    PiPi_k = double_integral(d)
    var = PiPi_k - (z.T @ K_inv @ z).item()

    return mean, var

def generate_unique_X(n, d, seed):
    np.random.seed(seed)
    all_states = np.array(list(product([0., 1.], repeat=d)))
    indices = np.random.choice(len(all_states), size=n, replace=False)
    return all_states[indices]

def run_experiment(f, n_vals, d, L, seed):
    np.random.seed(seed)
    J = J_matrix(L) 
    all_states = np.array(list(product([0., 1.], repeat=d)))
    true_expectation = np.mean(f(all_states, J, d, beta))

    bmc_means, bmc_lows, bmc_highs = [], [], []
    mc_means, mc_stds = [], []

    for n in n_vals:
        unique_X = generate_unique_X(n, d, seed)
        X = np.random.choice([0., 1.], size=(n, d))
        f_vals = f(X, J, d, beta)
        f_unique_vals = f(unique_X, J, d, beta)

        mu_bmc, var_bmc = bayesian_cubature(unique_X, f_unique_vals, d)
        df = len(unique_X)
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


def run_multiple_seeds(f, n_vals, d, L, seeds):
    bmc_means_all = []
    bmc_lows_all = []
    bmc_highs_all = []
    mc_means_all = []
    mc_stds_all = []

    for seed in range(seeds):
        result = run_experiment(f, n_vals, d, L, seed)
        
        bmc_means_all.append(result["bmc_means"])
        bmc_lows_all.append(result["bmc_lows"])
        bmc_highs_all.append(result["bmc_highs"])
        mc_means_all.append(result["mc_means"])
        mc_stds_all.append(result["mc_stds"])

    bmc_means_avg = np.mean(bmc_means_all, axis=0)
    bmc_lows_avg = np.mean(bmc_lows_all, axis=0)
    bmc_highs_avg = np.mean(bmc_highs_all, axis=0)
    mc_means_avg = np.mean(mc_means_all, axis=0)
    mc_stds_avg = np.mean(mc_stds_all, axis=0)
    
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
    plt.savefig(os.path.join(save_path, "abs_error_ising_0_1_tanimoto.png"), dpi=300)
    plt.close()

def main():
    global beta  # so it's accessible in f()
    k_b = 1
    T_c = 2.269
    beta = 1 / (k_b * T_c)
    L = 4
    d = L * L

    n_vals = np.array([10, 50, 100, 200, 300])

    seeds = 10
    results = run_multiple_seeds(f, n_vals, d, L, seeds)
    plot_results(n_vals, results, save_path="results")

if __name__ == "__main__":
    main()