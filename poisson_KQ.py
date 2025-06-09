import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, t
from scipy.special import gammainc, factorial
import os

def f(x):
    return x * np.exp(-np.sqrt(x))
    # return np.exp(-5 * x)

def brownian_kernel(X1, X2):
    X1 = np.atleast_1d(X1).reshape(-1, 1)
    X2 = np.atleast_1d(X2).reshape(-1, 1)
    return np.minimum(X1, X2.T)

def kernel_embedding_poisson(lambda_, xi):
    x_vals = np.arange(0, xi + 1)
    pmf = (lambda_**x_vals / factorial(x_vals)) * np.exp(-lambda_)
    term1 = np.sum(x_vals * pmf)
    tail_prob = gammainc(xi + 1, lambda_)
    term2 = xi * tail_prob
    return term1 + term2

def double_integral_poisson(lambda_, xmax=100):
    k_vals = np.arange(0, xmax + 1)
    tail_probs = 1 - gammainc(k_vals + 1, lambda_)  # P(X > k)
    return np.sum(tail_probs ** 2)

def bayesian_cubature_poisson(X, f_vals, lambda_, xmax=100):
    n = len(X)
    K = brownian_kernel(X, X).astype(float)
    z = np.array([kernel_embedding_poisson(lambda_, xi) for xi in X]).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)
    K += 1e-8 * np.eye(n)
    K_inv = np.linalg.inv(K)
    mean = (z.T @ K_inv @ f_vals).item()
    PiPi_k = double_integral_poisson(lambda_, xmax)
    var = PiPi_k - (z.T @ K_inv @ z).item()
    return mean, var

def run_experiment(f, n_vals, lambda_, xmax, seed):
    np.random.seed(seed)
    x_all = np.arange(0, xmax+1)
    true_expectation = np.sum(f(x_all) * poisson.pmf(x_all, mu=lambda_))

    bmc_means, bmc_lows, bmc_highs = [], [], []
    mc_means, mc_stds = [], []

    for n in n_vals:
        X = np.sort(np.random.poisson(lam=lambda_, size=n))
        unique_X = np.unique(X)
        f_vals = f(X)
        f_unique_vals = f(unique_X)

        mu_bmc, var_bmc = bayesian_cubature_poisson(unique_X, f_unique_vals, lambda_, xmax)
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

def run_multiple_seeds(f, n_vals, lambda_, xmax, seeds):
    bmc_means_all = []
    bmc_lows_all = []
    bmc_highs_all = []
    mc_means_all = []
    mc_stds_all = []

    for seed in range(seeds):
        result = run_experiment(f, n_vals, lambda_, xmax, seed)
        
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

    bmc_errors = np.clip(bmc_errors, 1e-10, None)
    mc_errors = np.clip(mc_errors, 1e-10, None)

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, mc_errors, 'ko-', label="MC Absolute Error")
    plt.plot(n_vals, bmc_errors, 'ro-', label="BQ Absolute Error")

    plt.title("Absolute Error: Bayesian Quadrature vs Monte Carlo\n(Brownian kernel, Poisson input)")
    plt.xlabel("n (number of points)")
    plt.ylabel("Absolute Error (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abs_error_poisson_bq_RMSE_optimal.png"), dpi=300)
    plt.close()


def main():
    lambda_val = 3
    xmax = 100
    seeds = 10
    n_vals = np.array([5, 20, 40, 60])

    results = run_multiple_seeds(f, n_vals, lambda_val, xmax, seeds)
    plot_results(n_vals, results, save_path="results")

if __name__ == "__main__":
    main()
