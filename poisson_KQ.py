import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, t
from scipy.special import gammainc, factorial


def f(x):
    return x * np.exp(-np.sqrt(x))
    # return np.exp(-5 * x)

def brownian_kernel(X1, X2):
    X1 = np.atleast_1d(X1).reshape(-1, 1)
    X2 = np.atleast_1d(X2).reshape(-1, 1)
    return np.minimum(X1, X2.T)

def kernel_embedding_poisson(lambda_, xi):
    x_vals = np.arange(0, xi + 1)
    pmf_vals = poisson.pmf(x_vals, mu=lambda_)
    term1 = np.sum(x_vals * pmf_vals)
    tail_prob = 1 - poisson.cdf(xi, mu=lambda_)
    term2 = xi * tail_prob
    return term1 + term2

def kernel_embedding_gamma(lambda_, xi):
    x_vals = np.arange(0, xi + 1)
    pmf = (lambda_**x_vals / factorial(x_vals)) * np.exp(-lambda_)
    term1 = np.sum(x_vals * pmf)
    tail_prob = gammainc(xi + 1, lambda_)
    term2 = xi * tail_prob
    return term1 + term2

def double_integral_poisson(lambda_, xmax=100):
    cdf_vals = poisson.cdf(np.arange(0, xmax + 1), mu=lambda_)
    tail_probs = 1 - cdf_vals
    return np.sum(tail_probs ** 2)

def double_integral_gamma(lambda_, xmax=100):
    k = np.arange(0, xmax + 1)
    tails = gammainc(k + 1, lambda_)
    return np.sum(tails ** 2)


def bayesian_cubature_poisson(X, f_vals, lambda_, xmax=100):
    n = len(X)
    K = brownian_kernel(X, X).astype(float)
    z = np.array([kernel_embedding_poisson(lambda_, xi) for xi in X]).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)
    K += 1e-8 * np.eye(n)  # regularization
    K_inv = np.linalg.inv(K)
    mean = (z.T @ K_inv @ f_vals).item()
    # scale = (f_vals.T @ K_inv @ f_vals).item() / n ### i will check this bit
    PiPi_k = double_integral_poisson(lambda_, xmax)
    var = PiPi_k - (z.T @ K_inv @ z).item()
    return mean, var


def run_experiment(f, n_vals, lambda_, xmax=100, seed=0):
    np.random.seed(seed)
    true_expectation = np.sum(f(np.arange(0, xmax+1)) * poisson.pmf(np.arange(0, xmax+1), mu=lambda_))

    bmc_means, bmc_lows, bmc_highs = [], [], []
    mc_means, mc_stds = [], []

    for n in n_vals:
        X = np.sort(np.random.poisson(lam=lambda_, size=n))
        X = np.unique(X)  # ensure Gram matrix is invertible
        f_vals = f(X)

        mu_bmc, var_bmc = bayesian_cubature_poisson(X, f_vals, lambda_, xmax)
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

# Run and plot
lambda_val = 20
n_vals = np.arange(5, 300, 5)
results = run_experiment(f, n_vals, lambda_val)

plt.figure(figsize=(10, 6))
plt.plot(n_vals, results["mc_means"], 'ko', label="MC")
plt.plot(n_vals, results["bmc_means"], 'ro', label="BQ (PMF + CDF)")
plt.fill_between(n_vals, results["bmc_lows"], results["bmc_highs"], color='red', alpha=0.2)
plt.axhline(results["true_val"], color='green', linestyle='--', label="True")
plt.title("Bayesian Quadrature vs Monte Carlo (Brownian kernel, Poisson input)")
plt.xlabel("n (number of points)")
plt.ylabel("Estimated Expectation")
plt.yscale('log') 
plt.legend()
plt.grid(True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

