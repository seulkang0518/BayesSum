import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jit, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc
from jax.scipy.stats import poisson as jax_poisson
import matplotlib.pyplot as plt
import os
import time
from functools import partial


def f(x):
    return x * jnp.exp(-jnp.sqrt(x))

def brownian_kernel(X1, X2):
    X1 = jnp.atleast_1d(X1).reshape(-1, 1)
    X2 = jnp.atleast_1d(X2).reshape(-1, 1)
    return jnp.minimum(X1, X2.T)

def kernel_embedding_poisson_fixed(lambda_, xi, xmax=200):
    xmax = int(xmax)  
    x_vals = jnp.arange(0, xmax + 1)
    mask = x_vals <= xi

    log_fact = jnp.cumsum(jnp.log(jnp.arange(1, xmax + 1)))
    fact = jnp.exp(jnp.concatenate([jnp.array([0.0]), log_fact]))

    pmf = (lambda_**x_vals / fact) * jnp.exp(-lambda_)
    term1 = jnp.sum(jnp.where(mask, x_vals * pmf, 0.0))
    term2 = xi * gammainc(xi + 1, lambda_)
    return term1 + term2

def kernel_embedding_poisson(lambda_, X, xmax):
    xmax = int(xmax)  
    def single_embed(xi):
        return kernel_embedding_poisson_fixed(lambda_, xi, xmax)
    return vmap(single_embed)(X)

def double_integral_poisson(lambda_, xmax=200):
    xmax = int(xmax)  
    k_vals = jnp.arange(0, xmax + 1)
    tail_probs = 1.0 - gammainc(k_vals + 1, lambda_)
    return jnp.sum(tail_probs ** 2)

@partial(jit, static_argnames=["xmax"])
def compute_integral_variance(X, lambda_, xmax):

    K = brownian_kernel(X, X) + 1e-4 * jnp.eye(len(X))
    L, lower = cho_factor(K)
    z = kernel_embedding_poisson(lambda_, X, xmax).reshape(-1, 1)
    K_inv_z = cho_solve((L, lower), z)

    return double_integral_poisson(lambda_, xmax) - (z.T @ K_inv_z)[0, 0]

@partial(jit, static_argnames=["xmax"])
def compute_max_variance(X_obs, candidates, lambda_, xmax):
    current_var = compute_integral_variance(X_obs, lambda_, xmax)

    def updated_var(x):
        X_aug = jnp.concatenate([X_obs, jnp.array([x])])
        return compute_integral_variance(X_aug, lambda_, xmax)

    updated_vars = vmap(updated_var)(candidates)
    kg_vals = current_var - updated_vars
    return candidates[jnp.argmax(kg_vals)]


def bayesian_cubature_poisson(X, f_vals, lambda_, xmax):
    # xmax = int(xmax) if isinstance(xmax, (int, float)) else int(jax.device_get(xmax))
    n = len(X)
    K = brownian_kernel(X, X) + 1e-4 * jnp.eye(n)
    L, lower = cho_factor(K)

    z = kernel_embedding_poisson(lambda_, X, xmax).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)

    K_inv_f = cho_solve((L, lower), f_vals)
    K_inv_z = cho_solve((L, lower), z)

    mean = (z.T @ K_inv_f)[0, 0]
    var = double_integral_poisson(lambda_, xmax) - (z.T @ K_inv_z)[0, 0]
    return mean, var

def run_experiment(f, n_vals, lambda_, xmax, seed, run_mc=True, run_bmc=True):
    key = jax.random.PRNGKey(seed)
    candidates = jnp.arange(0, xmax + 1)
    true_val = jnp.sum(f(candidates) * jax_poisson.pmf(candidates, mu=lambda_))
    bmc_means, mc_means, times = [], [], []

    for n in n_vals:
        start = time.time()

        if run_bmc:
            ## Deterministic BQ: fix x0 as the mode of Poisson(lambda)
            pmf = jax_poisson.pmf(candidates, mu=lambda_)
            x0 = candidates[jnp.argmax(pmf)]

            X = [x0]
            y = [f(x0)]

            while len(X) < n:
                key, subkey = random.split(key)
                used_set = set(map(int, X))
                unused = jnp.array([x for x in candidates if int(x) not in used_set])
                if len(unused) > 0:
                    x_next = compute_max_variance(jnp.array(X), unused, lambda_, xmax)
                else:
                    x_next = random.choice(subkey, candidates)
                y_next = f(x_next)
                X.append(x_next)
                y.append(y_next)

            X_bmc, f_vals_bmc = jnp.array(X), jnp.array(y)
            mu_bmc, _ = bayesian_cubature_poisson(X_bmc, f_vals_bmc, lambda_, xmax)
            jax.block_until_ready(mu_bmc)
            bmc_means.append(mu_bmc)

        else:
            bmc_means.append(jnp.nan)

        if run_mc:
            key, subkey = random.split(key)
            X_mc = jax.random.poisson(subkey, lam=lambda_, shape=(n, 1))
            f_vals_mc = f(X_mc)
            mu_mc = jnp.mean(f_vals_mc)
            jax.block_until_ready(mu_mc)
            mc_means.append(mu_mc)
        else:
            mc_means.append(jnp.nan)

        elapsed = time.time() - start
        times.append(elapsed)

        if run_bmc:
            print(f"n={n}, Time={elapsed:.3f}s, BMC err={float(jnp.abs(mu_bmc - true_val)):.10f}")
        if run_mc:
            print(f"n={n}, MC   err={float(jnp.abs(mu_mc - true_val)):.10f}")

    return {
        "true_val": true_val,
        "bmc_means": jnp.array(bmc_means),
        "mc_means": jnp.array(mc_means),
        "times": jnp.array(times)
    }


def run_multiple_seeds(f, n_vals, lambda_, xmax, num_seeds):
    bmc_all = []
    mc_all = []
    times_all = []

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")
        run_bmc = (seed == 0)
        run_mc = True

        result = run_experiment(
            f, n_vals, lambda_, xmax, seed,
             run_mc=run_mc, run_bmc=run_bmc)

        if run_bmc:
            bmc_all.append(result["bmc_means"])
            times_all.append(result["times"])

        mc_all.append(result["mc_means"])

    mc_all = jnp.stack(mc_all)
    mc_abs_error = jnp.mean(jnp.abs(mc_all - result["true_val"]), axis=0)

    if bmc_all:
        bmc_all = jnp.stack(bmc_all)
        bmc_abs_error = jnp.abs(bmc_all[0] - result["true_val"])
        avg_time = jnp.array(times_all[0])
    else:
        bmc_abs_error = None
        avg_time = None

    return {
        "true_val": result["true_val"],
        "bmc_mean_error": bmc_abs_error,
        "mc_mean_error": mc_abs_error,
        "times_mean": avg_time
    }

def plot_results(n_vals, results, lambda_val, save_path="results"):
    os.makedirs(save_path, exist_ok=True)

    true_val = results["true_val"]
    bmc_errors = jnp.abs(results["bmc_means"] - true_val)
    mc_errors = jnp.abs(results["mc_means"] - true_val)

    bmc_errors = jnp.clip(bmc_errors, 1e-10, None)
    mc_errors = jnp.clip(mc_errors, 1e-10, None)

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, mc_errors, 'ko-', label="MC Absolute Error")
    plt.plot(n_vals, bmc_errors, 'ro-', label="BQ Absolute Error")
    plt.title("Absolute Error: BQ vs MC\n(Brownian kernel, Poisson input)")
    plt.xlabel("n (number of points)")
    plt.ylabel("Absolute Error (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abs_error_poisson_bq_RMSE_optimal.png"), dpi=300)
    plt.close()


def main():
    lambda_val = 10
    xmax = 200
    n_vals = jnp.array([10, 20, 40, 60])
    seed = 0
    num_seeds = 10

    results = run_multiple_seeds(f, n_vals, lambda_val, xmax, num_seeds)


    print("\n=== Final Averaged Results ===")
    print("n =", n_vals)
    print("True value:", results["true_val"])
    print("BMC mean error:", results["bmc_mean_error"])
    print("MC  mean error:", results["mc_mean_error"])
    print("Avg runtime per n:", results["times_mean"])

if __name__ == "__main__":
    main()
