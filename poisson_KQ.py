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

    xmax = int(xmax)  
    K = brownian_kernel(X, X) + 1e-4 * jnp.eye(len(X))
    L, lower = cho_factor(K)
    z = kernel_embedding_poisson(lambda_, X, xmax).reshape(-1, 1)
    K_inv_z = cho_solve((L, lower), z)

    return double_integral_poisson(lambda_, xmax) - (z.T @ K_inv_z)[0, 0]

def kernel_vec(x, X):
    return jnp.minimum(x, X).reshape(-1, 1)

def compute_kg_poisson(X_obs, y_obs, candidates, lambda_, num_samples, xmax, rng_key):
    xmax = int(xmax) if isinstance(xmax, (int, float)) else int(jax.device_get(xmax))
    K = brownian_kernel(X_obs, X_obs) + 1e-4 * jnp.eye(len(X_obs))
    try:
        L, lower = cho_factor(K)
    except:
        raise ValueError("Cholesky decomposition failed. Kernel matrix might be singular.")

    z = v_kernel_embedding_poisson(lambda_, X_obs, xmax).reshape(-1, 1)
    K_inv_y = cho_solve((L, lower), y_obs.reshape(-1, 1))
    K_inv_z = cho_solve((L, lower), z)
    PiPi_k = double_integral_poisson(lambda_, xmax)
    current_var = PiPi_k - (z.T @ K_inv_z)[0, 0]

    def mu_sigma(x):
        kx = kernel_vec(x, X_obs)
        mu_x = (kx.T @ K_inv_y)[0, 0]
        sigma2_x = 1.0 - (kx.T @ cho_solve((L, lower), kx))[0, 0]
        return mu_x, jnp.sqrt(jnp.maximum(sigma2_x, 1e-10))

    def kg_for_candidate(x, subkey):
        mu_x, sigma_x = mu_sigma(x)
        f_samples = mu_x + sigma_x * random.normal(subkey, (num_samples,))

        def updated_var(f_j):
            X_aug = jnp.append(X_obs, x)
            f_aug = jnp.append(y_obs, f_j)
            K_aug = brownian_kernel(X_aug, X_aug) + 1e-4 * jnp.eye(len(X_aug))
            try:
                L_aug, lower_aug = cho_factor(K_aug)
                z_aug = v_kernel_embedding_poisson(lambda_, X_aug, xmax).reshape(-1, 1)
                K_inv_z_aug = cho_solve((L_aug, lower_aug), z_aug)
                return PiPi_k - (z_aug.T @ K_inv_z_aug)[0, 0]
            except:
                return current_var

        updated_vars = vmap(updated_var)(f_samples)
        return current_var - jnp.mean(updated_vars)

    subkeys = random.split(rng_key, len(candidates))
    kg_vals = vmap(kg_for_candidate)(candidates, subkeys)
    return candidates[jnp.argmax(kg_vals)]


def bayesian_cubature_poisson(X, f_vals, lambda_, xmax):
    # xmax = int(xmax) if isinstance(xmax, (int, float)) else int(jax.device_get(xmax))
    n = len(X)
    K = brownian_kernel(X, X) + 1e-4 * jnp.eye(n)
    try:
        L, lower = cho_factor(K)
    except:
        raise ValueError("Cholesky failed: kernel matrix not positive definite.")

    z = kernel_embedding_poisson(lambda_, X, xmax).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)

    K_inv_f = cho_solve((L, lower), f_vals)
    K_inv_z = cho_solve((L, lower), z)

    mean = (z.T @ K_inv_f)[0, 0]
    var = double_integral_poisson(lambda_, xmax) - (z.T @ K_inv_z)[0, 0]
    return mean, var

@partial(jit, static_argnames=["xmax"])
def compute_max_variance(X_obs, candidates, lambda_, xmax):
    current_var = compute_integral_variance(X_obs, lambda_, xmax)

    def updated_var(x):
        X_aug = jnp.append(X_obs, x)
        return compute_integral_variance(X_aug, lambda_, xmax)

    updated_vars = vmap(updated_var)(candidates)
    kg_vals = current_var - updated_vars
    return candidates[jnp.argmax(kg_vals)]

def run_experiment(f, n_vals, lambda_, xmax, seed, use_bo):
    rng = random.PRNGKey(seed)
    candidates = jnp.arange(0, xmax + 1)
    true_val = jnp.sum(f(candidates) * jax_poisson.pmf(candidates, mu=lambda_))

    bmc_means, mc_means, times = [], [], []

    for n in n_vals:
        start = time.time()

        if use_bo:
            key, subkey = random.split(rng)
            idx = random.choice(subkey, len(candidates))
            x0 = candidates[idx]
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

            X_bmc = jnp.array(X)
            f_vals_bmc = jnp.array(y)

        else:
            key, subkey = random.split(rng)
            X_bmc = random.poisson(subkey, lam=lambda_, shape=(n,))
            f_vals_bmc = f(X_bmc)

        mu_bmc, _ = bayesian_cubature_poisson(X_bmc, f_vals_bmc, lambda_, xmax)

        key, subkey = random.split(rng)
        X_mc = random.poisson(subkey, lam=lambda_, shape=(n,))
        f_vals_mc = f(X_mc)
        mu_mc = jnp.mean(f_vals_mc)

        jax.block_until_ready(mu_bmc)
        jax.block_until_ready(mu_mc)

        elapsed = time.time() - start
        times.append(elapsed)
        bmc_means.append(mu_bmc)
        mc_means.append(mu_mc)

        print(f"n={n}, Time={elapsed:.3f}s, BMC err={jnp.abs(mu_bmc - true_val):.10f}")
        print(f"n={n}, MC  err={jnp.abs(mu_mc - true_val):.10f}")

    return {
        "true_val": true_val,
        "bmc_means": jnp.array(bmc_means),
        "mc_means": jnp.array(mc_means),
        "times": jnp.array(times)
    }


def run_multiple_seeds(f, n_vals, lambda_, xmax, num_seeds, use_bo):
    bmc_all = []
    mc_all = []
    times_all = []

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")
        result = run_experiment(f, n_vals, lambda_, xmax, seed, use_bo)
        bmc_all.append(result["bmc_means"])
        mc_all.append(result["mc_means"])
        times_all.append(result["times"])

    bmc_all = jnp.stack(bmc_all)  
    mc_all = jnp.stack(mc_all)
    times_all = jnp.stack(times_all)

    # Compute mean of absolute errors (per seed, then averaged)
    true_val = result["true_val"]
    bmc_abs_error = jnp.mean(jnp.abs(bmc_all - true_val), axis=0)
    mc_abs_error = jnp.mean(jnp.abs(mc_all - true_val), axis=0)

    return {
        "true_val": true_val,
        "bmc_mean_error": bmc_abs_error,
        "mc_mean_error": mc_abs_error,
        "times_mean": jnp.mean(times_all, axis=0)
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
    use_bo = True
    num_mc = 10
    num_seeds = 10

    results = run_multiple_seeds(f, n_vals, lambda_val, xmax, num_seeds, use_bo)


    print("\n=== Final Averaged Results ===")
    print("n =", n_vals)
    print("True value:", results["true_val"])
    print("BMC mean error:", results["bmc_mean_error"])
    print("MC  mean error:", results["mc_mean_error"])
    print("Avg runtime per n:", results["times_mean"])

if __name__ == "__main__":
    main()
