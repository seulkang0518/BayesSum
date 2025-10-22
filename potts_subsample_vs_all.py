import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.linalg import cho_factor, cho_solve
from itertools import product
from functools import partial
import matplotlib.pyplot as plt
import time
from functools import lru_cache
from jax.scipy.stats import norm
import os
import matplotlib as mpl

@partial(jit, static_argnums=0)
def J_matrix(L):
    n = L * L
    idx = jnp.arange(n).reshape(L, L)

    right = jnp.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    down = jnp.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)

    pairs = jnp.concatenate([right, down], axis=0)

    i = pairs[:, 0]
    j = pairs[:, 1]
    J = jnp.zeros((n, n), dtype=jnp.float32).at[i, j].set(1.0).at[j, i].set(1.0)
    return 0.1 * J

@jit
def f_single(x, J, beta):
    diff = x[:, None] == x[None, :]
    energy = jnp.sum(J * diff) / 2
    return jnp.exp(-beta * energy)

@jit
def f_batch(X, J, beta):
    return vmap(lambda x: f_single(x, J, beta))(X)

@jit
def kernel_embedding(lambda_, d):
    return ((1 + 2 * jnp.exp(-lambda_)) / 3) ** d

@jit
def double_integral(lambda_, d):
    return kernel_embedding(lambda_, d)

@jit
def gram_matrix(X, lambda_, d):
    X1 = X[:, None, :]
    X2 = X[None, :, :]
    hamming = jnp.sum(X1 != X2, axis=2)
    return jnp.exp(-lambda_ * hamming)

@jit
def kernel_vec(x, X, lambda_, d):
    x1 = x.reshape(1, -1)
    hamming = jnp.sum(X != x1, axis=1)
    return jnp.exp(-lambda_ * hamming).reshape(-1, 1)

@jit
def compute_max_variance(X_obs, y_obs, candidates, lambda_, d):
    K = gram_matrix(X_obs, lambda_, d) + 1e-4 * jnp.eye(len(X_obs))
    L, lower = cho_factor(K)

    def posterior_variance(x):
        k_x = kernel_vec(x, X_obs, lambda_, d)
        K_inv_kx = cho_solve((L, lower), k_x)
        sigma2 = 1. - (k_x.T @ K_inv_kx).squeeze()
        return jnp.maximum(sigma2, 1e-10)

    variances = vmap(posterior_variance)(candidates)
    return candidates[jnp.argmax(variances)]

@jit
def bayesian_cubature(X, f_vals, lambda_, d):
    n = len(X)
    K = gram_matrix(X, lambda_, d) + 1e-4 * jnp.eye(n)

    try:
        L, lower = cho_factor(K)
    except:
        raise ValueError("Cholesky failed: kernel matrix not positive definite.")

    z = jnp.full((n, 1), kernel_embedding(lambda_, d))
    f_vals = f_vals.reshape(n, 1)

    K_inv_z = cho_solve((L, lower), z)
    K_inv_f = cho_solve((L, lower), f_vals)

    mean = (z.T @ K_inv_f)[0, 0]
    var = double_integral(lambda_, d) - (z.T @ K_inv_z)[0, 0]
    return mean, var

@lru_cache(maxsize=None)
def all_states_cached(d):
    states = jnp.array(list(product([0., 1., 2.], repeat=d)), dtype=jnp.float32)
    return states

def make_f_batch_fn(J, beta):
    return lambda xs: f_batch(xs, J, beta)

def batched_true_expectation(J, beta, all_states, batch_size=10000):
    f_eval = make_f_batch_fn(J, beta)
    n = all_states.shape[0]
    total_sum = 0.0

    for i in range(0, n, batch_size):
        chunk = all_states[i:i + batch_size]
        total_sum += jnp.sum(f_eval(chunk))

    return total_sum / n

def true_expectation(L, d, beta):
    J = J_matrix(L)
    all_states = all_states_cached(d)
    mean_val = batched_true_expectation(J, beta, all_states)
    return mean_val, all_states

def sample_states(key, d, num_samples):
    return jax.random.randint(key, shape=(num_samples, d), minval=0, maxval=3).astype(jnp.float32)

def run_experiment(f_single, n_vals, lambda_, d, L, seed, beta, t_e, all_states, run_mc, run_bmc, sub_sample):
    J = J_matrix(L)
    key = jax.random.PRNGKey(seed)
    bmc_means, bmc_sam_means = [], []

    for n in n_vals:
        start = time.time()

        #### no subsampling
        candidates = all_states
        key, subkey = jax.random.split(key)
        x0 = candidates[jax.random.choice(subkey, len(candidates))]
        X = [x0]
        y = [f_single(x0, J, beta)]

        for _ in range(1, n):
            x_next = compute_max_variance(jnp.array(X), jnp.array(y).reshape(-1, 1), candidates, lambda_, d)
            y_next = f_single(x_next, J, beta)
            X.append(x_next)
            y.append(y_next)

        X_bmc, f_vals_bmc = jnp.array(X), jnp.array(y)
        mu_bmc, _ = bayesian_cubature(X_bmc, f_vals_bmc, lambda_, d)
        jax.block_until_ready(mu_bmc)
        bmc_means.append(mu_bmc)

        #### subsampling
        X = jnp.zeros((n, d))
        y = jnp.zeros((n,))

        key, subkey = jax.random.split(key)
        x0 = sample_states(subkey, d, 1)[0]
        X = X.at[0].set(x0)
        y = y.at[0].set(f_single(x0, J, beta))

        for i in range(1, n):
            key, subkey = jax.random.split(key)
            candidates = sample_states(subkey, d, sub_sample or 1000)

            x_next = compute_max_variance(jnp.array(X), jnp.array(y).reshape(-1, 1), candidates, lambda_, d)
            y_next = f_single(x_next, J, beta)

            X = X.at[i].set(x_next)
            y = y.at[i].set(y_next)

        X_sam_bmc, f_sam_bmc = jnp.array(X), jnp.array(y)
        mu_sam_bmc, _ = bayesian_cubature(X_sam_bmc, f_sam_bmc, lambda_, d)
        jax.block_until_ready(mu_sam_bmc)
        bmc_sam_means.append(mu_sam_bmc)


    return {
        "true_val": t_e,
        "bmc_means": jnp.array(bmc_means),
        "bmc_sam_means": jnp.array(bmc_sam_means),
    }

def run_multiple_seeds(f_single, n_vals, lambda_, d, L, num_seeds, beta, t_e, all_states, run_mc, run_bmc, subsample):
    bmc_all = []
    bmc_sam_all = []

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")

        result = run_experiment(
            f_single, n_vals, lambda_, d, L,
            seed, beta, t_e, all_states,
            run_mc, run_bmc, subsample
        )

        bmc_all.append(result["bmc_means"])
        bmc_sam_all.append(result["bmc_sam_means"])

    bmc_all = jnp.stack(bmc_all)
    bmc_abs_error = jnp.mean(jnp.abs(bmc_all - result["true_val"]), axis=0)

    bmc_sam_all = jnp.stack(bmc_sam_all)
    bmc_sam_abs_error = jnp.mean(jnp.abs(bmc_sam_all - result["true_val"]), axis=0)



    return {
        "true_val": result["true_val"],
        "bmc_mean_error": bmc_abs_error,
        "bmc_sam_mean_error": bmc_sam_abs_error
    }


def main():
    k_b = 1
    T_c = 2.269
    beta = 1 / (k_b * T_c)
    L = 3
    d = L * L
    n_vals = jnp.array([10, 50, 100, 200])
    num_seeds = 10
    lambda_ = 0.01
    sub_sample = 200

    t_e, all_states = true_expectation(L, d, beta)
    bo_mc_subsampled_results = run_multiple_seeds(f_single, n_vals, lambda_, d, L, num_seeds, beta, t_e, all_states, True, True, sub_sample)

    print("\n=== Final Averaged Results ===")
    print("n =", n_vals)
    print("True value:", bo_mc_subsampled_results["true_val"])
    print("BMC mean error:", bo_mc_subsampled_results["bmc_mean_error"])
    print("BMC Subsampled mean error:", bo_mc_subsampled_results["bmc_sam_mean_error"])    

if __name__ == "__main__":
    main()







