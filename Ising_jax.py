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
import matplotlib.pyplot as plt

## done
@partial(jit, static_argnums=0)
def J_matrix(L):
    n = L * L
    idx = jnp.arange(n).reshape(L, L)

    right = jnp.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    down  = jnp.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)

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
    return jnp.exp(-lambda_ * d / 2) * (jnp.cosh(lambda_ / 2) ** d)

@jit
def double_integral(lambda_, d):
    return kernel_embedding(lambda_, d)

@jit
def gram_matrix(X, lambda_, d):
    X_rescaled = 2 * X - 1.
    inner = X_rescaled @ X_rescaled.T
    return jnp.exp(-lambda_ * 0.5 * (d - inner))

@jit
def kernel_vec(x, X, lambda_, d):
    x_rescaled = 2 * x - 1.
    X_rescaled = 2 * X - 1.
    inner = X_rescaled @ x_rescaled
    return jnp.exp(-lambda_ * 0.5 * (d - inner)).reshape(-1, 1)

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

@jit
def compute_integral_variance(X, lambda_, d):

    K = gram_matrix(X, lambda_, d) + 1e-4 * jnp.eye(len(X))
    z = jnp.full((len(X), 1), kernel_embedding(lambda_, d))
    L, lower = cho_factor(K)
    K_inv_z = cho_solve((L, lower), z)

    return double_integral(lambda_, d) - (z.T @ K_inv_z)[0, 0]


@partial(jit, static_argnums=(6,))
def compute_kg(X_obs, y_obs, candidates, lambda_, d, key, num_samples):
    K = gram_matrix(X_obs, lambda_, d) + 1e-4 * jnp.eye(len(X_obs))
    z = jnp.full((len(X_obs), 1), kernel_embedding(lambda_, d))
    L, lower = cho_factor(K)
    K_inv_y = cho_solve((L, lower), y_obs)
    K_inv_z = cho_solve((L, lower), z)

    PiPi_k = kernel_embedding(lambda_, d)
    current_Z_var = PiPi_k - (z.T @ K_inv_z)[0, 0]


    def mu_sigma2(x):
        k_x = kernel_vec(x, X_obs, lambda_, d)
        mu = (k_x.T @ K_inv_y).squeeze()
        K_inv_kx = cho_solve((L, lower), k_x)
        sigma2 = 1. - (k_x.T @ K_inv_kx).squeeze()
        return mu, jnp.maximum(sigma2, 1e-10)

    mus, sigma2s = vmap(mu_sigma2)(candidates)
    sigmas = jnp.sqrt(sigma2s)

    key, subkey = jax.random.split(key)
    std_normals = jax.random.normal(subkey, shape=(len(candidates), num_samples))
    f_samples = mus[:, None] + sigmas[:, None] * std_normals

    def compute_expected_augmented_var(x, y_samples):
        def single_aug_var(y_j):
            X_aug = jnp.vstack([X_obs, x])
            y_aug = jnp.vstack([y_obs, y_j.reshape(1, 1)])
            return compute_integral_variance(X_aug, lambda_, d)
        return jnp.mean(vmap(single_aug_var)(y_samples))

    expected_updated_vars = vmap(compute_expected_augmented_var)(candidates, f_samples)
    kg_values = current_Z_var - expected_updated_vars

    return candidates[jnp.argmax(kg_values)]

@partial(jit, static_argnums=(5,))
def compute_ei(X_obs, y_obs, candidates, lambda_, d, num_samples_unused=0):
    # GP posterior setup
    K = gram_matrix(X_obs, lambda_, d) + 1e-4 * jnp.eye(len(X_obs))
    L, lower = cho_factor(K)
    K_inv_y = cho_solve((L, lower), y_obs)

    def mu_sigma2(x):
        k_x = kernel_vec(x, X_obs, lambda_, d)
        mu = (k_x.T @ K_inv_y).squeeze()
        K_inv_kx = cho_solve((L, lower), k_x)
        sigma2 = 1. - (k_x.T @ K_inv_kx).squeeze()
        return mu, jnp.maximum(sigma2, 1e-10)

    mus, sigma2s = vmap(mu_sigma2)(candidates)
    sigmas = jnp.sqrt(sigma2s)

    y_best = jnp.max(y_obs)

    z = (mus - y_best) / sigmas
    ei = (mus - y_best) * norm.cdf(z) + sigmas * norm.pdf(z)
    return candidates[jnp.argmax(ei)]

@partial(jit, static_argnums=(4,))
def compute_ei_bq(X_obs, candidates, lambda_, d, key_unused=None):
    """
    Selects the next candidate that maximally reduces Var[Z],
    i.e., argmax_x Var[Z] - Var[Z | (x, f(x))]
    """
    current_var = compute_integral_variance(X_obs, lambda_, d)

    def reduction(x):
        X_aug = jnp.vstack([X_obs, x])
        updated_var = compute_integral_variance(X_aug, lambda_, d)
        return current_var - updated_var

    reductions = vmap(reduction)(candidates)
    return candidates[jnp.argmax(reductions)]

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

@lru_cache(maxsize=None)
def all_states_cached(d):
    states = jnp.array(list(product([0., 1.], repeat=d)), dtype=jnp.float32)
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

def run_experiment(f_single, n_vals, lambda_, d, L, seed, beta, t_e, all_states, run_mc=True, run_bmc=True):
    key = jax.random.PRNGKey(seed)
    J = J_matrix(L)
    bmc_means, mc_means, times = [], [], []

    for n in n_vals:
        start = time.time()

        if run_bmc:
            candidates = all_states
            # x0 = all_states[jax.random.choice(jax.random.PRNGKey(0), len(all_states))]  # fixed x0
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
        else:
            bmc_means.append(jnp.nan)

        if run_mc:
            key, subkey = jax.random.split(key)
            idxs_mc = jax.random.choice(subkey, len(all_states), shape=(n,), replace=True)
            X_mc = all_states[idxs_mc]
            f_vals_mc = f_batch(X_mc, J, beta)
            mu_mc = jnp.mean(f_vals_mc)
            jax.block_until_ready(mu_mc)
            mc_means.append(mu_mc)
        else:
            mc_means.append(jnp.nan)

        elapsed = time.time() - start
        times.append(elapsed)

        if run_bmc:
            print(f"n={n}, Time={elapsed:.3f}s, BMC err={float(jnp.abs(mu_bmc - t_e)):.10f}")
        if run_mc:
            print(f"n={n}, MC   err={float(jnp.abs(mu_mc - t_e)):.10f}")

    return {
        "true_val": t_e,
        "bmc_means": jnp.array(bmc_means),
        "mc_means": jnp.array(mc_means),
        "times": jnp.array(times)
    }

def run_multiple_seeds(f_single, n_vals, lambda_, d, L, num_seeds, beta, t_e, all_states):
    bmc_all = []
    mc_all = []
    times_all = []

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")
        run_bmc = (seed == 0)
        run_mc = True

        result = run_experiment(
            f_single, n_vals, lambda_, d, L,
            seed, beta, t_e, all_states,
            run_mc=run_mc, run_bmc=run_bmc
        )

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


def main():
    global beta
    k_b = 1
    T_c = 2.269
    beta = 1 / (k_b * T_c)
    lambda_ = 0.1
    L = 4
    d = L * L
    n_vals = jnp.array([10, 50, 100, 200, 300]) 
    num_seeds = 10

    t_e, all_states = true_expectation(L, d, beta)
    results = run_multiple_seeds(f_single, n_vals, lambda_, d, L, num_seeds, beta, t_e, all_states)

    print("\n=== Final Averaged Results ===")
    print("n =", n_vals)
    print("True value:", results["true_val"])
    print("BMC mean error:", results["bmc_mean_error"])
    print("MC  mean error:", results["mc_mean_error"])
    print("Avg runtime per n:", results["times_mean"])


if __name__ == "__main__":
    main()
