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
import os
import matplotlib as mpl

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

@partial(jit, static_argnums=(3,))
def compute_max_var_Z(X_obs, candidates, lambda_, d):
    current_var = compute_integral_variance(X_obs, lambda_, d)

    def reduction(x):
        X_aug = jnp.vstack([X_obs, x])
        updated_var = compute_integral_variance(X_aug, lambda_, d)
        return current_var - updated_var

    reductions = vmap(reduction)(candidates)
    return candidates[jnp.argmax(reductions)]


@partial(jit, static_argnums=0)
def all_states_cached(d):
    n_states = 3**d
    return jnp.array(((jnp.arange(n_states)[:, None] >> jnp.arange(d)) & 1), dtype=jnp.float32)

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
    if d > 30:
        raise ValueError("Too large to enumerate all states.")
    J = J_matrix(L)
    all_states = all_states_cached(d)
    mean_val = batched_true_expectation(J, beta, all_states)
    return mean_val, all_states

def sample_states(key, d, num_samples):
    return jax.random.randint(key, shape=(num_samples, d), minval=0, maxval=3).astype(jnp.float32)

def run_experiment(n_vals, lambda_, d, L, seed, beta, t_e, run_mc, run_bmc, sub_sample):
    key = jax.random.PRNGKey(seed)
    J = J_matrix(L)
    bmc_means, mc_means, times = [], [], []

    for n in n_vals:
        start = time.time()

        if run_bmc is not None:
            X = jnp.zeros((n, d))
            y = jnp.zeros((n,))

            init_points = 3  
            key, subkey = jax.random.split(key)
            x_init = sample_states(subkey, d, init_points)

            for i in range(init_points):
                X = X.at[i].set(x_init[i])
                y = y.at[i].set(f_single(x_init[i], J, beta))

            for i in range(1, n):
                key, subkey = jax.random.split(key)
                candidates = sample_states(subkey, d, sub_sample or 1000)

                if run_bmc == "max_var_f":
                    x_next = compute_max_variance(X[:i], y[:i].reshape(-1, 1), candidates, lambda_, d)
                elif run_bmc == "max_var_Z":
                    x_next = compute_max_var_Z(X[:i], candidates, lambda_, d)

                y_next = f_single(x_next, J, beta)

                X = X.at[i].set(x_next)
                y = y.at[i].set(y_next)

            mu_bmc, _ = bayesian_cubature(X, y, lambda_, d)
            jax.block_until_ready(mu_bmc)
            bmc_means.append(mu_bmc)

        else:
            key, subkey = jax.random.split(key)
            X = sample_states(subkey, d, n)
            y = f_batch(X, J, beta)
            mu_bmc, _ = bayesian_cubature(X, y, lambda_, d)
            jax.block_until_ready(mu_bmc)
            bmc_means.append(mu_bmc)

        if run_mc:
            key, subkey = jax.random.split(key)
            X_mc = sample_states(subkey, d, n)
            f_vals_mc = f_batch(X_mc, J, beta)
            mu_mc = jnp.mean(f_vals_mc)
            jax.block_until_ready(mu_mc)
            mc_means.append(mu_mc)
        else:
            mc_means.append(jnp.nan)

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"n={n}, Time={elapsed:.3f}s, BMC err={float(jnp.abs(mu_bmc - t_e)):.10f}")
        if run_mc:
            print(f"n={n}, MC   err={float(jnp.abs(mu_mc - t_e)):.10f}")

    return {
        "true_val": t_e,
        "bmc_means": jnp.array(bmc_means),
        "mc_means": jnp.array(mc_means),
        "times": jnp.array(times)
    }

def run_multiple_seeds(n_vals, lambda_, d, L, num_seeds, beta, t_e, run_mc, run_bmc, sub_sample):
    bmc_all = []
    mc_all = []
    times_all = []

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")

        result = run_experiment(
            n_vals, lambda_, d, L,
            seed, beta, t_e,
            run_mc, run_bmc, sub_sample
        )

        bmc_all.append(result["bmc_means"])
        times_all.append(result["times"])
        mc_all.append(result["mc_means"])

    mc_all = jnp.stack(mc_all)
    mc_abs_error = jnp.mean(jnp.abs(mc_all - result["true_val"]), axis=0)
   
    bmc_all = jnp.stack(bmc_all)
    bmc_abs_error = jnp.mean(jnp.abs(bmc_all - result["true_val"]), axis=0)

    avg_time = jnp.mean(jnp.stack(times_all))


    return {
        "true_val": result["true_val"],
        "bmc_mean_error": bmc_abs_error,
        "mc_mean_error": mc_abs_error,
        "times_mean": avg_time
    }


def plot_results(n_vals, bq_max_var_z_results, bq_max_var_f_results, bq_results, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    
    bq_max_var_z_errors = bq_max_var_z_results["bmc_mean_error"]
    bq_max_var_f_errors = bq_max_var_f_results["bmc_mean_error"]
    bq_random_errors = bq_results["bmc_mean_error"]
    mc_errors = bq_max_var_z_results["mc_mean_error"]

    bq_max_var_z_errors = jnp.clip(bq_max_var_z_errors, 1e-10, None)
    bq_max_var_f_errors = jnp.clip(bq_max_var_f_errors, 1e-10, None)
    bq_random_errors = jnp.clip(bq_random_errors, 1e-10, None)
    mc_errors = jnp.clip(mc_errors, 1e-10, None)

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, bq_max_var_z_errors, 'ko-', label="BQ (Argmax Var[Z|sample points]) Absolute Error")
    plt.plot(n_vals, bq_max_var_f_errors, 'bo-', label="BQ (Argmax Var[f|sample points]) Absolute Error")
    plt.plot(n_vals, bq_random_errors, 'ro-', label="BQ (random sample points) Absolute Error")
    plt.plot(n_vals, mc_errors, 'go-', label="MC Absolute Error")
    plt.title("Absolute Error: BQ vs MC\n(Pointwise comparison kernel, Potts model 4x4)")
    plt.xlabel("n (number of points)")
    plt.ylabel("Absolute Error (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abs_error_potts.png"), dpi=300)
    plt.close()

def plot_results_lambda(lambda_results, save_path="results"):
    os.makedirs(save_path, exist_ok=True)

    lambda_vals, results = zip(*lambda_results)  # unzip
    lambda_vals = jnp.array(lambda_vals)
    results = jnp.array(results)

    lambda_vals = jnp.clip(lambda_vals, 1e-10, None)
    results = jnp.clip(results, 1e-10, None)

    sort_idx = jnp.argsort(lambda_vals)
    lambda_vals = lambda_vals[sort_idx]
    results = results[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vals, results, 'ko-', label="BQ (random) Absolute Error")
    plt.title("Absolute Error: BQ (Random), Potts model (4 x 4 lattice)")
    plt.xlabel("Lambda values - parameters")
    plt.ylabel("Absolute Error (log scale)")
    plt.yscale('log')

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abs_error_potts_lambdas_4.png"), dpi=300)
    plt.close()

def main():
    global beta
    k_b = 1
    T_c = 5.0
    beta = 1 / (k_b * T_c)
    L = 2
    d = L * L

    lambda_ = 5.6 / d

    n_vals = jnp.array([10, 50, 100, 200, 300]) 
    num_seeds = 10
    sub_sample = 1000

    t_e, all_states = true_expectation(L, d, beta)

    ## BQ with max Var[Z|observed points] and MC
    bq_max_var_z_results = run_multiple_seeds(n_vals, lambda_, d, L, num_seeds, beta, t_e, True, "max_var_Z", sub_sample)

    ## BQ with max Var[f|observed points] 
    bq_max_var_f_results = run_multiple_seeds(n_vals, lambda_, d, L, num_seeds, beta, t_e, False, "max_var_f", sub_sample)

    ## BQ with random sample points
    bq_results = run_multiple_seeds(n_vals, lambda_, d, L, num_seeds, beta, t_e, False, None, sub_sample)

    print("\n=== Final Averaged Results ===")
    print("n =", n_vals)
    print("True value:", bq_max_var_z_results["true_val"])
    print("BMC (Max Var Z) mean error:", bq_max_var_z_results["bmc_mean_error"])
    print("BMC (Max Var f) mean error:", bq_max_var_f_results["bmc_mean_error"])
    print("BMC (random) mean error:", bq_results["bmc_mean_error"])
    print("MC  mean error:", bq_max_var_z_results["mc_mean_error"])

    plot_results(n_vals, bq_max_var_z_results, bq_max_var_f_results, bq_results, save_path="results")

    ## BQ (random) with lambda values
    lambda_vals = [1, 2, 3, 4, 5, 5.6, 6]
    lambda_results = []

    for lambda_ in lambda_vals:
        lambda_ /= d
        err = run_multiple_seeds(n_vals, lambda_, d, L, num_seeds, beta, t_e, False, None, sub_sample)["bmc_mean_error"][-1]
        lambda_results.append((lambda_, err))

    # print(lambda_results)

    plot_results_lambda(lambda_results, save_path="results")

if __name__ == "__main__":
    main()
