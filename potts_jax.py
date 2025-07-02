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

def run_experiment(f_single, n_vals, lambda_, d, L, seed, beta, t_e, all_states, run_mc=True, run_bmc=True):
    J = J_matrix(L)
    key = jax.random.PRNGKey(seed)
    bmc_means, mc_means, times = [], [], []

    for n in n_vals:
        start = time.time()

        if run_bmc:
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

def plot_results(n_vals, results, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    true_val = results["true_val"]
    bmc_errors = jnp.abs(results["bmc_means"] - true_val)
    mc_errors = jnp.abs(results["mc_means"] - true_val)
    bmc_errors = jnp.clip(bmc_errors, 1e-10)
    mc_errors = jnp.clip(mc_errors, 1e-10)

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, mc_errors, 'ko-', label="MC Absolute Error")
    plt.plot(n_vals, bmc_errors, 'ro-', label="BQ Absolute Error")
    plt.title("Absolute Error: BQ vs MC (BO via Max Variance)")
    plt.xlabel("n (number of points)")
    plt.ylabel("Absolute Error (log scale)")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abs_error_bo_maxvar.png"), dpi=300)
    plt.close()

def main():
    k_b = 1
    T_c = 2.269
    beta = 1 / (k_b * T_c)
    L = 4
    d = L * L
    n_vals = jnp.array([10, 50, 100, 200])
    num_seeds = 10
    lambda_val = 0.01

    t_e, all_states = true_expectation(L, d, beta)
    results = run_multiple_seeds(f_single, n_vals, lambda_val, d, L, num_seeds, beta, t_e, all_states)

    print("\n=== Final Averaged Results ===")
    print("n =", n_vals)
    print("True value:", results["true_val"])
    print("BMC mean error:", results["bmc_mean_error"])
    print("MC  mean error:", results["mc_mean_error"])
    print("Avg runtime per n:", results["times_mean"])


if __name__ == "__main__":
    main()







