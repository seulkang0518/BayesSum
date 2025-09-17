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
jax.config.update("jax_enable_x64", True)

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
def woodbury_inverse(K_inv, k_new, k_nn):
    k_new = k_new.reshape(-1, 1)
    denom = k_nn - (k_new.T @ K_inv @ k_new)[0, 0]
    denom = jnp.maximum(denom, 1e-10)
    factor = K_inv @ k_new
    top_left = K_inv + (factor @ factor.T) / denom
    top_right = -factor / denom
    bottom_left = top_right.T
    bottom_right = jnp.array([[1. / denom]])
    return jnp.block([[top_left, top_right], [bottom_left, bottom_right]])

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


@partial(jit, static_argnums=(3, 4))
def compute_max_variance(X_obs, y_obs_unused, candidates, lambda_, d, K_inv):
    X_rescaled = 2 * X_obs - 1.
    candidates_rescaled = 2 * candidates - 1.
    inner = candidates_rescaled @ X_rescaled.T
    K_x = jnp.exp(-lambda_ * 0.5 * (d - inner))  # (m, n)

    K_inv_K_xT = K_x @ K_inv.T
    k_xx = jnp.ones(K_x.shape[0])  # unit normed kernel
    sigma2 = k_xx - jnp.sum(K_x * K_inv_K_xT, axis=1)
    sigma2 = jnp.maximum(sigma2, 1e-10)

    return candidates[jnp.argmax(sigma2)]

@partial(jit, static_argnums=0)
def all_states_cached(d):
    n_states = 2**d
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

# --------------------
# Lengthscale ablation
# --------------------
def run_lengthscale_ablation(
    L, n_design, lam_start, lam_end, lam_num, seeds, lambda_ref, t_e, all_states, J, beta,
    sub_sample=2000
):
    d = L * L
    lam_grid = jnp.geomspace(lam_start, lam_end, lam_num)
    errs_seeds, sds_seeds = [], []
    key = jax.random.PRNGKey(0)

    for s in range(seeds):
        key, sk = jax.random.split(key)

        if sub_sample and sub_sample < all_states.shape[0]:
            key, sk_fixed = jax.random.split(key)
            idx_fixed = jax.random.choice(sk_fixed, all_states.shape[0], shape=(sub_sample,), replace=False)
            candidates_fixed = all_states[idx_fixed]
        else:
            candidates_fixed = all_states

        idx0 = jax.random.choice(sk, all_states.shape[0])
        x0 = all_states[idx0]
        X_list = [x0]
        y_list = [f_single(x0, J, beta)]
        K_inv = jnp.array([[1.0 / (1.0 + 1e-4)]])

        while len(X_list) < n_design:
            X_arr = jnp.array(X_list)
            y_arr = jnp.array(y_list).reshape(-1, 1)
            x_next = compute_max_variance(X_arr, y_arr, candidates_fixed, lambda_ref, d, K_inv)
            y_next = f_single(x_next, J, beta)
            X_list.append(x_next); y_list.append(y_next)

            k_x  = kernel_vec(x_next, X_arr,        lambda_ref, d).squeeze()
            k_xx = kernel_vec(x_next, x_next[None], lambda_ref, d).item() + 1e-4
            K_inv = woodbury_inverse(K_inv, k_x, k_xx)

        X_obs = jnp.array(X_list)
        y_obs = jnp.array(y_list)

        err_list, sd_list = [], []
        for lam in lam_grid:
            mu, var = bayesian_cubature(X_obs, y_obs, float(lam), d)
            sd = jnp.sqrt(jnp.maximum(var, 0.0))
            err_list.append(mu - t_e)
            sd_list.append(sd)
        errs_seeds.append(jnp.stack(err_list))
        sds_seeds.append(jnp.stack(sd_list))

    ERR = jnp.stack(errs_seeds)    # (seeds, lam_num)
    SD  = jnp.stack(sds_seeds)     # (seeds, lam_num)

    return {
        "lambda": lam_grid,
        "rmse": jnp.sqrt(jnp.mean(ERR**2, axis=0)),
        "mae":  jnp.mean(jnp.abs(ERR), axis=0),
        "avg_sd": jnp.mean(SD, axis=0),
        "true": t_e,
    }

# --------------------
# Calibration (Fig. 4 style)
# --------------------
def calibrate_safe(ground_truth, CBQ_mean, CBQ_std, step=0.05, eps=1e-12):
    """
    Confidence (x) vs empirical coverage (y), including 1.0.
    ground_truth, CBQ_mean, CBQ_std are vectors of same shape (S,)
    where each entry is one run/seed.
    """
    conf = jnp.arange(0.0, 1.0 + step/2, step)   # includes 1.0
    z_score = norm.ppf(1.0 - (1.0 - conf) / 2.0) # two-tailed; last entry = +inf

    # guard stds
    CBQ_std = jnp.where(jnp.isfinite(CBQ_std) & (CBQ_std > eps), CBQ_std, eps)
    abs_err = jnp.abs(ground_truth - CBQ_mean)

    coverage = []
    for z in z_score:
        covered = abs_err < (z * CBQ_std)  # (S,)
        coverage.append(covered.mean())
    return conf, jnp.array(coverage)

def collect_cbq_over_seeds(n_design, lambda_ref, lambda_eval, seeds, all_states, J, beta, d, sub_sample=2000):
    """
    Build a size-n_design design per seed (using lambda_ref),
    evaluate BQ at lambda_eval, return arrays of means/stds.
    """
    means, stds = [], []
    key = jax.random.PRNGKey(12345)

    for _ in range(seeds):
        key, sk = jax.random.split(key)

        # candidate pool
        if sub_sample and sub_sample < all_states.shape[0]:
            key, sk_fixed = jax.random.split(key)
            idx_fixed = jax.random.choice(sk_fixed, all_states.shape[0], shape=(sub_sample,), replace=False)
            candidates = all_states[idx_fixed]
        else:
            candidates = all_states

        # start design from candidates
        idx0 = jax.random.choice(sk, candidates.shape[0])
        x0   = candidates[idx0]
        X_list = [x0]
        y_list = [f_single(x0, J, beta)]
        K_inv = jnp.array([[1.0 / (1.0 + 1e-4)]])

        # greedy variance design at lambda_ref
        while len(X_list) < n_design:
            X_arr = jnp.array(X_list)
            y_arr = jnp.array(y_list).reshape(-1, 1)
            x_next = compute_max_variance(X_arr, y_arr, candidates, lambda_ref, d, K_inv)
            y_next = f_single(x_next, J, beta)
            X_list.append(x_next); y_list.append(y_next)
            k_x  = kernel_vec(x_next, X_arr, lambda_ref, d).squeeze()
            k_xx = kernel_vec(x_next, x_next[None], lambda_ref, d).item() + 1e-4
            K_inv = woodbury_inverse(K_inv, k_x, k_xx)

        X_obs = jnp.array(X_list)
        y_obs = jnp.array(y_list)

        mu, var = bayesian_cubature(X_obs, y_obs, lambda_eval, d)
        sd = jnp.sqrt(jnp.maximum(var, 0.0))

        # robust guards (avoid NaN/Inf in downstream calibration)
        mu = jnp.where(jnp.isfinite(mu), mu, 0.0)
        sd = jnp.where(jnp.isfinite(sd) & (sd > 0.0), sd, 0.0)

        means.append(float(mu))
        stds.append(float(sd))

    return jnp.array(means), jnp.array(stds)

def plot_multi_calibration(conf_cov_list, labels, save_path="results/calibration_multi.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6.5, 6.0))
    for (conf, cov), lab in zip(conf_cov_list, labels):
        plt.plot(conf, cov, marker="o", label=lab)
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Confidence level (α)")
    plt.ylabel("Empirical coverage")
    plt.title("Uncertainty Calibration (growing N)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --------------------
# Main
# --------------------
def main():
    # physics
    k_b = 1.0
    T_c = 2.269
    beta = 1.0 / (k_b * T_c)

    # problem size
    L = 4
    d = L * L

    # ground truth by enumeration
    t_e, all_states = true_expectation(L, d, beta)
    J = J_matrix(L)

    # --------------------
    # (Optional) run ablation to pick lambda_eval
    # --------------------
    n_design_ablation = 120
    lam_start, lam_end, lam_num = 0.005, 0.2, 15
    lambda_ref = 0.2  # used for design selection
    seeds_ablation = 10

    res = run_lengthscale_ablation(
        L, n_design_ablation, lam_start, lam_end, lam_num, seeds_ablation,
        lambda_ref, t_e, all_states, J, beta, sub_sample=2000
    )
    print("lambda\tRMSE\tMAE\tavg_sd")
    for i in range(len(res["lambda"])):
        print(f"{float(res['lambda'][i]):.4g}\t{float(res['rmse'][i]):.3e}\t"
              f"{float(res['mae'][i]):.3e}\t{float(res['avg_sd'][i]):.3e}")

    best_idx = int(jnp.argmin(res["mae"]))
    lambda_eval = float(res["lambda"][best_idx])
    print("Using lambda_eval from ablation:", lambda_eval)

    # --------------------
    # Fig-4 style calibration: multiple N curves
    # --------------------
    seeds = 200          # bump this up to smooth the staircase
    Ns = [20, 60, 120]   # growing design sizes (curves)
    conf_cov_list = []
    labels = []

    for n_design in Ns:
        print(f"\n[Calibration] N={n_design}, seeds={seeds}")
        cbq_means, cbq_stds = collect_cbq_over_seeds(
            n_design=n_design,
            lambda_ref=lambda_ref,
            lambda_eval=lambda_eval,
            seeds=seeds,
            all_states=all_states,
            J=J,
            beta=beta,
            d=d,
            sub_sample=2000
        )
        gt = jnp.full_like(cbq_means, t_e)
        conf, cov = calibrate_safe(gt, cbq_means, cbq_stds, step=0.05)
        conf_cov_list.append((conf, cov))
        labels.append(f"N={n_design}")

        print(f"  ㅈ1§coverage at 68%: {float(cov[jnp.argmin(jnp.abs(conf-0.68))]):.3f} | "
              f"at 95%: {float(cov[jnp.argmin(jnp.abs(conf-0.95))]):.3f}")

    os.makedirs("results", exist_ok=True)
    plot_multi_calibration(conf_cov_list, labels, save_path="results/calibration_multi.png")
    print("Saved multi-curve calibration to results/calibration_multi.png")

if __name__ == "__main__":
    main()

