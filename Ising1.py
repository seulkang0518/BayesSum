import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.linalg import cho_factor, cho_solve
from itertools import product
from functools import partial
import matplotlib.pyplot as plt
from jax import lax
from jax.scipy.special import erfinv
import scipy.stats 

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 26
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
# plt.rc('font', family='Arial', size=12)
plt.rc('axes', titlesize=26, labelsize=26, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=26, frameon=False)
plt.rc('xtick', labelsize=26, direction='in')
plt.rc('ytick', labelsize=26, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

import time
from functools import lru_cache
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches

jax.config.update("jax_enable_x64", True)

## done
def J_chain(d):
    J = jnp.zeros((d, d), dtype=jnp.float64)
    idx = jnp.arange(d - 1)
    J = J.at[idx,   idx + 1].set(1)
    J = J.at[idx+1, idx     ].set(1)
    return J

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
    K = gram_matrix(X, lambda_, d) + 1e-3 * jnp.eye(n)  # match jitter
    L, lower = cho_factor(K)
    z = jnp.full((n, 1), kernel_embedding(lambda_, d))
    y = f_vals.reshape(n, 1)
    K_inv_z = cho_solve((L, lower), z)
    K_inv_y = cho_solve((L, lower), y)
    mean = (z.T @ K_inv_y)[0, 0]
    var  = double_integral(lambda_, d) - (z.T @ K_inv_z)[0, 0]
    return mean, jnp.maximum(var, 1e-12)

@partial(jit, static_argnames=("d",))
def compute_integral_variance(X, lambda_, d, jitter=1e-3):  # bigger default jitter
    n = X.shape[0]
    K = gram_matrix(X, lambda_, d) + jitter * jnp.eye(n)
    L, lower = cho_factor(K, lower=True)
    z0 = kernel_embedding(lambda_, d)
    z  = jnp.full((n, 1), z0)
    K_inv_z = cho_solve((L, lower), z)
    kPP = double_integral(lambda_, d)
    var = kPP - (z.T @ K_inv_z)[0, 0]
    return jnp.maximum(var, 1e-12)  # hard clip

def _filter_duplicates(cands, X_obs):
    # mask out any cand that equals some row in X_obs
    eq = jnp.all(cands[:, None, :] == X_obs[None, :, :], axis=2)  # (m, n)
    keep = ~jnp.any(eq, axis=1)                                  # (m,)
    return cands[keep]

@partial(jit, static_argnames=("d",))
def compute_max_variance(X_obs, candidates, lambda_, d):
    # current variance
    cur = compute_integral_variance(X_obs, lambda_, d)

    # variance after adding each candidate
    def upd(x):
        X_aug = jnp.vstack([X_obs, x])
        return compute_integral_variance(X_aug, lambda_, d)
    updated = vmap(upd)(candidates)                     # (m,)
    gains = cur - updated                               # (m,)

    # mark duplicates (candidate equals some observed x)
    dup = jnp.any(jnp.all(candidates[:, None, :] == X_obs[None, :, :], axis=2), axis=1)  # (m,)
    gains = jnp.where(dup, -jnp.inf, gains)

    # if all are duplicates, just pick the first (or any) candidate
    all_dup = jnp.all(dup)
    best_idx = jnp.argmax(gains)
    fallback_idx = 0
    idx = jnp.where(all_dup, fallback_idx, best_idx)

    return candidates[idx]

# ----------- Unused Utility Functions ----------------

# @partial(jit, static_argnums=(6,))
# def compute_kg(X_obs, y_obs, candidates, lambda_, d, key, num_samples):
#     K = gram_matrix(X_obs, lambda_, d) + 1e-4 * jnp.eye(len(X_obs))
#     z = jnp.full((len(X_obs), 1), kernel_embedding(lambda_, d))
#     L, lower = cho_factor(K)
#     K_inv_y = cho_solve((L, lower), y_obs)
#     K_inv_z = cho_solve((L, lower), z)

#     PiPi_k = kernel_embedding(lambda_, d)
#     current_Z_var = PiPi_k - (z.T @ K_inv_z)[0, 0]

#     def mu_sigma2(x):
#         k_x = kernel_vec(x, X_obs, lambda_, d)
#         mu = (k_x.T @ K_inv_y).squeeze()
#         K_inv_kx = cho_solve((L, lower), k_x)
#         sigma2 = 1. - (k_x.T @ K_inv_kx).squeeze()
#         return mu, jnp.maximum(sigma2, 1e-10)

#     mus, sigma2s = vmap(mu_sigma2)(candidates)
#     sigmas = jnp.sqrt(sigma2s)

#     key, subkey = jax.random.split(key)
#     std_normals = jax.random.normal(subkey, shape=(len(candidates), num_samples))
#     f_samples = mus[:, None] + sigmas[:, None] * std_normals

#     def compute_expected_augmented_var(x, y_samples):
#         def single_aug_var(y_j):
#             X_aug = jnp.vstack([X_obs, x])
#             y_aug = jnp.vstack([y_obs, y_j.reshape(1, 1)])
#             return compute_integral_variance(X_aug, lambda_, d)
#         return jnp.mean(vmap(single_aug_var)(y_samples))

#     expected_updated_vars = vmap(compute_expected_augmented_var)(candidates, f_samples)
#     kg_values = current_Z_var - expected_updated_vars

#     return candidates[jnp.argmax(kg_values)]

# @partial(jit, static_argnums=(5,))
# def compute_ei(X_obs, y_obs, candidates, lambda_, d, num_samples_unused=0):
#     # GP posterior setup
#     K = gram_matrix(X_obs, lambda_, d) + 1e-4 * jnp.eye(len(X_obs))
#     L, lower = cho_factor(K)
#     K_inv_y = cho_solve((L, lower), y_obs)

#     def mu_sigma2(x):
#         k_x = kernel_vec(x, X_obs, lambda_, d)
#         mu = (k_x.T @ K_inv_y).squeeze()
#         K_inv_kx = cho_solve((L, lower), k_x)
#         sigma2 = 1. - (k_x.T @ K_inv_kx).squeeze()
#         return mu, jnp.maximum(sigma2, 1e-10)

#     mus, sigma2s = vmap(mu_sigma2)(candidates)
#     sigmas = jnp.sqrt(sigma2s)

#     y_best = jnp.max(y_obs)

#     z = (mus - y_best) / sigmas
#     ei = (mus - y_best) * norm.cdf(z) + sigmas * norm.pdf(z)
#     return candidates[jnp.argmax(ei)]

# @partial(jit, static_argnums=(4,))
# def compute_ei_bq(X_obs, candidates, lambda_, d, key_unused=None):
#     """
#     Selects the next candidate that maximally reduces Var[Z],
#     i.e., argmax_x Var[Z] - Var[Z | (x, f(x))]
#     """
#     current_var = compute_integral_variance(X_obs, lambda_, d)

#     def reduction(x):
#         X_aug = jnp.vstack([X_obs, x])
#         updated_var = compute_integral_variance(X_aug, lambda_, d)
#         return current_var - updated_var

#     reductions = vmap(reduction)(candidates)
#     return candidates[jnp.argmax(reductions)]

# --------------------------------------------------------------------------
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

def true_expectation(d, beta):
    if d > 30:
        raise ValueError("Too large to enumerate all states.")
    J = J_chain(d)                     # ← was J_matrix(L)
    all_states = all_states_cached(d)
    mean_val = batched_true_expectation(J, beta, all_states)
    return mean_val, all_states

def sample_states(key, d, num_samples):
    return jax.random.bernoulli(key, p=0.5, shape=(num_samples, d)).astype(jnp.float32)

# -------------------------------
# Russian Roulette (Rhee–Glynn) —
# debiased MC over levels
# -------------------------------

def rr_debiased_mc_estimator_linear(key, J, beta, d, rho=0.95, max_steps=1_000_000):
    """
    Unbiased estimator of mu = E[f(X)], X~Unif({0,1}^d)
    Levels m=0,1,... with n_m = m+1 samples, S_m = running mean.
    Survival q(m) = rho^m, continuation c_m = rho.
    Work ~ E[tau+1] = 1/(1-rho).
    """
    m = 0
    qm = 1.0
    total = 0.0
    used = 0
    sum_f = 0.0
    S_prev = 0.0
    subkey = key

    # level 0
    subkey, sk = jax.random.split(subkey)
    x = jax.random.bernoulli(sk, p=0.5, shape=(d,)).astype(jnp.float32)
    fx = float(f_single(x, J, beta))
    sum_f += fx
    S_m = sum_f / 1.0
    total += (S_m - S_prev) / qm
    S_prev = S_m
    used = 1

    # continue?
    while m < max_steps:
        subkey, sk = jax.random.split(subkey)
        cont = jax.random.bernoulli(sk, p=rho)
        if not cont:
            break
        m += 1
        qm *= rho
        # add one new sample at each level
        subkey, sk = jax.random.split(subkey)
        x = jax.random.bernoulli(sk, p=0.5, shape=(d,)).astype(jnp.float32)
        fx = float(f_single(x, J, beta))
        sum_f += fx
        S_m = sum_f / (m + 1)
        total += (S_m - S_prev) / qm
        S_prev = S_m
        used += 1

    return float(total), used  # unbiased estimate and number of samples used

# ------------------------------------------------------------
# Experiment runner — now with experiment_type selection
# experiment_type ∈ {"bq_bo", "bq_random", "mc", "rr"}
# ------------------------------------------------------------

def run_experiment(n_vals, lambda_, d, seed, beta, t_e, experiment_type, sub_sample=None):
    key = jax.random.PRNGKey(seed)
    J = J_chain(d)
    est_means, times = [], []

    for n in n_vals:
        start = time.time()
        key, subkey = jax.random.split(key)

        if experiment_type == "bq_bo":
            # greedy variance design of size n
            X = jnp.zeros((n, d))
            y = jnp.zeros((n,))
            x0 = sample_states(subkey, d, 1)[0]
            X = X.at[0].set(x0)
            y = y.at[0].set(f_single(x0, J, beta))

            for i in range(1, n):
                key, subkey = jax.random.split(key)
                candidates = sample_states(subkey, d, sub_sample or 1000) ## We are using subsampling trick in case d >= 16
                x_next = compute_max_variance(X[:i], candidates, lambda_, d)
                y_next = f_single(x_next, J, beta)
                X = X.at[i].set(x_next)
                y = y.at[i].set(y_next)

            mu_bmc, _ = bayesian_cubature(X, y, lambda_, d)
            est_mean = mu_bmc

        elif experiment_type == "bq_random":
            # random design of size n (without replacement)
            idxs = jax.random.choice(subkey, 2**d, shape=(int(n),), replace=False)
            X_rand = all_states_cached(d)[idxs]
            f_rand = f_batch(X_rand, J, beta)
            mu_bmc, _ = bayesian_cubature(X_rand, f_rand, lambda_, d)
            est_mean = mu_bmc

        elif experiment_type == "mc":
            # vanilla MC with n iid samples
            X_mc = sample_states(subkey, d, int(n))
            f_vals_mc = f_batch(X_mc, J, beta)
            est_mean = jnp.mean(f_vals_mc)

        elif experiment_type == "rr":
            # Russian Roulette debiased MC over levels, budget via rho
            rho = float(jnp.clip(1.0 - 1.0 / jnp.maximum(1, n), 0.0, 0.999999))
            est_rr, used = rr_debiased_mc_estimator_linear(subkey, J, beta, d, rho=rho)
            est_mean = jnp.asarray(est_rr)

        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        jax.block_until_ready(est_mean)
        elapsed = time.time() - start
        times.append(elapsed)
        est_means.append(est_mean)
        print(f"n={int(n)}, Type={experiment_type}, Time={elapsed:.3f}s, "
              f"Err={float(jnp.abs(est_mean - t_e)):.10f}")

    return {"true_val": t_e, "est_means": jnp.array(est_means), "times": jnp.array(times)}

def run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, experiment_type, sub_sample=None):
    abs_errors = []
    for seed in range(num_seeds):
        print(f"\n--- {experiment_type} | Seed {seed} ---")
        result = run_experiment(
            n_vals, lambda_, d, seed, beta, t_e, experiment_type, sub_sample  # <-- no L
        )
        abs_error = jnp.abs(result["est_means"] - result["true_val"])
        abs_errors.append(abs_error)

    all_abs_errors = jnp.stack(abs_errors)         # (seeds, len(n_vals))
    all_squared_errors = all_abs_errors**2
    mean_abs_error = jnp.mean(all_abs_errors, axis=0)
    se_abs_error = scipy.stats.sem(all_abs_errors, axis=0)
    mse  = jnp.mean(all_squared_errors, axis=0)
    rmse = jnp.sqrt(mse)

    return {
        "true_val": t_e,
        "abs_errors": all_abs_errors,
        "mean_abs_error": mean_abs_error,
        "se_abs_error": se_abs_error,
        "rmse": rmse
    }


def plot_results(n_vals, all_results, save_path="results"):
    import os
    os.makedirs(save_path, exist_ok=True)

    styles = {
        'MC':        {'color': 'k', 'marker': 'o', 'label': 'MC'},
        'DBQ':       {'color': 'b', 'marker': 's', 'label': 'DBQ'},
        'Active DBQ':{'color': 'g', 'marker': '^', 'label': 'Active DBQ'},
        # 'RR':        {'color': 'r', 'marker': 'D', 'label': 'RR'},
    }


    plt.figure(figsize=(10, 6))
    handles, labels = [], []

    for name in [m for m in styles.keys() if m in all_results]:
        data = all_results[name]

        mean_err = np.asarray(data["mean_abs_error"])
        se_err   = np.asarray(data["se_abs_error"])

        # for log y-scale, keep positive lower bound
        eps = 1e-12
        y_low  = np.clip(mean_err - 1.96 * se_err, eps, None)
        y_high = np.clip(mean_err + 1.96 * se_err, eps, None)
        y_line = np.clip(mean_err, eps, None)

        st = styles[name]
        (ln,) = plt.plot(n_vals, y_line, linestyle='-', color=st['color'],
                         marker=st['marker'], label=st['label'])
        plt.fill_between(n_vals, y_low, y_high, color=st['color'], alpha=0.15)
        handles.append(ln)
        labels.append(st['label'])

    plt.xlabel("Number of Points")
    plt.title("Ising Model")
    plt.ylabel("Absolute Error")
    plt.yscale('log') # Errors are best viewed on a log scale
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Ising_abs_err.pdf"), format='pdf')
    # plt.show()
    plt.close()

    fig_legend, ax_legend = plt.subplots(figsize=(6, 1))  # adjust size as needed
    ax_legend.axis('off')  # no axes
    ax_legend.legend(handles, labels, ncol=4, loc='center', fontsize=26, frameon=False)

    plt.savefig(os.path.join(save_path, "Ising_abs_err_legend.pdf"), bbox_inches='tight')
    plt.close(fig_legend)

 
def lengthscale_ablation(L, n_ablation, lam_start, lam_end, lam_num, seeds, t_e, J, beta, key_seed, experiment_type, sub_sample):
    d = L * L
    lam_grid = jnp.geomspace(lam_start, lam_end, lam_num)
    errs_seeds, sds_seeds = [], []
    key = jax.random.PRNGKey(key_seed)

    for s in range(seeds):

        if experiment_type == "bq_random":
            key, subkey = jax.random.split(key)
            idxs = jax.random.choice(subkey, 2**d, shape=(int(n_ablation),), replace=False)
            X = all_states_cached(d)[idxs]
            y = f_batch(X, J, beta)

            err_list, sd_list = [], []

            for lam in lam_grid:
                mu, var = bayesian_cubature(X, y, float(lam), d)
                sd = jnp.sqrt(jnp.maximum(var, 0.0))
                err_list.append(mu - t_e)
                sd_list.append(sd)

        elif experiment_type == "bq_bo":
            key, subkey = jax.random.split(key)
            X = jnp.zeros((n_ablation, d))
            y = jnp.zeros((n_ablation,))
            x0 = sample_states(subkey, d, 1)[0]
            X = X.at[0].set(x0)
            y = y.at[0].set(f_single(x0, J, beta))

            err_list, sd_list = [], []

            for lam in lam_grid:
                for i in range(1, n_ablation):
                    key, subkey = jax.random.split(key)
                    candidates = sample_states(subkey, d, sub_sample or 1000) ## We are using subsampling trick in case d >= 16
                    x_next = compute_max_variance(X[:i], candidates, lam, d)
                    y_next = f_single(x_next, J, beta)
                    X = X.at[i].set(x_next)
                    y = y.at[i].set(y_next)

                mu, var = bayesian_cubature(X, y, float(lam), d)
                sd = jnp.sqrt(jnp.maximum(var, 0.0))
                err_list.append(mu - t_e)
                sd_list.append(sd)

        errs_seeds.append(jnp.stack(err_list))  # (lam_num,)
        sds_seeds.append(jnp.stack(sd_list))    # (lam_num,)

    ERR = jnp.stack(errs_seeds)  # (seeds, lam_num)
    SD  = jnp.stack(sds_seeds)   # (seeds, lam_num)

    rmse = jnp.sqrt(jnp.mean(ERR**2, axis=0))
    mae  = jnp.mean(jnp.abs(ERR), axis=0)
    avg_sd = jnp.mean(SD, axis=0)
    best_idx = int(jnp.argmin(rmse))
    lam_star = float(lam_grid[best_idx])

    return {
        "lambda": lam_grid,
        "rmse": rmse,
        "mae": mae,
        "avg_sd": avg_sd,
        "true": t_e,
        "best_idx": best_idx,
        "lambda_star": lam_star,
    }

def calibration_erfin(n, seeds, lam, L, beta, t_e, all_states):
    J = J_matrix(L)
    d = L * L
    mus, vars_ = [], []
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        idx = jax.random.choice(key, all_states.shape[0], shape=(n,), replace=False)
        X = all_states[idx]
        y = f_batch(X, J, beta)
        mu, var = bayesian_cubature(X, y, lam, d)
        mus.append(mu); vars_.append(var)

    mus  = jnp.array(mus)
    vars_ = jnp.maximum(jnp.array(vars_), 1e-30)

    # nominal coverage grid and empirical coverage
    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = jnp.sqrt(2.0) * erfinv(C_nom)  # two-sided CI half-width multiplier
    half = z[:,None] * jnp.sqrt(vars_)[None,:]
    inside = (t_e >= mus[None,:] - half) & (t_e <= mus[None,:] + half)
    emp_cov = jnp.mean(inside, axis=1)

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(np.array(C_nom), np.array(emp_cov), "o-", label=f"N={n}")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal two-sided coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Uncertainty calibration (Poisson, fixed N)")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()

    return float(t_e), C_nom, emp_cov, mus, vars_


def calibration_ppf(n, seeds, lam, L, beta, t_e, all_states):
    J = J_matrix(L)
    d = L * L
    mus, vars_ = [], []
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        idx = jax.random.choice(key, all_states.shape[0], shape=(n,), replace=False)
        X = all_states[idx]
        y = f_batch(X, J, beta)
        mu, var = bayesian_cubature(X, y, lam, d)
        mus.append(mu); vars_.append(var)

    mus  = jnp.array(mus)
    vars_ = jnp.maximum(jnp.array(vars_), 1e-30)

    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = norm.ppf((1.0 + C_nom) / 2.0)  # two-sided CI half-width multiplier
    half = z[:, None] * jnp.sqrt(vars_)[None, :]
    inside = (t_e >= mus[None, :] - half) & (t_e <= mus[None, :] + half)
    emp_cov = jnp.mean(inside, axis=1)

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(np.array(C_nom), np.array(emp_cov), "o-", label=f"N={n}")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal two-sided coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Uncertainty calibration (Poisson, fixed N)")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()

    return float(t_e), C_nom, emp_cov, mus, vars_


def plot_abs_error_boxplots(
    rmses,
    L_list,
    methods=None,
    save_path="ising_abs_error_boxplot.png",
    logy=True,
    showfliers=True,
):

    if methods is None:
        methods = list(rmses.keys())

    # explicitly fix colors
    mcolors = {"Active DBQ": "g", "DBQ": "b", "MC": "k"}

    # fallback for any extra methods not specified
    palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, m in enumerate(methods):
        if m not in mcolors:
            mcolors[m] = palette[i % len(palette)]

    # collect data and positions
    data, positions, colors = [], [], []
    width = 0.35 if len(methods) == 2 else 0.8 / max(1, len(methods))

    for i, L in enumerate(L_list):
        for j, method in enumerate(methods):
            arr = jnp.asarray(rmses[method][L])
            if arr.ndim != 1:
                arr = jnp.ravel(arr)
            data.append(np.array(arr))
            positions.append(i + j * width)
            colors.append(mcolors[method])
    
    # draw
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(
        data,
        positions=positions,
        widths=width * 0.8,
        patch_artist=True,
        manage_ticks=False,
        showfliers=showfliers,
    )

    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.55, edgecolor="black")
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.6)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.0)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)

    centers = [i + (len(methods) - 1) * width / 2 for i in range(len(L_list))]
    plt.xticks(centers, [f"L={L} (d={L*L})" for L in L_list])

    if logy:
        plt.yscale("log")
    plt.ylabel("Absolute error")
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    handles = [mpatches.Patch(color=mcolors[m], label=m) for m in methods]
    plt.legend(handles=handles, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


# --------------------
# Main
# --------------------
def main():
    global beta
    k_b = 1
    T_c = 2.269
    beta = 1 / (k_b * T_c)
    lambda_ = 0.01
    d = 10
    n_vals = jnp.array([10, 50, 100, 200, 300]) 
    num_seeds = 10
    sub_sample = 1000
    

    t_e, all_states = true_expectation(d, beta)

    all_results = {
        "Active DBQ": run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, "bq_bo", sub_sample),
        "DBQ":        run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, "bq_random", sub_sample),
        "MC":         run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, "mc", None),
        # "RR":       run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, "rr", None),
    }

    plot_results(n_vals, all_results)
    

    # lam_start, lam_end, lam_num = 0.005, 0.2, 15
    # seeds_ablation = 10
    # seeds_calibration = 200
    
    # L_list = [3, 4, 5]
    # n_design = 200
    # key_seed = 0

    # n_ablation = 120
    # methods = [("mc", "MC"), ("bq_random", "DBQ"), ("bq_bo", "Active DBQ")]
    # rmses = {label: {} for _, label in methods}
    
    # bq_lambda_stars = {3: 0.053563101097399014, 4: 0.03162277660168381, 5: 0.0050000000000000044}
    # abq_lambda_stars = {3: 0.20000000000000007, 4: 0.09072592945609412, 5: 0.018669568779850945}

    # for L in L_list:
    #     d = L * L
    #     J = J_matrix(L)
    #     t_e, _ = true_expectation(L, d, beta)

    #     # bq_ablation = lengthscale_ablation(L, n_ablation, lam_start, lam_end, lam_num, seeds_ablation, t_e, J, beta, key_seed, "bq_random", None)
    #     # bq_lambda_star = bq_ablation["lambda_star"]
    #     # bq_lambda_stars[L] = bq_lambda_star

    #     # abq_ablation = lengthscale_ablation(L, n_ablation, lam_start, lam_end, lam_num, seeds_ablation, t_e, J, beta, key_seed, "bq_bo", sub_sample)
    #     # abq_lambda_star = abq_ablation["lambda_star"]
    #     # abq_lambda_stars[L] = abq_lambda_star

    #     mc_results = run_multiple_seeds([n_ablation], 0.0, d, L, num_seeds, beta, t_e, "mc", None)
    #     rmses["MC"][L] = mc_results["abs_errors"][:, 0]

    #     dbq_results = run_multiple_seeds([n_ablation], bq_lambda_stars[L], d, L, num_seeds, beta, t_e, "bq_random", sub_sample)
    #     rmses["DBQ"][L] = dbq_results["abs_errors"][:, 0]

    #     abq_results = run_multiple_seeds([n_ablation], abq_lambda_stars[L], d, L, num_seeds, beta, t_e, "bq_bo", sub_sample)
    #     rmses["Active DBQ"][L] = abq_results["abs_errors"][:, 0]

    # # print("\n--- All dimensions processed ---")
    # # print("Final lambda* for DBQ:", bq_lambda_stars)
    # # print("Final lambda* for Active DBQ:", abq_lambda_stars)
    
    # # 3. Plot the final results
    # plot_abs_error_boxplots(
    #     rmses,
    #     L_list,
    #     methods=[label for _, label in methods],
    #     save_path="ising_abs_error_boxplot.pdf",
    #     logy=True,
    #     showfliers=True
    # )

    # # print(rmses)

    # # # Your precomputed RMSE dict
    # # rmses = {
    # #     "MC": {
    # #         3: np.array([0.00230098, 0.00189215, 0.01255649, 0.00576246, 0.009094,
    # #                      0.00075471, 0.00610954, 0.00130999, 0.00086838, 0.00128865]),
    # #         4: np.array([0.00684047, 0.00886482, 0.00465614, 0.0061425, 0.00147098,
    # #                      0.00223464, 0.00231481, 0.00496465, 0.00139517, 0.00363827]),
    # #         5: np.array([0.00156072, 0.00923148, 0.00217, 0.00196546, 0.00826371,
    # #                      0.00135833, 0.00373673, 0.00261497, 0.00569257, 0.00723466])
    # #     },
    # #     "DBQ": {
    # #         3: np.array([1.11432577e-03, 1.95437318e-04, 5.59033455e-04, 5.87187012e-04,
    # #                      1.13779217e-04, 3.10724473e-04, 3.34748650e-04, 7.83192377e-04,
    # #                      4.31693111e-04, 7.27728740e-05]),
    # #         4: np.array([0.00012425, 0.00485936, 0.00407473, 0.00050779, 0.00120971,
    # #                      0.00286374, 0.00168464, 0.00212422, 0.00163115, 0.00324258]),
    # #         5: np.array([0.00100823, 0.00197031, 0.00316909, 0.00132123, 0.00037972,
    # #                      0.00468592, 0.00032563, 0.00613024, 0.00835903, 0.00722658])
    # #     }
    # # }

    # # # Which methods you want to compare
    # # methods = ["MC", "DBQ"]

    # # # Dimensions
    # # L_list = [3, 4, 5]

    # # # Call the plotting function
    # # plot_abs_error_boxplots(
    # #     rmses,
    # #     L_list,
    # #     methods=methods,
    # #     save_path="ising_abs_error_boxplot.pdf",
    # #     logy=True,
    # #     showfliers=True
    # # )

    # # calibrations = {}
    # # calibration_seeds = 200
    # # n_samples = 60
    # # t_true, C_nom, emp_cov, mus, vars_ = calibration_erfin(n_samples, calibration_seeds, lambda_, L, beta, t_e, all_states)
    # # print(t_true)
    # # print(C_nom)
    # # print(emp_cov)
    # # print(mus)
    # # print(vars_)

    # # t_true, C_nom, emp_cov, mus, vars_ = calibration_ppf(n_samples, calibration_seeds, lambda_, L, beta, t_e, all_states)
    # # print(t_true)
    # # print(C_nom)
    # # print(emp_cov)
    # # print(mus)
    # # print(vars_)

    # # jnp.savez("ising_calibration_results.npz",
    # #          t_true=t_true, C_nom=jnp.array(C_nom),
    # #          emp_cov=jnp.array(emp_cov),
    # #          mus=jnp.array(mus), vars=jnp.array(vars_))
    # # print("ising_calibration.npz")

if __name__ == "__main__":
    main()

