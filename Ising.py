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
    J = J.at[idx, idx + 1].set(1)
    J = J.at[idx+1, idx].set(1)
    return 0.1 * J

@jit
def f_single(x, J, beta):
    diff = x[:, None] == x[None, :]
    energy = jnp.sum(J * diff) / 2 ## to avoid double count
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

def kernel_vec(x, X, lambda_):
    X_rescaled = 2 * X - 1
    x_rescaled = 2 * x - 1
    inner = X_rescaled @ x_rescaled
    d = X_rescaled.shape[1]
    return jnp.exp(-0.5 * lambda_ * (d - inner))

@jit
def gram_matrix(X, lambda_, d):
    X_rescaled = 2 * X - 1
    inner = X_rescaled @ X_rescaled.T
    return jnp.exp(-lambda_ * 0.5 * (d - inner))


# ---------Bayesian Cubature-----------------------------------------------------------------
def precompute_bc_weights(X, lambda_, d):
    K = gram_matrix(X, lambda_, d) + 1e-8 * jnp.eye(X.shape[0], dtype=jnp.float64)
    Lc, lower = cho_factor(K, lower=True)
    z = jnp.full((X.shape[0], 1), kernel_embedding(lambda_, d), dtype=jnp.float64)
    w = cho_solve((Lc, lower), z)  
    return w 

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
# --------------------------------------------------------------------------------------------
# @partial(jit, static_argnames=("d",))
# def compute_integral_variance(X, lambda_, d, jitter=1e-3):  # bigger default jitter
#     n = X.shape[0]
#     K = gram_matrix(X, lambda_, d) + jitter * jnp.eye(n)
#     L, lower = cho_factor(K, lower=True)
#     z0 = kernel_embedding(lambda_, d)
#     z  = jnp.full((n, 1), z0)
#     K_inv_z = cho_solve((L, lower), z)
#     kPP = double_integral(lambda_, d)
#     var = kPP - (z.T @ K_inv_z)[0, 0]
#     return jnp.maximum(var, 1e-12)  # hard clip

# def _filter_duplicates(cands, X_obs):
#     # mask out any cand that equals some row in X_obs
#     eq = jnp.all(cands[:, None, :] == X_obs[None, :, :], axis=2)  # (m, n)
#     keep = ~jnp.any(eq, axis=1)                                  # (m,)
#     return cands[keep]

# @partial(jit, static_argnames=("d",))
# def compute_max_variance(X_obs, candidates, lambda_, d):
#     # current variance
#     cur = compute_integral_variance(X_obs, lambda_, d)

#     # variance after adding each candidate
#     def upd(x):
#         X_aug = jnp.vstack([X_obs, x])
#         return compute_integral_variance(X_aug, lambda_, d)
#     updated = vmap(upd)(candidates)                     # (m,)
#     gains = cur - updated                               # (m,)

#     # mark duplicates (candidate equals some observed x)
#     dup = jnp.any(jnp.all(candidates[:, None, :] == X_obs[None, :, :], axis=2), axis=1)  # (m,)
#     gains = jnp.where(dup, -jnp.inf, gains)

#     # if all are duplicates, just pick the first (or any) candidate
#     all_dup = jnp.all(dup)
#     best_idx = jnp.argmax(gains)
#     fallback_idx = 0
#     idx = jnp.where(all_dup, fallback_idx, best_idx)

#     return candidates[idx]


# --------Bayesian Optimisation ------------------------------------------------------------------
@jit
def woodbury_inverse(K_inv, inv_ones, X_obs, x_new, lambda_, jitter=1e-8):
    kx = kernel_vec(x_new, X_obs, lambda_)      
    u  = K_inv @ kx                             
    s  = jnp.maximum(1.0 + jitter - jnp.dot(kx, u), 1e-14)  

    u_col = u[:, None]                          

    TL = K_inv + (u_col @ u_col.T) / s          
    TR = -u_col / s                             
    BL = TR.T                                   
    BR = jnp.array([[1.0 / s]], dtype=K_inv.dtype)  

    K_inv_new = jnp.block([[TL, TR],
                           [BL, BR]])

    inv_ones_new = K_inv_new @ jnp.ones((K_inv_new.shape[0], 1), dtype=K_inv.dtype)
    return K_inv_new, inv_ones_new


@jit
def candidate_gain(x, X_obs, K_inv, inv_ones, lambda_, z0, jitter=1e-8):
    kx  = kernel_vec(x, X_obs, lambda_)         
    u   = K_inv @ kx                            
    s  = jnp.maximum(1.0 - jnp.dot(kx, u) + jitter, 1e-14)
    qx  = jnp.vdot(inv_ones.ravel(), kx)        
    eta = z0 * (1.0 - qx)
    return (eta * eta) / s


@jit
def compute_max_variance_using_inverse(X_obs, K_inv, inv_ones, candidates, lambda_, d, jitter=1e-8):
    z0 = kernel_embedding(lambda_, d)
    gains = vmap(lambda x: candidate_gain(x, X_obs, K_inv, inv_ones, lambda_, z0, jitter))(candidates)
    return candidates[jnp.argmax(gains)]

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
    J = J_chain(d)
    all_states = all_states_cached(d)
    mean_val = batched_true_expectation(J, beta, all_states)
    return mean_val, all_states

def sample_states(key, d, num_samples):
    return jax.random.bernoulli(key, p=0.5, shape=(num_samples, d)).astype(jnp.float32)
# --------------------------------------------------------------------------

def run_experiment(n_vals, lambda_, d, seed, beta, t_e, experiment_type, sub_sample=None):
    key = jax.random.PRNGKey(seed)
    J = J_chain(d)
    est_means, times = [], []

    for n in n_vals:
        start = time.time()
        key, key_bo, key_random = jax.random.split(key, 3)

        if experiment_type == "bq_bo":
            x0 = sample_states(key_bo, d, int(n))[0]
            X = [x0.astype(jnp.int32)]

            K_inv    = jnp.array([[1.0]], dtype=jnp.float64)
            inv_ones = jnp.array([[1.0]], dtype=jnp.float64)

            for i in range(1, n):
                key, subkey = jax.random.split(key)
                cand = sample_states(subkey, d, sub_sample or 1000) ## We are using subsampling trick in case d >= 16
                X_obs = jnp.stack(X, axis=0)    
                x_star = compute_max_variance_using_inverse(jnp.array(X_obs), K_inv, inv_ones, cand, lambda_, d)
                K_inv, inv_ones = woodbury_inverse(K_inv, inv_ones, jnp.array(X_obs), x_star, lambda_)
                X.append(x_star.astype(jnp.int32))

            X_obs = jnp.stack(X, axis=0)    
            f_vals = f_batch(X_obs, J, beta).reshape(-1, 1)
            w = precompute_bc_weights(X_obs, lambda_, d)
            est_mean = (w.T @ f_vals)[0, 0]

        elif experiment_type == "bq_random":
            X = sample_states(subkey, d, int(n))
            f_vals = f_batch(X, J, beta).reshape(-1, 1)
            w = precompute_bc_weights(X, lambda_, d)
            est_mean = (w.T @ f_vals)[0, 0]

        elif experiment_type == "mc":
            X_mc = sample_states(subkey, d, int(n))
            f_vals_mc = f_batch(X_mc, J, beta)
            est_mean = jnp.mean(f_vals_mc)

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


# --------------------------------------------------------------------------
# def lengthscale_ablation(d, n_ablation, lam_start, lam_end, lam_num, seeds, t_e, J, beta, key_seed, experiment_type, sub_sample):
#     lam_grid = jnp.geomspace(lam_start, lam_end, lam_num)
#     errs_seeds, sds_seeds = [], []
#     key = jax.random.PRNGKey(key_seed)

#     for s in range(seeds):

#         if experiment_type == "bq_random":
#             key, subkey = jax.random.split(key)
#             X = sample_states(subkey, d, int(n_ablation))
#             f_vals = f_batch(X, J, beta).reshape(-1, 1)

#             err_list, sd_list = [], []

#             for lam in lam_grid:
#                 mu, var = bayesian_cubature(X, f_vals, lam, d)
#                 sd = jnp.sqrt(jnp.maximum(var, 0.0))
#                 err_list.append(mu - t_e)
#                 sd_list.append(sd)

#         elif experiment_type == "bq_bo":

#             err_list, sd_list = [], []

#             for lam in lam_grid:

#                 key, subkey = jax.random.split(key)
#                 x0 = sample_states(subkey, d, int(n_ablation))[0]
#                 X = [x0.astype(jnp.int32)]

#                 K_inv    = jnp.array([[1.0]], dtype=jnp.float64)
#                 inv_ones = jnp.array([[1.0]], dtype=jnp.float64)

#                 for i in range(1, n_ablation):
#                     key, subkey = jax.random.split(key)
#                     cand = sample_states(subkey, d, sub_sample or 1000) ## We are using subsampling trick in case d >= 16
#                     X_obs = jnp.stack(X, axis=0)    
#                     x_star = compute_max_variance_using_inverse(jnp.array(X_obs), K_inv, inv_ones, cand, lambda_, d)
#                     K_inv, inv_ones = woodbury_inverse(K_inv, inv_ones, jnp.array(X_obs), x_star, lambda_)
#                     X.append(x_star.astype(jnp.int32))

#                 X_obs = jnp.stack(X, axis=0)    
#                 f_vals = f_batch(X_obs, J, beta).reshape(-1, 1)
#                 mu, var = bayesian_cubature(X_obs, f_vals, lam, d)

#                 sd = jnp.sqrt(jnp.maximum(var, 0.0))
#                 err_list.append(mu - t_e)
#                 sd_list.append(sd)

#         errs_seeds.append(jnp.stack(err_list))  # (lam_num,)
#         sds_seeds.append(jnp.stack(sd_list))    # (lam_num,)

#     ERR = jnp.stack(errs_seeds)  # (seeds, lam_num)
#     SD  = jnp.stack(sds_seeds)   # (seeds, lam_num)

#     rmse = jnp.sqrt(jnp.mean(ERR**2, axis=0))
#     mae  = jnp.mean(jnp.abs(ERR), axis=0)
#     avg_sd = jnp.mean(SD, axis=0)
#     best_idx = int(jnp.argmin(rmse))
#     lam_star = float(lam_grid[best_idx])

#     return {
#         "lambda": lam_grid,
#         "rmse": rmse,
#         "mae": mae,
#         "avg_sd": avg_sd,
#         "true": t_e,
#         "best_idx": best_idx,
#         "lambda_star": lam_star,
#     }


def calibration(n_calibration, seeds, lambda_, d, beta, t_e, J, key_seed):
    mus, vars_ = [], []
    key = jax.random.PRNGKey(key_seed)

    for s in range(seeds):
        key, subkey = jax.random.split(key)
        X = sample_states(subkey, d, int(n_calibration))
        y = f_batch(X, J, beta)
        mu, var = bayesian_cubature(X, y, lambda_, d)
        mus.append(mu); vars_.append(var)

    mus  = jnp.array(mus)
    vars_ = jnp.maximum(jnp.array(vars_), 1e-30)

    # nominal coverage grid and empirical coverage
    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = jnp.sqrt(2.0) * erfinv(C_nom)  # two-sided CI half-width multiplier
    half = z[:,None] * jnp.sqrt(vars_)[None,:]
    inside = (t_e >= mus[None,:] - half) & (t_e <= mus[None,:] + half)
    emp_cov = jnp.mean(inside, axis=1)

    # plt.figure(figsize=(6.2, 4.6))
    # plt.plot(np.array(C_nom), np.array(emp_cov), "o-", label=f"N={n}")
    # plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel("Nominal two-sided coverage")
    # plt.ylabel("Empirical coverage")
    # plt.title("Uncertainty calibration (Poisson, fixed N)")
    # plt.legend()
    # plt.grid(True, ls="--", alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return float(t_e), C_nom, emp_cov, mus, vars_


# def plot_abs_error_boxplots(
#     rmses,
#     d_list,
#     methods,
#     save_path="potts_abs_error_boxplot.png",
#     logy = True,
#     showfliers = True):

#     if methods is None:
#         methods = list(rmses.keys())

#     # explicitly fix colors
#     mcolors = {"Active DBQ": "g", "DBQ": "b", "MC": "k"}

#     # fallback for any extra methods not specified
#     palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
#     for i, m in enumerate(methods):
#         if m not in mcolors:
#             mcolors[m] = palette[i % len(palette)]

#     # collect data and positions
#     data, positions, colors = [], [], []
#     width = 0.35 if len(methods) == 2 else 0.8 / max(1, len(methods))

#     for i, L in enumerate(d_list):
#         for j, method in enumerate(methods):
#             arr = jnp.asarray(rmses[method][L])
#             if arr.ndim != 1:
#                 arr = jnp.ravel(arr)
#             data.append(np.array(arr))
#             positions.append(i + j * width)
#             colors.append(mcolors[method])
    
#     # draw
#     plt.figure(figsize=(10, 6))
#     bp = plt.boxplot(
#         data,
#         positions=positions,
#         widths=width * 0.8,
#         patch_artist=True,
#         manage_ticks=False,
#         showfliers=showfliers,
#     )

#     for box, c in zip(bp["boxes"], colors):
#         box.set(facecolor=c, alpha=0.55, edgecolor="black")
#     for med in bp["medians"]:
#         med.set(color="black", linewidth=1.6)
#     for whisker in bp["whiskers"]:
#         whisker.set(color="black", linewidth=1.0)
#     for cap in bp["caps"]:
#         cap.set(color="black", linewidth=1.0)

#     centers = [i + (len(methods) - 1) * width / 2 for i in range(len(d_list))]
#     plt.xticks(centers, [f"d={L}" for L in d_list])

#     if logy:
#         plt.yscale("log")
#     plt.ylabel("Absolute error")
#     plt.grid(True, which="major", linestyle="--", alpha=0.5)

#     handles = [mpatches.Patch(color=mcolors[m], label=m) for m in methods]
#     plt.legend(handles=handles, loc="best")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"Saved {save_path}")


# --------------------
# Main
# --------------------
def main():
    global beta
    k_b = 1
    T_c = 2.269
    beta = 1 / (k_b * T_c)
    lambda_ = 0.01
    d = 15
    n_vals = jnp.array([10, 50, 100, 200, 300, 500]) 
    num_seeds = 10
    sub_sample = 1000
    

    t_e, all_states = true_expectation(d, beta)

    # all_results = {
    #     "Active DBQ": run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, "bq_bo", sub_sample),
    #     "DBQ":        run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, "bq_random", sub_sample),
    #     "MC":         run_multiple_seeds(n_vals, lambda_, d, num_seeds, beta, t_e, "mc", None),
    # }

    # print("\nMean absolute errors:")
    # for k, v in all_results.items():
    #     print(k, np.asarray(v["mean_abs_error"]))

    # plot_results(n_vals, all_results)


    # ## Ablation
    # lam_start, lam_end, lam_num = 0.005, 0.2, 15
    # seeds_ablation = 10
    # seeds_calibration = 200

    # d_list = [8, 12, 15]
    # key_seed = 0

    # n_ablation = 400
    # methods = [("mc", "MC"), ("bq_random", "DBQ"), ("bq_bo", "Active DBQ")]
    # rmses = {label: {} for _, label in methods}

    # bq_lambda_stars = {}
    # abq_lambda_stars = {}

    # for d in d_list:

    #     J = J_chain(d)
    #     t_e = true_expectation(d, beta)

    #     bq_ablation = lengthscale_ablation(d, n_ablation, lam_start, lam_end, lam_num, num_seeds, t_e, J, beta, key_seed, "bq_random", None)
    #     bq_lambda_star = bq_ablation["lambda_star"]
    #     bq_lambda_stars[d] = bq_lambda_star

    #     abq_ablation = lengthscale_ablation(d, n_ablation, lam_start, lam_end, lam_num, num_seeds, t_e, J, h, beta, key_seed, "bq_bo", sub_sample)
    #     abq_lambda_star = abq_ablation["lambda_star"]
    #     abq_lambda_stars[d] = abq_lambda_star

    #     mc_results = run_multiple_seeds([n_ablation], 0.0, d, num_seeds, beta, t_e, "mc", None)
    #     rmses["MC"][d] = mc_results["abs_errors"][:, 0]
    #     # rmses["MC"][d] = jnp.sqrt(jnp.mean(mc_abs_errors**2, axis=0))

    #     dbq_results = run_multiple_seeds([n_ablation], bq_lambda_stars[d], d, num_seeds, beta, t_e, "bq_random", None)
    #     rmses["DBQ"][d] = dbq_results["abs_errors"][:, 0]
    #     # rmses["DBQ"][d] = jnp.sqrt(jnp.mean(dbq_abs_errors**2, axis=0))

    #     abq_results = run_multiple_seeds([n_ablation], abq_lambda_stars[d], d, num_seeds, beta, t_e, "bq_bo", sub_sample)
    #     rmses["Active DBQ"][d] = abq_results["abs_errors"][:, 0]
    #     # rmses["Active DBQ"][d] = jnp.sqrt(jnp.mean(abq_abs_errors**2, axis=0))


    # plot_abs_error_boxplots(
    #     rmses,
    #     d_list,
    #     methods=[label for _, label in methods],
    #     save_path="potts_abs_error_boxplot.pdf",
    #     logy=True,
    #     showfliers=True
    # )


    # Calibration
    calibrations = {}
    calibration_seeds = 200
    n_calibration = 60
    key_seed = 5

    J = J_chain(d)
    t_true, C_nom, emp_cov, mus, vars_ = calibration(n_calibration, calibration_seeds, lambda_, d, beta, t_e, J, key_seed)
    # print(t_true)
    # print(C_nom)
    # print(emp_cov)
    # print(mus)
    # print(vars_)

    jnp.savez("ising_calibration_results.npz",
             t_true=t_true, C_nom=jnp.array(C_nom),
             emp_cov=jnp.array(emp_cov),
             mus=jnp.array(mus), vars=jnp.array(vars_))
    print("ising_calibration.npz")

    

if __name__ == "__main__":
    main()

