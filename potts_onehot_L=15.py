import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.linalg import cho_factor, cho_solve
from itertools import product
from functools import partial
import matplotlib.pyplot as plt
import jax.random as jrand
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

@partial(jit, static_argnums=(0, 1))
def J_chain(d, q, alpha, gamma):
    M = jnp.full((q, q), gamma, dtype=jnp.float64).at[jnp.arange(q), jnp.arange(q)].set(alpha)
    J = jnp.zeros((d, d, q, q), dtype=jnp.float64)
    idx = jnp.arange(d - 1)
    J = J.at[idx,   idx+1, :, :].set(M)
    J = J.at[idx+1, idx,   :, :].set(M)
    return 0.1 * J

def J_mask(d):

    M = jnp.zeros((d, d))
    idx = jnp.arange(d - 1)
    M = M.at[idx, idx + 1].set(1.0)
    M = M.at[idx + 1, idx].set(1.0)

    return M[:, :, None, None]

@jit
def energy(x, J):
    return -0.5 * (jnp.einsum('ijab,ia,jb->', J, x, x))
@jit
def f_single(x, J, beta):
    return jnp.exp(-beta * energy(x, J))

@jit
def f_batch(X, J, beta):
    return vmap(lambda xi: f_single(xi, J, beta))(X)


def kernel_embedding(lambda_, d, q):
    return ((1.0 / q) + ((q - 1.0) / q) * jnp.exp(-lambda_)) ** d

@jit
def double_integral(lambda_, d, q):
    return kernel_embedding(lambda_, d, q)

@jit
def kernel_vec(x, X, lambda_):
    d = x.shape[0]
    dot = jnp.einsum('ndq,dq->n', X, x)  
    return jnp.exp(lambda_ * (dot - d))

@jit
def gram_matrix(X, lambda_):
    n, d, q = X.shape
    X_flat = X.reshape(n, d * q)
    dot = X_flat @ X_flat.T
    return jnp.exp(-lambda_ * d + lambda_ * dot)

# ---------Bayesian Cubature-----------------------------------------------------------------
def precompute_bc_weights(X, lambda_, d , q):
    K = gram_matrix(X, lambda_) + 1e-8 * jnp.eye(X.shape[0], dtype=jnp.float64)
    Lc, lower = cho_factor(K, lower=True)
    z = jnp.full((X.shape[0], 1), kernel_embedding(lambda_, d, q), dtype=jnp.float64)
    w = cho_solve((Lc, lower), z)  
    return w 

@jit
def bayesian_cubature(X, f_vals, lambda_, d, q):
    n = len(X)
    K = gram_matrix(X, lambda_) + 1e-8 * jnp.eye(n)  # match jitter
    L, lower = cho_factor(K)
    z = jnp.full((n, 1), kernel_embedding(lambda_, d, q))
    y = f_vals.reshape(n, 1)
    K_inv_z = cho_solve((L, lower), z)
    K_inv_y = cho_solve((L, lower), y)
    mean = (z.T @ K_inv_y)[0, 0]
    var  = double_integral(lambda_, d, q) - (z.T @ K_inv_z)[0, 0]
    return mean, var

# --------Bayesian Optimisation ------------------------------------------------------------------
# @jit
# def woodbury_inverse(K_inv, inv_ones, X_obs, x_new, lambda_, jitter=1e-8):
#     kx = kernel_vec(x_new, X_obs, lambda_)      
#     u  = K_inv @ kx                             
#     s  = jnp.maximum(1.0 + jitter - jnp.dot(kx, u), 1e-14)  

#     u_col = u[:, None]                          

#     TL = K_inv + (u_col @ u_col.T) / s          
#     TR = -u_col / s                             
#     BL = TR.T                                   
#     BR = jnp.array([[1.0 / s]], dtype=K_inv.dtype)  

#     K_inv_new = jnp.block([[TL, TR],
#                            [BL, BR]])

#     inv_ones_new = K_inv_new @ jnp.ones((K_inv_new.shape[0], 1), dtype=K_inv.dtype)
#     return K_inv_new, inv_ones_new


# @jit
# def candidate_gain(x, X_obs, K_inv, inv_ones, lambda_, z0, jitter=1e-8):
#     kx  = kernel_vec(x, X_obs, lambda_)         
#     u   = K_inv @ kx                            
#     s  = jnp.maximum(1.0 - jnp.dot(kx, u) + jitter, 1e-14)
#     qx  = jnp.vdot(inv_ones.ravel(), kx)        
#     eta = z0 * (1.0 - qx)
#     return (eta * eta) / s


# @jit
# def compute_max_variance_using_inverse(X_obs, K_inv, inv_ones, candidates, lambda_, d, q, jitter=1e-8):
#     z0 = kernel_embedding(lambda_, d, q)
#     gains = vmap(lambda x: candidate_gain(x, X_obs, K_inv, inv_ones, lambda_, z0, jitter))(candidates)
#     return candidates[jnp.argmax(gains)]

# --------------------------------------------------------------------------
@partial(jit, static_argnums=(1,2))
def one_hot_from_ints(ints, d, q):
    digits = q ** jnp.arange(d - 1, -1, -1, dtype=jnp.int64)
    X_int = ((ints[:, None] // digits) % q).astype(jnp.int32)  # (n,d)
    return jax.nn.one_hot(X_int, q, dtype=jnp.float64)     

def sample_uniform(key, n, d, q):
    ints = jrand.randint(key, (int(n),), 0, q**d, dtype=jnp.int64)
    return one_hot_from_ints(ints, d, q)

def true_expectation(q, d, beta, J, batch, return_logZ):
    total = int(q ** d)
    s = 0.0

    for start in range(0, total, batch):
        end = min(start + batch, total)
        idxs = jnp.arange(start, end, dtype=jnp.int64)     
        X = one_hot_from_ints(idxs, d, q)                  
        w = f_batch(X, J, beta)                   
        s += jnp.sum(w)

    if return_logZ:
        return jnp.log(s)                                  
    else:
        return s / total   

# -------Russian Roulette---------------------------------------------------------
# def sample_onehot_single(key, d, q):
#     idx = jrand.randint(key, (), 0, q**d, dtype=jnp.int64)
#     return one_hot_from_ints(idx[None], d, q)[0]

# def rr_debiased_mc_estimator_linear(key, J, h, beta, d, q, rho=0.95, max_steps=1_000_000):
#     m = 0
#     qm = 1.0
#     total = 0.0
#     used = 0
#     sum_f = 0.0
#     S_prev = 0.0
#     subkey = key

#     subkey, sk = jrand.split(subkey)
#     x = sample_onehot_single(sk, d, q)                 
#     fx = f_single(x, J, h, beta)
#     sum_f += fx
#     S_m = sum_f / 1.0
#     total += (S_m - S_prev) / qm
#     S_prev = S_m
#     used = 1

#     while m < max_steps:
#         subkey, sk = jrand.split(subkey)
#         cont = jrand.bernoulli(sk, p=jnp.asarray(rho))
#         if not bool(cont):
#             break
#         m += 1
#         qm *= rho

#         subkey, sk = jrand.split(subkey)
#         x = sample_onehot_single(sk, d, q)             
#         fx = f_single(x, J, h, beta)
#         sum_f += fx
#         S_m = sum_f / (m + 1)
#         total += (S_m - S_prev) / qm
#         S_prev = S_m
#         used += 1

#     return float(total), used

# # ----Stratified Sampling (4 Strata)-----------------------------------------------

# def _four_strata_bounds(d: int, q: int):
#     N = int(q ** d)
#     base = N // 4
#     rem  = N - 4 * base  # remainder 0..3

#     lengths = [base + (1 if j < rem else 0) for j in range(4)]
#     bounds  = []
#     start   = 0
#     for L in lengths:
#         end = start + L
#         bounds.append((jnp.int64(start), jnp.int64(end)))
#         start = end

#     probs = jnp.array([L / N for L in lengths], dtype=jnp.float64)  # π_j
#     return bounds, probs


# def stratified_mc_estimator_4(key, n: int, d: int, q: int, J, h, beta):
#     if n <= 0:
#         return 0.0, 0
#     if n % 4 != 0:
#         raise ValueError("Here we assume n is a multiple of 4.")

#     bounds, probs = _four_strata_bounds(d, q)  # π_j for each stratum
#     n_per = n // 4

#     mu_sum = 0.0
#     used   = 0
#     subkey = key
#     for j in range(4):
#         low, high = bounds[j]                         
#         subkey, kj = jrand.split(subkey)
#         idxs = jrand.randint(kj, (n_per,), low, high, dtype=jnp.int64)
#         Xj = one_hot_from_ints(idxs, d, q)           
#         fj = f_batch(Xj, J, h, beta)                 
#         mu_sum = mu_sum + probs[j] * jnp.mean(fj)
#         used   += n_per

#     return mu_sum, used

# --------------------------------------------------------------------------
# def run_experiment(n_vals, lambda_, d, q, seed, beta, J, h, t_true, experiment_type, sub_sample):
#     key = jax.random.PRNGKey(seed)
#     est_means, times = [], []

#     for n in n_vals:
#         start = time.time()
#         key, key_bo, key_random = jax.random.split(key, 3)

#         if experiment_type == "bq_bo":
#             x0 = sample_uniform(key_bo, int(n), d, q)[0]
#             X = [x0.astype(jnp.int32)]

#             K_inv    = jnp.array([[1.0]], dtype=jnp.float64)
#             inv_ones = jnp.array([[1.0]], dtype=jnp.float64)

#             for _ in range(1, n):
#                 key_bo, kc = jrand.split(key_bo)
#                 cand = sample_uniform(kc, sub_sample, d, q)
#                 X_obs = jnp.stack(X, axis=0)    
#                 x_star = compute_max_variance_using_inverse(jnp.array(X_obs), K_inv, inv_ones, cand, lambda_, d, q)
#                 K_inv, inv_ones = woodbury_inverse(K_inv, inv_ones, jnp.array(X_obs), x_star, lambda_)
#                 X.append(x_star.astype(jnp.int32))
            
#             X_obs = jnp.stack(X, axis=0)    
#             f_vals = f_batch(X_obs, J, h, beta).reshape(-1, 1)
#             w = precompute_bc_weights(X_obs, lambda_, d, q)
#             est_mean = (w.T @ f_vals)[0, 0]
#         else:

#             if experiment_type == "bq_random":
#                 X = sample_uniform(key_random, int(n), d, q)
#                 f_vals = f_batch(X, J, h, beta).reshape(-1, 1)     # (n,1)
#                 w = precompute_bc_weights(X, lambda_, d, q)     # (n,1)
#                 est_mean = (w.T @ f_vals)[0, 0]

#             elif experiment_type == "mc":
#                 X = sample_uniform(key_random, int(n), d, q)
#                 f_vals = f_batch(X, J, h, beta)                    # (n,)
#                 est_mean = jnp.mean(f_vals)

#             elif experiment_type == "rr":
#                 rho = float(jnp.clip(1.0 - 1.0 / jnp.maximum(1, n), 0.0, 0.999999))
#                 est_rr, used = rr_debiased_mc_estimator_linear(key_random, J, h, beta, d, q, rho=rho)
#                 est_mean = jnp.asarray(est_rr)

#             elif experiment_type == "strat":
#                 mu_hat, used = stratified_mc_estimator_4(key_random, int(n), d, q, J, h, beta)
#                 est_mean = jnp.asarray(mu_hat)

#         jax.block_until_ready(est_mean)
#         elapsed = time.time() - start
#         est_means.append(est_mean)
#         times.append(elapsed)
#         # print(f"n={int(n):5d} | {experiment_type.upper():9s} | time {elapsed:.3f}s | err={float(jnp.abs(est_mean - t_true)):.6e}")

#     return {"true_val": t_true * (q**d), "est_means": jnp.array(est_means) * (q**d), "times": jnp.array(times)}

# def run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_true, experiment_type, sub_sample):
#     abs_errors = []
#     for seed in range(num_seeds):
#         print(f"\n--- {experiment_type.upper()} | Seed {seed} ---")
#         res = run_experiment(n_vals, lambda_, d, q, seed, beta, J, h, t_true, experiment_type, sub_sample)
#         abs_errors.append(jnp.abs(res["est_means"] - res["true_val"])/res["true_val"])

#     abs_errors = jnp.stack(abs_errors)
#     mean_abs_error = jnp.mean(abs_errors, axis=0) 
#     se_abs_error = scipy.stats.sem(abs_errors, axis=0)

#     return {"true_val": t_true, "abs_errors": abs_errors, "mean_abs_error": mean_abs_error, "se_abs_error": se_abs_error}

# def plot_results(n_vals, all_results, save_path="results"):
#     os.makedirs(save_path, exist_ok=True)
#     styles = {
#         'MC':         {'color': 'k', 'marker': 'o', 'label': 'MC'},
#         'DBQ':        {'color': 'b', 'marker': 's', 'label': 'DBQ'},
#         'Active DBQ': {'color': 'g', 'marker': '^', 'label': 'Active DBQ'},
#         'RR': {'color': 'r', 'marker': '^', 'label': 'RR'},
#         'STRAT': {'color': 'c', 'marker': '^', 'label': 'STRAT'},
#     }

#     plt.figure(figsize=(10, 6))
#     handles, labels = [], []

#     for name in [m for m in styles.keys() if m in all_results]:
#         data = all_results[name]

#         mean_err = np.asarray(data["mean_abs_error"])
#         se_err   = np.asarray(data["se_abs_error"])

#         # for log y-scale, keep positive lower bound
#         eps = 1e-12
#         y_low  = np.clip(mean_err - 1.96 * se_err, eps, None)
#         y_high = np.clip(mean_err + 1.96 * se_err, eps, None)
#         y_line = np.clip(mean_err, eps, None)

#         st = styles[name]
#         (ln,) = plt.plot(n_vals, y_line, linestyle='-', color=st['color'],
#                          marker=st['marker'], label=st['label'])
#         plt.fill_between(n_vals, y_low, y_high, color=st['color'], alpha=0.15)
#         handles.append(ln)
#         labels.append(st['label'])

#     plt.xlabel("Number of Points")
#     plt.title("Potts Model")
#     plt.ylabel("Normalised Absolute Error")
#     plt.yscale('log') # Errors are best viewed on a log scale
#     plt.grid(True, which='major', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, "Potts_abs_err.pdf"), format='pdf')
#     # plt.show()
#     plt.close()

#     fig_legend, ax_legend = plt.subplots(figsize=(6, 1))  # adjust size as needed
#     ax_legend.axis('off')  # no axes
#     ax_legend.legend(handles, labels, ncol=5, loc='center', fontsize=26, frameon=False)

#     plt.savefig(os.path.join(save_path, "Potts_abs_err_legend.pdf"), bbox_inches='tight')
#     plt.close(fig_legend)

# # --------------------------------------------------------------------------
# def lengthscale_ablation(d, q, n_ablation, lam_start, lam_end, lam_num, seeds, t_e, J, h, beta, key_seed, experiment_type, sub_sample):
#     lam_grid = jnp.geomspace(lam_start, lam_end, lam_num)
#     errs_seeds, sds_seeds = [], []
#     key = jax.random.PRNGKey(key_seed)

#     for s in range(seeds):

#         if experiment_type == "bq_random":
#             key, subkey = jax.random.split(key)
#             X = sample_uniform(subkey, int(n), d, q)
#             y = f_batch(X, J, beta)

#             err_list, sd_list = [], []

#             for lam in lam_grid:
#                 mu, var = bayesian_cubature(X, y, float(lam), d)
#                 sd = jnp.sqrt(jnp.maximum(var, 0.0))
#                 err_list.append(mu - t_e)
#                 sd_list.append(sd)

#         elif experiment_type == "bq_bo":
#             key, subkey = jax.random.split(key)
#             X = jnp.zeros((n_ablation, d))
#             y = jnp.zeros((n_ablation,))
#             x0 = sample_states(subkey, d, 1)[0]
#             X = X.at[0].set(x0)
#             y = y.at[0].set(f_single(x0, J, beta))

#             err_list, sd_list = [], []

#             for lam in lam_grid:
#                 for i in range(1, n_ablation):
#                     key, subkey = jax.random.split(key)
#                     candidates = sample_states(subkey, d, sub_sample or 1000) ## We are using subsampling trick in case d >= 16
#                     x_next = compute_max_variance(X[:i], candidates, lam, d)
#                     y_next = f_single(x_next, J, beta)
#                     X = X.at[i].set(x_next)
#                     y = y.at[i].set(y_next)

#                 mu, var = bayesian_cubature(X, y, float(lam), d)
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

def calibration(n_calibration, seeds, lambda_, d, q, beta, t_e, J, key_seed):
    mus, vars_ = [], []
    key = jax.random.PRNGKey(key_seed)

    for s in range(seeds):
        key, subkey = jax.random.split(key)
        X = sample_uniform(subkey, int(n_calibration), d, q)
        y = f_batch(X, J, beta)
        mu, var = bayesian_cubature(X, y, lambda_, d, q)
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
# --------------------
# Main
# --------------------
def main():
    global beta
    k_b = 1
    T_c = 2.269
    beta = 1/2.269
    lambda_ = 0.01
    d = 15
    q = 3
    n_vals = jnp.array([40, 100, 200, 400, 1000]) 
    num_seeds = 10
    sub_sample = 2000

    # key = jax.random.PRNGKey(0)
    # key, kh, kJ = jax.random.split(key, 3)

    # h = jax.random.normal(kh, (d, q), dtype=jnp.float64) * 0.01

    # J = jax.random.normal(kJ, (d, d, q, q), dtype=jnp.float64) * 0.01 
    # Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    # mask = J_mask(d)
    # J = Jsym * mask
    alpha = 1.0
    gamma = 0.0

    J = J_chain(d, q, alpha, gamma)
    
    t_e = true_expectation(q, d, beta, J, batch=200_000, return_logZ = False)
    # t_e = 1.023081982216493
    print("True Z:", t_e * (q**d))

    # results = {
    #     "Active DBQ":        run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "bq_bo", sub_sample),
    #     "DBQ":        run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "bq_random", sub_sample),
    #     "MC":         run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "mc", sub_sample),
    #     "RR":         run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "rr", sub_sample),
    #     "STRAT":       run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "strat",  sub_sample),
    # }

    # print("\nMean absolute errors:")
    # for k, v in results.items():
    #     print(k, np.asarray(v["mean_abs_error"]))

    # plot_results(n_vals, results)

    calibrations = {}
    calibration_seeds = 200
    n_calibration = 60
    key_seed = 5
    t_true, C_nom, emp_cov, mus, vars_ = calibration(n_calibration, calibration_seeds, lambda_, d, q, beta, t_e, J, key_seed)

    jnp.savez("results/potts_calibration_results.npz",
             t_true=t_true, C_nom=jnp.array(C_nom),
             emp_cov=jnp.array(emp_cov),
             mus=jnp.array(mus), vars=jnp.array(vars_))
    print("potts_calibration.npz")


if __name__ == "__main__":
    main()

