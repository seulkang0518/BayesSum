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
@partial(jit, static_argnums=0)
def J_chain(d):
    J = jnp.zeros((d, d), dtype=jnp.float64)
    idx = jnp.arange(d - 1)
    J = J.at[idx, idx + 1].set(1)
    J = J.at[idx+1, idx].set(1)
    return 0.1 * J

def J_mask(d):
    M = jnp.zeros((d, d))
    idx = jnp.arange(d - 1)
    M = M.at[idx, idx + 1].set(1.0)
    M = M.at[idx + 1, idx].set(1.0)
    return M

@jit
def energy(x, J):
    same = (x[:, None] == x[None, :])  # (d,d)
    return -0.5 * jnp.sum(J * same)

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
def kernel_vec(x, X, lam):
    x1 = x.reshape(1, -1)
    hamming = jnp.sum(X != x1, axis=1)
    return jnp.exp(-lam * hamming).reshape(-1, 1)

@jit
def gram_matrix(X, lambda_):
    X1 = X[:, None, :]
    X2 = X[None, :, :]
    hamming = jnp.sum(X1 != X2, axis=2)
    return jnp.exp(-lambda_ * hamming)

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
    
# --------------------------------------------------------------------------
@partial(jit, static_argnums=(0,1))
def all_states(d,q):
    n = q ** d
    idx = jnp.arange(n, dtype=jnp.int64)[:, None]             
    digits = q ** jnp.arange(d-1, -1, -1, dtype=jnp.int64)    
    X_int = ((idx // digits) % q).astype(jnp.int32)           
    return X_int

def true_expectation(q, d, beta, J, batch):
    total = int(q ** d)
    s = 0.0
    cnt = 0
    digits = q ** jnp.arange(d-1, -1, -1, dtype=jnp.int64)

    for start in range(0, total, batch):
        count = min(batch, total - start)
       
        idx = jnp.arange(start, start + count, dtype=jnp.int64)[:, None] 
        X = ((idx // digits) % q).astype(jnp.int32)                   
        f_vals = f_batch(X, J, beta)

        s += jnp.sum(f_vals)
        cnt += count

    return (s / cnt).astype(jnp.float64)
# --------------------------------------------------------------------------
# @partial(jit, static_argnames=("d",))
# def compute_integral_variance(X, lambda_, d, q, jitter=1e-3):  # bigger default jitter
#     n = X.shape[0]
#     K = gram_matrix(X, lambda_) + jitter * jnp.eye(n)
#     L, lower = cho_factor(K, lower=True)
#     z0 = kernel_embedding(lambda_, d, q)
#     z  = jnp.full((n, 1), z0)
#     K_inv_z = cho_solve((L, lower), z)
#     kPP = double_integral(lambda_, d, q)
#     var = kPP - (z.T @ K_inv_z)[0, 0]
#     return jnp.maximum(var, 1e-12)

# @partial(jit, static_argnames=("d",))
# def compute_max_variance(X_obs, candidates, lambda_, d, q):
#     cur = compute_integral_variance(X_obs, lambda_, d, q)

#     def upd(x):
#         X_aug = jnp.vstack([X_obs, x])
#         return compute_integral_variance(X_aug, lambda_, d, q)

#     updated = vmap(upd)(candidates)                     
#     gains = cur - updated                               

#     # mark duplicates (candidate equals some observed x)
#     dup = jnp.any(jnp.all(candidates[:, None, :] == X_obs[None, :, :], axis=2), axis=1)  # (m,)
#     gains = jnp.where(dup, -jnp.inf, gains)

#     all_dup = jnp.all(dup)
#     best_idx = jnp.argmax(gains)
#     fallback_idx = 0
#     idx = jnp.where(all_dup, fallback_idx, best_idx)

#     return candidates[idx]
@jit
def woodbury_inverse(K_inv, inv_ones, X_obs, x_new, lam, jitter=1e-8):
    kx = kernel_vec(x_new, X_obs, lam)      # (n,1)
    u  = K_inv @ kx                         # (n,1)
    s  = jnp.maximum(1.0 + jitter - (kx.T @ u)[0,0], 1e-14)

    TL = K_inv + (u @ u.T) / s
    TR = -u / s
    BL = TR.T
    BR = jnp.array([[1.0 / s]], dtype=K_inv.dtype)

    top = jnp.hstack([TL, TR])
    bot = jnp.hstack([BL, BR])
    K_inv_new = jnp.vstack([top, bot])

    inv_ones_new = K_inv_new @ jnp.ones((K_inv_new.shape[0], 1), dtype=K_inv.dtype)
    return K_inv_new, inv_ones_new

@jit
def candidate_gain(x, X_obs, K_inv, inv_ones, lambda_, z0, jitter=1e-8):
    kx = kernel_vec(x, X_obs, lambda_)
    u  = K_inv @ kx
    s  = jnp.maximum(1.0 + jitter - (kx.T @ u)[0,0], 1e-14)
    qx = (kx.T @ inv_ones)[0,0]                     # k_x^T K^{-1} 1
    eta = z0 * (1.0 - qx)
    return (eta * eta) / s

@jit
def compute_max_variance_using_inverse(X_obs, K_inv, inv_ones, candidates, lam, d, q, jitter=1e-8):
    z0 = kernel_embedding(lam, d, q)
    gains = vmap(lambda x: candidate_gain(x, X_obs, K_inv, inv_ones, lam, z0, jitter))(candidates)
    return candidates[jnp.argmax(gains)]

def sample_states(key, d, num_samples, q):
    return jax.random.randint(key, shape=(num_samples, d), minval=0, maxval=q)
# --------------------------------------------------------------------------


def run_experiment(n_vals, lambda_, d, q, seed, beta, J, t_true, X_all, experiment_type):
    key = jax.random.PRNGKey(seed)
    est_means, times = [], []

    N_all = X_all.shape[0]

    for n in n_vals:
        start = time.time()
        key, subkey = jax.random.split(key)

        if experiment_type == "bq_bo":
            # X = jnp.zeros((n, d))
            # y = jnp.zeros((n,))
            # x0 = sample_states(subkey, d, 1, q)[0]

            # X = X.at[0].set(x0)
            # y = y.at[0].set(f_single(x0, J, beta))
            # K_inv    = jnp.array([[1.0]], dtype=jnp.float64)
            # inv_ones = jnp.array([[1.0]], dtype=jnp.float64)

            # for _ in range(1, n):
            #     key, kc = jrand.split(key)
            #     cand = sample_states(kc, q, d, n)
            #     x_star = compute_max_variance_using_inverse(jnp.array(X), K_inv, inv_ones, cand, lam, d, q, jitter)
            #     K_inv, inv_ones = woodbury_inverse(K_inv, inv_ones, jnp.array(X), x_star, lam, jitter)
            #     f_star = f_single(x_star.astype(jnp.float32), J, beta)
            #     X.append(x_star.astype(jnp.int32))
            #     f_vals.append(f_star.astype(jnp.float64))

            # f_vals = f_batch(X, J, beta).reshape(-1, 1)
            # w = precompute_bc_weights(X, lambda_, d, q)
            # est_mean = (w.T @ f_vals)[0, 0]
            break

        else:

            idxs = jax.random.choice(subkey, N_all, shape=(int(n),), replace=False)
            X = X_all[idxs]

            if experiment_type == "bq_random":
                f_vals = f_batch(X, J, beta).reshape(-1, 1)     # (n,1)
                w = precompute_bc_weights(X, lambda_, d, q)     # (n,1)
                est_mean = (w.T @ f_vals)[0, 0]

            elif experiment_type == "mc":
                f_vals = f_batch(X, J, beta)                    # (n,)
                est_mean = jnp.mean(f_vals)

        jax.block_until_ready(est_mean)
        elapsed = time.time() - start
        est_means.append(est_mean)
        times.append(elapsed)
        print(f"n={int(n):5d} | {experiment_type.upper():9s} | time {elapsed:.3f}s | err={float(jnp.abs(est_mean - t_true)):.6e}")

    return {"true_val": t_true, "est_means": jnp.array(est_means), "times": jnp.array(times)}

def run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, t_true, X_all, experiment_type):
    abs_errors = []
    for seed in range(num_seeds):
        print(f"\n--- {experiment_type.upper()} | Seed {seed} ---")
        res = run_experiment(n_vals, lambda_, d, q, seed, beta, J, t_true, X_all, experiment_type)
        abs_errors.append(jnp.abs(res["est_means"] - t_true))
    abs_errors = jnp.stack(abs_errors)
    mean_abs_error = jnp.mean(abs_errors, axis=0)
    se_abs_error = scipy.stats.sem(np.asarray(abs_errors), axis=0)
    rmse = jnp.sqrt(jnp.mean(abs_errors**2, axis=0))
    return {"true_val": t_true, "abs_errors": abs_errors, "mean_abs_error": mean_abs_error, "se_abs_error": se_abs_error, "rmse": rmse}

def calibration(n_calibration, seeds, lambda_, d, q, beta, X_all, t_e, J, key_seed):
    mus, vars_ = [], []
    key = jax.random.PRNGKey(key_seed)

    for s in range(seeds):
        key, subkey = jax.random.split(key)
        idxs = jax.random.choice(subkey, X_all.shape[0], shape=(int(n_calibration),), replace=False)
        X = X_all[idxs]
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
    beta = 1 / (k_b * T_c)
    lambda_ = 0.01
    d = 15
    q = 3
    n_vals = jnp.array([10, 50, 100, 300, 600]) 
    num_seeds = 10
    
    J = J_chain(d)
    # J = jax.random.normal(jax.random.PRNGKey(0), (d, d)) * 0.01
    # J = jnp.abs(J)        
    # Jsym = 0.5 * (J + J.T)
    # mask = J_mask(d)
    # J = Jsym*mask

    X_all = all_states(d, q)
    t_e = true_expectation(q, d, beta, J, batch=200_000)

    # results = {
    #     # "Active DBQ":         run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, t_e, X_all, "bq_bo"),
    #     "DBQ":        run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, t_e, X_all, "bq_random"),
    #     "MC":         run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, t_e, X_all, "mc")
    # }

    # print("\nMean absolute errors:")
    # for k, v in results.items():
    #     print(k, np.asarray(v["mean_abs_error"]))

    calibrations = {}
    calibration_seeds = 200
    n_calibration = 60
    key_seed = 5
    t_true, C_nom, emp_cov, mus, vars_ = calibration(n_calibration, calibration_seeds, lambda_, d, q, beta, X_all, t_e, J, key_seed)

    jnp.savez("results/potts_calibration_results.npz",
             t_true=t_true, C_nom=jnp.array(C_nom),
             emp_cov=jnp.array(emp_cov),
             mus=jnp.array(mus), vars=jnp.array(vars_))
    print("potts_calibration.npz")



if __name__ == "__main__":
    main()

