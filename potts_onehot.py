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
def energy(x, J, h):
    return 0.5 * (jnp.einsum('ijab,ia,jb->', J, x, x)) + jnp.einsum('ik,ik->', x, h)

@jit
def f_single(x, J, h, beta):
    return jnp.exp(beta * energy(x, J, h))

@jit
def f_batch(X, J, h, beta):
    return vmap(lambda xi: f_single(xi, J, h, beta))(X)


def kernel_embedding(lambda_, d, q):
    return ((1.0 / q) + ((q - 1.0) / q) * jnp.exp(-lambda_)) ** d

@jit
def double_integral(lambda_, d, q):
    return kernel_embedding(lambda_, d, q)

@jit
def gram_matrix(X, lambda_):
    n, d, q = X.shape
    X_flat = X.reshape(n, d * q)
    dot = X_flat @ X_flat.T
    return jnp.exp(-lambda_ * d + lambda_ * dot)


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
    return mean

# --------------------------------------------------------------------------
@partial(jit, static_argnums=(0,1))
def all_states(d,q):
    n = q ** d
    idx = jnp.arange(n, dtype=jnp.int64)[:, None]             
    digits = q ** jnp.arange(d-1, -1, -1, dtype=jnp.int64)    
    X_int = ((idx // digits) % q).astype(jnp.int32)           
    return jax.nn.one_hot(X_int, q, dtype=jnp.float64)

# def sample_uniform(key, q, d, n_samples):
#     ints = jrand.randint(key, shape=(n_samples, d), minval=0, maxval=q).astype(jnp.int32)
#     return one_hot_from_ints(ints, d, q)

def true_expectation(q, d, beta, J, h, batch):
    total = int(q ** d)
    s = 0.0
    cnt = 0
    digits = q ** jnp.arange(d-1, -1, -1, dtype=jnp.int64)

    for start in range(0, total, batch):
        count = min(batch, total - start)
       
        idx = jnp.arange(start, start + count, dtype=jnp.int64)[:, None] 
        X = ((idx // digits) % q).astype(jnp.int32)  
        X_onehot = jax.nn.one_hot(X, q, dtype=jnp.float64)                 
        f_vals = f_batch(X_onehot, J, h, beta)

        s += jnp.sum(f_vals)
        cnt += count

    return (s / cnt).astype(jnp.float64)


def run_experiment(n_vals, lambda_, d, q, seed, beta, J, h, t_true, X_all, experiment_type):
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

            # for i in range(1, n):
            #     key, subkey = jax.random.split(key)
            #     candidates = sample_states(subkey, d, n, q) ## We are using subsampling trick in case d >= 16
            #     x_next = compute_max_variance(X[:i], candidates, lambda_, d, q)
            #     X = X.at[i].set(x_next)

            # f_vals = f_batch(X, J, beta).reshape(-1, 1)
            # w = precompute_bc_weights(X, lambda_, d, q)
            # est_mean = (w.T @ f_vals)[0, 0]
            break

        else:

            idxs = jax.random.choice(subkey, N_all, shape=(int(n),), replace=False)
            X = X_all[idxs]

            if experiment_type == "bq_random":
                # f_vals = f_batch(X, J, h, beta).reshape(-1, 1)     # (n,1)
                # w = precompute_bc_weights(X, lambda_, d, q)     # (n,1)
                # est_mean = (w.T @ f_vals)[0, 0]
                E     = vmap(energy, in_axes=(0, None, None))(X, J, h)
                logf  = beta * E
                m     = jnp.max(logf)
                fshift= jnp.exp(logf - m)              

                w = precompute_bc_weights(X, lambda_, d, q)
                w_pos = jnp.maximum(w, 0.0)            # positive parts
                w_neg = jnp.maximum(-w, 0.0)           # abs(negative parts)
                s = jnp.clip((w_pos.T @ fshift)[0] - (w_neg.T @ fshift)[0], 1e-300, None)

                est_mean = jnp.log(s) + m + d * jnp.log(q)

            elif experiment_type == "mc":
                # f_vals = f_batch(X, J, h, beta)                    # (n,)
                # est_mean = jnp.mean(f_vals)
                E = vmap(energy, in_axes=(0, None, None))(X, J, h)
                logf = beta * E
                m = jnp.max(logf)
                log_Eu = jnp.log(jnp.clip(jnp.mean(jnp.exp(logf - m)), 1e-300)) + m
                est_mean = log_Eu + d * jnp.log(q)

        jax.block_until_ready(est_mean)
        elapsed = time.time() - start
        est_means.append(est_mean)
        times.append(elapsed)
        print(f"n={int(n):5d} | {experiment_type.upper():9s} | time {elapsed:.3f}s | err={float(jnp.abs(est_mean - t_true)):.6e}")

    return {"true_val": jnp.log((q**d) * t_true), "est_means": jnp.array(est_means), "times": jnp.array(times)}

def run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_true, X_all, experiment_type):
    abs_errors = []
    for seed in range(num_seeds):
        print(f"\n--- {experiment_type.upper()} | Seed {seed} ---")
        res = run_experiment(n_vals, lambda_, d, q, seed, beta, J, h, t_true, X_all, experiment_type)
        abs_errors.append(jnp.abs(res["est_means"] - res["true_val"]))

    abs_errors = jnp.stack(abs_errors)
    mean_abs_error = jnp.mean(abs_errors, axis=0) 
    return {"true_val": t_true, "abs_errors": abs_errors, "mean_abs_error": mean_abs_error}

# --------------------
# Main
# --------------------
def main():
    global beta
    k_b = 1
    T_c = 2.269
    beta = 5.0
    lambda_ = 0.3
    d = 12
    q = 3
    n_vals = jnp.array([100, 500, 1000]) 
    num_seeds = 10
    alpha = 1.0
    gamma = 0.0

    # J = J_chain(d, q, alpha, gamma)
    # print(J[0])
    key = jax.random.PRNGKey(0)
    key, kh, kJ = jax.random.split(key, 3)

    h = jax.random.normal(kh, (d, q), dtype=jnp.float64) * 0.01

    J = jax.random.normal(kJ, (d, d, q, q), dtype=jnp.float64) * 0.01 
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    mask = J_mask(d)
    J = Jsym * mask
    
    X_all = all_states(d, q)
    t_e = true_expectation(q, d, beta, J, h, batch=200_000)


    results = {
        "DBQ":        run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, X_all, "bq_random"),
        "MC":         run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, X_all, "mc")
    }

    print("\nMean absolute errors:")
    for k, v in results.items():
        print(k, np.asarray(v["mean_abs_error"]))

    print("True Z:")
    print(jnp.log(t_e * (q **d)))

if __name__ == "__main__":
    main()

