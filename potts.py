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
import math
import matplotlib.lines as mlines

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
from matplotlib.lines import Line2D
import os
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches
import argparse

jax.config.update("jax_enable_x64", True)



@partial(jit, static_argnums=(0,1))
def h_chain(d, q):
    a = jnp.arange(q, dtype=jnp.float64)   
    row = 0.1 * a                     
    return jnp.tile(row[None, :], (d, 1))  

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
    return -0.5 * (jnp.einsum('ijab,ia,jb->', J, x, x)) - jnp.einsum('ik,ik->', x, h)

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
def compute_max_variance_using_inverse(X_obs, K_inv, inv_ones, candidates, lambda_, d, q, jitter=1e-8):
    z0 = kernel_embedding(lambda_, d, q)
    gains = vmap(lambda x: candidate_gain(x, X_obs, K_inv, inv_ones, lambda_, z0, jitter))(candidates)
    return candidates[jnp.argmax(gains)]

# --------------------------------------------------------------------------
@partial(jit, static_argnums=(1,2))
def one_hot_from_ints(ints, d, q):
    digits = q ** jnp.arange(d - 1, -1, -1, dtype=jnp.int64)
    X_int = ((ints[:, None] // digits) % q).astype(jnp.int32)  # (n,d)
    return jax.nn.one_hot(X_int, q, dtype=jnp.float64)     

def sample_uniform(key, n, d, q):
    ints = jrand.randint(key, (int(n),), 0, q**d, dtype=jnp.int64)
    return one_hot_from_ints(ints, d, q)

def true_expectation(q, d, beta, J, h, batch, return_logZ):
    total = int(q ** d)
    s = 0.0

    for start in range(0, total, batch):
        end = min(start + batch, total)
        idxs = jnp.arange(start, end, dtype=jnp.int64)     
        X = one_hot_from_ints(idxs, d, q)                  
        w = f_batch(X, J, h, beta)                   
        s += jnp.sum(w)

    if return_logZ:
        return jnp.log(s)                                  
    else:
        return s / total   

# -------Russian Roulette---------------------------------------------------------
def sample_onehot_single(key, d, q):
    idx = jrand.randint(key, (), 0, q**d, dtype=jnp.int64)
    return one_hot_from_ints(idx[None], d, q)[0]

def rr_debiased_mc_estimator_linear(key, J, h, beta, d, q, rho=0.95, max_steps=1_000_000):
    rho = float(jnp.clip(rho, 0.0, 0.999999))
    qm = 1.0
    total = 0.0
    n_samples = 0
    S_prev = 0.0
    subkey = key

    
    subkey, sk = jrand.split(subkey)
    x0 = sample_onehot_single(sk, d, q)
    fx0 = f_single(x0, J, h, beta)
    sum_f = fx0
    S_m = sum_f / 1.0
    total += (S_m - 0.0) / qm  # q0 = 1
    S_prev = S_m
    n_samples = 1


    while n_samples < max_steps:
        
        subkey, sk_rr = jrand.split(subkey)
        if not bool(jrand.bernoulli(sk_rr, p=jnp.asarray(rho))):
            break

        qm *= rho  # q_m = rho^m

        subkey, sk_x = jrand.split(subkey)
        x = sample_onehot_single(sk_x, d, q)
        fx = f_single(x, J, h, beta)
        sum_f += fx
        S_m = sum_f / (n_samples + 1)

        total += (S_m - S_prev) / qm
        S_prev = S_m
        n_samples += 1

    return float(total), int(n_samples)

# ----Stratified Sampling (4 Strata)-----------------------------------------------

def _four_strata_bounds(d: int, q: int):
    N = int(q ** d)
    base = N // 4
    rem  = N - 4 * base  # remainder 0..3

    lengths = [base + (1 if j < rem else 0) for j in range(4)]
    bounds  = []
    start   = 0
    for L in lengths:
        end = start + L
        bounds.append((jnp.int64(start), jnp.int64(end)))
        start = end

    probs = jnp.array([L / N for L in lengths], dtype=jnp.float64)  # π_j
    return bounds, probs


def stratified_mc_estimator_4(key, n: int, d: int, q: int, J, h, beta):
    if n <= 0:
        return 0.0, 0
    if n % 4 != 0:
        raise ValueError("Here we assume n is a multiple of 4.")

    bounds, probs = _four_strata_bounds(d, q)  # π_j for each stratum
    n_per = n // 4

    mu_sum = 0.0
    used   = 0
    subkey = key
    for j in range(4):
        low, high = bounds[j]                         
        subkey, kj = jrand.split(subkey)
        idxs = jrand.randint(kj, (n_per,), low, high, dtype=jnp.int64)
        Xj = one_hot_from_ints(idxs, d, q)           
        fj = f_batch(Xj, J, h, beta)                 
        mu_sum = mu_sum + probs[j] * jnp.mean(fj)
        used   += n_per

    return mu_sum, used


# ------------ Importance Sampling: q(x) ∝ exp(-β ∑⟨h_i, x_i⟩), no interactions -----------

@jit
def proposal_log_probs(beta, h):
    logits = -beta * h
    m = jnp.max(logits, axis=1, keepdims=True)
    z = m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=1, keepdims=True))
    return logits - z  # (d, q)

@partial(jit, static_argnums=(1,))  # n is static if you want, but not required
def sample_from_proposal(key, n, log_p):

    d, q = log_p.shape
    keys = jrand.split(key, d)
    samples = [jrand.categorical(k, logits=log_p[i][None, :], shape=(n,)) for i, k in enumerate(keys)]
    X_int = jnp.stack(samples, axis=1)  # (n, d)
    return jax.nn.one_hot(X_int, q, dtype=jnp.float64)  # (n, d, q)

@jit
def log_q_of_X(X, log_p):
    return jnp.sum(jnp.einsum('ndq,dq->nd', X, log_p), axis=1)


@partial(jit, static_argnums=(1,))
def importance_sampling_uniform_mean(key, n, J, h, beta, d, q):

    log_p = proposal_log_probs(beta, h)                 # (d, q)
    X = sample_from_proposal(key, int(n), log_p)        # (n, d, q)

    lq = log_q_of_X(X, log_p)                           # (n,)
    fx = f_batch(X, J, h, beta)                         # (n,)  (already exp(β E))
    lw = jnp.log(fx) - lq                               # (n,)
    m = jnp.max(lw)
    Z_hat = jnp.exp(m) * jnp.mean(jnp.exp(lw - m))      # estimate of Z = Σ f(x)
    qd = (q ** d)
    mu_hat = Z_hat / qd
    return mu_hat

# --------------------------------------------------------------------------
def run_experiment(n_vals, lambda_, d, q, seed, beta, J, h, t_true, experiment_type, sub_sample):
    key = jax.random.PRNGKey(seed)
    est_means, times = [], []

    for n in n_vals:
        start = time.time()
        key, key_bo, key_random = jax.random.split(key, 3)

        if experiment_type == "bq_bo":
            x0 = sample_uniform(key_bo, int(n), d, q)[0]
            X = [x0.astype(jnp.int32)]

            K_inv    = jnp.array([[1.0]], dtype=jnp.float64)
            inv_ones = jnp.array([[1.0]], dtype=jnp.float64)

            for _ in range(1, n):
                key_bo, kc = jrand.split(key_bo)
                cand = sample_uniform(kc, sub_sample, d, q)
                X_obs = jnp.stack(X, axis=0)    
                x_star = compute_max_variance_using_inverse(jnp.array(X_obs), K_inv, inv_ones, cand, lambda_, d, q)
                K_inv, inv_ones = woodbury_inverse(K_inv, inv_ones, jnp.array(X_obs), x_star, lambda_)
                X.append(x_star.astype(jnp.int32))
            
            X_obs = jnp.stack(X, axis=0)    
            f_vals = f_batch(X_obs, J, h, beta).reshape(-1, 1)
            w = precompute_bc_weights(X_obs, lambda_, d, q)
            est_mean = (w.T @ f_vals)[0, 0]
        else:

            if experiment_type == "bq_random":
                X = sample_uniform(key_random, int(n), d, q)
                # a = len(X)
                # b = jnp.unique(X, axis = 0)
                f_vals = f_batch(X, J, h, beta).reshape(-1, 1)     # (n,1)
                w = precompute_bc_weights(X, lambda_, d, q)     # (n,1)
                est_mean = (w.T @ f_vals)[0, 0]

            elif experiment_type == "mc":
                X = sample_uniform(key_random, int(n), d, q)
                f_vals = f_batch(X, J, h, beta)                    # (n,)
                est_mean = jnp.mean(f_vals)

            elif experiment_type == "rr":
                rho = float(jnp.clip(1.0 - 1.0 / jnp.maximum(1, n), 0.0, 0.999999))
                est_rr, used = rr_debiased_mc_estimator_linear(key_random, J, h, beta, d, q, rho=rho)
                est_mean = jnp.asarray(est_rr)

            elif experiment_type == "strat":
                mu_hat, used = stratified_mc_estimator_4(key_random, int(n), d, q, J, h, beta)
                est_mean = jnp.asarray(mu_hat)

            elif experiment_type == "is":
                est_mean = importance_sampling_uniform_mean(key_random, int(n), J, h, beta, d, q)

        jax.block_until_ready(est_mean)
        elapsed = time.time() - start
        est_means.append(est_mean)
        times.append(elapsed)
        # print(f"n={int(n):5d} | {experiment_type.upper():9s} | time {elapsed:.3f}s | err={float(jnp.abs(est_mean - t_true)):.6e}")

    return {"true_val": t_true, "est_means": jnp.array(est_means), "times": jnp.array(times)}

def run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_true, experiment_type, sub_sample):
    """
    This is the corrected version of the function.
    """
    all_errors_across_seeds = []
    
    # This loop will now work, as num_seeds is an integer
    for seed in range(num_seeds):
        print(f"\n--- Running {experiment_type}, Seed {seed+1}/{num_seeds} ---")
        
        # Call run_experiment with the correct arguments
        result = run_experiment(
            n_vals, lambda_, d, q, seed, 
            beta, J, h, t_true, 
            experiment_type, sub_sample
        )
        
        abs_error = jnp.abs(result["est_means"] - result["true_val"])
        all_errors_across_seeds.append(abs_error)

    all_errors = jnp.stack(all_errors_across_seeds)  # shape: [num_seeds, len(n_vals)]

    # central tendency + dispersion
    mean_abs_error = jnp.mean(all_errors, axis=0)
    se_abs_error   = scipy.stats.sem(np.asarray(all_errors), axis=0)
    median_abs_error = jnp.median(all_errors, axis=0)
    q25_abs_error    = jnp.percentile(np.asarray(all_errors), 25, axis=0)
    q75_abs_error    = jnp.percentile(np.asarray(all_errors), 75, axis=0)

    out = {
        "mean_abs_error": np.asarray(mean_abs_error),
        "se_abs_error":   np.asarray(se_abs_error),
        "median_abs_error": np.asarray(median_abs_error),
        "q25_abs_error":    np.asarray(q25_abs_error),
        "q75_abs_error":    np.asarray(q75_abs_error),
        "true_val": float(result["true_val"]),
        "abs_errors": np.asarray(all_errors) # Added this for your boxplot data
    }
    
    return out

def format_log_label(n):

    if n <= 0:
        return r"$0$"
    e = math.floor(math.log10(n))
    m = n / (10**e)
    
    m_str = "{:.1f}".format(m)
    if m_str == "1.0":
        return r"$10^{{{}}}$".format(e)
    else:
        return r"${} \times 10^{{{}}}$".format(m_str, e)

def plot_results(n_vals, all_results, unique, filename, save_path="results"):
    os.makedirs(save_path, exist_ok=True)

    if unique:
        styles = {
            'MC': {'color': 'k', 'marker': 'o', 'label': 'MC', 'linestyle': '-'},
            'SS': {'color': 'c', 'marker': '^', 'label': 'SS', 'linestyle': '-'},
            'BayesSum': {'color': 'b', 'marker': 's', 'label': 'DBQ', 'linestyle': '-'},
            'Active BayesSum': {'color': 'g', 'marker': '^', 'label': 'Active DBQ', 'linestyle': '-'},
            'RR': {'color': 'r', 'marker': 'D', 'label': 'RR', 'linestyle': '-'},
            'IS': {'color': '#A52A2A', 'marker': 'o', 'label': 'IS', 'linestyle': '-'},
        }
    else:
        styles = {
            'BayesSum': {'color': 'blue', 'marker': 'D', 'label': 'BayesSum', 'linestyle': '-'},
            'BayesSum (IID)': {'color': 'blue', 'marker': 'D', 'label': 'BayesSum (IID)', 'linestyle': '--'}
        }

    plt.figure(figsize=(10, 6))
    handles, labels = [], []
    eps = 1e-12

    for name in [m for m in styles.keys() if m in all_results]:
        data = all_results[name]

        # Prefer median + IQR; fallback to mean ± 1.96*SE if needed
        if ("median_abs_error" in data) and ("q25_abs_error" in data) and ("q75_abs_error" in data):
            y_line = np.asarray(data["median_abs_error"])
            y_low  = np.asarray(data["q25_abs_error"])
            y_high = np.asarray(data["q75_abs_error"])
        else:
            mean_err = np.asarray(data["mean_abs_error"])
            se_err   = np.asarray(data["se_abs_error"])
            y_line = mean_err
            y_low  = mean_err - 1.96 * se_err
            y_high = mean_err + 1.96 * se_err

        y_line = np.clip(y_line, eps, None)
        y_low  = np.clip(y_low,  eps, None)
        y_high = np.clip(y_high, eps, None)

        st = styles[name]
        (ln,) = plt.plot(n_vals, y_line, linestyle='-', color=st['color'],
                         marker=st['marker'], label=st['label'])
        plt.fill_between(n_vals, y_low, y_high, color=st['color'], alpha=0.15)
        handles.append(ln); labels.append(st['label'])

    plt.xlabel("Number of Points", fontsize=32)
    plt.title("Uniform", fontsize=32)
    plt.ylabel("Absolute Error", fontsize=32)
    plt.yscale('log')
    plt.xscale('log')
    log_labels = [format_log_label(n) for n in n_vals]
    
    plt.xticks(n_vals, log_labels, fontsize=28) 
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(n_vals))
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), format='pdf')
    plt.close()

    fig_legend, ax_legend = plt.subplots(figsize=(2, 1))
    ax_legend.axis('off')

    legend_handles = []
    legend_labels  = []

    for name in styles:
        st = styles[name]
        h = Line2D(
            [0], [0],
            color=st['color'],
            linestyle=st['linestyle'],   # <--- DIFFERENT FOR EACH METHOD
            marker=st['marker'],
            markersize=10,
            label=st['label']
        )
        legend_handles.append(h)
        legend_labels.append(st['label'])

    ax_legend.legend(
        legend_handles,
        legend_labels,
        ncol=len(styles.keys()),
        loc='center',
        fontsize=16,
        frameon=False
    )

    if not unique:
        plt.savefig(os.path.join(save_path, "potts_unique_vs_iid_legend.pdf"), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_path, "abs_error_potts_legend.pdf"), bbox_inches='tight')

    plt.close(fig_legend)


def save_results_portable(filename, n_vals, all_results):
    save_dict = {'n_vals': np.array(n_vals, dtype=int)}
    for method, data in all_results.items():
        for field in ["mean_abs_error","se_abs_error","median_abs_error","q25_abs_error","q75_abs_error","true_val"]:
            if field in data:
                v = data[field]
                if field == "true_val":
                    v = np.array([float(v)], dtype=float)
                save_dict[f"{method}_{field}"] = np.asarray(v, dtype=float)
    np.savez(filename, **save_dict)
    print(f"Saved results to {filename}")

def load_results_portable(filename):
    data = np.load(filename)
    n_vals = data['n_vals']
    all_results = {}
    for key in data.files:
        if key == 'n_vals':
            continue
        # keys look like "<method>_<field>"
        if "_" not in key:
            continue
        method, field = key.split('_', 1)
        val = data[key]
        if field == "true_val":
            val = val[0]
        all_results.setdefault(method, {})[field] = val
    return n_vals, all_results


# --------------------------------------------------------------------------
def lengthscale_ablation(d, q, n_ablation, lam_start, lam_end, lam_num, seeds, t_e, J, h, beta, key_seed, experiment_type, sub_sample):
    lam_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    errs_seeds, sds_seeds = [], []
    key = jax.random.PRNGKey(key_seed)

    for s in range(seeds):

        if experiment_type == "bq_random":
            key, subkey = jax.random.split(key)
            X = sample_uniform(subkey, int(n_ablation), d, q)
            f_vals = f_batch(X, J, h, beta)

            err_list, sd_list = [], []

            for lam in lam_grid:
                mu, var = bayesian_cubature(X, f_vals, lam, d, q)
                sd = jnp.sqrt(jnp.maximum(var, 0.0))
                err_list.append(mu - t_e)
                sd_list.append(sd)

        elif experiment_type == "bq_bo":

            err_list, sd_list = [], []

            for lam in lam_grid:

                key, subkey = jax.random.split(key)
                x0 = sample_uniform(subkey, int(n_ablation), d, q)[0]
                X = [x0.astype(jnp.int32)]

                K_inv    = jnp.array([[1.0]], dtype=jnp.float64)
                inv_ones = jnp.array([[1.0]], dtype=jnp.float64)

                for i in range(1, n_ablation):
                    key, subkey = jax.random.split(key)
                    cand = sample_uniform(subkey, sub_sample, d, q)
                    X_obs = jnp.stack(X, axis=0)    
                    x_star = compute_max_variance_using_inverse(jnp.array(X_obs), K_inv, inv_ones, cand, lam, d, q)
                    K_inv, inv_ones = woodbury_inverse(K_inv, inv_ones, jnp.array(X_obs), x_star, lam)
                    X.append(x_star.astype(jnp.int32))

                X_obs = jnp.stack(X, axis=0)    
                f_vals = f_batch(X_obs, J, h, beta)

                mu, var = bayesian_cubature(X_obs, f_vals, lam, d, q)
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
        "ERR": ERR
    }

@jit
def _scale_to_Z(mu, var, d, q):
    qd = (q ** d).astype(jnp.float64) if isinstance(q, jnp.ndarray) else (q ** d)
    return qd * mu, (qd * qd) * jnp.maximum(var, 0.0)

def calibration(n_calibration, seeds, lambda_, d, q, beta, t_e, J, h, key_seed):
    mus, vars_ = [], []

    key = jax.random.PRNGKey(key_seed)

    for s in range(seeds):
        key, subkey = jax.random.split(key)
        X = sample_uniform(subkey, int(n_calibration), d, q)
        y = f_batch(X, J, h, beta)

        mu, var = bayesian_cubature(X, y, lambda_, d, q)
        mus.append(mu); vars_.append(var)
        
    mus  = jnp.array(mus)
    vars_ = jnp.maximum(jnp.array(vars_), 1e-30)

    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = jnp.sqrt(2.0) * erfinv(C_nom) 

    half = z[:,None] * jnp.sqrt(vars_)[None,:]
    inside = (t_e >= mus[None,:] - half) & (t_e <= mus[None,:] + half)
    emp_cov = jnp.mean(inside, axis=1)

    return float(t_e), C_nom, emp_cov, mus, vars_


def plot_abs_error_boxplots(
    rmses,
    d_list,
    methods,
    save_path="potts_abs_error_boxplot.png",
    logy = True,
    showfliers = True):

    if methods is None:
        methods = list(rmses.keys())

    # explicitly fix colors
    mcolors = {"Active BayesSum": "g", "BayesSum": "b", "MC": "k"}

    # fallback for any extra methods not specified
    palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, m in enumerate(methods):
        if m not in mcolors:
            mcolors[m] = palette[i % len(palette)]

    # collect data and positions
    data, positions, colors = [], [], []
    width = 0.35 if len(methods) == 2 else 0.8 / max(1, len(methods))

    for i, L in enumerate(d_list):
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

    centers = [i + (len(methods) - 1) * width / 2 for i in range(len(d_list))]
    plt.xticks(centers, [f"d={L}*3" for L in d_list], fontsize=32)

    if logy:
        plt.yscale("log")
    plt.ylabel("Absolute Error", fontsize=32)
    plt.xlabel("Dimension", fontsize=32)
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    handles = [mpatches.Patch(color=mcolors[m], label=m) for m in methods]
    plt.legend(handles=handles, loc="best", fontsize=32)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def plot_lambda_boxplot(result_dict, n_show=10, logy=True,
                        save_path="results/potts_ablation_boxplot.pdf"):
    lam_grid = np.array(result_dict["lambda"])
    ERR = np.array(result_dict["ERR"])  # shape (seeds, lam_num)
    ERR = np.abs(ERR)

    lam_num = len(lam_grid)
    if lam_num > n_show:
        # Uniform subsample across the lambda grid
        idxs = np.linspace(0, lam_num - 1, n_show, dtype=int)
    else:
        idxs = np.arange(lam_num)

    # here you just decide to keep first 5
    idxs = idxs[:5]
    lam_sub = lam_grid[idxs]        # <--- these are your x-axis values
    ERR_sub = ERR[:, idxs]

    # boxplot data per lambda
    data = [ERR_sub[:, i] for i in range(ERR_sub.shape[1])]

    plt.figure(figsize=(10, 6))

    # positions = lam_sub -> box centers are exactly at those lambdas
    bp = plt.boxplot(
        data,
        positions=lam_sub,
        widths=0.5 * lam_sub,   # small width relative to value (so it doesn't look crazy near 0.01)
        patch_artist=True,
        showfliers=False
    )

    # style
    for box in bp['boxes']:
        box.set(facecolor='blue', alpha=0.7, edgecolor='black')
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)

    # x-axis: ticks at lam_sub, labels = lam_sub, NO rotation
    plt.xscale('log')
    # plt.xticks(lam_sub, [f"{lam:.2g}" for lam in lam_sub], fontsize=32)

    plt.xlabel("Lengthscale", fontsize=32)
    plt.ylabel("Error", fontsize=32)

    if logy:
        plt.yscale('log')

    plt.title("Uniform: Lengthscale Ablation", fontsize=32)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "potts_ablation_boxplot.pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")



def estimate_alpha(N, MAE, min_index=1):
    N_fit = N[min_index:]
    mae_fit = MAE[min_index:]

    x = np.log(N_fit)
    y = np.log(mae_fit)

    a, b = np.polyfit(x, y, 1)
    alpha = -a
    return alpha, a, b


# =========================
# Figure generators
# =========================

POTTS_RESULTS_FILE = "potts.npz"

def ensure_uniform_results():
    """
    Make sure POTTS_RESULTS_FILE exists.
    If not, run the experiments and save it.
    """

    if os.path.exists(POTTS_RESULTS_FILE):
        print(f"Found {POTTS_RESULTS_FILE}, reusing it.")
        return

    print(f"{POTTS_RESULTS_FILE} not found. Running experiments to create it...")

    # ---- set up parameters exactly as in your original main() ----
    k_b = 1
    T_c = 2.269
    beta = 1 / T_c
    d = 15
    q = 3
    n_vals = jnp.array([8, 28, 84, 220, 1000])
    num_seeds = int(50)
    sub_sample = 2000

    alpha, gamma = 1.0, 0.0
    h = h_chain(d, q)
    J = J_chain(d, q, alpha, gamma)

    # true expectation (per-state)
    t_e = true_expectation(q, d, beta, J, h, batch=200_000, return_logZ=False)
    print("True Z:", float(t_e * (q**d)))

    # lengthscale ablation (or hard-code lambda_ if you prefer)
    lam_start, lam_end, lam_num = 0.001, 10, 50
    seeds_ablation = 50
    n_ablation = 400
    ablation = lengthscale_ablation(
        d, q, n_ablation,
        lam_start, lam_end, lam_num,
        seeds_ablation,
        t_e, J, h, beta,
        key_seed=0,
        experiment_type="bq_random",
        sub_sample=None,
    )
    lambda_ = ablation["lambda_star"]
    print("Chosen lambda* =", float(lambda_))

    results = {
        "Active BayesSum": run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "bq_bo", sub_sample),
        "BayesSum":        run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "bq_random", sub_sample),
        "MC":              run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "mc", sub_sample),
        "RR":              run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "rr", sub_sample),
        "SS":              run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "strat", sub_sample),
        "IS":              run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "is", sub_sample),
    }

    results["BayesSum (IID)"] = results["BayesSum"]

    save_results_portable(POTTS_RESULTS_FILE, n_vals, results)

def generate_uniform_abs_error():
    """
    Figure: potts_abs_err.pdf + potts_abs_err_legend.pdf
    """
    ensure_uniform_results()
    n_vals, all_results = load_results_portable(POTTS_RESULTS_FILE)
    plot_results(n_vals, all_results, True, "potts_abs_err.pdf")


def generate_comparison_unique_vs_iid():
    """
    Figure: potts_abs_err_iid_vs_unique.pdf
    """
    ensure_uniform_results()
    n_vals, all_results = load_results_portable(POTTS_RESULTS_FILE)
    plot_results(n_vals, all_results, False, "potts_abs_err_iid_vs_unique.pdf")


def generate_slope_table():
    """
    Prints alpha exponents for each method.
    """
    ensure_uniform_results()
    n_vals, all_results = load_results_portable(POTTS_RESULTS_FILE)
    methods = ["MC", "BayesSum", "Active BayesSum", "RR", "SS", "IS"]
    for m in methods:
        alpha, a, b = estimate_alpha(n_vals, all_results[m]["mean_abs_error"], min_index=1)
        print(f"{m}: α ≈ {alpha:.3f}")


def generate_lengthscale_ablation():
    """
    Figure: potts_ablation_boxplot.pdf
    """
    d = 15
    q = 3
    k_b = 1
    T_c = 2.269
    beta = 1 / T_c
    alpha, gamma = 1.0, 0.0

    h = h_chain(d, q)
    J = J_chain(d, q, alpha, gamma)
    t_e = true_expectation(q, d, beta, J, h, batch=200_000, return_logZ=False)

    lam_start, lam_end, lam_num = 0.001, 10, 50
    seeds_ablation = 50
    n_ablation = 400

    result = lengthscale_ablation(
        d, q, n_ablation,
        lam_start, lam_end, lam_num,
        seeds_ablation,
        t_e,
        J, h, beta,
        key_seed=0,
        experiment_type="bq_random",
        sub_sample=None
    )

    plot_lambda_boxplot(result, save_path="potts_ablation_boxplot.pdf")


def generate_calibration():
    """
    Saves calibration npz; you can have a separate script to plot it.
    """
    d = 15
    q = 3
    k_b = 1
    T_c = 2.269
    beta = 1 / T_c
    alpha, gamma = 1.0, 0.0
    lambda_ = 0.005

    h = h_chain(d, q)
    J = J_chain(d, q, alpha, gamma)
    t_e = true_expectation(q, d, beta, J, h, batch=200_000, return_logZ=False)

    calibration_seeds = 50
    n_calibration = 60
    key_seed = 5
    t_true, C_nom, emp_cov, mus, vars_ = calibration(
        n_calibration, calibration_seeds,
        lambda_,
        d=d, q=q, beta=beta,
        t_e=t_e, J=J, h=h,
        key_seed=key_seed
    )

    os.makedirs("results", exist_ok=True)
    jnp.savez(
        "results/potts_calibration_results.npz",
        t_true=t_true,
        C_nom=jnp.array(C_nom),
        emp_cov=jnp.array(emp_cov),
        mus=jnp.array(mus),
        vars=jnp.array(vars_),
    )
    print("Saved results/potts_calibration_results.npz")

def generate_dimension_boxplot():
    """
    Figure: potts_abs_error_boxplot.pdf
    """
    d_list = [5, 10, 15]
    q = 3
    k_b = 1
    T_c = 2.269
    beta = 1 / T_c
    alpha, gamma = 1.0, 0.0

    lam_start, lam_end, lam_num = 0.001, 10, 50
    seeds_ablation = 50
    n_ablation = 400

    methods = [("mc", "MC"), ("bq_random", "BayesSum"), ("bq_bo", "Active BayesSum")]
    rmses = {label: {} for _, label in methods}
    bq_lambda_stars = {}

    for d in d_list:

        h = h_chain(d, q)
        J = J_chain(d, q, alpha, gamma)
        t_e = true_expectation(q, d, beta, J, h, batch=200_000, return_logZ = False)
        print(t_e)

        bq_ablation = lengthscale_ablation(d, q, n_ablation, lam_start, lam_end, lam_num, seeds_ablation, t_e, J, h, beta, d, "bq_random", None)
        bq_lambda_star = bq_ablation["lambda_star"]
        bq_lambda_stars[d] = bq_lambda_star

        mc_results = run_multiple_seeds([n_ablation], 0.0, d, q, seeds_ablation, beta, J, h, t_e, "mc", None)
        rmses["MC"][d] = mc_results["abs_errors"][:, 0]

        dbq_results = run_multiple_seeds([n_ablation], bq_lambda_star, d, q, seeds_ablation, beta, J, h, t_e, "bq_random", None)
        rmses["BayesSum"][d] = dbq_results["abs_errors"][:, 0]

        abq_results = run_multiple_seeds([n_ablation], bq_lambda_star, d, q, seeds_ablation, beta, J, h, t_e, "bq_bo", sub_sample)
        rmses["Active BayesSum"][d] = abq_results["abs_errors"][:, 0]

    # lambda_stars = [0.004558104601977957, 0.004558104601977957, 0.028593018398391068]

    # print(rmses)
    #     rmses = {
#     "MC": {
#         5: [2.54166095e-04, 7.07450705e-03, 8.04885205e-03, 6.93353856e-03, 2.47596441e-03,
#             1.49401660e-02, 9.29559119e-04, 9.74872029e-03, 6.10219785e-03, 1.73635589e-04],
#         10: [1.03597373e-03, 8.68107533e-03, 7.27703871e-03, 1.01829601e-03, 3.27493977e-04,
#              1.20005946e-02, 7.99650907e-06, 1.46295412e-02, 3.05741087e-03, 3.80251672e-03],
#         15: [1.42315616e-03, 1.05462090e-02, 1.09413407e-02, 2.57800959e-03, 5.62285734e-03,
#              9.52050022e-03, 5.54054485e-03, 1.51984579e-02, 4.32610621e-04, 6.49570287e-03],
#     },
#     "BayesSum": {
#         5: [3.41480972e-05, 2.25149492e-05, 1.45841488e-04, 2.51880068e-06, 8.49946952e-06,
#             2.33356808e-05, 2.03995727e-06, 1.58398211e-05, 2.44664033e-06, 6.33773609e-05],
#         10: [1.31345453e-04, 3.38322736e-04, 1.25630877e-04, 3.24524426e-04, 6.76157720e-05,
#              4.16162600e-04, 1.43796280e-04, 6.14573156e-04, 6.02121570e-05, 7.82360206e-04],
#         15: [9.08454434e-04, 2.15210888e-03, 2.86778805e-03, 3.52999969e-03, 2.13046158e-03,
#              6.84320681e-04, 9.40602877e-04, 7.16832566e-05, 9.97242303e-04, 1.78999419e-03],
#     },
#     "Active BayesSum": {
#         5: [2.20436918e-06, 2.04733721e-06, 3.91577211e-06, 3.53205905e-06, 6.34696651e-06,
#             2.76420486e-06, 2.46079382e-06, 2.28937861e-07, 1.53039999e-05, 9.76466477e-06],
#         10: [1.36588159e-04, 1.32593100e-04, 1.27642270e-04, 1.09296333e-04, 4.19209910e-05,
#              1.36949995e-04, 6.82333574e-05, 1.05065468e-04, 1.97872117e-04, 2.08293830e-04],
#         15: [2.15650487e-04, 3.96974087e-04, 1.80159348e-04, 6.57238145e-04, 8.68516740e-04,
#              3.62030438e-04, 3.78704875e-04, 1.37290209e-04, 3.80317304e-04, 2.15475960e-06],
#     },
# }


    plot_abs_error_boxplots(
        rmses,
        d_list,
        methods=[label for _, label in methods],
        save_path="potts_abs_error_boxplot.pdf",
        logy=True,
        showfliers=True
    )

# --------------------
# Main
# --------------------
# def main():
#     global beta
#     k_b = 1
#     T_c = 2.269
#     beta = 1/2.269
#     # lambda_ = 0.001
#     d = 15
#     q = 3
#     # n_vals = jnp.array([8, 48, 216, 1000]) 
#     n_vals = jnp.array([8, 28, 84, 220, 1000]) 
#     num_seeds = int(50)
#     sub_sample = 2000

    # key = jax.random.PRNGKey(0)
    # key, kh, kJ = jax.random.split(key, 3)

    # h = jax.random.normal(kh, (d, q), dtype=jnp.float64) * 0.01

    # J = jax.random.normal(kJ, (d, d, q, q), dtype=jnp.float64) * 0.01 
    # Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    # mask = J_mask(d)
    # J = Jsym * mask

    # alpha, gamma = 1.0, 0.0
    # h = h_chain(d, q)
    # J = J_chain(d, q, alpha, gamma)
    
    # t_e = true_expectation(q, d, beta, J, h, batch=200_000, return_logZ = False)
    # print("True Z:", t_e * (q**d))

    # # Ablation
    # lam_start, lam_end, lam_num = 0.001, 10, 50
    # seeds_ablation = 50
    # n_ablation = 400
    # ablation = lengthscale_ablation(d, q, n_ablation, lam_start, lam_end, lam_num, seeds_ablation, t_e, J, h, beta, 0, "bq_random", None)

    # # plot_lambda_boxplot(ablation)
    
    # lambda_ = ablation["lambda_star"]
    # # print(lambda_)
    # results = {
    #     "Active BayesSum":        run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "bq_bo", sub_sample),
    #     "BayesSum":        run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "bq_random", sub_sample),
    #     "MC":         run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "mc", sub_sample),
    #     "RR":         run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "rr", sub_sample),
    #     "SS":       run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "strat",  sub_sample),
    #     "IS":       run_multiple_seeds(n_vals, lambda_, d, q, num_seeds, beta, J, h, t_e, "is",  sub_sample),
    # }


    # print("\nMean absolute errors:")
    # for k, v in results.items():
    #     print(k, np.asarray(v["mean_abs_error"]))

    # save_results_portable("potts.npz", n_vals, results)
    # n_vals, all_results = load_results_portable("potts.npz")
    # all_results["BayesSum (IID)"] = all_results["BayesSum"]
    # print(all_results)
    # plot_results(n_vals, all_results, False, "potts_abs_err_iid_vs_unique.pdf")
    # plot_results(n_vals, all_results, True, "potts_abs_err.pdf")

    # methods = ["MC", "BayesSum", "Active BayesSum", "RR", "SS", "IS"]

    # for m in methods:
    #     alpha, a, b = estimate_alpha(n_vals, all_results[m]["mean_abs_error"], min_index=1)
    #     print(f"{m}: α ≈ {alpha:.3f}")

    # # Ablation

    # d_list = [5, 10, 15]
    # methods = [("mc", "MC"), ("bq_random", "BayesSum"), ("bq_bo", "Active BayesSum")]
    # rmses = {label: {} for _, label in methods}
    # n_ablation = 400
    # bq_lambda_stars = {}
    # abq_lambda_stars = {}

    # for d in d_list:

    #     h = h_chain(d, q)
    #     J = J_chain(d, q, alpha, gamma)
    #     t_e = true_expectation(q, d, beta, J, h, batch=200_000, return_logZ = False)
    #     print(t_e)

    #     bq_ablation = lengthscale_ablation(d, q, n_ablation, lam_start, lam_end, lam_num, seeds_ablation, t_e, J, h, beta, d, "bq_random", None)
    #     bq_lambda_star = bq_ablation["lambda_star"]
    #     bq_lambda_stars[d] = bq_lambda_star

    #     mc_results = run_multiple_seeds([n_ablation], 0.0, d, q, seeds_ablation, beta, J, h, t_e, "mc", None)
    #     rmses["MC"][d] = mc_results["abs_errors"][:, 0]

    #     dbq_results = run_multiple_seeds([n_ablation], bq_lambda_star, d, q, seeds_ablation, beta, J, h, t_e, "bq_random", None)
    #     rmses["BayesSum"][d] = dbq_results["abs_errors"][:, 0]

    #     abq_results = run_multiple_seeds([n_ablation], bq_lambda_star, d, q, seeds_ablation, beta, J, h, t_e, "bq_bo", sub_sample)
    #     rmses["Active BayesSum"][d] = abq_results["abs_errors"][:, 0]

    # lambda_stars = [0.004558104601977957, 0.004558104601977957, 0.028593018398391068]

    # print(rmses)

    # # plot_abs_error_boxplots(
    # #     rmses,
    # #     d_list,
    # #     methods=[label for _, label in methods],
    # #     save_path="potts_abs_error_boxplot.pdf",
    # #     logy=True,
    # #     showfliers=True
    # # )

#     rmses = {
#     "MC": {
#         5: [2.54166095e-04, 7.07450705e-03, 8.04885205e-03, 6.93353856e-03, 2.47596441e-03,
#             1.49401660e-02, 9.29559119e-04, 9.74872029e-03, 6.10219785e-03, 1.73635589e-04],
#         10: [1.03597373e-03, 8.68107533e-03, 7.27703871e-03, 1.01829601e-03, 3.27493977e-04,
#              1.20005946e-02, 7.99650907e-06, 1.46295412e-02, 3.05741087e-03, 3.80251672e-03],
#         15: [1.42315616e-03, 1.05462090e-02, 1.09413407e-02, 2.57800959e-03, 5.62285734e-03,
#              9.52050022e-03, 5.54054485e-03, 1.51984579e-02, 4.32610621e-04, 6.49570287e-03],
#     },
#     "BayesSum": {
#         5: [3.41480972e-05, 2.25149492e-05, 1.45841488e-04, 2.51880068e-06, 8.49946952e-06,
#             2.33356808e-05, 2.03995727e-06, 1.58398211e-05, 2.44664033e-06, 6.33773609e-05],
#         10: [1.31345453e-04, 3.38322736e-04, 1.25630877e-04, 3.24524426e-04, 6.76157720e-05,
#              4.16162600e-04, 1.43796280e-04, 6.14573156e-04, 6.02121570e-05, 7.82360206e-04],
#         15: [9.08454434e-04, 2.15210888e-03, 2.86778805e-03, 3.52999969e-03, 2.13046158e-03,
#              6.84320681e-04, 9.40602877e-04, 7.16832566e-05, 9.97242303e-04, 1.78999419e-03],
#     },
#     "Active BayesSum": {
#         5: [2.20436918e-06, 2.04733721e-06, 3.91577211e-06, 3.53205905e-06, 6.34696651e-06,
#             2.76420486e-06, 2.46079382e-06, 2.28937861e-07, 1.53039999e-05, 9.76466477e-06],
#         10: [1.36588159e-04, 1.32593100e-04, 1.27642270e-04, 1.09296333e-04, 4.19209910e-05,
#              1.36949995e-04, 6.82333574e-05, 1.05065468e-04, 1.97872117e-04, 2.08293830e-04],
#         15: [2.15650487e-04, 3.96974087e-04, 1.80159348e-04, 6.57238145e-04, 8.68516740e-04,
#              3.62030438e-04, 3.78704875e-04, 1.37290209e-04, 3.80317304e-04, 2.15475960e-06],
#     },
# }

#     d_list = [5, 10, 15]
#     plot_abs_error_boxplots(
#         rmses,
#         d_list,
#         methods=[label for _, label in methods],
#         save_path="potts_abs_error_boxplot.pdf",
#         logy=True,
#         showfliers=True
#     )

    # # Calibration
    # calibrations = {}
    # calibration_seeds = 50
    # n_calibration = 60
    # key_seed = 5
    # t_e, C_nom, emp_cov, mus, vars_= calibration(n_calibration, calibration_seeds, 0.005, d, q, beta, t_e, J, h, key_seed)

    # jnp.savez("results/potts_calibration_results.npz",
    #          t_true=t_e, C_nom=jnp.array(C_nom),
    #          emp_cov=jnp.array(emp_cov),
    #          mus=jnp.array(mus), vars=jnp.array(vars_))
    # print("potts_calibration.npz")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "unique_vs_iid",
            "slope",
            "dim_boxplot",
            "lambda_ablation",
            "calibration",
            "all",
        ],
        help="Which figure/results to generate.",
    )
    args = parser.parse_args()

    if args.figure in ("uniform", "all"):
        generate_uniform_abs_error()
    if args.figure in ("unique_vs_iid", "all"):
        generate_comparison_unique_vs_iid()
    if args.figure in ("slope", "all"):
        generate_slope_table()
    if args.figure in ("dim_boxplot", "all"):
        generate_dimension_boxplot()
    if args.figure in ("lambda_ablation", "all"):
        generate_lengthscale_ablation()
    if args.figure in ("calibration", "all"):
        generate_calibration()

if __name__ == "__main__":
    main()
