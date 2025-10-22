# import jax
# jax.config.update("jax_enable_x64", True)

# import jax.numpy as jnp
# import numpy as np
# from jax import vmap, jit, random
# from jax.scipy.linalg import cho_solve, cho_factor
# from jax.scipy.special import gammainc, erfinv
# from jax.scipy.stats import poisson as jax_poisson
# import matplotlib.pyplot as plt
# import os
# import time
# from functools import partial
# import scipy.stats # NEW: Import scipy.stats for sem
# from jax.scipy.stats import norm
# import json
# from jax.scipy.special import gammaln, gammainc
# import optax

# # --- All your preamble and helper functions remain the same ---
# plt.rcParams['axes.grid'] = True
# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.labelsize'] = 26
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
# plt.rc('axes', titlesize=26, labelsize=26, grid=True)
# plt.rc('lines', linewidth=2)
# plt.rc('legend', fontsize=26, frameon=False)
# plt.rc('xtick', labelsize=26, direction='in')
# plt.rc('ytick', labelsize=26, direction='in')
# plt.rc('figure', figsize=(6, 4), dpi=100)


# # -----------------
# # Sales Data Loading
# # -----------------

# def data_loading(filename = "sales_hist.json"):
#     with open(filename) as f:
#         data = json.load(f)              # keys come back as strings
#     sales_hist = {int(k): int(v) for k, v in data.items()}  # make keys/vals ints
#     return sales_hist

# def params_init(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     fs = np.array([sales_hist[k] for k in xs])
#     n  = fs.sum()
#     mean = (xs*fs).sum()/n
#     var  = (xs**2*fs).sum()/n - mean**2
#     nu0 = mean/var
#     lam0 = (mean + (nu0-1)/(2*nu0))**nu0
#     return nu0, lam0

# def inv_softplus(y):
#     y = jnp.asarray(y)
#     return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))

# from scipy.stats import poisson as sp_poiss

# def build_design(lam, tail_eps=1e-12, min_cap=50, max_cap=5000):
#     q = int(sp_poiss.ppf(1.0 - tail_eps, lam))  # quantile
#     xmax = int(np.clip(q, min_cap, max_cap))
#     X = jnp.arange(0, xmax + 1, dtype=jnp.int32)
#     return X, xmax

# # -----------------
# # Sales Data Loading
# # -----------------

# def f(X, nu):
#     X = jnp.asarray(X)
#     return jnp.exp((1.0 - nu) * gammaln(X + 1.0)).reshape(-1, 1)

# def brownian_kernel(X1, X2):
#     X1 = jnp.atleast_1d(X1).reshape(-1, 1)
#     X2 = jnp.atleast_1d(X2).reshape(-1, 1)
#     return jnp.minimum(X1, X2.T)

# @partial(jit, static_argnames=('xmax',))
# def kernel_embedding_poisson_fixed(lambda_, xi, xmax):
#     xmax   = int(xmax)                       # static OK
#     x_vals = jnp.arange(0, xmax + 1)

#     # Poisson pmf over x_vals (stable log form)
#     log_pmf = -lambda_ + x_vals * jnp.log(jnp.maximum(lambda_, 1e-300)) - gammaln(x_vals + 1.0)
#     pmf     = jnp.exp(log_pmf)

#     # sum_{y <= xi} y * pmf(y)
#     term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals * pmf, 0.0))

#     # tail Pr(Y > xi) = gammainc(xi+1, lambda_)  (lower regularized)
#     tail  = gammainc(xi + 1.0, lambda_)

#     # ❗️use JAX casting, not Python float()
#     xi_f  = xi.astype(jnp.float64)           # or jnp.asarray(xi, jnp.float64)
#     term2 = xi_f * tail

#     return term1 + term2

# @partial(jit, static_argnames=('xmax',))
# def kernel_embedding_poisson(lambda_, X, xmax):
#     X = jnp.atleast_1d(X)
#     return vmap(lambda xi: kernel_embedding_poisson_fixed(lambda_, xi, xmax))(X)

# def precompute_chol(X):
#     K = brownian_kernel(X, X) + 1e-6 * jnp.eye(X.shape[0], dtype=jnp.float64)
#     Lc, _ = cho_factor(K, lower=True)   # discard the bool here
#     return Lc

# # MC
# @jit
# def logZ_mc_full_pool(X_mc, nu, lambda_):
#     logf = (1.0 - nu) * gammaln(X_mc + 1.0)
#     m = jnp.max(logf)
#     log_Eu = jnp.log(jnp.clip(jnp.mean(jnp.exp(logf - m)), 1e-300)) + m
#     return log_Eu + lambda_


# @partial(jit, static_argnames=('xmax',))
# def logZ_bc_on_pool(X_poi, nu, lambda_, Lc, xmax):
#     mu = kernel_embedding_poisson(lambda_, X_poi, xmax).reshape(-1, 1)
#     w  = cho_solve((Lc, True), mu)              # weights
#     logf = (1.0 - nu) * gammaln(X_poi + 1.0)
#     m    = jnp.max(logf)
#     fexp = jnp.exp(logf - m)
#     s    = jnp.dot(w.ravel(), fexp)             # single dot
#     s    = jnp.clip(s, 1e-300, None)            # guard
#     return jnp.log(s) + m + lambda_

# def cmp_suff_stats(sales_hist):
#     if isinstance(sales_hist, dict):
#         ys = jnp.array(list(sales_hist.keys()))
#         cs = jnp.array(list(sales_hist.values()))
#         n = cs.sum()
#         s1 = jnp.sum(ys * cs)
#         s2 = jnp.sum(cs * gammaln(ys + 1.0))
#     else:
#         x = jnp.asarray(sales_hist)
#         n = x.size
#         s1 = jnp.sum(x)
#         s2 = jnp.sum(gammaln(x + 1.0))
#     return n, s1, s2

# def loss_and_aux(params, sales_hist, X_poi, Lc, xmax, run_bq, key):
#     lambda_ = jax.nn.softplus(params[1]) + 1e-8
#     nu  = jax.nn.softplus(params[0]) + 1e-8
#     n, s1, s2 = cmp_suff_stats(sales_hist)

#     if run_bq:
#         logZ = logZ_bc_on_pool(X_poi, nu, lambda_, Lc, xmax)
#     else:
#         assert key is not None
#         X_mc = random.poisson(key, lam=lambda_, shape=(int(len(X_poi)),))
#         logZ = logZ_mc_full_pool(X_mc, nu, lambda_)

#     nll = n * logZ - s1 * jnp.log(lambda_) + nu * s2
#     return nll, {"logZ": logZ, "nll": nll}


# def run_experiment(seed, sales_hist, n_poi, lr, run_bq, num_steps, n_std):

#     print(f"\n--- Seed {seed} | n(BQ or MC)={n_poi}, run_bq={run_bq} ---")

#     key = jax.random.PRNGKey(seed)

#     nu0, lam0 = params_init(sales_hist)
#     raw_params = jnp.array([inv_softplus(nu0), inv_softplus(lam0)])

#     # X_poi
#     X_poi, xmax = build_design(lam0, tail_eps=1e-12, min_cap=50)
#     # Use full grid for BQ:
#     if run_bq:
#         pass
#     else:
#         key, subkey = random.split(key)
#         n_budget = int(min(n_poi, X_poi.size))
#         X_poi = random.choice(subkey, X_poi.size, shape=(n_budget,), replace=False)

#     if run_bq:
#         Lc = precompute_chol(X_poi)
#     else:
#         Lc = None

#     # Optimizer & step fn 
#     optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
#     opt_state = optimizer.init(raw_params)

#     @jit
#     def step(raw_params, opt_state, sales_hist, key):
#         (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
#             raw_params, sales_hist, X_poi, Lc, xmax, run_bq, key
#         )
#         updates, opt_state = optimizer.update(grads, opt_state, raw_params)
#         raw_params = optax.apply_updates(raw_params, updates)
#         return raw_params, opt_state, loss, aux


#     for t in range(num_steps):
#         key, k = random.split(key)
#         raw_params, opt_state, loss, aux = step(raw_params, opt_state, sales_hist, k)
#         if (t % 50) == 0:
#             lam = float(jax.nn.softplus(raw_params[1]) + 1e-8)
#             nu  = float(jax.nn.softplus(raw_params[0]) + 1e-8)
#             print(f"step {t:04d} | NLL={float(loss):.6f} | logZ={float(aux['logZ']):.6f} | nu={nu:.6f} | lambda={lam:.6f}")

#     lam = float(jax.nn.softplus(raw_params[1]) + 1e-8)
#     nu  = float(jax.nn.softplus(raw_params[0]) + 1e-8)
#     return nu, lam


# if __name__ == "__main__":
#     n_poi = 50
#     seed = 0    
#     lr = 1e-3
#     num_steps = 1000
#     run_bq = False
#     n_std = 8.0

#     sales_hist = data_loading(filename = "sales_hist.json")

#     nu_hat, lam_hat = run_experiment(seed, sales_hist, n_poi, lr, run_bq, num_steps, n_std)
#     print("Final:", {"nu": nu_hat, "lambda": lam_hat})


import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammaln, gammainc
import jax.nn as jnn

import numpy as np
import json
import optax
from functools import partial
from scipy.stats import poisson as sp_poiss
import matplotlib.pyplot as plt

# -----------------
# I/O + init utils
# -----------------
def data_loading(filename="sales_hist.json"):
    with open(filename) as f:
        data = json.load(f)
    # keys come as strings -> to ints
    return {int(k): int(v) for k, v in data.items()}

def params_init(sales_hist):
    xs = np.array(sorted(sales_hist.keys()))
    fs = np.array([sales_hist[k] for k in xs])
    n  = fs.sum()
    mean = (xs * fs).sum() / n
    var  = (xs**2 * fs).sum() / n - mean**2
    # crude moment-based init (reasonable starting point)
    nu0  = max(1e-3, mean / max(var, 1e-8))
    lam0 = (mean + (nu0 - 1.0) / (2.0 * nu0)) ** nu0
    lam0 = float(np.clip(lam0, 1e-6, 1e6))
    return float(nu0), float(lam0)

def inv_softplus(y):
    y = np.asarray(y, dtype=np.float64)
    # numerically stable inverse of softplus
    return np.where(y > 20.0, y, np.log(np.expm1(y)))

# -----------------
# Design (fixed generous grid)
# -----------------
def build_design(lam, tail_eps=1e-10, min_cap=100, max_cap=5000):
    q = int(sp_poiss.ppf(1.0 - tail_eps, lam))
    xmax = int(np.clip(q, min_cap, max_cap))
    X = jnp.arange(0, xmax + 1, dtype=jnp.int32)
    return X, xmax


# -----------------
# Kernels / embeddings for BQ
# -----------------
def brownian_kernel(X1, X2):
    X1 = jnp.atleast_1d(X1).reshape(-1, 1)
    X2 = jnp.atleast_1d(X2).reshape(-1, 1)
    return jnp.minimum(X1, X2.T)

@partial(jit, static_argnames=('xmax',))
def kernel_embedding_poisson_fixed(lambda_, xi, xmax):
    xmax   = int(xmax)
    x_vals = jnp.arange(0, xmax + 1)

    log_pmf = -lambda_ + x_vals * jnp.log(jnp.maximum(lambda_, 1e-300)) - gammaln(x_vals + 1.0)
    pmf     = jnp.exp(log_pmf)

    term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals * pmf, 0.0))
    tail  = gammainc(xi + 1.0, lambda_)
    xi_f  = xi.astype(jnp.float64)
    term2 = xi_f * tail
    return term1 + term2

@partial(jit, static_argnames=('xmax',))
def kernel_embedding_poisson(lambda_, X, xmax):
    X = jnp.atleast_1d(X)
    return vmap(lambda xi: kernel_embedding_poisson_fixed(lambda_, xi, xmax))(X)

def precompute_chol(X):
    K = brownian_kernel(X, X) + 1e-6 * jnp.eye(X.shape[0], dtype=jnp.float64)
    Lc, _ = cho_factor(K, lower=True)   # keep only the factor
    return Lc

@partial(jit, static_argnames=('xmax',))
def logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax):
    mu   = kernel_embedding_poisson(lam, X_grid, xmax).reshape(-1, 1)
    w    = cho_solve((Lc, True), mu)              # (N,1)

    logf = (1.0 - nu) * gammaln(X_grid + 1.0)     # (N,)
    m    = jnp.max(logf)
    fexp = jnp.exp(logf - m)
    s    = jnp.dot(w.ravel(), fexp)               # scalar
    s    = jnp.clip(s, 1e-300, None)
    return jnp.log(s) + m + lam


# -----------------
# MC logZ helper
# -----------------
@jit
def logZ_mc_from_samples(X_mc, nu, lam):
    logf = (1.0 - nu) * gammaln(X_mc + 1.0)
    m    = jnp.max(logf)
    logE = jnp.log(jnp.clip(jnp.mean(jnp.exp(logf - m)), 1e-300)) + m
    return logE + lam

def logZ_mc_once(nu, lam, mc_n=200000, seed=0):
    key = random.PRNGKey(seed)
    x = random.poisson(key, lam=lam, shape=(mc_n,))
    return float(logZ_mc_from_samples(x, nu, lam))


# -----------------
# Brute-force truncation (diagnostic truth proxy)
# -----------------
def logZ_trunc(nu, lam, y_max=4000):
    xs = jnp.arange(0, y_max+1)
    # p(y|lam)
    log_p = -lam + xs * jnp.log(jnp.maximum(lam,1e-300)) - gammaln(xs+1.0)
    # f_nu(y) = (y!)^{1-nu}
    log_f = (1.0 - nu) * gammaln(xs + 1.0)
    # log E[f(Y)] + lam
    m = jnp.max(log_f + log_p)
    logE = jnp.log(jnp.sum(jnp.exp(log_f + log_p - m))) + m
    return float(logE + lam)

def nll_true(nu, lam, sales_hist, y_max=8000):
    if isinstance(sales_hist, dict):
        ys = jnp.array(list(sales_hist.keys()))
        cs = jnp.array(list(sales_hist.values()))
        n  = float(cs.sum())
        s1 = float(jnp.sum(ys * cs))
        s2 = float(jnp.sum(cs * gammaln(ys + 1.0)))
    else:
        x = jnp.asarray(sales_hist)
        n  = float(x.size)
        s1 = float(jnp.sum(x))
        s2 = float(jnp.sum(gammaln(x + 1.0)))
    # accurate logZ via truncation proxy you already have
    logZ = logZ_trunc(nu, lam, y_max=y_max)
    return n * logZ - s1 * np.log(lam) + nu * s2

# -----------------
# Sufficient stats + loss
# -----------------
def cmp_suff_stats(sales_hist):
    if isinstance(sales_hist, dict):
        ys = jnp.array(list(sales_hist.keys()))
        cs = jnp.array(list(sales_hist.values()))
        n = cs.sum()
        s1 = jnp.sum(ys * cs)
        s2 = jnp.sum(cs * gammaln(ys + 1.0))
    else:
        x = jnp.asarray(sales_hist)
        n = x.size
        s1 = jnp.sum(x)
        s2 = jnp.sum(gammaln(x + 1.0))
    return n, s1, s2

def loss_and_aux(raw_params, sales_hist, run_bq, *, X_grid, Lc, xmax, key, mc_n):
    lam = jnp.clip(jnn.softplus(raw_params[1]), 1e-12, 1e12)
    nu  = jnp.clip(jnn.softplus(raw_params[0]), 1e-12, 1e12)

    n, s1, s2 = cmp_suff_stats(sales_hist)

    if run_bq:
        logZ = logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax)
    else:
        X_mc = random.poisson(key, lam=lam, shape=(mc_n,))
        logZ = logZ_mc_from_samples(X_mc, nu, lam)

    nll = n * logZ - s1 * jnp.log(lam) + nu * s2
    return nll, {"logZ": logZ, "nll": nll, "nu": nu, "lam": lam}


# -----------------
# Training loop
# -----------------
def run_experiment(seed, sales_hist, lr=1e-3, num_steps=1000, run_bq=True,
                   tail_eps=1e-10, min_cap=100, max_cap=5000):
    print(f"\n--- Seed {seed} | run_bq={run_bq} ---")
    key = random.PRNGKey(seed)

    # init params from data
    # nu0, lam0 = 1.0, 1.2
    nu0, lam0 = params_init(sales_hist)
    # # Poisson-ish start is safest
    # ys = jnp.asarray(sorted(sales_hist.keys()), dtype=jnp.float64)
    # cs = jnp.asarray([sales_hist[int(k)] for k in ys], dtype=jnp.float64)
    # mean = float((ys * cs).sum() / jnp.maximum(cs.sum(), 1.0))
    # nu0  = 1.0
    # lam0 = float(jnp.clip(mean, 1e-6, 1e12))
    raw_params = jnp.array([inv_softplus(nu0), inv_softplus(lam0)], dtype=jnp.float64)

    # fixed generous grid from init λ0
    xmax = 200
    X_grid = jnp.arange(0, xmax + 1, dtype=jnp.int32)
    Lc = precompute_chol(X_grid) if run_bq else None

    # MC budget = BQ grid size (to match effort)
    MC_N = int(X_grid.size)

    # optimizer
    optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
    opt_state = optimizer.init(raw_params)

    @jit
    def step(raw_params, opt_state, key):
        (loss, aux), grads = jax.value_and_grad(
            lambda p, k: loss_and_aux(
                p, sales_hist, run_bq,
                X_grid=X_grid, Lc=Lc, xmax=int(xmax), key=k, mc_n=MC_N
            ),
            has_aux=True
        )(raw_params, key)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        return raw_params, opt_state, loss, aux

    for t in range(num_steps):
        key, k = random.split(key)
        raw_params, opt_state, loss, aux = step(raw_params, opt_state, k)
        if (t % 50) == 0:
            print(f"step {t:04d} | NLL={float(loss):.6f} | "
                  f"logZ={float(aux['logZ']):.6f} | "
                  f"nu={float(aux['nu']):.6f} | lambda={float(aux['lam']):.6f}")

    lam = float(jnn.softplus(raw_params[1]))
    nu  = float(jnn.softplus(raw_params[0]))
    return nu, lam, (X_grid, xmax, Lc), nu0, lam0

# ============== Plot helpers ==============
def cmp_logpmf(y, nu, lam, logZ):
    y = jnp.asarray(y)
    return y * jnp.log(lam) - nu * gammaln(y + 1.0) - logZ

def empirical_pmf_from_hist(sales_hist):
    xs = np.array(sorted(sales_hist.keys()))
    cs = np.array([sales_hist[k] for k in xs], dtype=float)
    p  = cs / cs.sum()
    return xs, p, int(cs.sum())

# -----------------
# Main + diagnostics
# -----------------
if __name__ == "__main__":
    # Toggle here
    RUN_BQ    = False   # set False to compare MC
    LR        = 5e-3
    NUM_STEPS = 2000
    SEED      = 0

    # Load data
    sales_hist = data_loading("sales_hist.json")

    # Train
    nu_hat, lam_hat, (X_grid, xmax, Lc), nu0, lam0 = run_experiment(
        SEED, sales_hist, lr=LR, num_steps=NUM_STEPS, run_bq=RUN_BQ,
        tail_eps=1e-12, min_cap=400, max_cap=5000
    )
    print("Final:", {"nu": nu_hat, "lambda": lam_hat})

    # Compare training-time logZ diag (you already do this) ...
    # Now add the true NLL using the truncation proxy:
    nll_star = nll_true(nu_hat, lam_hat, sales_hist, y_max=8000)
    print(f"True NLL (trunc) at final params = {nll_star:.6f}")

    nll_init = nll_true(nu0, lam0, sales_hist, y_max=8000)
    print(f"True NLL (trunc) at init params = {nll_init:.6f}")

# ---- Plot empirical vs model PMF ----
    if RUN_BQ:
        logZ_model = logZ_bc_on_grid(X_grid, jnp.asarray(nu_hat), jnp.asarray(lam_hat), Lc, int(xmax))
        logZ_init = logZ_bc_on_grid(X_grid, jnp.asarray(nu0), jnp.asarray(lam0), Lc, int(xmax))
    else:
        logZ_model = logZ_trunc(nu_hat, lam_hat, y_max=4000)
        logZ_init = logZ_trunc(nu0, lam0, y_max=4000)

    xs_emp, p_emp, n_emp = empirical_pmf_from_hist(sales_hist)
    y_max_plot = max(xs_emp.max(), int(lam_hat + 8*np.sqrt(max(lam_hat, 1e-8))), 60)
    ys = jnp.arange(0, y_max_plot + 1)

    logp_model = cmp_logpmf(ys, jnp.asarray(nu_hat), jnp.asarray(lam_hat), jnp.asarray(logZ_model))
    pmf_model  = np.asarray(jnp.exp(logp_model))

    logp_model_init = cmp_logpmf(ys, jnp.asarray(nu0), jnp.asarray(lam0), jnp.asarray(logZ_init))
    pmf_model_init  = np.asarray(jnp.exp(logp_model_init))

    plt.figure(figsize=(8, 4.6))
    plt.bar(xs_emp, p_emp, width=0.9, alpha=0.5, label="data (empirical pmf)")
    plt.plot(np.asarray(ys), pmf_model, marker="o", lw=1.8,
             label=f"CMP model (ν={nu_hat:.3f}, λ={lam_hat:.3f})")
    plt.plot(np.asarray(ys), pmf_model_init, marker="s", lw=1.8,
             label=f"CMP model (ν={nu0:.3f}, λ={lam0:.3f})")
    plt.xlabel("y")
    plt.ylabel("Probability")
    plt.title("Empirical vs CMP model")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ### ADAPTIVE SAMPLING
# # ---------- put these 2 lines FIRST (before importing jax) ----------
# import jax
# jax.config.update("jax_enable_x64", True)
# # -------------------------------------------------------------------

# import jax.numpy as jnp
# from jax import jit, vmap, random
# from jax.scipy.linalg import cho_solve, cho_factor
# from jax.scipy.special import gammaln, gammainc
# import jax.nn as jnn

# import numpy as np
# import json
# import optax
# from functools import partial
# from scipy.stats import poisson as sp_poiss


# # =========================
# # I/O + initialization
# # =========================
# def data_loading(filename="sales_hist.json"):
#     with open(filename) as f:
#         data = json.load(f)
#     return {int(k): int(v) for k, v in data.items()}

# def params_init(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     fs = np.array([sales_hist[k] for k in xs])
#     n  = fs.sum()
#     mean = (xs * fs).sum() / n
#     var  = (xs**2 * fs).sum() / n - mean**2
#     # crude but stable
#     nu0  = max(1e-3, mean / max(var, 1e-8))
#     lam0 = (mean + (nu0 - 1.0) / (2.0 * nu0)) ** nu0
#     lam0 = float(np.clip(lam0, 1e-6, 1e6))
#     return float(nu0), float(lam0)

# def inv_softplus(y):
#     y = np.asarray(y, dtype=np.float64)
#     return np.where(y > 20.0, y, np.log(np.expm1(y)))


# # =========================
# # Design / grid utils
# # =========================
# def build_design_from_lambda(lam, *, tail_eps=1e-12, min_cap=200, max_cap=8000):
#     """Contiguous grid X={0,...,xmax} with P(Y>xmax)≈tail_eps for Y~Pois(lam)."""
#     lam = float(lam)
#     if not np.isfinite(lam) or lam <= 0.0:
#         lam = 1.0  # fallback
#     q = int(sp_poiss.ppf(1.0 - tail_eps, lam))
#     xmax = int(np.clip(q, min_cap, max_cap))
#     X = jnp.arange(0, xmax + 1, dtype=jnp.int32)
#     return X, xmax

# def need_expand(old_xmax, lam_now, tail_eps=1e-12, safety=1.2, max_cap=8000):
#     """Return (do_expand, new_X, new_xmax)."""
#     if not np.isfinite(lam_now) or lam_now <= 0.0:
#         return False, None, None
#     q = int(sp_poiss.ppf(1.0 - tail_eps, float(lam_now)))
#     target = int(min(int(q * safety), max_cap))
#     if target > int(old_xmax):
#         X = jnp.arange(0, target + 1, dtype=jnp.int32)
#         return True, X, target
#     return False, None, None


# # =========================
# # Kernels / embeddings
# # =========================
# def brownian_kernel(X1, X2):
#     X1 = jnp.atleast_1d(X1).reshape(-1, 1)
#     X2 = jnp.atleast_1d(X2).reshape(-1, 1)
#     return jnp.minimum(X1, X2.T)

# @partial(jit, static_argnames=("xmax",))
# def kernel_embedding_poisson_fixed(lam, xi, xmax):
#     """
#     μ(xi) = E[min(Y, xi)], Y~Pois(lam)
#           = sum_{y<=xi} y p(y) + xi * P(Y > xi).
#     We use gammainc(a,x) = P(a,x) (lower regularized),
#     and P(Y>xi) = gammainc(xi+1, lam).
#     """
#     xmax = int(xmax)
#     x_vals = jnp.arange(0, xmax + 1)

#     # pmf over x_vals (log-stable)
#     log_pmf = -lam + x_vals * jnp.log(jnp.maximum(lam, 1e-300)) - gammaln(x_vals + 1.0)
#     pmf     = jnp.exp(log_pmf)

#     term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals * pmf, 0.0))
#     tail  = gammainc(xi + 1.0, lam)        # = P(Y>xi)
#     term2 = xi.astype(jnp.float64) * tail
#     return term1 + term2

# @partial(jit, static_argnames=("xmax",))
# def kernel_embedding_poisson(lam, X, xmax):
#     X = jnp.atleast_1d(X)
#     return vmap(lambda xi: kernel_embedding_poisson_fixed(lam, xi, xmax))(X)

# def precompute_chol(X):
#     # small jitter for stability
#     K = brownian_kernel(X, X) + 1e-6 * jnp.eye(X.shape[0], dtype=jnp.float64)
#     Lc, _ = cho_factor(K, lower=True)
#     return Lc

# @partial(jit, static_argnames=("xmax",))
# def logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax):
#     """
#     log Z(ν,λ) = log E[(Y!)^{1-ν}] + λ.
#     BQ with Brownian kernel; weights w = K^{-1} μ.
#     """
#     mu = kernel_embedding_poisson(lam, X_grid, xmax).reshape(-1, 1)
#     w  = cho_solve((Lc, True), mu)  # (N,1)

#     logf = (1.0 - nu) * gammaln(X_grid + 1.0)
#     m    = jnp.max(logf)
#     fexp = jnp.exp(logf - m)
#     s    = jnp.dot(w.ravel(), fexp)
#     s    = jnp.clip(s, 1e-300, None)
#     return jnp.log(s) + m + lam


# # =========================
# # MC helper (for diagnostics or MC training)
# # =========================
# @jit
# def logZ_mc_from_samples(X_mc, nu, lam):
#     logf = (1.0 - nu) * gammaln(X_mc + 1.0)
#     m    = jnp.max(logf)
#     logE = jnp.log(jnp.clip(jnp.mean(jnp.exp(logf - m)), 1e-300)) + m
#     return logE + lam


# # =========================
# # Sufficient stats + loss
# # =========================
# def cmp_suff_stats(sales_hist):
#     if isinstance(sales_hist, dict):
#         ys = jnp.array(list(sales_hist.keys()))
#         cs = jnp.array(list(sales_hist.values()))
#         n = cs.sum()
#         s1 = jnp.sum(ys * cs)
#         s2 = jnp.sum(cs * gammaln(ys + 1.0))
#     else:
#         x = jnp.asarray(sales_hist)
#         n = x.size
#         s1 = jnp.sum(x)
#         s2 = jnp.sum(gammaln(x + 1.0))
#     return n, s1, s2

# def loss_and_aux(raw_params, sales_hist, *, X_grid, Lc, xmax, key=None, run_bq=True, mc_n=None):
#     # positivity via softplus
#     lam = jnp.clip(jnn.softplus(raw_params[1]), 1e-12, 1e12)
#     nu  = jnp.clip(jnn.softplus(raw_params[0]),  1e-12, 1e12)

#     n, s1, s2 = cmp_suff_stats(sales_hist)

#     if run_bq:
#         logZ = logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax)
#     else:
#         assert key is not None and mc_n is not None
#         X_mc = random.poisson(key, lam=lam, shape=(int(mc_n),))
#         logZ = logZ_mc_from_samples(X_mc, nu, lam)

#     nll = n * logZ - s1 * jnp.log(lam) + nu * s2
#     return nll, {"logZ": logZ, "nll": nll, "nu": nu, "lam": lam}


# # =========================
# # Build a jitted step with concrete (static) args
# # =========================
# def make_step_fn(sales_hist, X_grid, Lc, xmax, optimizer):
#     # close over concrete Python ints
#     xmax_int = int(xmax)
#     MC_N     = int(X_grid.shape[0])

#     @jit
#     def step(raw_params, opt_state, key):
#         (loss, aux), grads = jax.value_and_grad(
#             lambda p, k: loss_and_aux(
#                 p, sales_hist,
#                 X_grid=X_grid, Lc=Lc, xmax=xmax_int,
#                 key=k, run_bq=True, mc_n=MC_N
#             ),
#             has_aux=True
#         )(raw_params, key)
#         updates, opt_state = optimizer.update(grads, opt_state, raw_params)
#         raw_params = optax.apply_updates(raw_params, updates)
#         return raw_params, opt_state, loss, aux

#     return step


# # =========================
# # Adaptive training loop (BQ)
# # =========================
# def train_with_adaptive_grid(
#     seed,
#     sales_hist,
#     *,
#     lr=1e-3,
#     num_steps=1000,
#     init_tail_eps=1e-12,
#     expand_tail_eps=5e-13,
#     expand_safety=1.2,
#     min_cap=200,
#     max_cap=8000,
#     print_every=50
# ):
#     print(f"\n--- Seed {seed} | run_bq=True ---")
#     key = random.PRNGKey(seed)

#     # init params
#     nu0, lam0 = params_init(sales_hist)
#     raw_params = jnp.array([inv_softplus(nu0), inv_softplus(lam0)], dtype=jnp.float64)

#     # initial grid from λ0
#     X_grid, xmax = build_design_from_lambda(lam0, tail_eps=init_tail_eps, min_cap=min_cap, max_cap=max_cap)
#     Lc = precompute_chol(X_grid)

#     # optimizer + initial step
#     optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
#     opt_state = optimizer.init(raw_params)
#     step = make_step_fn(sales_hist, X_grid, Lc, xmax, optimizer)

#     for t in range(num_steps):
#         key, k = random.split(key)
#         raw_params, opt_state, loss, aux = step(raw_params, opt_state, k)

#         if (t % print_every) == 0:
#             print(f"step {t:04d} | NLL={float(loss):.6f} | "
#                   f"logZ={float(aux['logZ']):.6f} | "
#                   f"nu={float(aux['nu']):.6f} | lambda={float(aux['lam']):.6f} | "
#                   f"xmax={int(xmax)} | |X|={int(X_grid.size)}")

#         # consider expanding if λ grew
#         lam_now = float(aux["lam"])
#         do_expand, new_X, new_xmax = need_expand(xmax, lam_now, tail_eps=expand_tail_eps,
#                                                  safety=expand_safety, max_cap=max_cap)
#         if do_expand:
#             X_grid, xmax = new_X, new_xmax
#             Lc = precompute_chol(X_grid)
#             # rebuild jitted step with new static args
#             step = make_step_fn(sales_hist, X_grid, Lc, xmax, optimizer)

#     lam = float(jnn.softplus(raw_params[1]))
#     nu  = float(jnn.softplus(raw_params[0]))
#     return nu, lam, (X_grid, xmax, Lc)


# # =========================
# # Main
# # =========================
# if __name__ == "__main__":
#     SEED      = 0
#     LR        = 1e-3
#     NUM_STEPS = 2000

#     sales_hist = data_loading("sales_hist.json")

#     nu_hat, lam_hat, (X_grid, xmax, Lc) = train_with_adaptive_grid(
#         SEED,
#         sales_hist,
#         lr=LR,
#         num_steps=NUM_STEPS,
#         init_tail_eps=1e-12,     # initial coverage
#         expand_tail_eps=5e-13,   # stricter on expand
#         expand_safety=1.2,       # modest overshoot
#         min_cap=200,
#         max_cap=8000,
#         print_every=50
#     )

#     print("Final:", {"nu": nu_hat, "lambda": lam_hat})
