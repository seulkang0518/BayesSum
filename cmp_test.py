# # cmp_train_and_plot.py
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# import jax
# jax.config.update("jax_enable_x64", True)

# import jax.numpy as jnp
# import jax.nn as jnn
# from jax import jit, vmap, random
# from jax.scipy.linalg import cho_solve, cho_factor
# from jax.scipy.special import gammaln, gammainc
# import optax
# from functools import partial


# # -----------------
# # I/O + init utils
# # -----------------
# def data_loading(filename="sales_hist.json"):
#     with open(filename) as f:
#         data = json.load(f)
#     # keys come back as strings -> convert to ints
#     return {int(k): int(v) for k, v in data.items()}

# def params_init(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     fs = np.array([sales_hist[k] for k in xs])
#     n  = fs.sum()
#     mean = (xs * fs).sum() / n
#     var  = (xs**2 * fs).sum() / n - mean**2
#     nu0  = max(1e-3, mean / max(var, 1e-8))           # crude but stable
#     lam0 = (mean + (nu0 - 1.0) / (2.0 * nu0)) ** nu0  # moment-ish init
#     lam0 = float(np.clip(lam0, 1e-6, 1e6))
#     return float(nu0), float(lam0)

# def inv_softplus(y):
#     y = np.asarray(y, dtype=np.float64)
#     # numerically stable inverse softplus
#     return np.where(y > 20.0, y, np.log(np.expm1(y)))


# # -----------------
# # Fixed grid (use a single, generous grid)
# # -----------------
# def build_fixed_grid(xmax=200):
#     X_grid = jnp.arange(0, int(xmax) + 1, dtype=jnp.int32)
#     return X_grid, int(xmax)


# # -----------------
# # Brownian kernel + Poisson mean embedding μ(x)=E[min(Y,x)]
# # -----------------
# def brownian_kernel(X1, X2):
#     X1 = jnp.atleast_1d(X1).reshape(-1, 1)
#     X2 = jnp.atleast_1d(X2).reshape(-1, 1)
#     return jnp.minimum(X1, X2.T)

# @partial(jit, static_argnames=("xmax",))
# def kernel_embedding_poisson_fixed(lambda_, xi, xmax):
#     """
#     μ(xi) = E[min(Y, xi)] for Y ~ Poisson(lambda)
#           = sum_{y<=xi} y p(y) + xi * P(Y > xi)
#     Using regularized lower incomplete gamma: P(Y > xi) = gammainc(xi+1, lambda)
#     """
#     xmax   = int(xmax)
#     x_vals = jnp.arange(0, xmax + 1)

#     # Poisson pmf(y) for y in x_vals (log-stable)
#     log_pmf = -lambda_ + x_vals * jnp.log(jnp.maximum(lambda_, 1e-300)) - gammaln(x_vals + 1.0)
#     pmf     = jnp.exp(log_pmf)

#     term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals * pmf, 0.0))
#     tail  = gammainc(xi + 1.0, lambda_)   # = P(a, x) (lower regularized)
#     term2 = xi.astype(jnp.float64) * tail
#     return term1 + term2

# @partial(jit, static_argnames=("xmax",))
# def kernel_embedding_poisson(lambda_, X, xmax):
#     X = jnp.atleast_1d(X)
#     return vmap(lambda xi: kernel_embedding_poisson_fixed(lambda_, xi, xmax))(X)

# def precompute_chol(X):
#     K = brownian_kernel(X, X) + 1e-6 * jnp.eye(X.shape[0], dtype=jnp.float64)
#     Lc, _ = cho_factor(K, lower=True)   # keep only the factor
#     return Lc

# # -----------------
# # log Z for BQ and MC
# # -----------------
# @partial(jit, static_argnames=('xmax',))
# def logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax):
#     mu   = kernel_embedding_poisson(lam, X_grid, xmax).reshape(-1, 1)
#     w    = cho_solve((Lc, True), mu)              # (N,1)

#     logf = (1.0 - nu) * gammaln(X_grid + 1.0)     # (N,)
#     m    = jnp.max(logf)
#     fexp = jnp.exp(logf - m)
#     s    = jnp.dot(w.ravel(), fexp)               # scalar
#     s    = jnp.clip(s, 1e-300, None)
#     return jnp.log(s) + m + lam


# @jit
# def logZ_mc_from_samples(X_mc, nu, lam):
#     logf = (1.0 - nu) * gammaln(X_mc + 1.0)
#     m    = jnp.max(logf)
#     logE = jnp.log(jnp.clip(jnp.mean(jnp.exp(logf - m)), 1e-300)) + m
#     return logE + lam


# # -----------------
# # Sufficient stats + loss (+ optional regularization)
# # -----------------
# def cmp_suff_stats(sales_hist):
#     if isinstance(sales_hist, dict):
#         ys = jnp.array(list(sales_hist.keys()))
#         cs = jnp.array(list(sales_hist.values()))
#         n  = cs.sum()
#         s1 = jnp.sum(ys * cs)
#         s2 = jnp.sum(cs * gammaln(ys + 1.0))
#     else:
#         x  = jnp.asarray(sales_hist)
#         n  = x.size
#         s1 = jnp.sum(x)
#         s2 = jnp.sum(gammaln(x + 1.0))
#     return n, s1, s2

# def loss_and_aux(raw_params, sales_hist, run_bq, *,
#                  X_grid, K_chol, xmax, key, mc_n,
#                  use_reg, log_nu0, log_lam0, w_log_nu, w_log_lam):
#     lam = jnp.clip(jnn.softplus(raw_params[1]), 1e-12, 1e12)
#     nu  = jnp.clip(jnn.softplus(raw_params[0]), 1e-12, 1e12)

#     n, s1, s2 = cmp_suff_stats(sales_hist)

#     if run_bq:
#         logZ = logZ_bc_on_grid(X_grid, nu, lam, K_chol, int(xmax))
#     else:
#         X_mc = random.poisson(key, lam=lam, shape=(mc_n,))
#         logZ = logZ_mc_from_samples(X_mc, nu, lam)

#     # CMP negative log-likelihood up to constant terms
#     nll = n * logZ - s1 * jnp.log(lam) + nu * s2

#     if use_reg:
#         log_nu  = jnp.log(nu)
#         log_lam = jnp.log(lam)
#         reg = w_log_nu * (log_nu  - log_nu0)**2 + w_log_lam * (log_lam - log_lam0)**2
#     else:
#         reg = 0.0

#     total = nll + reg
#     return total, {"logZ": logZ, "nll": nll, "reg": reg, "nu": nu, "lam": lam}


# # -----------------
# # Training loop (fixed grid)
# # -----------------
# def train_fixed(seed, sales_hist,
#                 xmax=200,          # fixed grid 0..xmax
#                 lr=1e-3, num_steps=1000,
#                 run_bq=True,
#                 use_reg=True, w_log_nu=1e-3, w_log_lam=1e-3):
#     print(f"\n--- Seed {seed} | run_bq={run_bq} ---")
#     key = random.PRNGKey(seed)

#     # init params & anchors for regularizer
#     nu0, lam0 = params_init(sales_hist)
#     raw_params = jnp.array([inv_softplus(nu0), inv_softplus(lam0)], dtype=jnp.float64)
#     log_nu0  = jnp.log(jnp.asarray(nu0)  + 1e-12)
#     log_lam0 = jnp.log(jnp.asarray(lam0) + 1e-12)

#     # fixed grid
#     X_grid, xmax = build_fixed_grid(xmax=xmax)
#     K_chol = precompute_chol(X_grid) if run_bq else None
#     MC_N   = int(X_grid.size)  # match effort for MC baseline

#     # optimizer
#     optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
#     opt_state = optimizer.init(raw_params)

#     @jit
#     def step(raw_params, opt_state, key):
#         (loss, aux), grads = jax.value_and_grad(
#             lambda p, k: loss_and_aux(
#                 p, sales_hist, run_bq,
#                 X_grid=X_grid, K_chol=K_chol, xmax=int(xmax), key=k, mc_n=MC_N,
#                 use_reg=use_reg, log_nu0=log_nu0, log_lam0=log_lam0,
#                 w_log_nu=w_log_nu, w_log_lam=w_log_lam
#             ),
#             has_aux=True
#         )(raw_params, key)
#         updates, opt_state = optimizer.update(grads, opt_state, raw_params)
#         raw_params = optax.apply_updates(raw_params, updates)
#         return raw_params, opt_state, loss, aux

#     for t in range(num_steps):
#         key, k = random.split(key)
#         raw_params, opt_state, total, aux = step(raw_params, opt_state, k)
#         if (t % 50) == 0:
#             print(f"step {t:04d} | NLL={float(aux['nll']):.6f} | "
#                   f"reg={float(aux['reg']):.3e} | total={float(total):.6f} | "
#                   f"logZ={float(aux['logZ']):.6f} | nu={float(aux['nu']):.6f} | "
#                   f"lambda={float(aux['lam']):.6f}")

#     lam = float(jnn.softplus(raw_params[1]))
#     nu  = float(jnn.softplus(raw_params[0]))
#     return nu, lam, (X_grid, xmax, K_chol)


# # -----------------
# # Diagnostics
# # -----------------
# def logZ_trunc(nu, lam, y_max=4000):
#     xs    = jnp.arange(0, y_max + 1)
#     log_p = -lam + xs * jnp.log(jnp.maximum(lam, 1e-300)) - gammaln(xs + 1.0)
#     log_f = (1.0 - nu) * gammaln(xs + 1.0)
#     m     = jnp.max(log_f + log_p)
#     logE  = jnp.log(jnp.sum(jnp.exp(log_f + log_p - m))) + m
#     return float(logE + lam)

# def logZ_mc_big(nu, lam, mc_n=200000, seed=123):
#     key = random.PRNGKey(seed)
#     x   = random.poisson(key, lam=lam, shape=(mc_n,))
#     logf = (1.0 - nu) * gammaln(x + 1.0)
#     m    = jnp.max(logf)
#     logE = jnp.log(jnp.clip(jnp.mean(jnp.exp(logf - m)), 1e-300)) + m
#     return float(logE + lam)


# # -----------------
# # Plot helpers
# # -----------------
# def cmp_logpmf(y, nu, lam, logZ):
#     y = jnp.asarray(y)
#     # log p(y) = y log λ − ν log(y!) − logZ
#     return y * jnp.log(lam) - nu * gammaln(y + 1.0) - logZ

# def empirical_pmf_from_hist(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     cs = np.array([sales_hist[k] for k in xs], dtype=float)
#     p  = cs / cs.sum()
#     return xs, p, int(cs.sum())


# # -----------------
# # Main
# # -----------------
# if __name__ == "__main__":
#     # Toggles / hyperparams
#     RUN_BQ     = True     # set False to train the MC baseline
#     USE_REG    = True     # tiny log-scale ridge regularization
#     W_LOG_NU   = 5e-4
#     W_LOG_LAM  = 5e-4
#     LR         = 1e-2
#     NUM_STEPS  = 1000
#     SEED       = 0
#     XMAX       = 200      # fixed grid 0..XMAX (200 is safe for small λ)

#     # Load data
#     sales_hist = data_loading("sales_hist.json")

#     # Train
#     nu_hat, lam_hat, (X_grid, xmax, K_chol) = train_fixed(
#         SEED, sales_hist,
#         xmax=XMAX, lr=LR, num_steps=NUM_STEPS, run_bq=RUN_BQ,
#         use_reg=USE_REG, w_log_nu=W_LOG_NU, w_log_lam=W_LOG_LAM
#     )
#     print("Final:", {"nu": nu_hat, "lambda": lam_hat})

#     # Diagnostics (compare logZ estimates)
#     print("\nDiagnostics at final params:")
#     print(f"  trunc : {logZ_trunc(nu_hat, lam_hat, y_max=4000):.6f}")
#     if RUN_BQ:
#         logZ_bq = float(logZ_bc_on_grid(X_grid, jnp.asarray(nu_hat), jnp.asarray(lam_hat), K_chol, int(xmax)))
#         print(f"  BQ    : {logZ_bq:.6f}")
#     print(f"  MCbig : {logZ_mc_big(nu_hat, lam_hat, mc_n=200000, seed=123):.6f}")

#     # ----------------- Plot empirical vs. CMP model -----------------
#     # Choose logZ used for plotting
#     if RUN_BQ:
#         logZ_for_plot = float(logZ_bc_on_grid(X_grid, jnp.asarray(nu_hat), jnp.asarray(lam_hat), K_chol, int(xmax)))
#     else:
#         # if training with MC, just use truncation proxy for plotting
#         logZ_for_plot = logZ_trunc(nu_hat, lam_hat, y_max=4000)

#     xs_emp, p_emp, n_emp = empirical_pmf_from_hist(sales_hist)
#     y_max_plot = max(
#         xs_emp.max(),
#         int(lam_hat + 8 * np.sqrt(max(lam_hat, 1e-8))),
#         60
#     )
#     ys = jnp.arange(0, y_max_plot + 1)

#     logp_model = cmp_logpmf(ys, jnp.asarray(nu_hat), jnp.asarray(lam_hat), jnp.asarray(logZ_for_plot))
#     pmf_model  = np.asarray(jnp.exp(logp_model))

#     plt.figure(figsize=(8, 4.6))
#     plt.bar(xs_emp, p_emp, width=0.9, alpha=0.5, label="data (empirical pmf)")
#     plt.plot(np.asarray(ys), pmf_model, marker="o", lw=1.8,
#              label=f"CMP model (ν={nu_hat:.3f}, λ={lam_hat:.3f})")
#     plt.xlabel("y")
#     plt.ylabel("Probability")
#     plt.title("Empirical vs CMP model")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

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
    Lc, _ = cho_factor(K, lower=True)  # keep only the factor
    return Lc

@partial(jit, static_argnames=('xmax',))
def logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax):
    mu   = kernel_embedding_poisson(lam, X_grid, xmax).reshape(-1, 1)
    w    = cho_solve((Lc, True), mu)          # (N,1)

    logf = (1.0 - nu) * gammaln(X_grid + 1.0)  # (N,)
    m    = jnp.max(logf)
    fexp = jnp.exp(logf - m)
    s    = jnp.dot(w.ravel(), fexp)            # scalar
    s    = jnp.clip(s, 1e-300, None)
    return jnp.log(s) + m + lam


# -----------------
# MC logZ helper (NEW FIXED-BASE IMPLEMENTATION)
# -----------------
@jit
def logZ_mc_fixed_base_from_samples(X_mc, nu, lam, lam0):
    log_f = X_mc * jnp.log(jnp.maximum(lam, 1e-300)) - \
            X_mc * jnp.log(lam0) + \
            (1.0 - nu) * gammaln(X_mc + 1.0)

    m = jnp.max(log_f)
    log_E = jnp.log(jnp.clip(jnp.mean(jnp.exp(log_f - m)), 1e-300)) + m
    return log_E + lam0

# -----------------
# Brute-force truncation (diagnostic truth proxy)
# -----------------
def logZ_trunc(nu, lam, y_max=8000): # Increased consistency
    xs = jnp.arange(0, y_max+1)
    log_p = -lam + xs * jnp.log(jnp.maximum(lam,1e-300)) - gammaln(xs+1.0)
    log_f = (1.0 - nu) * gammaln(xs + 1.0)
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

# MODIFIED: loss_and_aux now takes fixed MC items
def loss_and_aux(raw_params, sales_hist, run_bq, *, 
                 X_grid, Lc, xmax, lam0, X_mc):
    # MODIFIED: Removed redundant clipping
    lam = jnn.softplus(raw_params[1]) + 1e-12
    nu  = jnn.softplus(raw_params[0]) + 1e-3

    n, s1, s2 = cmp_suff_stats(sales_hist)

    if run_bq:
        logZ = logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax)
    else:
        # MODIFIED: Call the new fixed-base estimator
        logZ = logZ_mc_fixed_base_from_samples(X_mc, nu, lam, lam0)

    nll = n * logZ - s1 * jnp.log(lam) + nu * s2
    return nll, {"logZ": logZ, "nll": nll, "nu": nu, "lam": lam}


# -----------------
# Training loop
# -----------------
def run_experiment(seed, sales_hist, lr=1e-3, num_steps=1000, run_bq=True,
                   tail_eps=1e-10, min_cap=100, max_cap=5000):
    print(f"\n--- Seed {seed} | run_bq={run_bq} ---")
    key = random.PRNGKey(seed)

    nu0, lam0 = params_init(sales_hist)

    # ys = jnp.asarray(sorted(sales_hist.keys()), dtype=jnp.float64)
    # cs = jnp.asarray([sales_hist[int(k)] for k in ys], dtype=jnp.float64)
    # mean = float((ys * cs).sum() / jnp.maximum(cs.sum(), 1.0))
    # nu0  = 1.0
    # lam0 = float(jnp.clip(mean, 1e-6, 1e12))
    # nu0, lam0 = 1.0, 1.2
    raw_params = jnp.array([inv_softplus(nu0), inv_softplus(lam0)], dtype=jnp.float64)

    # X_grid, xmax = build_design(lam0, tail_eps=tail_eps, min_cap=min_cap, max_cap=max_cap)
    xmax = 200
    X_grid = jnp.arange(0, xmax + 1, dtype=jnp.int32)
    Lc = precompute_chol(X_grid) if run_bq else None
    
    # NEW: Generate a large, fixed set of MC samples ONCE before the loop
    if not run_bq:
        MC_N = int(X_grid.size) 
        key, subkey = random.split(key)
        X_mc = random.poisson(subkey, lam=lam0, shape=(MC_N,))
    else:
        X_mc = None

    optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
    opt_state = optimizer.init(raw_params)

    # MODIFIED: The step function is simpler as fixed items are passed to the loss
    @jit
    def step(raw_params, opt_state):
        (loss, aux), grads = jax.value_and_grad(
            lambda p: loss_and_aux(
                p, sales_hist, run_bq,
                X_grid=X_grid, Lc=Lc, xmax=int(xmax), 
                lam0=lam0, X_mc=X_mc
            ),
            has_aux=True
        )(raw_params)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        return raw_params, opt_state, loss, aux

    for t in range(num_steps):
        # MODIFIED: Key is no longer needed in the loop for the MC method
        raw_params, opt_state, loss, aux = step(raw_params, opt_state)
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
    RUN_BQ    = False   # Set to False to use the new fixed-base MC estimator
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

    # Compare with true NLL using the truncation proxy
    nll_star = nll_true(nu_hat, lam_hat, sales_hist, y_max=8000)
    print(f"True NLL (trunc) at final params = {nll_star:.6f}")

    # ---- Plot empirical vs model PMF ----
    # For plotting, always use the most accurate logZ method (truncation)
    logZ_model = logZ_trunc(nu_hat, lam_hat, y_max=8000)

    xs_emp, p_emp, n_emp = empirical_pmf_from_hist(sales_hist)
    y_max_plot = max(xs_emp.max(), int(lam_hat + 8*np.sqrt(max(lam_hat, 1e-8))), 60)
    ys = jnp.arange(0, y_max_plot + 1)

    logp_model = cmp_logpmf(ys, jnp.asarray(nu_hat), jnp.asarray(lam_hat), jnp.asarray(logZ_model))
    pmf_model  = np.asarray(jnp.exp(logp_model))

    logp_model_init = cmp_logpmf(ys, jnp.asarray(nu0), jnp.asarray(lam0), jnp.asarray(3.527730))
    pmf_model_init  = np.asarray(jnp.exp(logp_model_init))

    plt.figure(figsize=(8, 4.6))
    plt.bar(xs_emp, p_emp, width=0.9, alpha=0.5, label="data (empirical pmf)")
    plt.plot(np.asarray(ys), pmf_model, marker="o", lw=1.8,
             label=f"CMP model (ν={nu_hat:.3f}, λ={lam_hat:.3f})")
    plt.plot(np.asarray(ys), pmf_model_init, marker="s", lw=1.8,
             label=f"CMP model (ν={nu_hat:.3f}, λ={lam_hat:.3f})")
    plt.xlabel("y")
    plt.ylabel("Probability")
    plt.title("Empirical vs CMP model")
    plt.legend()
    plt.tight_layout()
    plt.show()