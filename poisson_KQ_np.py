# -----------------------------------------------
# Poisson BQ with Brownian kernel (Option A scale)
# Reliability curve with uniform vs Poisson designs
# -----------------------------------------------
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, vmap, jit
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc
from jax.scipy.stats import poisson as jax_poisson, norm
import matplotlib.pyplot as plt
from functools import partial
from jax.scipy.special import erfinv
# -----------------
# Matplotlib config
# -----------------
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.rc('axes', titlesize=12, labelsize=12, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=12, frameon=False)
plt.rc('xtick', labelsize=12, direction='in')
plt.rc('ytick', labelsize=12, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

# ----------------
# Test integrand f
# ----------------
def f(x, mu=15.0, sigma=2.0):
    return jnp.exp(-((x - mu)**2) / (2.0 * sigma**2))

# -------------------------------------------
# Unscaled Brownian kernel helpers on Z_{\ge0}
# -------------------------------------------
def brownian_kernel_raw(X1, X2):
    X1 = jnp.atleast_1d(X1).reshape(-1, 1)
    X2 = jnp.atleast_1d(X2).reshape(-1, 1)
    return jnp.minimum(X1, X2.T)

def kernel_embedding_poisson_fixed(lambda_, xi, xmax=200):
    xmax = int(xmax)
    x_vals = jnp.arange(0, xmax + 1)
    mask = x_vals <= xi
    log_fact = jnp.cumsum(jnp.log(jnp.arange(1, xmax + 1)))
    fact = jnp.exp(jnp.concatenate([jnp.array([0.0]), log_fact]))
    pmf = (lambda_**x_vals / fact) * jnp.exp(-lambda_)
    term1 = jnp.sum(jnp.where(mask, x_vals * pmf, 0.0))
    term2 = xi * gammainc(xi + 1, lambda_)
    return term1 + term2

def kernel_embedding_poisson(lambda_, X, xmax):
    X = jnp.atleast_1d(X)
    return vmap(lambda xi: kernel_embedding_poisson_fixed(lambda_, xi, xmax))(X)

def double_integral_poisson(lambda_, xmax=200):
    xmax = int(xmax)
    k_vals = jnp.arange(0, xmax + 1)
    tail_probs = 1.0 - gammainc(k_vals + 1, lambda_)
    return jnp.sum(tail_probs ** 2)

# --------------------------
# Option A: scaling by 1 / s
# --------------------------
def choose_scale(lambda_, xmax, method="sigma", k=5.0, q=0.999):
    if method == "sigma":
        s = int(jnp.ceil(lambda_ + k * jnp.sqrt(lambda_)))
    elif method == "quantile":
        t = 0
        cdf = jax_poisson.cdf(t, mu=lambda_)
        while (cdf < q) and (t < int(xmax)):
            t += 1
            cdf = jax_poisson.cdf(t, mu=lambda_)
        s = t
    else:
        s = int(xmax)
    return float(min(int(xmax), max(1, s)))

def brownian_kernel_scaled(X1, X2, s):
    return brownian_kernel_raw(X1, X2) / s

def kme_poisson_scaled(lambda_, X, xmax, s):
    return kernel_embedding_poisson(lambda_, X, xmax) / s

def J_poisson_scaled(lambda_, xmax, s):
    return double_integral_poisson(lambda_, xmax) / s

# --------------------------
# Bayesian Cubature (scaled)
# --------------------------
def bayesian_cubature_poisson_scaled(X, f_vals, lambda_, xmax, s, jitter_rel=1e-6):
    n = X.shape[0]
    K = brownian_kernel_scaled(X, X, s)
    diag_mean = jnp.maximum(jnp.mean(jnp.diag(K)), 1e-16)
    K = K + (jitter_rel * diag_mean) * jnp.eye(n)
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
    z = kme_poisson_scaled(lambda_, X, xmax, s).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)
    K_inv_f = cho_solve((L, lower), f_vals, check_finite=False)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)
    mean = (z.T @ K_inv_f)[0, 0]
    var = J_poisson_scaled(lambda_, xmax, s) - (z.T @ K_inv_z)[0, 0]
    return float(mean), float(jnp.maximum(var, 1e-30))

# --------------------------
# Acquisition (max variance)
# --------------------------
@partial(jit, static_argnames=["xmax"])
def compute_integral_variance_scaled(X, lambda_, xmax, s, jitter_rel=1e-6):
    n = X.shape[0]
    K = brownian_kernel_scaled(X, X, s)
    diag_mean = jnp.maximum(jnp.mean(jnp.diag(K)), 1e-16)
    K = K + (jitter_rel * diag_mean) * jnp.eye(n)
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
    z = kme_poisson_scaled(lambda_, X, xmax, s).reshape(n, 1)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)
    return J_poisson_scaled(lambda_, xmax, s) - (z.T @ K_inv_z)[0, 0]

@partial(jit, static_argnames=["xmax"])
def compute_max_variance_scaled(X_obs, candidates, lambda_, xmax, s):
    current_var = compute_integral_variance_scaled(X_obs, lambda_, xmax, s)
    def updated_var(x):
        X_aug = jnp.concatenate([X_obs, jnp.array([x])])
        return compute_integral_variance_scaled(X_aug, lambda_, xmax, s)
    updated_vars = vmap(updated_var)(candidates)
    kg_vals = current_var - updated_vars
    return candidates[jnp.argmax(kg_vals)]

# --------------------------
# Reliability Curve (scaled)
# --------------------------
def reliability_curve_scaled(n, seeds, lam, xmax):
    
    s = 10000
    all_x = jnp.arange(0, xmax + 1, dtype=jnp.int64)
    I_true = jnp.sum(f(all_x) * jax_poisson.pmf(all_x, mu=lam))

    mus, vars_ = [], []
    for seed in range(seeds):
        key = random.PRNGKey(seed)
        idx = random.choice(key, all_x.shape[0], shape=(n,), replace=False)
        X, y = all_x[idx], f(all_x[idx])

        mu, var = bayesian_cubature_poisson_scaled(X, y, lam, xmax, s)
        mus.append(mu); vars_.append(var)

    mus, vars_ = jnp.array(mus), jnp.array(vars_)
    C_nom = jnp.linspace(0.05, 0.99, 20)
    z = jnp.sqrt(2.0) * erfinv(C_nom)  # two-sided CI half-width multiplier
    half = z[:,None] * jnp.sqrt(vars_)[None,:]
    inside = (I_true >= mus[None,:] - half) & (I_true <= mus[None,:] + half)
    emp_cov = jnp.mean(inside, axis=1)

    return float(I_true), C_nom, emp_cov, mus, vars_

# --------------------------
# main
# --------------------------
def main():
    lam, xmax = 10.0, 200
    # Uniform design
    calibration_seeds = 200
    n_samples = 60
    t_true, C_nom, emp_cov, mus, vars_  = reliability_curve_scaled(n_samples, calibration_seeds, lam, xmax)
    print(mus)
    print(vars_)
    # # Poisson-weighted design
    # reliability_curve_scaled(N=60, seeds=50, lam=lam, xmax=xmax,
    #                          design="poisson", out_png="relcurve_poisson.png")
    # # BO design
    # reliability_curve_scaled(N=60, seeds=20, lam=lam, xmax=xmax,
    #                          design="bo", out_png="relcurve_bo.png")

if __name__ == "__main__":
    main()