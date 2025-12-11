import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import vmap, jit, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc, erfinv, gammaln
from jax.scipy.stats import poisson as jax_poisson
import matplotlib.pyplot as plt
import os
from functools import partial
import scipy.stats
from jax.scipy.stats import norm
import math
import matplotlib.ticker as mticker
from matplotlib.ticker import LogFormatterMathtext

# ---------- Matplotlib style ----------
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 26
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.rc('axes', titlesize=26, labelsize=26, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=26, frameon=False)
plt.rc('xtick', labelsize=26, direction='in')
plt.rc('ytick', labelsize=26, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

# ============================================================
#                    Test integrand & Brownian BQ
# ============================================================

def f(x, mu=15.0, sigma=2.0):
    return jnp.exp(-((x - mu)**2) / (2.0 * sigma**2))

def brownian_kernel(X1, X2, amp=1.0):
    X1 = jnp.atleast_1d(X1).reshape(-1, 1)
    X2 = jnp.atleast_1d(X2).reshape(-1, 1)
    return amp * jnp.minimum(X1, X2.T)

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
    xmax = int(xmax)
    X = jnp.atleast_1d(X)
    return vmap(lambda xi: kernel_embedding_poisson_fixed(lambda_, xi, xmax))(X)

def double_integral_poisson(lambda_, xmax=200):
    xmax = int(xmax)
    k_vals = jnp.arange(0, xmax + 1)
    tail_probs = 1.0 - gammainc(k_vals + 1, lambda_)
    return jnp.sum(tail_probs ** 2)

def kernel_embedding_poisson_amp(lambda_, X, xmax, amp=1.0):
    return amp * kernel_embedding_poisson(lambda_, X, xmax)

def double_integral_poisson_amp(lambda_, xmax, amp=1.0):
    return amp * double_integral_poisson(lambda_, xmax)

@partial(jit, static_argnames=["xmax"])
def compute_integral_variance(X, lambda_, xmax, amp=1.0, jitter=1e-4):
    n = len(X)
    K = brownian_kernel(X, X, amp=amp) + (max(jitter, 5e-3) * amp + 1e-8) * jnp.eye(n)
    L, lower = cho_factor(K, lower=True)
    z = kernel_embedding_poisson_amp(lambda_, X, xmax, amp=amp).reshape(-1, 1)
    K_inv_z = cho_solve((L, lower), z)
    J = double_integral_poisson_amp(lambda_, xmax, amp=amp)
    return J - (z.T @ K_inv_z)[0, 0]

@partial(jit, static_argnames=["xmax"])
def compute_max_variance(X_obs, candidates, lambda_, xmax, amp=1.0):
    current_var = compute_integral_variance(X_obs, lambda_, xmax, amp=amp)
    updated_vars = vmap(lambda x: compute_integral_variance(
        jnp.concatenate([X_obs, jnp.array([x])]), lambda_, xmax, amp=amp
    ))(candidates)
    # pmf_vals = jax_poisson.pmf(candidates, mu=lambda_)
    kg_vals = (current_var - updated_vars) 
    # weighted_acq = kg_vals * pmf_vals
    # jax.debug.print("current_var = {}",current_var)
    # jax.debug.print("kg_vals max = {}", jnp.max(kg_vals))
    # jax.debug.print("kg_vals min = {}", jnp.min(kg_vals))
    return candidates[jnp.argmax(kg_vals)]

def bayesian_cubature_poisson(X, f_vals, lambda_, xmax, amp=1.0, jitter=1e-4):
    n = X.shape[0]
    K = brownian_kernel(X, X, amp=amp) + (jitter * amp + 1e-10) * jnp.eye(n)
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
    z = kernel_embedding_poisson_amp(lambda_, X, xmax, amp=amp).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)
    K_inv_f = cho_solve((L, lower), f_vals, check_finite=False)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)
    mean = (z.T @ K_inv_f)[0, 0]
    var = double_integral_poisson_amp(lambda_, xmax, amp=amp) - (z.T @ K_inv_z)[0, 0]
    return float(mean), float(jnp.maximum(var, 1e-30))

# ============================================================
#                Polynomial kernel 
# ============================================================

def _poly_center_scale_params(lambda_):
    mu = float(lambda_)
    s  = max(1.0, 9.0 * float(jnp.sqrt(lambda_)))
    alpha = 1.0 / (s * s)
    return mu, s, alpha
    

def _stirling2_table(max_d: int) -> jnp.ndarray:
    S = np.zeros((max_d + 1, max_d + 1), dtype=np.float64)
    S[0, 0] = 1.0
    for n in range(1, max_d + 1):
        for k in range(1, n + 1):
            S[n, k] = k * S[n - 1, k] + S[n - 1, k - 1]
    return jnp.asarray(S)

_S2 = _stirling2_table(8)  # good up to degree 8

def _poisson_raw_moments(lambda_: float, deg: int) -> jnp.ndarray:
    lam_pows = jnp.array([lambda_**k for k in range(deg + 1)], dtype=jnp.float64)
    S = _S2[:deg + 1, :deg + 1]
    m = (S * lam_pows[None, :]).sum(axis=1)
    return m.at[0].set(1.0)

def _binom_int(n, k):
    n = int(n); k = int(k)
    if k < 0 or k > n: return 0.0
    if k == 0 or k == n: return 1.0
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return float(num // den)

def polynomial_kernel(X1, X2, degree: int = 2, c: float = 0.3, amp: float = 1.0, lambda_=10.0):
    mu, _, alpha = _poly_center_scale_params(lambda_)
    X1 = jnp.atleast_1d(X1).reshape(-1, 1).astype(jnp.float64)
    X2 = jnp.atleast_1d(X2).reshape(1, -1).astype(jnp.float64)
    base = c + alpha * (X1 - mu) * (X2 - mu)
    K = amp * jnp.power(base, degree)
    return jnp.nan_to_num(K, posinf=1e300, neginf=-1e300)

def poly_kernel_embedding_poisson(lambda_, X, degree: int, c: float = 1.0, amp: float = 1.0):
    X = jnp.atleast_1d(X).astype(jnp.float64)
    mu, _, alpha = _poly_center_scale_params(lambda_)

    m_raw = _poisson_raw_moments(lambda_, degree)
    powers = jnp.arange(degree + 1, dtype=jnp.float64)
    # centered moments c_r = sum_{k=0}^r binom(r,k) (-mu)^{r-k} m_k
    B = jnp.array([[_binom_int(r, k) for k in range(degree + 1)] for r in range(degree + 1)],
                  dtype=jnp.float64)
    MU = jnp.array([((-mu) ** (r - k) if k <= r else 0.0)
                    for r in range(degree + 1) for k in range(degree + 1)],
                   dtype=jnp.float64).reshape(degree + 1, degree + 1)
    c_mom = (B * MU) @ m_raw  # shape (d+1,)

    xpow = (X - mu)[:, None] ** powers[None, :]
    binoms = jnp.array([_binom_int(degree, int(p)) for p in powers], dtype=jnp.float64)
    coeff = binoms * (c ** (degree - powers)) * (alpha ** powers)
    z = amp * (xpow * (coeff * c_mom)[None, :]).sum(axis=1)
    return jnp.nan_to_num(z)

def poly_double_integral_poisson(lambda_, degree: int, c: float = 1.0, amp: float = 1.0):
    mu, _, alpha = _poly_center_scale_params(lambda_)
    m_raw = _poisson_raw_moments(lambda_, degree)
    powers = jnp.arange(degree + 1, dtype=jnp.float64)

    B = jnp.array([[_binom_int(r, k) for k in range(degree + 1)] for r in range(degree + 1)],
                  dtype=jnp.float64)
    MU = jnp.array([((-mu) ** (r - k) if k <= r else 0.0)
                    for r in range(degree + 1) for k in range(degree + 1)],
                   dtype=jnp.float64).reshape(degree + 1, degree + 1)
    c_mom = (B * MU) @ m_raw

    binoms = jnp.array([_binom_int(degree, int(p)) for p in powers], dtype=jnp.float64)
    coeff = binoms * (c ** (degree - powers)) * (alpha ** powers)
    J = amp * jnp.sum(coeff * (c_mom ** 2))
    return float(J)


def bayesian_cubature_poisson_poly(X, f_vals, lambda_, degree: int, c: float = 1.0, amp: float = 1.0, jitter=1e-4):
    X = jnp.atleast_1d(X).astype(jnp.float64)
    f_vals = jnp.atleast_1d(f_vals).astype(jnp.float64)
    n = X.shape[0]

    K = polynomial_kernel(X, X, degree=degree, c=c, amp=amp, lambda_=lambda_)
    K = jnp.nan_to_num(K, posinf=1e300, neginf=-1e300)
    # jitter = 1e-3 if degree == 5 else (3e-3 if degree == 8 else 5e-4)
    K = K + (jitter * amp + 1e-10) * jnp.eye(n)

    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
    z = poly_kernel_embedding_poisson(lambda_, X, degree=degree, c=c, amp=amp).reshape(n, 1)
    z = jnp.nan_to_num(z)

    f_vals = f_vals.reshape(n, 1)
    K_inv_f = cho_solve((L, lower), f_vals, check_finite=False)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)

    mean = (z.T @ K_inv_f)[0, 0]
    var = poly_double_integral_poisson(lambda_, degree=degree, c=c, amp=amp) - (z.T @ K_inv_z)[0, 0]
    return float(mean), float(jnp.maximum(var, 1e-30))

# ============================================================
#                    RR / IS / Stratified
# ============================================================

@jit
def a_term(n, lambda_):
    return f(jnp.asarray(n)) * jax_poisson.pmf(n, mu=lambda_)

@jit
def poisson_tail_survival(n, lam_star):
    return jnp.where(n == 0, 1.0, 1.0 - jax_poisson.cdf(n - 1, lam_star))

def rr_poisson_tail_estimator(key, lambda_, lam_star, n_max=100000):
    total, k = 0.0, 0
    qk = 1.0
    subkey = key
    while True:
        total += a_term(k, lambda_) / qk

        if k >= n_max: 
            k += 1
            break

        qk1 = poisson_tail_survival(k + 1, lam_star)
        ck = jnp.clip(qk1 / qk, 0.0, 1.0)

        subkey, draw = random.split(subkey)
        if not random.bernoulli(draw, ck): 
            k += 1
            break
        k += 1    
        qk = qk1
    return total, k

def rr_mc_poisson(key, n, lambda_, rho=0.95):
    # rho sets continuation probability
    key, k = random.split(key)
    
    X0 = random.poisson(k, lam=lambda_)
    f0 = f(X0)
    
    S = f0
    g_prev = f0
    total = g_prev  # because g0 - g_{-1} = g0
    q = 1.0
    t = 1
    
    while True:
        # Bernoulli continuation
        key, kb = random.split(key)
        cont = random.bernoulli(kb, rho)
        if not cont or t >= n:
            break
        
        # draw next sample
        key, kx = random.split(key)
        Xt = random.poisson(kx, lam=lambda_)
        ft = f(Xt)
        
        S += ft
        g_t = S / (t + 1)
        
        # RR correction
        total += (g_t - g_prev) / q
        
        q *= rho
        g_prev = g_t
        t += 1

    return total


def nb_p_from_mean(lambda_, r):
    return float(lambda_) / (float(lambda_) + float(r))

def logpmf_nb(k, r, p):
    k = jnp.asarray(k, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    return (
        gammaln(k + r) - gammaln(r) - gammaln(k + 1.0)
        + r * jnp.log1p(-p) + k * jnp.log(p)
    )

def logpmf_poisson(k, lam):
    k = jnp.asarray(k, dtype=jnp.float64)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    return k * jnp.log(lam) - lam - gammaln(k + 1.0)

def sample_nb(key, n, r, eta):
    key_g, key_p = random.split(key)
    theta = float(eta) / float(r)
    y = random.gamma(key_g, a=float(r), shape=(n,)) * theta
    k = random.poisson(key_p, y)
    return k

def sample_nb_direct(key, n, r, eta):
    # eta plays role of λ in your code
    p = eta / (eta + r)
    return np.random.negative_binomial(r, p, n)

def is_nb_estimate(key, n, lambda_, r, f):
    eta = float(lambda_)
    p = nb_p_from_mean(eta, r)
    ks = sample_nb(key, n, r=r, eta=eta)
    f_vals = f(ks)
    lw = logpmf_poisson(ks, lam=lambda_) - logpmf_nb(ks, r=r, p=p)
    w = jnp.exp(lw)
    w_sum = jnp.sum(w)
    w2_sum = jnp.sum(w * w)
    est = jnp.mean(w * f_vals)
    ess = (w_sum * w_sum) / jnp.maximum(w2_sum, 1e-300)
    w_mean = jnp.mean(w)
    w_var  = jnp.var(w)
    return float(est), float(ess), float(w_mean), float(w_var)

# -------- Stratified node utilities (for bq_poly optional) --------

@partial(jit, static_argnames=("xmax",))
def _poisson_cdf_table(lambda_, xmax):
    ks   = jnp.arange(0, xmax + 1, dtype=jnp.int64)
    cdfk = jax_poisson.cdf(ks, mu=lambda_)
    return ks, cdfk

def poisson_region_masses(lambda_, cutoff):
    p1 = float(jax_poisson.cdf(int(cutoff) - 1, mu=lambda_))
    p2 = 1.0 - p1
    return p1, p2

def _draw_in_slab_by_cdf(key, m, a, b, lambda_, xmax):
    ks, cdfk = _poisson_cdf_table(lambda_, xmax)
    u = random.uniform(key, shape=(m,), minval=a, maxval=b)
    idx = jnp.searchsorted(cdfk, u, side="left")
    idx = jnp.clip(idx, 0, xmax)
    return ks[idx]

def two_region_samples(key, n_total, lambda_, xmax, cutoff=100):
    p1, _ = poisson_region_masses(lambda_, cutoff)
    n1 = int(np.round(n_total * p1))
    n1 = max(0, min(n1, n_total))
    n2 = n_total - n1
    key1, key2 = random.split(key)
    X1 = _draw_in_slab_by_cdf(key1, n1, 0.0, p1, lambda_, xmax)   # X < cutoff
    X2 = _draw_in_slab_by_cdf(key2, n2, p1, 1.0, lambda_, xmax)   # X >= cutoff
    return X1, X2, p1, 1.0 - p1

# ============================================================
#                       Runner & Plots
# ============================================================

def run_experiment(f, X, n_vals, lambda_, r, xmax, seed, all_states, experiment_type, amp, poly_deg=None, poly_c=0.3):
    key = jax.random.PRNGKey(seed)
    est_means = []
    diag_cond, diag_lmin, diag_wnorm, diag_wmax = [], [], [], []
    for n in n_vals:
        key, subkey = random.split(key)

        if experiment_type == "bq_poly":
            # stratified-ish nodes to reduce across-seed SE
            X1, X2, _, _ = two_region_samples(subkey, int(n), lambda_, xmax)
            X_eval = jnp.concatenate([X1, X2], axis=0)
            f_vals_eval = f(X_eval)
            est_mean, _ = bayesian_cubature_poisson_poly(
                X_eval, f_vals_eval, lambda_, degree=int(poly_deg), c=poly_c, amp=amp
            )

        elif experiment_type == "bq_bo":
            x0 = all_states[random.choice(subkey, len(all_states))]
            X, y = [x0], [f(x0)]
            while len(X) < n:
                used_set = set(map(int, X))
                unused = jnp.array([x for x in all_states if int(x) not in used_set])
                x_next = compute_max_variance(jnp.array(X), unused, lambda_, xmax, amp) if len(unused) > 0 else random.choice(key, all_states)
                y.append(f(x_next)); X.append(x_next)
            est_mean, _ = bayesian_cubature_poisson(jnp.array(X), jnp.array(y), lambda_, xmax, amp)

        elif experiment_type == "bq_random":
            n_py = int(n)
            X_n = X[n_py]
            X_eval = X_n[seed]
            f_vals_eval = f(X_eval)
            est_mean, _ = bayesian_cubature_poisson(X_eval, f_vals_eval, lambda_, xmax, amp)

        elif experiment_type == "bq_iid":
            idxs = random.choice(subkey, len(all_states), shape=(int(n),), replace=False)
            X_eval = all_states[idxs]
            f_vals_eval = f(X_eval)
            est_mean, _ = bayesian_cubature_poisson(X_eval, f_vals_eval, lambda_, xmax, amp)

        elif experiment_type == "mc":
            X_mc = random.poisson(subkey, lam=lambda_, shape=(int(n),))
            f_vals_mc = f(X_mc)
            est_mean = jnp.mean(f_vals_mc)


        elif experiment_type == "rr_pois":
            # lam_star = lambda_ + 6 * jnp.sqrt(lambda_) 
            # est_mean, _ = rr_poisson_tail_estimator(subkey, lambda_, lam_star=lam_star)
            rho = float(jnp.clip(1.0 - 1.0 / jnp.maximum(1, n), 0.0, 0.999999))
            est_mean = rr_mc_poisson(subkey, n, lambda_, rho=rho)

        elif experiment_type == "is":
            est_mean, ess, w_mean, w_var = is_nb_estimate(subkey, int(n), lambda_, r, f)

        elif experiment_type == "ss":
            X1, X2, p1, p2 = two_region_samples(subkey, int(n), lambda_, xmax)
            m1 = float(jnp.mean(f(X1))) if len(X1) > 0 else 0.0
            m2 = float(jnp.mean(f(X2))) if len(X2) > 0 else 0.0
            est_mean = p1 * m1 + p2 * m2

        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        est_means.append(est_mean)

    true_val = jnp.sum(f(all_states) * jax_poisson.pmf(all_states, mu=lambda_))
    return {
        "est_means": jnp.array(est_means),
        "true_val": true_val
    }

def run_multiple_seeds(f, X, n_vals, lambda_, r, xmax, num_seeds, all_states, experiment_type, amp=1.0, poly_deg=None, poly_c=0.3):
    num_seeds = int(num_seeds)
    all_errors_across_seeds = []

    for seed in range(num_seeds):
        print(f"\n--- Running {experiment_type}, Seed {seed+1}/{num_seeds} ---")
        result = run_experiment(f, X, n_vals, lambda_, r, xmax, seed, all_states, experiment_type, amp=amp, poly_deg=poly_deg, poly_c=poly_c)
        abs_error = jnp.abs(result["est_means"] - result["true_val"])
        all_errors_across_seeds.append(abs_error)
    
    all_errors = jnp.stack(all_errors_across_seeds)

    mean_abs_error = jnp.mean(all_errors, axis=0)
    se_abs_error = scipy.stats.sem(np.asarray(all_errors), axis=0)

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
        (ln,) = plt.plot(n_vals, y_line, linestyle=st['linestyle'], color=st['color'],
                         marker=st['marker'], label=st['label'])
        plt.fill_between(n_vals, y_low, y_high, color=st['color'], alpha=0.15)
        handles.append(ln); labels.append(st['label'])

    plt.xlabel("Number of Points", fontsize=32)
    plt.title("Poisson", fontsize=32)
    plt.ylabel("Absolute Error", fontsize=32)
    plt.yscale('log')
    plt.xscale('log')
    
    # Generate the log-formatted labels for the n_vals
    log_labels = [format_log_label(n) for n in n_vals]
    
    # Set the x-ticks to n_vals and use the custom log labels
    plt.xticks(n_vals, log_labels, fontsize=28) 

    plt.gca().xaxis.set_major_locator(plt.FixedLocator(n_vals))
    
    # 2. Use NullLocator to remove default MINOR ticks
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    # keep only these ticks
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), format='pdf')
    plt.close()
    # fig_legend, ax_legend = plt.subplots(figsize=(6, 1))
    # ax_legend.axis('off')
    # ax_legend.legend(handles, labels, ncol=4, loc='center', fontsize=26, frameon=False)
    # plt.savefig(os.path.join(save_path, "potts_abs_err_legend.pdf"), bbox_inches='tight')
    # plt.close(fig_legend)

def plot_results_poly(n_vals, all_results, unique, filename, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    if unique:
        styles = {
            'Poly d=2 (Active)': {'color': '#E75480', 'marker': 'D', 'label': 'Poly 2', 'linestyle': '-'},
            'Poly d=4 (Active)': {'color': '#FF69B4', 'marker': 'D', 'label': 'Poly 4', 'linestyle': '-'},
            'Poly d=8 (Active)': {'color': '#800080', 'marker': 'D', 'label': 'Poly 8', 'linestyle': '-'},
            'BayesSum': {'color': 'blue', 'marker': 'D', 'label': 'Brownian Kernel', 'linestyle': '-'}
        }
    else:
        styles = {
            'BayesSum': {'color': 'blue', 'marker': 'D', 'label': 'BayesSum', 'linestyle': '-'},
            'BayesSum (IID)': {'color': 'blue', 'marker': 'D', 'label': 'BayesSum (IID)', 'linestyle': '--'}
        }

    plt.figure(figsize=(10, 6))
    handles, labels = [], []
    for name in [m for m in styles.keys() if m in all_results]:
        data = all_results[name]
        mean_err = np.asarray(data["mean_abs_error"])
        se_err   = np.asarray(data["se_abs_error"])
        eps = 1e-12
        y_low  = np.clip(mean_err - 1.96 * se_err, eps, None)
        y_high = np.clip(mean_err + 1.96 * se_err, eps, None)
        y_line = np.clip(mean_err, eps, None)
        st = styles[name]
        (ln,) = plt.plot(n_vals, y_line, linestyle='-', color=st['color'],
                         marker=st['marker'], label=st['label'])
        plt.fill_between(n_vals, y_low, y_high, color=st['color'], alpha=0.15)
        handles.append(ln); labels.append(st['label'])
    plt.xlabel("Number of Points", fontsize=32)
    plt.ylabel("Absolute Error", fontsize=32)
    plt.title("Poisson", fontsize=32)
    plt.yscale('log')
    plt.xscale('log')
    # Generate the log-formatted labels for the n_vals
    log_labels = [format_log_label(n) for n in n_vals]
    
    # Set the x-ticks to n_vals and use the custom log labels
    plt.xticks(n_vals, log_labels, fontsize=28) 

    plt.gca().xaxis.set_major_locator(plt.FixedLocator(n_vals))
    
    # 2. Use NullLocator to remove default MINOR ticks
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend(handles, labels, loc='best', fontsize=32, frameon=False, ncol=1)
    plt.savefig(os.path.join(save_path, filename), format='pdf')
    plt.close()

def save_results_portable(filename, n_vals, all_results):
    save_dict = {'n_vals': np.array(n_vals, dtype=int)}
    for method, data in all_results.items():
        save_dict[f"{method}_mean_abs_error"] = np.asarray(data["mean_abs_error"], dtype=float)
        save_dict[f"{method}_se_abs_error"]   = np.asarray(data["se_abs_error"], dtype=float)
        save_dict[f"{method}_true_val"]       = np.array([data["true_val"]], dtype=float)
    np.savez(filename, **save_dict)
    print(f"Saved results to {filename}")

def load_results_portable(filename):
    data = np.load(filename)
    n_vals = data['n_vals']
    all_results = {}
    for key in data.files:
        if key == 'n_vals':
            continue
        method, field = key.split('_', 1)
        all_results.setdefault(method, {})[field] = data[key] if field != "true_val" else data[key][0]
    return n_vals, all_results

def estimate_alpha(N, MAE, min_index=1):
    N_fit = N[min_index:]
    mae_fit = MAE[min_index:]

    x = np.log(N_fit)
    y = np.log(mae_fit)

    a, b = np.polyfit(x, y, 1)
    alpha = -a
    return alpha, a, b

def calibration(n, X, seeds, lambda_val, xmax, all_states):
    
    t_e = jnp.sum(f(all_states) * jax_poisson.pmf(all_states, mu=lambda_val))
    mus, vars_ = [], []
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        target_n = int(n)

        X_eval = X[s]
        f_vals = f(X_eval)
        mu, var = bayesian_cubature_poisson(X_eval, f_vals, lambda_val, xmax)
        mus.append(mu); vars_.append(var)

    mus  = jnp.array(mus)
    vars_ = jnp.maximum(jnp.array(vars_), 1e-30)

    # nominal coverage grid and empirical coverage
    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = norm.ppf((1.0 + C_nom) / 2.0)  # two-sided CI half-width multiplier
    half = z[:, None] * jnp.sqrt(vars_)[None, :]
    inside = (t_e >= mus[None, :] - half) & (t_e <= mus[None, :] + half)
    emp_cov = jnp.mean(inside, axis=1)

    return float(t_e), C_nom, emp_cov, mus, vars_


# ============================================================
#                           Main
# ============================================================

def main():
    lambda_val = 30.0
    xmax = 200
    n_vals = jnp.array([10, 16, 24, 38, 60])
    num_seeds = 100
    all_states = jnp.arange(0, xmax + 1)
    r = 5.0

    result60 = np.load("poisson_bq_unique.npz")
    result = np.load("poisson_bq_unique_under38.npz")
    X10 = result["X10"]
    X16 = result["X16"]
    X24 = result["X24"]
    X38 = result["X38"]
    X60 = result60["X"]
    

    X = {10: X10, 16:X16, 24:X24, 38:X38, 60:X60}
    # all_results = {}

    # all_results['Active BayesSum'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "bq_bo", 1.0)
    # all_results['BayesSum'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "bq_random", 1.0)
    # all_results['BayesSum (IID)'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "bq_iid", 1.0)
    # all_results['MC'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "mc", 1.0)
    # all_results['RR'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "rr_pois", 1.0)
    # all_results['IS'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "is", 1.0)
    # all_results['SS'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "ss", 1.0)

    # save_results_portable("poisson.npz", n_vals, all_results)
    # print("poisson.npz")
    
    n_vals_loaded, all_results_loaded = load_results_portable("poisson.npz")
    # print(all_results_loaded)
    # plot_results(n_vals_loaded, all_results_loaded, True, "poisson_abs_err.pdf")
    # plot_results(n_vals_loaded, all_results_loaded, False, "poisson_abs_err_iid_vs_unique.pdf")


    methods = ["MC", "RR", "SS", "IS"]
    n_vals = jnp.array([10, 16, 24, 38, 60, 150, 300, 500, 1000])
    all_results = {}

    all_results['MC'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "mc", 1.0)
    all_results['RR'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "rr_pois", 1.0)
    all_results['IS'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "is", 1.0)
    all_results['SS'] = run_multiple_seeds(f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states, "ss", 1.0)

    for m in methods:
        alpha, a, b = estimate_alpha(n_vals, all_results[m]["mean_abs_error"], min_index=2)
        print(f"{m}: α ≈ {alpha:.3f}")

    # all_results_poly = {}
    # for deg in [2, 5, 8]:
    #     label = f"Poly d={deg} (Active)"
    #     all_results_poly[label] = run_multiple_seeds(
    #         f, X, n_vals, lambda_val, r, xmax, num_seeds, all_states,
    #         experiment_type="bq_poly", amp=1.0, poly_deg=deg, poly_c=0.3 
    #     )

    # save_results_portable("poisson_poly.npz", n_vals, all_results)

    # n_vals_loaded, all_results_loaded = load_results_portable("poisson_poly.npz")
    # all_results_loaded["BayesSum"] = all_results['BayesSum']

    # for k, v in all_results_loaded.items():
    #     print(k, "mean_abs_error:", np.asarray(v["mean_abs_error"]))
    #     print(k, "se_abs_error:  ", np.asarray(v["se_abs_error"]))
    # plot_results_poly(n_vals_loaded, all_results_loaded)


    # calibration_seeds = 200
    # n_samples = 60
    # unique_results = np.load("poisson_bq_unique.npz")
    # X = unique_results['X']
    # t_true, C_nom, emp_cov, mus, vars_ = calibration(n_samples, X, calibration_seeds, lambda_val, xmax, all_states)
   

    # jnp.savez("poisson_calibration_results.npz",
    #          t_true=t_true, C_nom=jnp.array(C_nom),
    #          emp_cov=jnp.array(emp_cov),
    #          mus=jnp.array(mus), vars=jnp.array(vars_))
    # print("poisson_calibration.npz")

if __name__ == "__main__":
    main()
