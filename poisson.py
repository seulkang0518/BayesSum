import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import vmap, jit, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc, erfinv
from jax.scipy.stats import poisson as jax_poisson
import matplotlib.pyplot as plt
import os
import time
from functools import partial
import scipy.stats # NEW: Import scipy.stats for sem
from jax.scipy.stats import norm

# --- All your preamble and helper functions remain the same ---
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

# @partial(jit, static_argnames=["xmax"])
# def compute_integral_variance(X, lambda_, xmax):
#     K = brownian_kernel(X, X) + 1e-4 * jnp.eye(len(X))
#     L, lower = cho_factor(K, lower=True)
#     z = kernel_embedding_poisson(lambda_, X, xmax).reshape(-1, 1)
#     K_inv_z = cho_solve((L, lower), z)
#     return double_integral_poisson(lambda_, xmax) - (z.T @ K_inv_z)[0, 0]

@partial(jit, static_argnames=["xmax"])
def compute_integral_variance(X, lambda_, xmax, amp=1.0):
    n = len(X)
    # Scale the kernel and jitter sensibly
    K = brownian_kernel(X, X, amp=amp) + (1e-4 * amp + 1e-10) * jnp.eye(n)
    L, lower = cho_factor(K, lower=True)
    z = kernel_embedding_poisson_amp(lambda_, X, xmax, amp=amp).reshape(-1, 1)
    K_inv_z = cho_solve((L, lower), z)
    J = double_integral_poisson_amp(lambda_, xmax, amp=amp)
    return J - (z.T @ K_inv_z)[0, 0]

# @partial(jit, static_argnames=["xmax"])
# def compute_max_variance(X_obs, candidates, lambda_, xmax):
#     current_var = compute_integral_variance(X_obs, lambda_, xmax)
#     updated_vars = vmap(lambda x: compute_integral_variance(jnp.concatenate([X_obs, jnp.array([x])]), lambda_, xmax))(candidates)
#     kg_vals = current_var - updated_vars
#     return candidates[jnp.argmax(kg_vals)]

@partial(jit, static_argnames=["xmax"])
def compute_max_variance(X_obs, candidates, lambda_, xmax, amp=1.0):
    # Argmax is invariant to amp, but we keep it for completeness.
    current_var = compute_integral_variance(X_obs, lambda_, xmax, amp=amp)
    updated_vars = vmap(lambda x: compute_integral_variance(
        jnp.concatenate([X_obs, jnp.array([x])]), lambda_, xmax, amp=amp
    ))(candidates)
    kg_vals = current_var - updated_vars  # scaled by amp; argmax unchanged
    return candidates[jnp.argmax(kg_vals)]

# def bayesian_cubature_poisson(X, f_vals, lambda_, xmax, jitter=1e-4):
#     n = X.shape[0]
#     K = brownian_kernel(X, X) + jitter * jnp.eye(n)
#     L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
#     z = kernel_embedding_poisson(lambda_, X, xmax).reshape(n, 1)
#     f_vals = f_vals.reshape(n, 1)
#     K_inv_f = cho_solve((L, lower), f_vals, check_finite=False)
#     K_inv_z = cho_solve((L, lower), z)
#     mean = (z.T @ K_inv_f)[0, 0]
#     var = double_integral_poisson(lambda_, xmax) - (z.T @ K_inv_z)[0, 0]
#     return float(mean), float(jnp.maximum(var, 1e-30))

def bayesian_cubature_poisson(X, f_vals, lambda_, xmax, amp=1.0, jitter=1e-4):
    n = X.shape[0]
    K = brownian_kernel(X, X, amp=amp) + (jitter * amp + 1e-10) * jnp.eye(n)
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
    z = kernel_embedding_poisson_amp(lambda_, X, xmax, amp=amp).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)
    K_inv_f = cho_solve((L, lower), f_vals, check_finite=False)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)
    mean = (z.T @ K_inv_f)[0, 0]  # invariant to amp (up to numerical noise)
    var = double_integral_poisson_amp(lambda_, xmax, amp=amp) - (z.T @ K_inv_z)[0, 0]
    return float(mean), float(jnp.maximum(var, 1e-30))

@jit
def a_term(n, lambda_):
    return f(jnp.asarray(n)) * jax_poisson.pmf(n, mu=lambda_)

def rr_geometric_estimator(key, lambda_, rho=0.97, n_max=100000):
    total, k = 0.0, 0
    qk = 1.0
    subkey = key
    while True:
        total += a_term(k, lambda_) / qk
        if k >= n_max: break
        subkey, draw = random.split(subkey)
        if not random.bernoulli(draw, rho): break
        k += 1
        qk *= rho
    return total, (k + 1)

@jit
def poisson_tail_survival(n, lam_star):
    return jnp.where(n == 0, 1.0, 1.0 - jax_poisson.cdf(n - 1, lam_star))

def rr_poisson_tail_estimator(key, lambda_, lam_star, n_max=100000):
    total, k = 0.0, 0
    qk = 1.0
    subkey = key
    while True:
        total += a_term(k, lambda_) / qk
        if k >= n_max: break
        qk1 = poisson_tail_survival(k + 1, lam_star)
        ck = jnp.clip(qk1 / qk, 0.0, 1.0)
        subkey, draw = random.split(subkey)
        if not random.bernoulli(draw, ck): break
        k += 1
        qk = qk1
    return total, (k + 1)

# This is your original experiment runner, returning just the means from one seed.
def run_experiment(f, n_vals, lambda_, xmax, seed, all_states, experiment_type, amp):
    key = jax.random.PRNGKey(seed)
    est_means = []
    for n in n_vals:
        key, subkey = random.split(key)
        if experiment_type == "bq_bo":
            x0 = all_states[random.choice(subkey, len(all_states))]
            X, y = [x0], [f(x0)]
            while len(X) < n:
                used_set = set(map(int, X))
                unused = jnp.array([x for x in all_states if int(x) not in used_set])
                x_next = compute_max_variance(jnp.array(X), unused, lambda_, xmax, amp) if len(unused) > 0 else random.choice(key, all_states)
                y.append(f(x_next)); X.append(x_next)
            est_mean, _ = bayesian_cubature_poisson(jnp.array(X), jnp.array(y), lambda_, xmax)
        elif experiment_type == "bq_random":
            idxs = random.choice(subkey, len(all_states), shape=(int(n),), replace=False)
            X_eval = all_states[idxs]
            f_vals_eval = f(X_eval)
            est_mean, _ = bayesian_cubature_poisson(X_eval, f_vals_eval, lambda_, xmax, amp)
        elif experiment_type == "mc":
            X_mc = random.poisson(subkey, lam=lambda_, shape=(int(n),))
            f_vals_mc = f(X_mc)
            est_mean = jnp.mean(f_vals_mc)
        elif experiment_type == "rr_geom":
            rho = jnp.clip(1.0 - 1.0 / jnp.maximum(1, n), 0.0, 0.999999)
            est_mean, _ = rr_geometric_estimator(subkey, lambda_, rho=float(rho))
        elif experiment_type == "rr_pois":
            lam_star = max(float(n) - 1.0, 1e-6)
            est_mean, _ = rr_poisson_tail_estimator(subkey, lambda_, lam_star=lam_star)
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")
        est_means.append(est_mean)
    true_val = jnp.sum(f(all_states) * jax_poisson.pmf(all_states, mu=lambda_))
    return {"est_means": jnp.array(est_means), "true_val": true_val}

# --- MODIFIED: This function now computes the mean and SE of the ABSOLUTE ERROR ---
def run_multiple_seeds(f, n_vals, lambda_, xmax, num_seeds, all_states, experiment_type, amp):
    all_errors_across_seeds = []
    
    for seed in range(num_seeds):
        print(f"\n--- Running {experiment_type}, Seed {seed+1}/{num_seeds} ---")
        result = run_experiment(f, n_vals, lambda_, xmax, seed, all_states, experiment_type, amp)
        
        # Calculate absolute error for this seed
        abs_error = jnp.abs(result["est_means"] - result["true_val"])
        all_errors_across_seeds.append(abs_error)

    # Stack errors: shape becomes (num_seeds, num_n_vals)
    all_errors = jnp.stack(all_errors_across_seeds)
    
    # Calculate mean and standard error of the absolute error
    mean_abs_error = jnp.mean(all_errors, axis=0)
    se_abs_error = scipy.stats.sem(all_errors, axis=0)
    
    return {"mean_abs_error": mean_abs_error, "se_abs_error": se_abs_error, "true_val": result["true_val"]}

# --- MODIFIED: This function now plots the mean absolute error with its SE ---
def plot_results(n_vals, all_results, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    styles = {
        'MC': {'color': 'k', 'marker': 'o', 'label': 'MC'},
        'DBQ': {'color': 'b', 'marker': 's', 'label': 'DBQ'},
        'Active DBQ': {'color': 'g', 'marker': '^', 'label': 'Active DBQ'},
        'RR': {'color': 'r', 'marker': 'D', 'label': 'RR'},
        # 'RR (PoisTail)': {'color': 'c', 'marker': '*', 'label': 'RR (PoisTail)'},
    }
    
    plt.figure(figsize=(10, 6))
    handles, labels = [], []

    # only plot methods that exist in all_results and are in styles
    for name in [m for m in styles.keys() if m in all_results]:
        data = all_results[name]

        # mean abs error and its SE across seeds
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
    plt.title("Poisson")
    plt.ylabel("Absolute Error")
    plt.yscale('log') # Errors are best viewed on a log scale
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "poisson_abs_err.pdf"), format='pdf')
    # plt.show()
    plt.close()

    fig_legend, ax_legend = plt.subplots(figsize=(6, 1))  # adjust size as needed
    ax_legend.axis('off')  # no axes
    ax_legend.legend(handles, labels, ncol=4, loc='center', fontsize=26, frameon=False)

    plt.savefig(os.path.join(save_path, "poisson_abs_err_legend.pdf"), bbox_inches='tight')
    plt.close(fig_legend)

def calibration(n, seeds, lambda_, xmax, amp, out_png="poisson_reliability_N60.png"):
    
    all_states = jnp.arange(0, xmax+1, dtype=jnp.int64)
    t_e = jnp.sum(f(all_states) * jax_poisson.pmf(all_states, mu=lambda_))
    mus, vars_ = [], []
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        idx = jax.random.choice(key, all_states.shape[0], shape=(n,), replace=False)
        X = all_states[idx]
        y = f(X, mu=15.0, sigma=2.0)
        mu, var = bayesian_cubature_poisson(X, y, lambda_, xmax, amp)
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

def variance_curve(n_vals, lambda_, xmax, amps, num_seeds, all_states):
    out = {}
    for amp in amps:
        vars_per_n = []
        for n in n_vals:
            seed_vars = []
            for s in range(num_seeds):
                key = jax.random.PRNGKey(s)
                idx = jax.random.choice(key, all_states.shape[0], shape=(int(n),), replace=False)
                X = all_states[idx]
                y = f(X)
                _, var = bayesian_cubature_poisson(X, y, lambda_, xmax, amp=amp)
                seed_vars.append(var)
            vars_per_n.append(float(np.mean(seed_vars)))
        out[amp] = np.array(vars_per_n)
    return out

def plot_variance_results(n_vals, var_dict, save_path="results", title="BQ posterior variance vs n"):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10,6))
    for amp, vals in var_dict.items():
        plt.plot(n_vals, np.asarray(vals), marker='o', label=f"amp={amp}")
    plt.yscale('log')
    plt.xlabel("Number of Points")
    plt.ylabel("Posterior Variance (log-scale)")
    plt.title(title)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "poisson_bq_variance_vs_n.pdf"), format='pdf')
    plt.close()


def main():
    lambda_val = 10.0
    xmax = 200
    n_vals = jnp.array([10, 20, 40, 60])
    num_seeds = 10 # Using 50 seeds for a reliable SE calculation
    all_states = jnp.arange(0, xmax + 1)

    # all_results = {}
    # all_results['Active BayesSum'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, num_seeds, all_states, "bq_bo")
    # all_results['BayesSum'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, num_seeds, all_states, "bq_random")
    # all_results['MC'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, num_seeds, all_states, "mc")
    # all_results['RR'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, num_seeds, all_states, "rr_geom")
    # # all_results['RR (PoisTail)'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, num_seeds, all_states, "rr_pois")

    # plot_results(n_vals, all_results)

    amps = [0.1, 0.5, 1.0, 1.5, 2.0, 4.0]
    var_dict = variance_curve(n_vals, lambda_val, xmax, amps, num_seeds, all_states)
    print(var_dict)
    # plot_variance_results(n_vals, var_dict, title="BQ posterior variance vs n (Brownian, Poisson prior)")

    # calibration_seeds = 200
    # n_samples = 60
    # t_true, C_nom, emp_cov, mus, vars_ = calibration(n_samples, calibration_seeds, lambda_val, xmax, all_states)
   

    # jnp.savez("poisson_calibration_results.npz",
    #          t_true=t_true, C_nom=jnp.array(C_nom),
    #          emp_cov=jnp.array(emp_cov),
    #          mus=jnp.array(mus), vars=jnp.array(vars_))
    # print("poisson_calibration.npz")

if __name__ == "__main__":
    main()