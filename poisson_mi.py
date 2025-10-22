import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jit, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc
from jax.scipy.stats import poisson as jax_poisson
import matplotlib.pyplot as plt
from jax.scipy.special import erfinv

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 22
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
# plt.rc('font', family='Arial', size=12)
plt.rc('axes', titlesize=22, labelsize=22, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=22, frameon=False)
plt.rc('xtick', labelsize=18, direction='in')
plt.rc('ytick', labelsize=18, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

import os
import time
from functools import partial

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------

def f(x, mu=15.0, sigma=2.0):
    return jnp.exp(-((x - mu)**2) / (2.0 * sigma**2))

def brownian_kernel_raw(X1, X2):
    X1 = jnp.atleast_1d(X1).reshape(-1, 1)
    X2 = jnp.atleast_1d(X2).reshape(-1, 1)
    return jnp.minimum(X1, X2.T)

def kernel_embedding_poisson_fixed_raw(lambda_, xi, xmax=200):
    xmax = int(xmax)
    x_vals = jnp.arange(0, xmax + 1)
    mask = x_vals <= xi
    log_fact = jnp.cumsum(jnp.log(jnp.arange(1, xmax + 1)))
    fact = jnp.exp(jnp.concatenate([jnp.array([0.0]), log_fact]))
    pmf = (lambda_**x_vals / fact) * jnp.exp(-lambda_)
    term1 = jnp.sum(jnp.where(mask, x_vals * pmf, 0.0))
    term2 = xi * gammainc(xi + 1, lambda_)
    return term1 + term2

def kernel_embedding_poisson_raw(lambda_, X, xmax):
    xmax = int(xmax)
    X = jnp.atleast_1d(X)
    def single_embed(xi):
        return kernel_embedding_poisson_fixed_raw(lambda_, xi, xmax)
    return vmap(single_embed)(X)

def double_integral_poisson_raw(lambda_, xmax=200):
    xmax = int(xmax)
    k_vals = jnp.arange(0, xmax + 1)
    tail_probs = 1.0 - gammainc(k_vals + 1, lambda_)
    return jnp.sum(tail_probs ** 2)

# @partial(jit, static_argnames=["xmax"])
# def compute_integral_variance_raw(X, lambda_, xmax):
#     K = brownian_kernel(X, X) + 1e-4 * jnp.eye(len(X))
#     L, lower = cho_factor(K, lower=True)
#     z = kernel_embedding_poisson(lambda_, X, xmax).reshape(-1, 1)
#     K_inv_z = cho_solve((L, lower), z)
#     return double_integral_poisson(lambda_, xmax) - (z.T @ K_inv_z)[0, 0]

# @partial(jit, static_argnames=["xmax"])
# def compute_max_variance_raw(X_obs, candidates, lambda_, xmax):
#     current_var = compute_integral_variance(X_obs, lambda_, xmax)
#     def updated_var(x):
#         X_aug = jnp.concatenate([X_obs, jnp.array([x])])
#         return compute_integral_variance(X_aug, lambda_, xmax)
#     updated_vars = vmap(updated_var)(candidates)
#     kg_vals = current_var - updated_vars
#     return candidates[jnp.argmax(kg_vals)]
# ------------------------------------------------------------
# Acquisition functions for Bayesian Optimization (your code)
# ------------------------------------------------------------

def brownian_kernel(X1, X2, s):
    return brownian_kernel_raw(X1, X2) / s

def kernel_embedding_poisson(lambda_, X, xmax, s):
    return kernel_embedding_poisson_raw(lambda_, X, xmax) / s

def double_integral_poisson(lambda_, xmax, s):
    return double_integral_poisson_raw(lambda_, xmax) / s

@partial(jit, static_argnames=["xmax"])
def compute_integral_variance(X, lambda_, xmax, s, jitter_rel=1e-6):
    K = brownian_kernel(X, X, s) + 1e-4 * jnp.eye(len(X))
    L, lower = cho_factor(K, lower=True)
    z = kernel_embedding_poisson(lambda_, X, xmax, s).reshape(-1, 1)
    K_inv_z = cho_solve((L, lower), z)
    return double_integral_poisson(lambda_, xmax, s) - (z.T @ K_inv_z)[0, 0]

@partial(jit, static_argnames=["xmax"])
def compute_max_variance(X_obs, candidates, lambda_, xmax, s):
    current_var = compute_integral_variance(X_obs, lambda_, xmax, s)
    def updated_var(x):
        X_aug = jnp.concatenate([X_obs, jnp.array([x])])
        return compute_integral_variance(X_aug, lambda_, xmax, s)
    updated_vars = vmap(updated_var)(candidates)
    kg_vals = current_var - updated_vars
    return candidates[jnp.argmax(kg_vals)]

# @jit
# def gp_posterior_variance(x_cand, X_obs, L_factor):
#     k_star_star = brownian_kernel(x_cand, x_cand)[0, 0]
#     k_star = brownian_kernel(X_obs, x_cand)
#     v = cho_solve((L_factor, True), k_star)
#     return k_star_star - jnp.dot(k_star.T, v)

# @partial(jit, static_argnames=["xmax"])
# def compute_max_mi(X_obs, candidates, lambda_, xmax):
#     """
#     Mutual Information acquisition using eta(x*) = mu_P(x*) - k(x*,X) K^{-1} z
#     """
#     sigma2_bq = compute_integral_variance(X_obs, lambda_, xmax)
#     K = brownian_kernel(X_obs, X_obs) + 1e-4 * jnp.eye(len(X_obs))
#     L, _ = cho_factor(K, lower=True)

#     z_obs = kernel_embedding_poisson(lambda_, X_obs, xmax).reshape(-1, 1)
#     K_inv_z = cho_solve((L, True), z_obs)

#     def mi_acquisition(x_cand):
#         k_tilde = gp_posterior_variance(x_cand, X_obs, L)
#         mu_p_cand = kernel_embedding_poisson(lambda_, x_cand, xmax)[0]
#         k_star = brownian_kernel(X_obs, x_cand)
#         eta = mu_p_cand - jnp.dot(k_star.T, K_inv_z)
#         numerator = sigma2_bq * k_tilde
#         denominator = numerator - eta**2
#         safe_ratio = numerator / jnp.maximum(denominator, 1e-20)
#         return 0.5 * jnp.log(jnp.maximum(safe_ratio, 1.0))

#     mi_vals = vmap(mi_acquisition)(candidates)
#     return candidates[jnp.argmax(mi_vals)]

# ------------------------------------------------------------
# Bayesian Cubature (your code)
# ------------------------------------------------------------

def bayesian_cubature_poisson(X, f_vals, lambda_, xmax, s, jitter=1e-4):
    n = X.shape[0]
    K = brownian_kernel(X, X, s) + jitter * jnp.eye(n)
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
    z = kernel_embedding_poisson(lambda_, X, xmax, s).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)
    K_inv_f = cho_solve((L, lower), f_vals, check_finite=False)
    K_inv_z = cho_solve((L, lower), z)
    mean = (z.T @ K_inv_f)[0, 0]
    var = double_integral_poisson(lambda_, xmax, s) - (z.T @ K_inv_z)[0, 0]
    return float(mean), float(jnp.maximum(var, 1e-30))

# ------------------------------------------------------------
# Russian Roulette estimators (NEW)
# ------------------------------------------------------------

@jit
def a_term(n, lambda_):
    nn = jnp.asarray(n)
    return f(nn) * jax_poisson.pmf(nn, mu=lambda_)

def rr_geometric_estimator(key, lambda_, rho=0.97, n_max=100000):

    total = 0.0
    k = 0
    qk = 1.0  # q(0)=1
    subkey = key
    while True:
        total += a_term(k, lambda_) / qk
        if k >= n_max:
            break
        subkey, draw = random.split(subkey)
        cont = random.bernoulli(draw, rho)
        if not cont:
            break
        k += 1
        qk *= rho
    return total, (k + 1)

@jit
def poisson_tail_survival(n, lam_star):
    
    return jnp.where(n == 0, 1.0, 1.0 - jax_poisson.cdf(n - 1, lam_star))

def rr_poisson_tail_estimator(key, lambda_, lam_star, n_max=100000):

    total = 0.0
    k = 0
    qk = 1.0  # q(0)=1
    subkey = key
    while True:
        total += a_term(k, lambda_) / qk
        if k >= n_max:
            break
        qk1 = poisson_tail_survival(k + 1, lam_star)
        ck = jnp.clip(qk1 / qk, 0.0, 1.0)
        subkey, draw = random.split(subkey)
        cont = random.bernoulli(draw, ck)
        if not cont:
            break
        k += 1
        qk = qk1
    return total, (k + 1)

# ------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------

def run_experiment(f, n_vals, lambda_, xmax, s, seed, all_states, experiment_type):
    key = jax.random.PRNGKey(seed)
    true_val = jnp.sum(f(all_states) * jax_poisson.pmf(all_states, mu=lambda_))
    est_means, times = [], []

    for n in n_vals:
        start = time.time()
        key, subkey = random.split(key)

        if experiment_type == "bq_bo":
        
            x0 = all_states[jax.random.choice(subkey, len(all_states))]
            X, y = [x0], [f(x0)]
            while len(X) < n:
                used_set = set(map(int, X))
                unused = jnp.array([x for x in all_states if int(x) not in used_set])
                if len(unused) > 0:
                    x_next = compute_max_variance(jnp.array(X), unused, lambda_, xmax, s)
                else:
                    x_next = random.choice(subkey, all_states)
                y.append(f(x_next)); X.append(x_next)
            X_eval, f_vals_eval = jnp.array(X), jnp.array(y)
            est_mean, _ = bayesian_cubature_poisson(X_eval, f_vals_eval, lambda_, xmax, s)

        elif experiment_type == "bq_random":
            idxs = random.choice(subkey, len(all_states), shape=(int(n),), replace=False)
            X_eval = all_states[idxs].reshape(-1, 1)
            f_vals_eval = f(X_eval)
            est_mean, _ = bayesian_cubature_poisson(X_eval, f_vals_eval, lambda_, xmax, s)

        elif experiment_type == "mc":
            X_mc = jax.random.poisson(subkey, lam=lambda_, shape=(int(n), 1))
            f_vals_mc = f(X_mc)
            est_mean = jnp.mean(f_vals_mc)

        elif experiment_type == "rr_geom":
            # Choose rho so E[tau+1] ≈ n  -> rho = 1 - 1/n
            rho = jnp.clip(1.0 - 1.0 / jnp.maximum(1, n), 0.0, 0.999999)
            est_mean, used_terms = rr_geometric_estimator(subkey, lambda_, rho=float(rho))

        elif experiment_type == "rr_pois":
            # Choose lam_star so E[tau+1] = E[Z]+1 = lam_star+1 ≈ n
            lam_star = max(float(n) - 1.0, 1e-6)
            est_mean, used_terms = rr_poisson_tail_estimator(subkey, lambda_, lam_star=lam_star)

        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        jax.block_until_ready(est_mean)
        est_means.append(est_mean)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"n={int(n)}, Type={experiment_type}, Time={elapsed:.3f}s, "
              f"Err={float(jnp.abs(est_mean - true_val)):.8f}")

    return {"true_val": true_val, "est_means": jnp.array(est_means), "times": jnp.array(times)}

def run_multiple_seeds(f, n_vals, lambda_, xmax, s, num_seeds, all_states, experiment_type):
    all_means, all_times = [], []
    for seed in range(num_seeds):
        print(f"\n--- Running {experiment_type}, Seed {seed} ---")
        result = run_experiment(f, n_vals, lambda_, xmax, s, seed, all_states, experiment_type)
        all_means.append(result["est_means"])
        all_times.append(result["times"])

    all_means = jnp.stack(all_means)
    mean_abs_error = jnp.mean(jnp.abs(all_means - result["true_val"]), axis=0)
    avg_time = jnp.mean(jnp.array(all_times), axis=0)
    return {"mean_abs_error": mean_abs_error, "times_mean": avg_time, "true_val": result["true_val"]}

# ------------------------------------------------------------
# Plotting and main
# ------------------------------------------------------------
def reliability_curve(N=60, seeds=200, lam=10.0, xmax=200,
                      out_png="poisson_reliability_N60.png"):
    all_x = jnp.arange(0, xmax+1, dtype=jnp.int64)
    # exact ground truth under Poisson(λ)
    I_true = jnp.sum(f(all_x) * jax_poisson.pmf(all_x, mu=lam))

    mus, vars_ = [], []
    for s in range(seeds):
        key = random.PRNGKey(s)
        idx = random.choice(key, all_x.shape[0], shape=(N,), replace=False)
        X = all_x[idx]
        y = f(X)
        mu, var = bayesian_cubature_poisson(X, y, lam, xmax, s)
        mus.append(mu); vars_.append(var)

    mus   = jnp.array(mus)
    vars_ = jnp.maximum(jnp.array(vars_), 1e-30)

    C_nom = jnp.linspace(0.05, 0.99, 20)                 # nominal two-sided coverages
    z = jnp.sqrt(2.0) * erfinv(C_nom)                    # Gaussian half-width multipliers
    half = z[:,None] * jnp.sqrt(vars_)[None,:]
    inside = (I_true >= mus[None,:] - half) & (I_true <= mus[None,:] + half)
    emp_cov = jnp.mean(inside, axis=1)

    # plot
    plt.figure(figsize=(6.0,4.5))
    plt.plot(C_nom, emp_cov, "o-", label=f"N={N}")
    plt.plot([0,1],[0,1], "k--", label="Ideal")
    plt.xlabel("Nominal two-sided coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Uncertainty calibration (Poisson, fixed N)")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Poisson: true E[f(X)]={float(I_true):.6f}  |  saved {out_png}")

def plot_results(n_vals, all_results, save_path="results"):
    os.makedirs(save_path, exist_ok=True)

    styles = {
        'MC':        {'color': 'k', 'marker': 'o', 'label': 'MC'},
        'DBQ':       {'color': 'b', 'marker': 's', 'label': 'DBQ'},
        'Active DBQ':{'color': 'g', 'marker': '^', 'label': 'Active DBQ'},
        'RR (Geom)': {'color': 'r', 'marker': 'D', 'label': 'RR (Geom)'},
        'RR (PoisTail)': {'color': 'c', 'marker': '*', 'label': 'RR (PoisTail)'},
    }

    plt.figure(figsize=(10, 6))
    handles, labels = [], []
    for name, data in all_results.items():
        errors = jnp.clip(data["mean_abs_error"], 1e-10, None)
        line, = plt.plot(n_vals, errors, **styles[name], linestyle='-')
        handles.append(line)
        labels.append(styles[name]['label'])

    plt.xlabel("Number of Points")
    plt.ylabel("Absolute Error")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abs_err_poisson.pdf"), format='pdf')
    plt.close()

    fig_legend, ax_legend = plt.subplots(figsize=(6, 1))  # adjust size as needed
    ax_legend.axis('off')  # no axes
    ax_legend.legend(handles, labels, ncol=2, loc='center', fontsize=14, frameon=False)

    plt.savefig(os.path.join(save_path, "abs_err_poisson_legend.pdf"), bbox_inches='tight')
    plt.close(fig_legend)

def main():
    lambda_val = 10.0
    xmax = 200
    n_vals = jnp.array([10, 20, 40, 60])  # budgets
    num_seeds = 10
    all_states = jnp.arange(0, xmax + 1)
    s = 10000

    all_results = {}
    all_results['Active DBQ'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, s, num_seeds, all_states, "bq_bo")
    all_results['DBQ'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, s, num_seeds, all_states, "bq_random")
    all_results['MC'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, s, num_seeds, all_states, "mc")
    all_results['RR (Geom)'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, s, num_seeds, all_states, "rr_geom")
    all_results['RR (PoisTail)'] = run_multiple_seeds(f, n_vals, lambda_val, xmax, s, num_seeds, all_states, "rr_pois")

    plot_results(n_vals, all_results)

    # print("\n\n" + "="*45)
    # print("=== Final Averaged Results ===")
    # print("="*45)
    # print(f"n = {n_vals}")
    # print(f"True value: {all_results['MC']['true_val']:.10f}")
    # print("-" * 45)
    # for name, data in all_results.items():
    #     print(f"{name:<20} mean error: {data['mean_abs_error']}")
    # print("-" * 45)
    # reliability_curve(N=60, seeds=200, lam=10.0, xmax=200)

if __name__ == "__main__":
    main()
