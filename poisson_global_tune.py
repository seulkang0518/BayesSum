import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import vmap, jit, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc
from jax.scipy.stats import poisson as jax_poisson
import matplotlib.pyplot as plt
import os
import time
from functools import partial
import scipy.stats
from jax.scipy.stats import norm
from scipy.optimize import minimize

# --- Matplotlib Preamble ---
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

def brownian_kernel(X1, X2, amplitude):
    X1 = jnp.atleast_1d(X1).reshape(-1, 1)
    X2 = jnp.atleast_1d(X2).reshape(-1, 1)
    return amplitude * jnp.minimum(X1, X2.T)

def kernel_embedding_poisson(lambda_, X, xmax):
    xmax = int(xmax)
    X = jnp.atleast_1d(X)
    x_vals = jnp.arange(0, xmax + 1)
    log_fact = jnp.cumsum(jnp.log(jnp.arange(1, xmax + 1)))
    fact = jnp.exp(jnp.concatenate([jnp.array([0.0]), log_fact]))
    pmf = (lambda_**x_vals / fact) * jnp.exp(-lambda_)
    
    def single_embedding(xi):
        mask = x_vals <= xi
        term1 = jnp.sum(jnp.where(mask, x_vals * pmf, 0.0))
        term2 = xi * gammainc(xi + 1, lambda_)
        return term1 + term2
        
    return vmap(single_embedding)(X)

def double_integral_poisson(lambda_, xmax=200):
    xmax = int(xmax)
    k_vals = jnp.arange(0, xmax + 1)
    tail_probs = 1.0 - gammainc(k_vals + 1, lambda_)
    return jnp.sum(tail_probs ** 2)

# #### UPDATED DECORATOR: Mark xmax as a static argument ####
@partial(jit, static_argnames='xmax')
def bq_with_fixed_amplitude(X, f_vals, lambda_, xmax, fixed_amplitude, jitter=1e-8):
    K = brownian_kernel(X, X, fixed_amplitude) + jitter * jnp.eye(X.shape[0])
    z_base = kernel_embedding_poisson(lambda_, X, xmax).reshape(-1, 1)
    integral_base = double_integral_poisson(lambda_, xmax)
    
    z_scaled = fixed_amplitude * z_base
    integral_sq_norm = fixed_amplitude * integral_base
    
    L, lower = cho_factor(K, lower=True)
    K_inv_f = cho_solve((L, lower), f_vals)
    K_inv_z = cho_solve((L, lower), z_scaled)
    
    mean = jnp.dot(z_scaled.T, K_inv_f)
    var = integral_sq_norm - jnp.dot(z_scaled.T, K_inv_z)
    
    # Ensure output shapes are scalar
    return mean[0], jnp.maximum(var[0,0], 1e-30)

def find_global_amplitude(f, all_states, lambda_, xmax):
    print("\n--- Finding global optimal amplitude on all_states ---")
    f_vals = f(all_states)
    
    @jit
    def estimate_and_logml_global(K, y):
        n = K.shape[0]
        C = K + 1e-8 * jnp.eye(n)
        y = y.reshape(-1)
        L, lower = cho_factor(C, lower=True)
        ones = jnp.ones_like(y)
        C_inv_ones = cho_solve((L, lower), ones)
        c_star = jnp.dot(ones, cho_solve((L, lower), y)) / jnp.dot(ones, C_inv_ones)
        r = y - c_star
        log_det_C = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        logml = -0.5 * jnp.dot(r, cho_solve((L, lower), r)) - 0.5 * log_det_C - 0.5 * n * jnp.log(2 * jnp.pi)
        return logml

    amplitude_initial = jnp.maximum(jnp.var(f_vals), 1e-4)
    initial_params = jnp.log(amplitude_initial)

    @jit
    def nll(log_amplitude):
        amplitude = jnp.exp(log_amplitude)
        K = brownian_kernel(all_states, all_states, amplitude)
        return -estimate_and_logml_global(K, f_vals)

    result = minimize(fun=nll, x0=initial_params, method='L-BFGS-B', jac=jit(jax.grad(nll)))
    amplitude_opt = jnp.exp(result.x[0])
    print(f"Found global optimal amplitude: {amplitude_opt:.4f}")
    return amplitude_opt

@partial(jit, static_argnames=["xmax"])
def compute_integral_variance(X, lambda_, xmax, amplitude=1.0):
    K = brownian_kernel(X, X, amplitude) + 1e-4 * jnp.eye(len(X))
    z = kernel_embedding_poisson(lambda_, X, xmax).reshape(-1, 1)
    z_scaled = amplitude * z
    integral_sq_norm = double_integral_poisson(lambda_, xmax) * amplitude
    L, lower = cho_factor(K, lower=True)
    K_inv_z = cho_solve((L, lower), z_scaled)
    return integral_sq_norm - (z_scaled.T @ K_inv_z)[0, 0]

@partial(jit, static_argnames=["xmax"])
def compute_max_variance(X_obs, candidates, lambda_, xmax):
    amplitude_bo = 1.0 
    current_var = compute_integral_variance(X_obs, lambda_, xmax, amplitude_bo)
    updated_vars = vmap(lambda x: compute_integral_variance(jnp.concatenate([X_obs, jnp.array([x])]), lambda_, xmax, amplitude_bo))(candidates)
    gains = current_var - updated_vars
    return candidates[jnp.argmax(gains)]

@jit
def a_term(n, lambda_):
    return f(jnp.asarray(n)) * jax_poisson.pmf(n, mu=lambda_)

def rr_geometric_estimator(key, lambda_, rho=0.97, n_max=100000):
    total, k = 0.0, 0; qk = 1.0; subkey = key
    while True:
        total += a_term(k, lambda_) / qk
        if k >= n_max: break
        subkey, draw = random.split(subkey)
        if not random.bernoulli(draw, rho): break
        k += 1; qk *= rho
    return total, (k + 1)

def run_experiment(f, n_vals, lambda_, xmax, seed, all_states, experiment_type, global_amplitude):
    key = jax.random.PRNGKey(seed)
    est_means = []
    for n in n_vals:
        key, subkey = random.split(key)
        if experiment_type == "bq_bo":
            x0 = all_states[random.choice(subkey, len(all_states))]
            X_list, y_list = [x0], [f(x0)]
            while len(X_list) < n:
                X_arr = jnp.array(X_list); used_set = set(map(int, X_arr))
                unused = jnp.array([x for x in all_states if int(x) not in used_set])
                if len(unused) == 0: break
                x_next = compute_max_variance(X_arr, unused, lambda_, xmax)
                y_list.append(f(x_next)); X_list.append(x_next)
            X_eval, f_vals_eval = jnp.array(X_list), jnp.array(y_list)
            est_mean, _ = bq_with_fixed_amplitude(X_eval, f_vals_eval, lambda_, xmax, global_amplitude)
        elif experiment_type == "bq_random":
            idxs = random.choice(subkey, len(all_states), shape=(int(n),), replace=False)
            X_eval, f_vals_eval = all_states[idxs], f(all_states[idxs])
            est_mean, _ = bq_with_fixed_amplitude(X_eval, f_vals_eval, lambda_, xmax, global_amplitude)
        elif experiment_type == "mc":
            X_mc = random.poisson(subkey, lam=lambda_, shape=(int(n),))
            f_vals_mc = f(X_mc)
            est_mean = jnp.mean(f_vals_mc)
        elif experiment_type == "rr_geom":
            rho = jnp.clip(1.0 - 1.0 / jnp.maximum(1, n), 0.0, 0.999999)
            est_mean, _ = rr_geometric_estimator(subkey, lambda_, rho=float(rho))
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")
        est_means.append(est_mean)
    true_val = jnp.sum(f(all_states) * jax_poisson.pmf(all_states, mu=lambda_))
    return {"est_means": jnp.array(est_means), "true_val": true_val}

def run_multiple_seeds(f, n_vals, lambda_, xmax, num_seeds, all_states, experiment_type, global_amplitude):
    all_errors_across_seeds = []
    for seed in range(num_seeds):
        print(f"\n--- Running {experiment_type}, Seed {seed+1}/{num_seeds} ---")
        result = run_experiment(f, n_vals, lambda_, xmax, seed, all_states, experiment_type, global_amplitude)
        abs_error = jnp.abs(result["est_means"] - result["true_val"])
        all_errors_across_seeds.append(abs_error)
    all_errors = jnp.stack(all_errors_across_seeds)
    mean_abs_error = jnp.mean(all_errors, axis=0)
    se_abs_error = scipy.stats.sem(all_errors, axis=0)
    return {"mean_abs_error": mean_abs_error, "se_abs_error": se_abs_error, "true_val": result["true_val"]}

def plot_results(n_vals, all_results, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    styles = {
        'MC': {'color': 'k', 'marker': 'o', 'label': 'MC'},
        'DBQ': {'color': 'b', 'marker': 's', 'label': 'DBQ'},
        'Active DBQ': {'color': 'g', 'marker': '^', 'label': 'Active DBQ'},
        'RR': {'color': 'r', 'marker': 'D', 'label': 'RR'},
    }
    plt.figure(figsize=(10, 6))
    handles, labels = [], []
    for name in [m for m in styles.keys() if m in all_results]:
        data = all_results[name]
        mean_err, se_err = np.asarray(data["mean_abs_error"]), np.asarray(data["se_abs_error"])
        eps = 1e-12
        y_low, y_high = np.clip(mean_err - 1.96 * se_err, eps, None), np.clip(mean_err + 1.96 * se_err, eps, None)
        y_line = np.clip(mean_err, eps, None)
        st = styles[name]
        (ln,) = plt.plot(n_vals, y_line, linestyle='-', color=st['color'], marker=st['marker'], label=st['label'])
        plt.fill_between(n_vals, y_low, y_high, color=st['color'], alpha=0.15)
        handles.append(ln); labels.append(st['label'])
    plt.xlabel("Number of Points")
    plt.title("Poisson Integration (Fixed Global Amplitude)")
    plt.ylabel("Absolute Error")
    plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend(handles=handles, labels=labels, loc='best')
    plt.savefig(os.path.join(save_path, "poisson_abs_err_v2.pdf"), format='pdf')
    plt.close()

def calibration(f, n, seeds, lambda_, xmax, global_amplitude, out_png="poisson_reliability_v2.pdf"):
    all_states = jnp.arange(0, xmax+1, dtype=jnp.int64)
    t_e = jnp.sum(f(all_states) * jax_poisson.pmf(all_states, mu=lambda_))
    mus, vars_ = [], []
    print(f"\n--- Running calibration with N={n}, Seeds={seeds}, Fixed Global Amplitude ---")
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        idx = jax.random.choice(key, all_states.shape[0], shape=(n,), replace=False)
        X = all_states[idx]
        y = f(X)
        mu, var = bq_with_fixed_amplitude(X, y, lambda_, xmax, global_amplitude)
        mus.append(mu); vars_.append(var)

    mus, vars_ = jnp.array(mus), jnp.maximum(jnp.array(vars_), 1e-30)
    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = norm.ppf((1.0 + C_nom) / 2.0)
    half = z[:, None] * jnp.sqrt(vars_)[None, :]
    inside = (t_e >= mus[None, :] - half) & (t_e <= mus[None, :] + half)
    emp_cov = jnp.mean(inside, axis=1)

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(np.asarray(C_nom), np.asarray(emp_cov), "o-", label=f"DBQ (Fixed Amp), N={n}")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Nominal Credible Interval")
    plt.ylabel("Empirical Coverage")
    plt.title("Uncertainty Calibration")
    plt.legend(loc='best')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, format='pdf')
    print(f"Calibration plot saved as {out_png}")
    plt.close()

def main():
    lambda_val = 10.0
    xmax = 200
    n_vals = jnp.array([10, 20, 40, 60, 80])
    num_seeds = 10 
    all_states = jnp.arange(0, xmax + 1)

    jitted_f = jit(f, static_argnames=['mu', 'sigma'])
    f_exp = partial(jitted_f, mu=15.0, sigma=2.0)
    
    global_amplitude = find_global_amplitude(f_exp, all_states, lambda_val, xmax)

    # all_results = {}
    # all_results['Active DBQ'] = run_multiple_seeds(f_exp, n_vals, lambda_val, xmax, num_seeds, all_states, "bq_bo", global_amplitude)
    # all_results['DBQ'] = run_multiple_seeds(f_exp, n_vals, lambda_val, xmax, num_seeds, all_states, "bq_random", global_amplitude)
    # all_results['MC'] = run_multiple_seeds(f_exp, n_vals, lambda_val, xmax, num_seeds, all_states, "mc", global_amplitude)
    # all_results['RR'] = run_multiple_seeds(f_exp, n_vals, lambda_val, xmax, num_seeds, all_states, "rr_geom", global_amplitude)
    
    # plot_results(n_vals, all_results)
    
    calibration_seeds = 200
    n_samples_calib = 60
    calibration(f_exp, n_samples_calib, calibration_seeds, lambda_val, xmax, global_amplitude)

if __name__ == "__main__":
    main()