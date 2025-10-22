import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jit, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc
from jax.scipy.stats import poisson as jax_poisson
import os
from functools import partial

## --------- Basic Settings ----------

def f(X):
    X = jnp.atleast_2d(X)
    log_terms = jnp.log(X+1) - jnp.sqrt(X)
    return jnp.exp(jnp.sum(log_terms, axis=1))

def brownian_kernel(X1, X2):
    # Product of 1D Brownian kernels
    X1 = jnp.atleast_2d(X1)
    X2 = jnp.atleast_2d(X2)
    mins = jnp.minimum(X1[:, None, :], X2[None, :, :])  # (n, m, d)
    return jnp.prod(mins, axis=-1)

## --------- KME and Integral Calculations ----------

def make_precomp(xmaxs):
    # Precomputes factorials to speed up KME calculation
    x_vals_list = []
    fact_list = []
    for m in xmaxs:
        m = int(m)
        x_vals = jnp.arange(0, m + 1)
        if m >= 1:
            log_fact = jnp.cumsum(jnp.log(jnp.arange(1, m + 1)))
            fact = jnp.exp(jnp.concatenate([jnp.array([0.0]), log_fact]))
        else:
            fact = jnp.array([1.0])
        x_vals_list.append(x_vals)
        fact_list.append(fact)
    return (x_vals_list, fact_list)

def kme_poisson_1d_pre(lambda_, xi, x_vals, fact):
    # 1D Kernel Mean Embedding E[min(X, xi)] for X ~ Poisson(lambda)
    pmf = (lambda_ ** x_vals / fact) * jnp.exp(-lambda_)
    term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals * pmf, 0.0))
    term2 = xi * gammainc(xi + 1, lambda_) # This is P(X > xi)
    return term1 + term2

def double_integral_poisson_1d_pre(lambda_, x_vals):
    # 1D double integral of the kernel
    k = x_vals
    tails = 1.0 - gammainc(k + 1, lambda_)  # P(X > k)
    return jnp.sum(tails ** 2)

def kernel_embedding_poisson(lambdas, Y, precomp):
    # Multi-dimensional KME using product structure
    x_vals_list, fact_list = precomp
    d = len(x_vals_list)

    def per_point(y):
        vals = []
        for j in range(d):
            vals.append(kme_poisson_1d_pre(lambdas[j], y[j],
                                          x_vals_list[j], fact_list[j]))
        return jnp.prod(jnp.stack(vals))
    return vmap(per_point)(Y)

def double_integral_poisson(lambdas, precomp):
    # Multi-dimensional double integral using product structure
    x_vals_list, _ = precomp
    d = len(x_vals_list)
    prod_val = 1.0
    for j in range(d):
        prod_val = prod_val * double_integral_poisson_1d_pre(lambdas[j], x_vals_list[j])
    return prod_val

def bayesian_cubature_poisson(X, f_vals, lambdas, precomp, jitter=1e-4):
    # Main BQ calculation
    n = X.shape[0]
    K = brownian_kernel(X, X) + jitter * jnp.eye(n)
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)

    z = kernel_embedding_poisson(lambdas, X, precomp).reshape(n, 1)
    f_vals = f_vals.reshape(n, 1)

    K_inv_f = cho_solve((L, lower), f_vals, check_finite=False)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)

    mean = (z.T @ K_inv_f)[0, 0]
    var  = double_integral_poisson(lambdas, precomp) - (z.T @ K_inv_z)[0, 0]
    return mean, var

## --------- True Reference Value ----------

def grid_candidates(xmaxs):
    # Creates a full grid of points
    axes = [jnp.arange(0, int(m) + 1) for m in xmaxs]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    G = jnp.stack([m.reshape(-1) for m in mesh], axis=1)
    return G

def true_integral_md(f, lambdas, xmaxs):
    # Calculates the "true" integral by summing over a fine grid
    G = grid_candidates(xmaxs)
    pmfs = jnp.prod(jax.vmap(lambda lam, col: jax_poisson.pmf(col, mu=lam))(lambdas, G.T), axis=0)
    return jnp.sum(f(G) * pmfs)

## --------- Point Selection (BO) ----------

@jit
def compute_integral_variance_md(X, lambdas, precomp, jitter=1e-4):
    # Computes the posterior variance of the integral
    K = brownian_kernel(X, X) + jitter * jnp.eye(X.shape[0])
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)
    z = kernel_embedding_poisson(lambdas, X, precomp).reshape(-1, 1)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)
    return double_integral_poisson(lambdas, precomp) - (z.T @ K_inv_z)[0, 0]

@jit
def compute_max_variance_md(X_obs, candidates, lambdas, precomp, jitter=1e-4):
    # Finds the candidate point that maximizes the reduction in variance
    current_var = compute_integral_variance_md(X_obs, lambdas, precomp, jitter=jitter)

    def updated_var(x):
        X_aug = jnp.concatenate([X_obs, x[None, :]], axis=0)
        return compute_integral_variance_md(X_aug, lambdas, precomp, jitter=jitter)

    updated_vars = vmap(updated_var)(candidates)
    kg_vals = current_var - updated_vars
    return candidates[jnp.argmax(kg_vals)]

## --------- Experiment ----------

def monte_carlo_poisson(f, n, lambdas, key):
    # Standard Monte Carlo estimator
    d = lambdas.shape[0]
    keys = random.split(key, d)
    draws = jnp.stack(
        [random.poisson(keys[i], lam=lambdas[i], shape=(n,)) for i in range(d)],
        axis=1
    )
    f_vals = f(draws)
    return jnp.mean(f_vals), draws

def run_experiment_md(f, n_vals, lambdas, xmaxs, seed, use_bo=True, run_mc=True):
    # Main experiment loop
    key = random.PRNGKey(seed)
    precomp = make_precomp(xmaxs)
    true_val = true_integral_md(f, lambdas, xmaxs)

    bq_estimates, mc_estimates = [], []
    abs_err_bq, abs_err_mc = [], []

    for n in n_vals:
        # --- BQ/BO Section ---
        modes = jnp.floor(lambdas)
        X = [modes]
        y = [f(modes[None, :])[0]]

        while len(X) < n:
            if use_bo:
                # MODIFICATION: Generate random candidates instead of using a fixed grid
                key, subkey = random.split(key)
                # Draw 1000 random candidates to select from at each step
                _, random_candidates = monte_carlo_poisson(f, 1000, lambdas, subkey)
                
                # Filter out points that have already been selected
                used_tuples = set(map(lambda xi: tuple(map(int, xi)), X))
                mask = jnp.array([tuple(map(int, c)) not in used_tuples for c in random_candidates])
                cand = random_candidates[mask]
                
                # Fallback in case all random candidates were already used
                if cand.shape[0] == 0:
                    key, subkey = random.split(key, 1)
                    _, cand = monte_carlo_poisson(f, 1, lambdas, subkey)

                x_next = compute_max_variance_md(jnp.array(X), cand, lambdas, precomp)
            else: # Random selection BQ
                key, subkey = random.split(key)
                _, x_next_arr = monte_carlo_poisson(f, 1, lambdas, subkey)
                x_next = x_next_arr[0]
                
            X.append(x_next)
            y.append(f(x_next[None, :])[0])

        X_final = jnp.stack(X)
        f_vals_final = jnp.array(y)
        mu_bq, _ = bayesian_cubature_poisson(X_final, f_vals_final, lambdas, precomp)

        bq_estimates.append(mu_bq)
        abs_err_bq.append(jnp.abs(mu_bq - true_val))

        # --- MC baseline ---
        if run_mc:
            key, subkey = random.split(key)
            mu_mc, _ = monte_carlo_poisson(f, n, lambdas, subkey)
            mc_estimates.append(mu_mc)
            abs_err_mc.append(jnp.abs(mu_mc - true_val))

    return {
        "true_val": true_val,
        "abs_err_bq": jnp.array(abs_err_bq),
        "abs_err_mc": jnp.array(abs_err_mc) if run_mc else None,
    }

def run_multiple_seeds(f, n_vals, lambdas, xmaxs, num_seeds, run_mc=True, use_bo=True):
    # Repeats experiment for several seeds and averages the errors
    bq_errs, mc_errs = [], []
    for seed in range(num_seeds):
        print(f"--- Running Seed {seed} (use_bo={use_bo}) ---")
        res = run_experiment_md(f, n_vals, lambdas, xmaxs,
                                seed=seed, use_bo=use_bo, run_mc=run_mc)
        bq_errs.append(res["abs_err_bq"])
        if run_mc:
            mc_errs.append(res["abs_err_mc"])

    bq_mean_err = jnp.mean(jnp.stack(bq_errs), axis=0)
    mc_mean_err = jnp.mean(jnp.stack(mc_errs), axis=0) if run_mc else None

    return {
        "true_val": res["true_val"],
        "bq_mean_err": bq_mean_err,
        "mc_mean_err": mc_mean_err,
    }

def demo_md():
    # Configuration
    lambdas = jnp.array([10.0, 6.0, 8.0])
    xmaxs   = (40, 30, 40)
    n_vals  = [10, 30, 70, 100]
    num_seeds = 10

    # Run experiments for both BQ-BO and the MC baseline
    bo_results = run_multiple_seeds(f, n_vals, lambdas, xmaxs, num_seeds, 
                                     run_mc=True, use_bo=True)

    print("\n=== Final Averaged Results ===")
    print(f"n = {n_vals}")
    print(f"True value: {bo_results['true_val']:.6f}")
    print(f"BQ-BO mean error: {bo_results['bq_mean_err']}")
    if bo_results["mc_mean_err"] is not None:
        print(f"MC mean error:    {bo_results['mc_mean_err']}")

if __name__ == "__main__":
    demo_md()