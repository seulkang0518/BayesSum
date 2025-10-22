import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jit, random
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammainc
from jax.scipy.stats import poisson as jax_poisson
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
from functools import partial

## --------- Basic Settings ----------

def f(X): 
    return jnp.prod(X * jnp.exp(-jnp.sqrt(X + 1e-12)), axis=1)

def brownian_kernel(X1, X2):
    X1 = jnp.atleast_2d(X1)
    X2 = jnp.atleast_2d(X2)
    mins = jnp.minimum(X1[:, None, :], X2[None, :, :])  # (n, m, d)
    return jnp.prod(mins, axis=-1)


def make_precomp(xmaxs):
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
    # return a tuple, not dict
    return (x_vals_list, fact_list)

def kme_poisson_1d_pre(lambda_, xi, x_vals, fact):
    # xi may be traced; x_vals and fact are static arrays provided from outside
    pmf = (lambda_ ** x_vals / fact) * jnp.exp(-lambda_)
    term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals * pmf, 0.0))
    # IMPORTANT: tail, not CDF
    # term2 = xi * (1 - gammainc(xi + 1, lambda_))
    term2 = xi * gammainc(xi + 1, lambda_)
    return term1 + term2

def double_integral_poisson_1d_pre(lambda_, x_vals):
    # k indexes the same grid as x_vals (0..m)
    k = x_vals  # integer grid
    tails = 1.0 - gammainc(k + 1, lambda_)  # P(X > k)
    return jnp.sum(tails ** 2)

def kernel_embedding_poisson(lambdas, Y, precomp):
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
    x_vals_list, _ = precomp
    d = len(x_vals_list)
    prod_val = 1.0
    for j in range(d):
        prod_val = prod_val * double_integral_poisson_1d_pre(lambdas[j], x_vals_list[j])
    return prod_val


def bayesian_cubature_poisson(X, f_vals, lambdas, precomp, jitter=1e-4):
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
    axes = [jnp.arange(0, int(m) + 1) for m in xmaxs]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    G = jnp.stack([m.reshape(-1) for m in mesh], axis=1)  # (prod(m+1), d)
    return G

def true_integral_md(f, lambdas, xmaxs):
    G = grid_candidates(xmaxs)  # (M, d)
    pmfs = jnp.prod(jax.vmap(lambda lam, col: jax_poisson.pmf(col, mu=lam))(lambdas, G.T), axis=0)
    return jnp.sum(f(G) * pmfs)


## --------- Samplings ----------

def poisson_sample_candidates(key, lambdas, num, clip_each=None):
    """
    Sample 'num' candidates from product Poisson, optionally clip per-dim to avoid very large values.
    """
    d = lambdas.shape[0]
    key, *sub = random.split(key, d + 1)
    sub = jnp.array(sub)
    draws = jnp.stack([random.poisson(sub[i], lam=lambdas[i], shape=(num,)) for i in range(d)], axis=1)
    if clip_each is not None:
        clip_each = jnp.asarray(clip_each)
        draws = jnp.minimum(draws, clip_each)
    return key, draws

@jit
def compute_integral_variance_md(X, lambdas, precomp, jitter=1e-4):
    # Gram
    K = brownian_kernel(X, X) + jitter * jnp.eye(X.shape[0])
    L, lower = cho_factor(K, overwrite_a=False, check_finite=False)

    z = kernel_embedding_poisson(lambdas, X, precomp).reshape(-1, 1)
    K_inv_z = cho_solve((L, lower), z, check_finite=False)

    return double_integral_poisson(lambdas, precomp) - (z.T @ K_inv_z)[0, 0]

@jit
def compute_max_variance_md(X_obs, candidates, lambdas, precomp, jitter=1e-4):
    current_var = compute_integral_variance_md(X_obs, lambdas, precomp, jitter=jitter)

    def updated_var(x):
        X_aug = jnp.concatenate([X_obs, x[None, :]], axis=0)
        return compute_integral_variance_md(X_aug, lambdas, precomp, jitter=jitter)

    updated_vars = vmap(updated_var)(candidates)
    kg_vals = current_var - updated_vars
    return candidates[jnp.argmax(kg_vals)]

## --------- Experiment ----------
def monte_carlo_poisson(f, n, lambdas, key):
    """
    Monte Carlo estimate of E[f(X)] where X ~ product Poisson(lambdas).
    """
    d = lambdas.shape[0]
    # split keys per dimension
    keys = random.split(key, d)
    # sample independently per dimension
    draws = jnp.stack(
        [random.poisson(keys[i], lam=lambdas[i], shape=(n,)) for i in range(d)],
        axis=1
    )  # shape (n, d)
    f_vals = f(draws)
    return jnp.mean(f_vals), draws

def run_experiment_md(f, n_vals, lambdas, xmaxs, seed, candidates,
                      use_bo=True, run_mc=True):
    key = random.PRNGKey(seed)
    precomp = make_precomp(xmaxs)
    true_val = true_integral_md(f, lambdas, xmaxs)

    bq_estimates = []
    mc_estimates = []
    abs_err_bq = []
    abs_err_mc = []

    for n in n_vals:
        # --- BQ/BO as before ---
        modes = jnp.floor(lambdas)
        X = [modes]
        y = [f(modes[None, :])[0]]

        while len(X) < n:
            if use_bo:
                used = set(map(tuple, [tuple(map(int, xi)) for xi in X]))
                mask = jnp.array([tuple(map(int, c)) not in used for c in candidates])
                cand = candidates[mask]
                if cand.shape[0] == 0:
                    key, samp = poisson_sample_candidates(key, lambdas, 1, clip_each=jnp.array(xmaxs))
                    cand = samp
                x_next = compute_max_variance_md(jnp.array(X), cand, lambdas, precomp)
            else:
                used = set(map(tuple, [tuple(map(int, xi)) for xi in X]))
                cand = candidates[jnp.array([tuple(map(int, c)) not in used for c in candidates])]
                if cand.shape[0] == 0:
                    key, cand = poisson_sample_candidates(key, lambdas, 1, clip_each=jnp.array(xmaxs))
                idx = random.randint(key, (), 0, cand.shape[0])
                key, _ = random.split(key)
                x_next = cand[idx]

            X.append(x_next)
            y.append(f(x_next[None, :])[0])

        X = jnp.stack(X)
        f_vals = jnp.array(y)
        mu_bq, _ = bayesian_cubature_poisson(X, f_vals, lambdas, precomp)

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
        "bq_estimates": jnp.array(bq_estimates),
        "abs_err_bq": jnp.array(abs_err_bq),
        "mc_estimates": jnp.array(mc_estimates) if run_mc else None,
        "abs_err_mc": jnp.array(abs_err_mc) if run_mc else None,
    }


def run_multiple_seeds(f, n_vals, lambdas, xmaxs, num_seeds, candidates,
                       run_mc=True, use_bo=True):
    """
    Repeat experiments for several seeds, return averaged absolute errors.
    """
    bq_errs = []
    mc_errs = []
    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")
        res = run_experiment_md(f, n_vals, lambdas, xmaxs,
                                seed=seed,
                                candidates=candidates,
                                use_bo=use_bo,
                                run_mc=run_mc)
        bq_errs.append(res["abs_err_bq"])
        if run_mc:
            mc_errs.append(res["abs_err_mc"])

    bq_errs = jnp.stack(bq_errs)  # (num_seeds, len(n_vals))
    bq_mean_err = jnp.mean(bq_errs, axis=0)

    if run_mc:
        mc_errs = jnp.stack(mc_errs)
        mc_mean_err = jnp.mean(mc_errs, axis=0)
    else:
        mc_mean_err = None

    return {
        "true_val": res["true_val"],
        "bq_mean_err": bq_mean_err,
        "mc_mean_err": mc_mean_err,
        "n_vals": n_vals,
    }

def demo_md():
    lambdas = jnp.array([10.0, 6.0])
    xmaxs   = (40, 30)
    n_vals  = [10, 20, 30, 40, 50, 60, 70]
    num_seeds = 10

    candidates = grid_candidates(xmaxs)

    bo_results   = run_multiple_seeds(f, n_vals, lambdas, xmaxs, num_seeds, candidates,
                                      run_mc=True, use_bo=True)
    # rand_results = run_multiple_seeds(f, n_vals, lambdas, xmaxs, num_seeds, candidates,
    #                                   run_mc=True, use_bo=False)

    # plot_results(n_vals, bo_results, rand_results, lambdas, save_path="results")

    print("\n=== Final Averaged Results ===")
    print("n =", n_vals)
    print("True value:", bo_results["true_val"])
    print("BQ/BO mean error:", bo_results["bq_mean_err"])
    # print("BQ Random mean error:", rand_results["bq_mean_err"])
    if bo_results["mc_mean_err"] is not None:
        print("MC mean error:", bo_results["mc_mean_err"])

if __name__ == "__main__":
    demo_md()



