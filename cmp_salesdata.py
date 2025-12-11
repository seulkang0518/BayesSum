import json
import jax
import os, time
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from jax.scipy.special import gammaln, gammainc
from jax.scipy.linalg import cho_factor, cho_solve
import optax
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import poisson as jax_poisson
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

def data_loading(filename="sales_hist.json"):
    with open(filename) as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.items()}


def empirical_mean(sales_hist):
    xs = np.array(sorted(sales_hist.keys()))
    cs = np.array([sales_hist[k] for k in xs])
    return (xs * cs).sum() / cs.sum()


def params_init(sales_hist):
    xs = np.array(sorted(sales_hist.keys()))
    fs = np.array([sales_hist[k] for k in xs])
    n  = fs.sum()
    mean = (xs * fs).sum() / n
    var  = (xs**2 * fs).sum() / n - mean**2
    # crude moment-based init (reasonable starting point)
    nu0  = max(1e-3, mean / max(var, 1e-8))
    return float(nu0)

def cmp_suff_stats(sales_hist):
    ys = jnp.array(list(sales_hist.keys()))
    cs = jnp.array(list(sales_hist.values()))
    n = cs.sum()
    s1 = jnp.sum(ys * cs)
    s2 = jnp.sum(cs * gammaln(ys + 1.0))
    return n, s1, s2


# ================================================================
#               FIXED REFERENCE POISSON + SAMPLES
# ================================================================
def get_reference_nodes(seed, lam_ref, n_samples):
    key = jax.random.PRNGKey(seed)
    X = jax.random.poisson(key, lam=lam_ref, shape=(n_samples,))
    # oversample = n_samples * 5
    # X_proposed = random.poisson(key, lam=lam_ref, shape=(oversample,))
    # X_unique = jnp.unique(X_proposed)

    # while X_unique.shape[0] < n_samples / 2:
    #     subkey, extra_key = random.split(key)
    #     extra = random.poisson(extra_key, lam=lam_ref, shape=(oversample,))
    #     X_unique = jnp.unique(jnp.concatenate([X_unique, extra]))

    # X = X_unique[:n_samples]
    return jnp.sort(X.astype(jnp.float64))

def poisson_ppf_jax(u, lam_ref, xmax=100):

    ks = jnp.arange(0, xmax + 1)
    cdfk = jax_poisson.cdf(ks, lam_ref)
    idx = jnp.searchsorted(cdfk, u, side="left")
    idx = jnp.clip(idx, 0, xmax)
    return idx.astype(jnp.float64)

def stratified_poisson_nodes(seed, lam_ref, n_samples, xmax=100):
    key = random.PRNGKey(seed)
    eps = random.uniform(key, shape=(n_samples,))
    u = (jnp.arange(n_samples) + eps) / n_samples
    u = jnp.clip(u, 1e-12, 1 - 1e-12)

    X = poisson_ppf_jax(u, lam_ref, xmax)
    return jnp.sort(X.astype(jnp.float64))

# ================================================================
#                        BROWN KERNEL + KME
# ================================================================
def brownian_kernel(X1, X2):
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    return jnp.minimum(X1, X2.T)


def kme_single(lam_ref, x, xmax=200):
    xs = jnp.arange(0, xmax + 1)
    mask = xs <= x

    log_fact = jnp.cumsum(jnp.log(jnp.arange(1, xmax + 1)))
    fact = jnp.exp(jnp.concatenate([jnp.array([0.]), log_fact]))

    pmf = (lam_ref**xs / fact) * jnp.exp(-lam_ref)

    term1 = jnp.sum(jnp.where(mask, xs * pmf, 0.0))
    term2 = x * gammainc(x + 1, lam_ref)
    return term1 + term2


def compute_kme(lam_ref, X, xmax):
    return vmap(lambda xi: kme_single(lam_ref, xi, xmax))(X)


# ================================================================
#                IMPORTANCE SAMPLING FUNCTION f(j)
# ================================================================
def cmp_f_vals(X, lam, nu, lam_ref):
    return (jnp.exp((1 - nu) * gammaln(X + 1.0)) * jnp.exp(X * jnp.log(lam / lam_ref)))

def cmp_logf_vals(X, lam, nu, lam_ref):
    return (1 - nu) * gammaln(X + 1.0) + X*jnp.log(lam / lam_ref)
# ================================================================
#                LOG Z ESTIMATORS (MC-IS and BQ)
# ================================================================
def logZ_mc_is(log_fvals, lam_ref):
    m = jnp.max(log_fvals)
    return lam_ref + (jnp.log(jnp.mean(jnp.exp(log_fvals - m))) + m)
    # return lam_ref + jnp.mean(log_fvals)

def logZ_bq(fvals, w, lam_ref):
    return lam_ref + jnp.log(jnp.dot(w, fvals))
# ================================================================
#                NLL (MC or BQ)
# ================================================================
def nll(params, suffstats, X, w, lam_ref, use_bq):
    lam, nu = params
    lam = jnp.maximum(lam, 1e-8)
    nu = jnp.maximum(nu, 1e-8)

    if use_bq:
        fvals = cmp_f_vals(X, lam, nu, lam_ref)
        logZ = logZ_bq(fvals, w, lam_ref)
    else:
        log_fvals = cmp_logf_vals(X, lam, nu, lam_ref)
        logZ = logZ_mc_is(log_fvals, lam_ref)

    n, s1, s2 = suffstats
    return -(s1*jnp.log(lam) - nu*s2 - n*logZ)


loss_grad = value_and_grad(nll, argnums=0)


# ================================================================
#                       TRAINING LOOP
# ================================================================
def train_cmp(suffstats, X, w, lam_ref, lam_init, nu_init, use_bq, seed):
    params = jnp.array([lam_init, nu_init])
    opt = optax.adam(1e-3)
    state = opt.init(params)

    traj = []
    traj.append(params)

    for i in range(1000):
        loss, grads = loss_grad(params, suffstats, X, w, lam_ref, use_bq)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        traj.append(params)

        if i % 50 == 0:
            print(f"[seed {seed:2d} | iter {i}] "
                  f"{'BQ' if use_bq else 'MC'} "
                  f"lam={float(params[0]):.4f}, nu={float(params[1]):.4f}, loss={float(loss):.4f}")

    traj = jnp.array(traj)

    return float(params[0]), float(params[1]), traj


def empirical_pmf_from_hist(sales_hist):
    xs = np.array(sorted(sales_hist.keys()))
    cs = np.array([sales_hist[k] for k in xs], dtype=float)
    p  = cs / cs.sum()
    return xs, p, int(cs.sum())

def cmp_logpmf(y, nu, lam, logZ):
    y = jnp.asarray(y)
    return y * jnp.log(lam) - nu * gammaln(y + 1.0) - logZ

def logZ_trunc(nu, lam, y_max=8000):
    xs = jnp.arange(0, y_max+1, dtype=jnp.int32)
    xsf = xs.astype(jnp.float64)
    # p(y|lam)
    log_p = -lam + xsf * jnp.log(jnp.maximum(lam,1e-300)) - gammaln(xsf+1.0)
    # f_nu(y) = (y!)^{1-nu}
    log_f = (1.0 - nu) * gammaln(xsf + 1.0)
    # log E[f(Y)] + lam
    m = jnp.max(log_f + log_p)
    logE = jnp.log(jnp.sum(jnp.exp(log_f + log_p - m))) + m
    return float(logE + lam)

def cmp_pmf(y, nu, lam):
    logZ = logZ_trunc(nu, lam)
    return jnp.exp(cmp_logpmf(y, nu, lam, logZ))

def plot_results_with_se(
    sales_hist,
    bq_nu_mean, bq_lam_mean, bq_nu_se, bq_lam_se,
    mc_nu_mean, mc_lam_mean, mc_nu_se, mc_lam_se,
    nu_init, lam_init
):
    xs_emp, p_emp, _ = empirical_pmf_from_hist(sales_hist)

    # plotting range
    ymax = max(
        xs_emp.max(),
        int(bq_lam_mean + 8*np.sqrt(bq_lam_mean)),
        int(mc_lam_mean + 8*np.sqrt(mc_lam_mean)),
    )
    ys = jnp.arange(0, ymax+1)

    # Compute central PMFs
    bq_pmf_mean = cmp_pmf(ys, bq_nu_mean, bq_lam_mean)
    mc_pmf_mean = cmp_pmf(ys, mc_nu_mean, mc_lam_mean)
    init_pmf = cmp_pmf(ys, nu_init, lam_init)

    # Compute upper/lower bands (+/- 1 SE)
    bq_pmf_upper = cmp_pmf(ys, bq_nu_mean + bq_nu_se, bq_lam_mean + bq_lam_se)
    bq_pmf_lower = cmp_pmf(ys, bq_nu_mean - bq_nu_se, bq_lam_mean - bq_lam_se)

    mc_pmf_upper = cmp_pmf(ys, mc_nu_mean + mc_nu_se, mc_lam_mean + mc_lam_se)
    mc_pmf_lower = cmp_pmf(ys, mc_nu_mean - mc_nu_se, mc_lam_mean - mc_lam_se)

    # plot
    plt.figure(figsize=(10,6))

    # empirical data
    plt.bar(xs_emp, p_emp, alpha=0.4, label="Empirical pmf (data)")

    # BQ mean
    plt.plot(ys, bq_pmf_mean, color="blue", lw=2,
             label="BayesSum")

    # BQ band
    plt.fill_between(
        ys, bq_pmf_lower, bq_pmf_upper,
        color="blue", alpha=0.15
    )

    # MC mean
    plt.plot(ys, mc_pmf_mean, color="black", lw=2,
             label="MC")

    # MC band
    plt.fill_between(
        ys, mc_pmf_lower, mc_pmf_upper,
        color="gray", alpha=0.15
    )

    # plt.plot(np.asarray(ys), init_pmf, linestyle = '--', color = '#ff7f0e', marker="s", lw=1.5, alpha=0.8,
    #          label="Initial Parameters")

    plt.xlim(0, 30)
    plt.xlabel("Count", fontsize = 32)
    plt.ylabel("Probability", fontsize = 32)
    plt.legend(fontsize = 32)
    plt.tight_layout()
    plt.savefig("cmp_sales_data_with_se.pdf")
    plt.close()


def expand_hist_to_vector(hist):
    ys = []
    for k, count in hist.items():
        ys += [k] * count
    return jnp.array(ys, dtype=jnp.float64)


def logZ_bayesum(eta, nu, X_Z, w, lam_ref):
    fvals = cmp_f_vals(X_Z, eta, nu, lam_ref)
    return logZ_bq(fvals, w, lam_ref)

@jit
def loglik_point(eta, nu, data, X_Z, w, lam_ref):

    logZ = logZ_bayesum(eta, nu, X_Z, w, lam_ref)
    term1 = jnp.sum(data * jnp.log(eta))
    term2 = -nu * jnp.sum(gammaln(data + 1))

    return term1 + term2 - len(data) * logZ


def make_loglik_grid(data, X_Z, w, lam_ref, eta_min=0.1, eta_max=3.0, nu_min=0.0, nu_max=0.8,
                     n_eta=80, n_nu=80):
    """
    Generate 2D grid (ETA, NU) and log-likelihood values.
    """
    eta_vals = jnp.linspace(eta_min, eta_max, n_eta)
    nu_vals  = jnp.linspace(nu_min, nu_max, n_nu)

    ETA, NU = jnp.meshgrid(eta_vals, nu_vals)

    # Vectorized evaluation of loglik over the grid
    loglik_grid = vmap(
        lambda eta_row, nu_row:
            vmap(lambda e, n: loglik_point(e, n, data, X_Z, w, lam_ref))(eta_row, nu_row)
    )(ETA, NU)

    return ETA, NU, loglik_grid


# -----------------------------------------------------------
# 5. PLOT EVERYTHING
# -----------------------------------------------------------
def plot_traj(ETA, NU, loglik_vals, traj_bq, traj_mc, eta_mle, nu_mle):
    plt.figure(figsize=(9,5))

    # Heatmap background
    plt.contourf(ETA, NU, loglik_vals, levels=50, cmap="viridis")
    # plt.colorbar(label="log-likelihood")

    # Trajectories
    plt.plot(traj_bq[:,0], traj_bq[:,1], '-o', color='blue',
             label="BayesSum", alpha=0.9)
    plt.plot(traj_mc[:,0], traj_mc[:,1], '-o', color='black',
             label="MC", alpha=0.9)

    # True MLE (grid maximizer)
    plt.scatter(eta_mle, nu_mle, s=160, c="yellow",
                edgecolor="black", label="Optimum")

    plt.xlabel(r"$\theta_1$", fontsize=32)
    plt.ylabel(r"$\theta_2$", fontsize=32)
    # plt.ylim(0.0, 0.6)
    plt.title("Optimisation Trajectories", fontsize=32)
    plt.legend(
        loc='upper right',   # change to 'lower left' / 'upper left' if overlapping data
        fontsize=32,
        frameon=False,
        ncol=1,
        bbox_to_anchor=(1.0, 1.05)  # x,y in axes fraction coords
        ,columnspacing=0.5
    )
    plt.tight_layout()
    plt.savefig("cmp_trajectories.pdf")
    plt.close()

# ================================================================
#                             MAIN
# ================================================================
if __name__ == "__main__":

    # -------------------------
    # Load data
    # -------------------------
    sales_hist = data_loading("sales_hist.json")
    data = expand_hist_to_vector(sales_hist)
    xmax = int(jnp.max(data)) + 20
    suffstats  = cmp_suff_stats(sales_hist)

    lam_Z = empirical_mean(sales_hist)  
    # print(lam_Z)    
    lam_init = 1.2
    nu_init = 0.5

    num_seeds = 10
    n_samples = 30

    mc_lam_list, mc_nu_list = [], []
    bq_lam_list, bq_nu_list = [], []
    traj_bq_list, traj_mc_list = [], []
    mc_time, bq_time = [], []

    # for seed in range(num_seeds):

    #     X_mc = get_reference_nodes(seed, lam_Z, n_samples)
    #     X_bq = jnp.arange(10)
    #     # X_Z = stratified_poisson_nodes(seed, lam_Z, n_samples)
    #     # print(X_Z)

    #     # ----------------------------------------------------------
    #     # MC-IS training
    #     # ----------------------------------------------------------
    #     start_time = time.perf_counter()
    #     lam_mc, nu_mc, traj_mc = train_cmp(
    #         suffstats, X_mc, None, lam_Z,
    #         lam_init, nu_init, use_bq=False, seed=seed)
    #     elapsed = time.perf_counter() - start_time

    #     mc_lam_list.append(lam_mc)
    #     mc_nu_list.append(nu_mc)
    #     mc_time.append(elapsed)
    #     # ----------------------------------------------------------
    #     # BQ training
    #     # ----------------------------------------------------------
    #     if seed == 0:
    #         start_time = time.perf_counter()

    #         K = brownian_kernel(X_bq, X_bq) + 1e-5*jnp.eye(len(X_bq))
    #         L, _ = cho_factor(K, lower=True)

    #         mu = compute_kme(lam_Z, X_bq, xmax=200).reshape(-1, 1)
    #         w = cho_solve((L, True), mu).ravel()       

    #         lam_bq, nu_bq, traj_bq = train_cmp(
    #             suffstats, X_bq, w, lam_Z,
    #             lam_init, nu_init, use_bq=True, seed=seed)

    #         elapsed = time.perf_counter() - start_time

    #         bq_lam_list.append(lam_bq)
    #         bq_nu_list.append(nu_bq)
    #         bq_time.append(elapsed)

    #     if seed == 0:
    #         traj_mc_list = traj_mc
    #         traj_bq_list = traj_bq

    # lam_mc = np.mean(mc_lam_list)
    # nu_mc = np.mean(mc_nu_list)
    # mc_time = np.mean(mc_time)

    # lam_bq = np.mean(bq_lam_list)
    # nu_bq = np.mean(bq_nu_list)     
    # bq_time = np.mean(bq_time)


    # traj_bq_list = jnp.stack(traj_bq_list)
    # traj_mc_list = jnp.stack(traj_mc_list)

    # print("\n================ SUMMARY ================\n")
    # print("MC-IS mean:", lam_mc, nu_mc)
    # print("MC time   :", mc_time)
    # print("BQ mean   :", lam_bq, nu_bq)
    # print("BQ time   :", bq_time)

    # np.savez(
    # "cmp_results_fixed.npz",
    # bq_lam_list=bq_lam_list,
    # bq_nu_list=bq_nu_list,
    # traj_bq_list = traj_bq_list,
    # mc_lam_list=mc_lam_list,
    # mc_nu_list=mc_nu_list,
    # traj_mc_list = traj_mc_list,
    # mc_time_list = mc_time,
    # bq_time_list = bq_time
    # )

    results = np.load("cmp_results_fixed.npz")

    # bq_lam_list = jnp.array(results["bq_lam_list"])
    # mc_lam_list = jnp.array(results["mc_lam_list"])

    # bq_nu_list = jnp.array(results["bq_nu_list"])
    # mc_nu_list = jnp.array(results["mc_nu_list"])

    # bq_lam_mean = np.mean(bq_lam_list)
    # bq_lam_se   = np.std(bq_lam_list) / jnp.sqrt(num_seeds)

    # bq_nu_mean  = np.mean(bq_nu_list)
    # bq_nu_se    = np.std(bq_nu_list) / jnp.sqrt(num_seeds)

    # mc_lam_mean = np.mean(mc_lam_list)
    # mc_lam_se   = np.std(mc_lam_list) / jnp.sqrt(num_seeds)

    # mc_nu_mean  = np.mean(mc_nu_list)
    # mc_nu_se    = np.std(mc_nu_list) / jnp.sqrt(num_seeds)

    # print("\n================ SUMMARY ================\n")
    # print("MC-IS mean:", mc_lam_mean, mc_nu_mean)
    # print("BQ mean   :", bq_lam_mean, bq_nu_mean)
    # plot_results_with_se(sales_hist, bq_nu_mean, bq_lam_mean, bq_nu_se, bq_lam_se, mc_nu_mean, mc_lam_mean, mc_nu_se, mc_lam_se, nu_init, lam_init)

    traj_bq = jnp.array(results["traj_bq_list"])
    traj_mc = jnp.array(results["traj_mc_list"])

    X_Z = jnp.arange(10)
    K = brownian_kernel(X_Z, X_Z) + 1e-6*jnp.eye(len(X_Z))
    L, _ = cho_factor(K, lower=True)

    mu = compute_kme(lam_Z, X_Z, xmax=200).reshape(-1, 1)
    w  = cho_solve((L, True), mu).ravel()

    ETA, NU, loglik_vals = make_loglik_grid(data, X_Z, w, lam_Z)
    flat_idx = jnp.argmax(loglik_vals)
    i, j = jnp.unravel_index(flat_idx, loglik_vals.shape)
    eta_mle = ETA[i, j]
    nu_mle = NU[i, j]
    plot_traj(ETA, NU, loglik_vals, traj_bq, traj_mc, eta_mle, nu_mle)
