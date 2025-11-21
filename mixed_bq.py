import math
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
    
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 26
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
# plt.rc('font', family='Arial', size=12)
plt.rc('axes', titlesize=26, labelsize=26, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=26, frameon=False)
plt.rc('xtick', labelsize=26, direction='in')
plt.rc('ytick', labelsize=26, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

import os
import jax
import jax.numpy as jnp
import jax.scipy.linalg as sla
from jax.scipy.special import erf, erfinv
from scipy.special import iv
jax.config.update("jax_enable_x64", True)

# ----------------------------
# Ground Truth
# ----------------------------
def _J1_series(k, L, N=80):
    s = np.i0(1.0)  # I_0(1)
    t = 0.0
    for n in range(1, N+1):
        t += (iv(n, 1.0)/n) * np.sin(n * k * L)
    return s + (2.0/(k*L)) * t

def ackley2c_truth_d1_exact(L, a=20.0, b=0.2, c=2.0*np.pi, levels=17):
    H = np.linspace(-1.0, 1.0, levels)
    total = 0.0
    for h1, h2 in product(H, H):
        alpha = 1.0 + 0.1*(h1 + h2)
        beta  = 1.0 + 0.05*(h1*h1 + h2*h2)

        # term1 (closed form)
        lam = b * alpha
        E1  = -a * (1.0 - np.exp(-lam*L)) / (lam * L)

        # term2 via J1
        kappa = c * beta
        key = float(np.round(kappa, 14))
        E2  = -1 * _J1_series(kappa, L, N=80)

        total += (E1 + E2 + a + np.e)
        # total += (E1 + E2)

    return total / (levels**2)

def ackley2c_truth_d1_mc_joint(L, key, levels=17, N=200_000):
    kx, kh = jax.random.split(key)

    X = jax.random.uniform(kx, (N, 1), minval=-L, maxval=L)  

    levels_lin = jnp.linspace(-1.0, 1.0, levels)
    i1 = jax.random.randint(kh, (N,), 0, levels)
    i2 = jax.random.randint(kh, (N,), 0, levels)  
    H = jnp.stack([levels_lin[i1], levels_lin[i2]], axis=1)  

    fvals = v_ackley_2C(X, H)  
    return float(jnp.mean(fvals))

# ----------------------------
# Ackley-2C integrand
# ----------------------------
def ackley_2C_single(x, h, a=20.0, b=0.2, c=2*math.pi):
    x = jnp.atleast_1d(x)
    h = jnp.atleast_1d(h)
    assert h.shape[-1] == 2, "Ackley-2C expects two categorical variables"

    d = x.shape[-1]
    alpha = 1.0 + 0.1 * jnp.sum(h)         # small amplitude modulation
    beta  = 1.0 + 0.05 * jnp.sum(h**2)     # small frequency modulation

    term1 = -a * jnp.exp(-b * jnp.sqrt(jnp.mean((alpha * x)**2)))
    term2 = -jnp.exp(jnp.mean(jnp.cos(c * beta * x)))
    return term1 + term2 + a + math.e
    # return term1 + term2

v_ackley_2C = jax.vmap(ackley_2C_single, in_axes=(0, 0))  

# ----------------------------
# RBF kernel on R^d (continuous)
# ----------------------------
def rbf_kx(x, xp, ell):
    diff = x - xp
    return jnp.exp(-0.5 * jnp.sum(diff * diff, axis=-1) / (ell**2))

# ----------------------------
# Categorical kernels on 2-D categorical vectors
# ----------------------------
def kc_equal_vec(h, hp):
    return jnp.all(h == hp, axis=-1).astype(jnp.float64)
i
def kc_exp_hamming_vec(h, hp, lambda_c):
    mismatches = jnp.sum(h != hp, axis=-1)   
    return jnp.exp(-lambda_c * mismatches).astype(jnp.float64)

# ----------------------------
# Mixed kernels
# ----------------------------
def k_mixed_product_2c(xh, xhp, ell, cat="equal", lambda_c=0.0):
    x,  h  = xh
    xp, hp = xhp
    kx = rbf_kx(x, xp, ell)
    if cat == "equal":
        kc = kc_equal_vec(h, hp)
    else:
        kc = kc_exp_hamming_vec(h, hp, lambda_c)
    return kx * kc

def k_mixed_shared_2c(xh, xhp, ell, cat="equal", lambda_c=0.0):
    x,  h  = xh
    xp, hp = xhp
    kx = rbf_kx(x, xp, ell)
    if cat == "equal":
        kc = kc_equal_vec(h, hp)
    else:
        kc = kc_exp_hamming_vec(h, hp, lambda_c)
    return kx + kc + kx * kc

# ----------------------------
# Closed-form continuous KMEs on Uniform([-L,L]^d)
#   m_x(x') = E_x[kx(x,x')] and I_xx' = E_{x,x'}[kx(x,x')]
# ----------------------------
def rbf_kme_uniform_1d(xp_scalar, ell, L):
    s2 = jnp.sqrt(2.0) * ell
    return (jnp.sqrt(jnp.pi / 2.0) * ell / (2.0 * L)) * (
        erf((L - xp_scalar) / s2) + erf((L + xp_scalar) / s2)
    )

def rbf_kme_uniform_box(Xp, ell, L):
    def per_row(xrow):
        m1d_vals = jax.vmap(lambda z: rbf_kme_uniform_1d(z, ell, L))(xrow)
        return jnp.prod(m1d_vals)
    return jax.vmap(per_row)(Xp)

def rbf_kmean_uniform_1d(ell, L):
    a = jnp.sqrt(2.0) * L / ell
    term1 = jnp.sqrt(jnp.pi / 2.0) * ell / L * erf(a)
    term2 = (ell**2) / (2.0 * L**2) * (-jnp.expm1(-a*a))
    return term1 - term2

def rbf_kmean_uniform_box(ell, L, d):
    e1 = rbf_kmean_uniform_1d(ell, L)
    return e1 ** d

# ----------------------------
# Expected categorical factors under uniform prior over 17x17 combos
# ----------------------------
def Eh_kc_uniform_2c(cat, lambda_c, q=17):
    if cat == "equal":
        return (1.0 / q) ** 2
    else:
        return ((1.0 + (q - 1.0) * jnp.exp(-lambda_c)) / q) ** 2

def EhEh_kc_uniform_2c(cat, lambda_c, q=17):
    return Eh_kc_uniform_2c(cat=cat, lambda_c=lambda_c, q=q)

# ----------------------------
# Build Gram and z for 2 categorical dims (uniform over 17x17 combos)
# ----------------------------
def build_gram_and_z_box_2c(X, H, ell, L, kernel, cat, lambda_c, q, ridge=1e-8):
    n = X.shape[0]
    if kernel == "product":
        kfun = lambda a, b: k_mixed_product_2c(a, b, ell, cat=cat, lambda_c=lambda_c)
    else:
        kfun = lambda a, b: k_mixed_shared_2c(a, b, ell, cat=cat, lambda_c=lambda_c)

    def row(i):
        xi, hi = X[i], H[i]
        return jax.vmap(lambda j: kfun((xi, hi), (X[j], H[j])))(jnp.arange(n))

    K = jax.vmap(row)(jnp.arange(n))
    K = K + ridge * jnp.eye(n)

    # continuous kernel mean m_x(x_i)
    mx = rbf_kme_uniform_box(X, ell, L)  # (n,)

    # categorical expectation (uniform prior) -- same scalar for all i
    Ehkc = Eh_kc_uniform_2c(cat=cat, lambda_c=lambda_c, q=q)

    if kernel == "product":
        # z_i = E[kx] * E[kc] = m_x(x_i) * Eh[kc]
        z = mx * Ehkc
    else:
        # z_i = E[kx] + E[kc] + E[kx]*E[kc] = mx + Ehkc + mx*Ehkc
        z = mx + Ehkc + mx * Ehkc

    return K, z

def mu_kk_2c(ell, L, d, kernel, cat, lambda_c, q):

    Ikxkx = rbf_kmean_uniform_box(ell, L, d)      
    Sc    = EhEh_kc_uniform_2c(cat, lambda_c, q)  
    if kernel == "product":
        return Ikxkx * Sc
    else:
        return Ikxkx + Sc + Ikxkx * Sc

# ----------------------------
# Core BQ solver
# ----------------------------
def bq_estimate_and_var(K, z, fvals, mu_kk):
    Lc, lower = sla.cho_factor(K, lower=True)
    w = sla.cho_solve((Lc, lower), z)
    I_hat = jnp.dot(w, fvals)
    qv = sla.cho_solve((Lc, lower), z)
    Var = mu_kk - jnp.dot(z, qv)
    return I_hat, Var, w

# ----------------------------
# Designs, MC & RBQ helpers
# ----------------------------
def sample_levels(key, n, q):
    levels = jnp.linspace(-1.0, 1.0, q)
    k1, k2 = jax.random.split(key)
    idx1 = jax.random.randint(k1, (n,), 0, q)
    idx2 = jax.random.randint(k2, (n,), 0, q)
    return levels[idx1], levels[idx2]  # (n,), (n,)

def make_random_design(key, n, d, L, q):
    kx, kh = jax.random.split(key)
    X = jax.random.uniform(kx, (n, d), minval=-L, maxval=L)
    h1, h2 = sample_levels(kh, n, q=q)
    H = jnp.stack([h1, h2], axis=1)  # (n,2)
    return X, H

def mc_estimate_single(key, n, d, L, q):
    X, H = make_random_design(key, n, d, L, q)
    fvals = v_ackley_2C(X, H)
    return float(fvals.mean())

def mc_estimate_repeated(base_key, n, d, L, q, reps, I_true):
    errs = []
    k = base_key
    for _ in range(reps):
        k, kk = jax.random.split(k)
        Ihat = mc_estimate_single(kk, n, d, L, q)
        if I_true is not None:
            errs.append(abs(Ihat - I_true))
        else:
            errs.append(Ihat)
    if I_true is not None:
        return float(np.mean(errs)), float(np.std(errs)), float(I_true)
    else:
        m = float(np.mean(errs))
        s = float(np.std(errs))
        return m, s, m

def rbq_once(key, n, d, L, ell, kernel, cat, lambda_c, q):
    X, H = make_random_design(key, n, d, L, q)
    fvals = v_ackley_2C(X, H)
    K, z = build_gram_and_z_box_2c(X, H, ell, L, kernel, cat, lambda_c, q)
    mu_kk = mu_kk_2c(ell, L, d, kernel, cat, lambda_c, q)
    I_hat, Var, _ = bq_estimate_and_var(K, z, fvals, mu_kk)
    return float(I_hat), float(jnp.sqrt(jnp.maximum(Var, 0.0)))

def rbq_repeated(base_key, n, d, L, ell, kernel, cat, lambda_c, reps, q, I_true):
    errs, sds = [], []
    k = base_key
    for _ in range(reps):
        k, kk = jax.random.split(k)
        Ihat, sd = rbq_once(kk, n, d, L, ell, kernel, cat, lambda_c, q)
        if I_true is not None:
            errs.append(abs(Ihat - I_true))
        else:
            errs.append(Ihat)
        sds.append(sd)
    return float(np.mean(errs)), float(np.std(errs)), float(np.mean(sds)), float(I_true if I_true is not None else np.mean(errs))


def plot_results(n_vals, all_results, save_path="results", filename="ackley2c_abs_err.pdf"):
    os.makedirs(save_path, exist_ok=True)

    styles = {
        "MC":         {"color": "k", "marker": "o", "label": "MC"},
        "RBQ": {"color": "b", "marker": "s", "label": "BayesSum + BQ"},
    }

    plt.figure(figsize=(10, 6))
    handles, labels = [], []

    for name in [m for m in styles.keys() if m in all_results]:
        data = all_results[name]
        mean_err = np.asarray(data["mean_abs_error"], dtype=float)
        se_err   = np.asarray(data["se_abs_error"], dtype=float)

        # 95% CI using normal approximation
        eps    = 1e-12  # keep strictly positive for log y-scale
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
    # plt.title(title, fontsize=20)
    plt.yscale("log")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.legend(loc="best", fontsize=32)
    out_path = os.path.join(save_path, filename)
    plt.savefig(out_path, format="pdf")
    # plt.show()
    plt.close()
    print(f"Saved plot to: {out_path}")

def save_results(filename, n_vals, results):
    out = {"n_vals": np.asarray(n_vals, dtype=int)}
    for method, d in results.items():
        out[f"{method}_mean_abs_error"] = np.asarray(d["mean_abs_error"], dtype=float)
        out[f"{method}_se_abs_error"]   = np.asarray(d["se_abs_error"],   dtype=float)
    np.savez(filename, **out)
    print(f"Saved {filename} (portable)")

def load_results(filename):
    data = np.load(filename)
    n_vals = data["n_vals"]
    all_results = {}
    for key in data.files:
        if key == "n_vals": 
            continue
        method, field = key.split("_", 1)  
        val = data[key]
        all_results.setdefault(method, {})
        all_results[method][field] = val[0] if field == "true_val" else val
    return n_vals, all_results

def calibration(n_calibration, seeds, key_seed, ell, lambda_, d, L, q, I_true, kernel, cat):
    mus, stds = [], []
    key = jax.random.PRNGKey(key_seed)

    for s in range(seeds):
        key, subkey = jax.random.split(key)
        mu, std = rbq_once(subkey, n_calibration, d, L, ell, kernel, cat, lambda_, q)
        mus.append(mu)
        stds.append(std)

    mus  = jnp.array(mus)
    stds = jnp.maximum(jnp.array(stds), 1e-30)

    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = jnp.sqrt(2.0) * erfinv(C_nom)

    half = z[:, None] * stds[None, :]
    inside = (I_true >= mus[None, :] - half) & (I_true <= mus[None, :] + half)
    emp_cov = jnp.mean(inside, axis=1)

    return float(I_true), C_nom, emp_cov, mus, stds

# ----------------------------
# Demo / Experiment
# ----------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # settings
    d        = 1
    L        = 1.0
    ell      = 0.8
    q     = 17
    ns       = [20, 40, 60, 80, 100]
    seeds  = 10

    kernel = "shared"
    cat      = "exp"
    lambda_c = 1.0   # only used if cat="exp"

    I_true = ackley2c_truth_d1_exact(L)
    I_true_mc   = ackley2c_truth_d1_mc_joint(L, jax.random.PRNGKey(42), q, 300_000)
    print(f"Large-MC ground truth: I_true = {I_true:+.8f}")
    print(f"Large-MC truth (approx): I_true = {I_true_mc:+.8f}")
    # # storage
    # err_mc_mean,    err_mc_std    = [], []
    # err_rbq_shr_m,  err_rbq_shr_s, sd_rbq_shr_m = [], [], []

    # for n in ns:
    #     # RBQ (randomized designs, mean±sd over reps)
    #     m_e_s, s_e_s, m_sd_s, _ = rbq_repeated(jax.random.PRNGKey(8000 + n), n, d, L, ell, kernel, cat, lambda_c, seeds, q, I_true)
    #     err_rbq_shr_m.append(m_e_s)
    #     err_rbq_shr_s.append(s_e_s)
    #     sd_rbq_shr_m.append(m_sd_s)

    #     # MC baseline (mean±sd over reps)
    #     mc_mean_err, mc_std_err, _ = mc_estimate_repeated(jax.random.PRNGKey(9000 + n), n, d, L, q, seeds, I_true)
    #     err_mc_mean.append(mc_mean_err)
    #     err_mc_std.append(mc_std_err)

    #     print(
    #         f"n={n:3d} | "
    #         f"RBQ_shared[{cat}]: {m_e_s:.3e} ± {s_e_s:.3e} (mean SD={m_sd_s:.3e}) | "
    #         f"MC: {mc_mean_err:.3e} ± {mc_std_err:.3e}"
    #     )

    # # Package results for the new plotting helper.
    # # Convert (std across reps) -> standard error for 95% CIs: se = std / sqrt(reps)
    # all_results = {
    #     "RBQ": {
    #         "mean_abs_error": np.asarray(err_rbq_shr_m, dtype=float),
    #         "se_abs_error":   np.asarray(err_rbq_shr_s, dtype=float) / np.sqrt(seeds),
    #     },
    #     "MC": {
    #         "mean_abs_error": np.asarray(err_mc_mean, dtype=float),
    #         "se_abs_error":   np.asarray(err_mc_std, dtype=float) / np.sqrt(seeds),
    #     },
    # }

    # save_results("mixed.npz", ns, all_results)
    # n_vals, all_results = load_results("mixed.npz")
    # print(all_results)

    # plot_results(ns, all_results, save_path="results", filename="ackley2c_abs_err.pdf")


    # # Calibration
    # calibrations = {}
    # calibration_seeds = 50
    # n_calibration = 60
    # key_seed = 5
    # t_e, C_nom, emp_cov, mus, stds = calibration(n_calibration, calibration_seeds, key_seed, ell, lambda_c, d, L, q, I_true, "shared", cat)

    # jnp.savez("results/mixed_calibration_results.npz",
    #          t_true=t_e, C_nom=jnp.array(C_nom),
    #          emp_cov=jnp.array(emp_cov),
    #          mus=jnp.array(mus), vars=jnp.array(stds))
    # print("mixed_calibration.npz")
