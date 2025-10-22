# ising_boxplot_rmse_vs_dimension.py
import os, time
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, random as jrand
from jax.scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# -------------------------
# Ising on LxL grid
# -------------------------
@partial(jit, static_argnums=(0,))
def J_matrix(L):
    n = L * L
    idx = jnp.arange(n).reshape(L, L)
    right = jnp.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    down  = jnp.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    pairs = jnp.concatenate([right, down], axis=0)
    i, j = pairs[:, 0], pairs[:, 1]
    J = jnp.zeros((n, n), dtype=jnp.float32).at[i, j].set(1.0).at[j, i].set(1.0)
    return 0.1 * J

@jit
def f_single(x, J, beta):
    diff = (x[:, None] == x[None, :]).astype(J.dtype)
    energy = jnp.sum(J * diff) / 2.0
    return jnp.exp(-beta * energy)

@jit
def f_batch(X, J, beta):
    return vmap(lambda x: f_single(x, J, beta))(X)

# -------------------------
# Hamming kernel on {0,1}^d
# -------------------------
@jit
def hamming_kernel_vec(x, X, lam):
    x1 = x.reshape(1, -1)
    h = jnp.sum(X != x1, axis=1)
    return jnp.exp(-lam * h).reshape(-1, 1)

@jit
def hamming_gram(X, lam):
    X1, X2 = X[:, None, :], X[None, :, :]
    h = jnp.sum(X1 != X2, axis=2)
    return jnp.exp(-lam * h)

def kernel_mean_scalar(lam, d):
    # U ~ Uniform({0,1}^d), E[e^{-lam * Hamming(x,U)}] = ((1+e^{-lam})/2)^d
    return ((1.0 + jnp.exp(-lam)) / 2.0) ** d

# -------------------------
# Woodbury (block inverse) update
# -------------------------
@jit
def woodbury_append(K_inv, inv_ones, X_obs, x_new, lam, jitter=1e-8):
    kx = hamming_kernel_vec(x_new, X_obs, lam)          # (n,1)
    u  = K_inv @ kx                                     # (n,1)
    s  = jnp.maximum(1.0 + jitter - (kx.T @ u)[0,0], 1e-14)
    TL = K_inv + (u @ u.T) / s
    TR = -u / s
    BL = TR.T
    BR = jnp.array([[1.0 / s]], dtype=K_inv.dtype)
    top = jnp.hstack([TL, TR])
    bot = jnp.hstack([BL, BR])
    K_inv_new = jnp.vstack([top, bot])
    inv_ones_new = K_inv_new @ jnp.ones((K_inv_new.shape[0], 1), dtype=K_inv.dtype)
    return K_inv_new, inv_ones_new

@jit
def candidate_gain(x, X_obs, K_inv, inv_ones, lam, z0, jitter=1e-8):
    kx = hamming_kernel_vec(x, X_obs, lam)
    u  = K_inv @ kx
    s  = jnp.maximum(1.0 + jitter - (kx.T @ u)[0,0], 1e-14)
    qx = (kx.T @ inv_ones)[0,0]
    eta = z0 * (1.0 - qx)
    return (eta * eta) / s

@jit
def pick_greedy(X_obs, K_inv, inv_ones, candidates, lam, d):
    z0 = kernel_mean_scalar(lam, d)
    gains = vmap(lambda x: candidate_gain(x, X_obs, K_inv, inv_ones, lam, z0))(candidates)
    return candidates[jnp.argmax(gains)]

def subsample_candidates(key, d, m):
    return jrand.bernoulli(key, p=0.5, shape=(m, d)).astype(jnp.int32)

def bq_bo_fast(lam, d, beta, L, N, m_cand, key, jitter=1e-8):
    J = J_matrix(L)
    key, k0 = jrand.split(key)
    x0 = subsample_candidates(k0, d, 1)[0]
    f0 = f_single(x0.astype(jnp.float32), J, beta)
    X = [x0.astype(jnp.int32)]
    y = [f0.astype(jnp.float64)]
    K_inv    = jnp.array([[1.0]], dtype=jnp.float64)
    inv_ones = jnp.array([[1.0]], dtype=jnp.float64)

    for t in range(1, N):
        key, kc = jrand.split(key)
        cand = subsample_candidates(kc, d, m_cand)
        x_star = pick_greedy(jnp.array(X), K_inv, inv_ones, cand, lam, d)
        K_inv, inv_ones = woodbury_append(K_inv, inv_ones, jnp.array(X), x_star, lam, jitter)
        X.append(x_star.astype(jnp.int32))
        y.append(f_single(x_star.astype(jnp.float32), J, beta).astype(jnp.float64))

    # μ = z0 * 1^T K^{-1} y ;  var = z0 - z0^2 * 1^T K^{-1} 1 (not used for RMSE)
    z0 = kernel_mean_scalar(lam, d)
    yv = jnp.array(y).reshape(-1,1)
    mu = (z0 * (jnp.ones((1, K_inv.shape[0])) @ (K_inv @ yv)))[0,0]
    return float(mu)

def bq_random(lam, d, beta, L, N, key, jitter=1e-8):
    J = J_matrix(L)
    X = jrand.bernoulli(key, p=0.5, shape=(N, d)).astype(jnp.int32)
    y = f_batch(X.astype(jnp.float32), J, beta).reshape(N,1).astype(jnp.float64)
    K = hamming_gram(X, lam) + jitter * jnp.eye(N)
    Lc, lower = cho_factor(K)
    z0 = kernel_mean_scalar(lam, d)
    z  = jnp.full((N,1), z0, dtype=jnp.float64)
    Kinv_y = cho_solve((Lc, lower), y)
    mu = (z.T @ Kinv_y)[0,0]
    return float(mu)

def mc_estimate(d, beta, L, N, key):
    J = J_matrix(L)
    X = jrand.bernoulli(key, p=0.5, shape=(N, d)).astype(jnp.float32)
    return float(jnp.mean(f_batch(X, J, beta)))

# Russian roulette (same as before)
def rr_debiased_mc(key, d, beta, L, N):
    # choose rho so E[tau+1] ≈ N
    rho = float(jnp.clip(1.0 - 1.0 / jnp.maximum(1, N), 0.0, 0.999999))
    J = J_matrix(L)
    m = 0; qm = 1.0; total = 0.0; used = 0; s = 0.0; S_prev = 0.0
    k = key
    k, sk = jrand.split(k); x = jrand.bernoulli(sk, 0.5, (d,)).astype(jnp.float32)
    fx = float(f_single(x, J, beta)); s += fx
    S_m = s / 1.0; total += (S_m - S_prev) / qm; S_prev = S_m; used = 1
    while True:
        k, sk = jrand.split(k)
        if not bool(jrand.bernoulli(sk, rho)): break
        m += 1; qm *= rho
        k, sk = jrand.split(k); x = jrand.bernoulli(sk, 0.5, (d,)).astype(jnp.float32)
        fx = float(f_single(x, J, beta)); s += fx
        S_m = s / (m+1); total += (S_m - S_prev) / qm; S_prev = S_m; used += 1
    return float(total)

# exact mean (enumeration) for small d (<=20–22 is OK on CPU)
@partial(jit, static_argnums=(0,))
def all_states_cached(d):
    n_states = 2**d
    return jnp.array(((jnp.arange(n_states)[:, None] >> jnp.arange(d)) & 1), dtype=jnp.float32)

def exact_true_expectation(L, d, beta, batch=10000):
    J = J_matrix(L)
    X = all_states_cached(d)
    n = X.shape[0]
    s = 0.0
    for i in range(0, n, batch):
        s += jnp.sum(f_batch(X[i:i+batch], J, beta))
    return float(s / n)

# -------------------------
# Experiment: RMSE vs dimension (boxplots)
# -------------------------
def rmse_vs_dimension_boxplot(
    L_list=(2,3,4),   # dimensions are d=L^2  → 4,9,16
    N=60,             # fixed design size
    seeds=40,
    lam=0.08,         # kernel lengthscale (kept fixed for fairness)
    beta=1.0/2.269,
    m_cand=600,
    out_png="ising_rmse_vs_dimension.png"
):
    rng0 = jrand.PRNGKey(0)
    dims = [L*L for L in L_list]

    # collect per-dimension per-method errors across seeds
    methods = ["MC", "BQ-Random", "BQ-BO", "RR"]
    per_dim_errs = {m: [] for m in methods}
    per_dim_rmse = {m: [] for m in methods}

    for L in L_list:
        d = L*L
        t_true = exact_true_expectation(L, d, beta)
        errs = {m: [] for m in methods}

        for s in range(seeds):
            key = jrand.fold_in(rng0, s + 137*L)
            k1, k2, k3, k4 = jrand.split(key, 4)

            mu_mc  = mc_estimate(d, beta, L, N, k1)
            mu_rnd = bq_random(lam, d, beta, L, N, k2)
            mu_bo  = bq_bo_fast(lam, d, beta, L, N, m_cand, k3)
            mu_rr  = rr_debiased_mc(k4, d, beta, L, N)

            errs["MC"].append(abs(mu_mc  - t_true))
            errs["BQ-Random"].append(abs(mu_rnd - t_true))
            errs["BQ-BO"].append(abs(mu_bo  - t_true))
            errs["RR"].append(abs(mu_rr  - t_true))

        # stash epoch results
        for m in methods:
            arr = jnp.array(errs[m])
            per_dim_errs[m].append(arr)
            per_dim_rmse[m].append(float(jnp.sqrt(jnp.mean(arr**2))))

    print(per_dim_errs["BQ-BO"][0]) 

    # ----------------- plot grouped boxplots -----------------
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(9,4.6))
    colors = ["#4e79a7","#f28e2b","#59a14f","#e15759"]  # MC, BQ-Rnd, BQ-BO, RR
    box_w = 0.16
    x0 = jnp.arange(len(dims))

    for j, m in enumerate(methods):
        positions = x0 + (j - 1.5)*box_w
        data = [jnp.asarray(per_dim_errs[m][i]) for i in range(len(dims))]
        bp = plt.boxplot(
            data, positions=positions, widths=box_w*0.9,
            patch_artist=True, showfliers=True
        )
        for box in bp['boxes']:
            box.set(facecolor=colors[j], edgecolor="black", linewidth=1.0)
        for elem in ['whiskers','caps','medians','fliers']:
            for art in bp[elem]:
                art.set(color="black", linewidth=0.9)

        # RMSE markers on top of boxes
        plt.plot(positions, per_dim_rmse[m], "kD", ms=3)

    plt.yscale("log")
    plt.xticks(x0, dims)
    plt.xlabel("Dimension")
    plt.ylabel("RMSE")
    plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
    # legend patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[i], edgecolor='k', label=methods[i]) for i in range(len(methods))]
    plt.legend(handles=handles, ncol=len(methods), loc="upper left", bbox_to_anchor=(0,1.02))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved {out_png}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # choose the grid sizes you want to compare.
    # (L=2,3,4 → d=4,9,16 keeps exact mean feasible.)
    rmse_vs_dimension_boxplot(
        L_list=(2,3,4),
        N=60,
        seeds=60,
        lam=0.08,
        beta=1.0/2.269,
        m_cand=600,
        out_png="ising_rmse_vs_dimension.png"
    )
