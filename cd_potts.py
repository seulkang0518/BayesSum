# potts_bq_full.py
# 2D Potts: exact grad vs Gibbs, MH, and Stein-BQ (KDSD with RBF on Hamming)
# beta is stored on the PottsOracle (no need to pass it around).

import os
os.environ["JAX_ENABLE_X64"] = "1"   # must be set before importing jax

import itertools
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, jit
import jax.scipy.linalg as sla
from functools import partial
from scipy.optimize import minimize
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# -----------------------------
# Utilities
# -----------------------------
def lattice_J_mask(Lside: int) -> jnp.ndarray:
    """Nearest-neighbor mask for Lside x Lside grid."""
    d = Lside * Lside
    idx = jnp.arange(d).reshape(Lside, Lside)
    rights = jnp.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    downs  = jnp.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    edges = jnp.concatenate([rights, downs], axis=0)
    mask = jnp.zeros((d, d, 1, 1), dtype=jnp.float64)
    mask = mask.at[edges[:, 0], edges[:, 1], 0, 0].set(1.0)
    mask = mask.at[edges[:, 1], edges[:, 0], 0, 0].set(1.0)
    diag = jnp.arange(d)
    mask = mask.at[diag, diag, 0, 0].set(0.0)
    return mask

# -----------------------------
# Potts Oracle (raw energy; stores beta)
# -----------------------------
class PottsOracle:
    def __init__(self, h, J, J_mask=None, beta: float = 1.0):
        self.h = jnp.array(h, dtype=jnp.float64)         # (d,q)
        self.beta = float(beta)
        d, q = self.h.shape
        self.d, self.q = int(d), int(q)

        J = jnp.array(J, dtype=jnp.float64)              # (d,d,q,q)
        Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
        if J_mask is not None:
            Jsym = Jsym * J_mask
        diag = jnp.arange(d)
        Jsym = Jsym.at[diag, diag, :, :].set(0.0)        # ensure no self-coupling
        self.J = Jsym

    def energy(self, x):
        """E_raw(x) = <h,x> + 1/2 * <J, x⊗x>   (no -beta here)."""
        x_oh = jax.nn.one_hot(x, self.q, dtype=jnp.float64)
        e_h = jnp.einsum('iq,iq->', x_oh, self.h)
        e_J = 0.5 * jnp.einsum('iq,ijar,jr->', x_oh, self.J, x_oh)
        return e_h + e_J

    def log_unnorm(self, x):
        return -self.beta * self.energy(x)

    def suff_stats(self, x):
        """∇θ U(x) = beta * features (for log p ∝ -beta * E_raw)."""
        x_oh = jax.nn.one_hot(x, self.q, dtype=jnp.float64)
        gh = self.beta * x_oh                                  # (d,q)
        gJ = self.beta * jnp.einsum('ia,jr->ijar', x_oh, x_oh) # (d,d,q,q)
        diag = jnp.arange(self.d)
        gJ = gJ.at[diag, diag, :, :].set(0.0)
        return gh, gJ

    def cond_probs(self, x, i):
        base = self.h[i]                   # (q,)
        Ji   = self.J[i]                   # (d, q, q)
        xoh  = jax.nn.one_hot(x, self.q, dtype=jnp.float64)  # (d, q)
        xoh  = xoh.at[i].set(0.0)          # exclude self
        add = jnp.einsum('jkq,jq->k', Ji, xoh)      # (q,)
        logits = -self.beta * (base + add)
        return jax.nn.softmax(logits)

    @staticmethod
    def neighbors_cyclic(x, q, shift):
        d = x.shape[0]
        x_shift = (x + shift) % q
        X = jnp.tile(x, (d, 1))
        return X.at[jnp.arange(d), jnp.arange(d)].set(x_shift)

    def cyclic_score(self, x):
        nbh = self.neighbors_cyclic(x, self.q, shift=+1)
        Ex = self.energy(x)
        Enb = vmap(self.energy)(nbh)
        return 1.0 - jnp.exp(-self.beta * (Enb - Ex))

# -----------------------------
# Exact enumeration
# -----------------------------
def enumerate_states(L, q):
    d = L*L
    return jnp.array(list(itertools.product(range(q), repeat=d)), dtype=jnp.int32)

def exact_mu(oracle: PottsOracle, L, q):
    X = enumerate_states(L, q)
    logw = jax.vmap(oracle.log_unnorm)(X)
    m = jnp.max(logw)
    w = jnp.exp(logw - m)
    Z = jnp.sum(w)
    probs = w / Z
    gh, gJ = jax.vmap(oracle.suff_stats)(X)
    E_gh = jnp.tensordot(probs, gh, axes=1)  # (d,q)
    E_gJ = jnp.tensordot(probs, gJ, axes=1)  # (d,d,q,q)
    return E_gh, E_gJ

def _idx_to_states_np(start, end, q, d):
    """Map integers [start,end) to base-q states of length d (NumPy)."""
    idx = np.arange(start, end, dtype=np.int64)
    out = np.empty((idx.size, d), dtype=np.int32)
    x = idx.copy()
    for k in range(d-1, -1, -1):
        out[:, k] = (x % q).astype(np.int32)
        x //= q
    return out

def exact_mu_batched(oracle, L: int, q: int, batch_size: int = 200_000):
    d = L * L
    total_states = q ** d

    # Running log-max and shifted sums for stability:
    M = -np.inf
    sum_w_shift = 0.0
    sum_gh_shift = None   # will be initialized on first batch
    sum_gJ_shift = None

    for start in range(0, total_states, batch_size):
        end = min(start + batch_size, total_states)

        # states for this batch
        X = jnp.array(_idx_to_states_np(start, end, q, d), dtype=jnp.int32)   # (B, d)

        # log-weights and sufficient stats for this batch
        E = vmap(oracle.energy)(X)                      # (B,)
        logw = -oracle.beta * E                             # (B,)
        gh_batch, gJ_batch = vmap(oracle.suff_stats)(X)     # gh: (B,d,q), gJ: (B,d,d,q,q)

        # update running max
        m = float(jnp.max(logw))
        if m > M:
            scale = np.exp(M - m)
            sum_w_shift *= scale
            if sum_gh_shift is not None:
                sum_gh_shift = sum_gh_shift * scale
                sum_gJ_shift = sum_gJ_shift * scale
            M = m

        # accumulate shifted sums for this batch
        w_shift = jnp.exp(logw - M)                         # (B,)
        gh_contrib = jnp.tensordot(w_shift, gh_batch, axes=1)   # (d,q)
        gJ_contrib = jnp.tensordot(w_shift, gJ_batch, axes=1)   # (d,d,q,q)

        if sum_gh_shift is None:
            sum_gh_shift = gh_contrib
            sum_gJ_shift = gJ_contrib
        else:
            sum_gh_shift = sum_gh_shift + gh_contrib
            sum_gJ_shift = sum_gJ_shift + gJ_contrib

        sum_w_shift += float(jnp.sum(w_shift))

    # finalize expectations
    E_gh = sum_gh_shift / sum_w_shift
    E_gJ = sum_gJ_shift / sum_w_shift
    return E_gh, E_gJ

# -------------------------
# MCMC Samplers
# -------------------------
def gibbs_sample(oracle: PottsOracle, key, n, burnin=300, thinning=2, x0=None):
    d, q = oracle.d, oracle.q
    if x0 is None:
        x0 = jrand.randint(key, (d,), 0, q)
    total = burnin + n * thinning
    x = jnp.array(x0)
    out = []
    k = key
    for t in range(total):
        for i in range(d):
            k, sub = jrand.split(k)
            probs = oracle.cond_probs(x, i)
            xi = int(jrand.categorical(sub, jnp.log(probs)))
            x = x.at[i].set(xi)
        if t >= burnin and ((t - burnin) % thinning == 0):
            out.append(x)
    return jnp.stack(out, axis=0)

def metropolis_hastings_sample(oracle: PottsOracle, key, n, burnin=300, thinning=2, x0=None):
    d, q = oracle.d, oracle.q
    if x0 is None:
        key, subkey = jrand.split(key)
        x0 = jrand.randint(subkey, (d,), 0, q)

    total = burnin + n * thinning
    x_current = jnp.array(x0)
    E_current = oracle.energy(x_current)
    out = []
    for t in range(total):
        key, s1, s2, s3 = jrand.split(key, 4)
        flip_idx = jrand.randint(s1, shape=(), minval=0, maxval=d)
        a_old = int(x_current[flip_idx])
        cands = jnp.delete(jnp.arange(q), a_old)
        new_val = jrand.choice(s2, cands)
        x_prop = x_current.at[flip_idx].set(new_val)

        E_prop = oracle.energy(x_prop)
        delta_E = E_prop - E_current
        acc = jnp.minimum(1.0, jnp.exp(-oracle.beta * delta_E))
        u = jrand.uniform(s3)
        accept = u < acc
        x_current = jnp.where(accept, x_prop, x_current)
        E_current = jnp.where(accept, E_prop, E_current)
        if t >= burnin and ((t - burnin) % thinning == 0):
            out.append(x_current)
    return jnp.stack(out, axis=0)

# -----------------------------
# KDSD Stein kernel (RBF on Hamming)
# -----------------------------
def rbf_kernel(x, y, gamma):
    """exp(-gamma * ||onehot(x)-onehot(y)||^2) == exp(-2*gamma*Hamming(x,y))."""
    h = jnp.sum(x != y)
    return jnp.exp(-2.0 * gamma * h)

@partial(jit, static_argnames=('oracle',))
def stein_kernel_pair(oracle: PottsOracle, x, y, gamma: float):
    d, q = oracle.d, oracle.q
    sx = oracle.cyclic_score(x)                      # (d,)
    sy = oracle.cyclic_score(y)                      # (d,)
    k_xy = rbf_kernel(x, y, gamma)                   # scalar

    x_inv = oracle.neighbors_cyclic(x, q, shift=-1)  # (d,d)
    y_inv = oracle.neighbors_cyclic(y, q, shift=-1)  # (d,d)

    k_x_yinv   = vmap(lambda yr: rbf_kernel(x,  yr, gamma))(y_inv)  # (d,)
    k_xinv_y   = vmap(lambda xr: rbf_kernel(xr, y,  gamma))(x_inv)  # (d,)
    K_xinv_yinv= vmap(lambda xr: vmap(lambda yr: rbf_kernel(xr, yr, gamma))(y_inv))(x_inv)  # (d,d)

    trace_term = jnp.trace(K_xinv_yinv) - jnp.sum(k_x_yinv) - jnp.sum(k_xinv_y) + d * k_xy
    t1 = jnp.dot(sx, sy) * k_xy
    t2 = - jnp.dot(sx, (k_xy - k_x_yinv))
    t3 = - jnp.dot((k_xy - k_xinv_y), sy)
    return t1 + t2 + t3 + trace_term

def stein_kernel_matrix(oracle: PottsOracle, X, gamma: float):
    pair = lambda xi, xj: stein_kernel_pair(oracle, xi, xj, gamma)
    return vmap(vmap(pair, in_axes=(None, 0)), in_axes=(0, None))(X, X)

def median_gamma_on_int(X, q, n_subset=500, key=None):
    """Median heuristic on one-hot Euclidean distances (same as 2*Hamming)."""
    n = X.shape[0]
    if key is not None and n > n_subset:
        idx = jrand.choice(key, n, shape=(n_subset,), replace=False)
        Xs = X[idx]
    else:
        Xs = X if n <= n_subset else X[:n_subset]
    Z = jax.nn.one_hot(Xs, q, dtype=jnp.float64).reshape(Xs.shape[0], -1)
    norms = jnp.sum(Z**2, axis=1, keepdims=True)
    D2 = norms + norms.T - 2.0 * (Z @ Z.T)
    iu = jnp.triu_indices(Z.shape[0], k=1)
    med_sq = jnp.median(D2[iu])
    # For Gaussian RBF: k = exp(-gamma * ||z-z'||^2); use gamma = 1 / median(||z-z'||^2)
    return float(1.0 / jnp.clip(med_sq, 1e-12, None))

def neg_log_marginal_likelihood(params, X, Y, oracle, jitter=1e-8):
    log_gamma, log_c = params
    gamma = jnp.exp(log_gamma)
    c     = jnp.exp(log_c)

    n, p = Y.shape
    Ks = stein_kernel_matrix(oracle, X, gamma)       
    K  = Ks + c * jnp.ones((n, n)) + jitter * jnp.eye(n)

    L, lower = sla.cho_factor(K, lower=True)
    alpha = sla.cho_solve((L, lower), Y)            
    trace_term = jnp.sum(Y * alpha)                 
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    nll = 0.5*trace_term + 0.5*p*logdetK + 0.5*n*p*jnp.log(2*jnp.pi)
    return nll

def tune_and_estimate_stein_bq(oracle: PottsOracle, X, fvals, key=None, jitter=1e-8):
    gamma_init = median_gamma_on_int(X, oracle.q, key=key)
    c_init     = jnp.maximum(jnp.var(fvals), 1e-6)
    init       = jnp.log(jnp.array([gamma_init, c_init], dtype=jnp.float64))

    def nll_jax(p):
        return neg_log_marginal_likelihood(p, X, fvals, oracle, jitter=jitter)

    grad_nll = jax.jit(jax.grad(nll_jax))
    def fun_np(p_np): return float(nll_jax(jnp.array(p_np)))
    def jac_np(p_np): return np.array(grad_nll(jnp.array(p_np)), dtype=np.float64)

    # no bounds:
    res = minimize(fun=fun_np, x0=np.array(init, dtype=np.float64),
                   method='L-BFGS-B', jac=jac_np)

    log_gamma_opt, log_c_opt = res.x
    gamma_opt = float(np.exp(log_gamma_opt))
    c_opt     = float(np.exp(log_c_opt))

    n, p = fvals.shape
    Ks = stein_kernel_matrix(oracle, X, gamma_opt)
    K  = Ks + c_opt * jnp.ones((n, n)) + jitter * jnp.eye(n)
    L, lower = sla.cho_factor(K, lower=True)
    alpha = sla.cho_solve((L, lower), fvals)             # (n,p) = K^{-1} fvals

    mu_hat = c_opt * jnp.sum(alpha, axis=0)              # μ̂ = c * 1^T α  ∈ R^p
    info = {'gamma': gamma_opt, 'c': c_opt, 'opt_result': res}
    return mu_hat, info


def flatten_stats(gh, gJ):
    return jnp.concatenate([gh.ravel(), gJ.ravel()], axis=0)

def grad_from_mu(gh_obs, gJ_obs, mu_vec):
    obs = flatten_stats(gh_obs, gJ_obs)
    return -obs + mu_vec

# -----------------------------
# Sweep + plot
# -----------------------------

def run_sweep_and_plot(L=2, q=3, beta=1.0, seeds=10,
                       ns=(25, 50, 100, 200, 400),
                       burnin=500, thinning=5):
    d = L * L
    key = jrand.PRNGKey(0)

    # random params (fixed across runs for fairness)
    key, k1, k2, k3 = jrand.split(key, 4)
    h = 0.3 * jrand.normal(k1, (d, q))
    J = 0.05 * jrand.normal(k2, (d, d, q, q))
    J_mask = lattice_J_mask(L)
    oracle = PottsOracle(h, J, J_mask, beta=beta)

    # observed config
    x_obs = jrand.randint(k3, (d,), 0, q)

    # exact gradient
    if L <= 2:
        E_gh, E_gJ = exact_mu(oracle, L, q)
    else:
        E_gh, E_gJ = exact_mu_batched(oracle, L, q, batch_size=200_000)

    gh_obs, gJ_obs = oracle.suff_stats(x_obs)
    grad_true = grad_from_mu(gh_obs, gJ_obs, flatten_stats(E_gh, E_gJ))
    norm_true = float(jnp.linalg.norm(grad_true)) + 1e-12

    print(grad_true)
    print(norm_true)

    errs_gibbs, errs_mh, errs_bq = [], [], []

    for n in ns:
        # collect errors over seeds
        e_gibbs, e_mh, e_bq = [], [], []
        for s in range(seeds):
            ks = jrand.fold_in(key, s)
            kg, km, kb = jrand.split(ks, 3)

            # Gibbs
            X_gibbs = gibbs_sample(oracle, kg, n=n, burnin=burnin, thinning=thinning)
            gh_g, gJ_g = vmap(oracle.suff_stats)(X_gibbs)
            mu_g = flatten_stats(jnp.mean(gh_g, axis=0), jnp.mean(gJ_g, axis=0))
            grad_gibbs = grad_from_mu(gh_obs, gJ_obs, mu_g)
            e_gibbs.append(float(jnp.linalg.norm(grad_gibbs - grad_true) / norm_true))

            # MH
            X_mh = metropolis_hastings_sample(oracle, km, n=n, burnin=burnin, thinning=thinning)
            gh_m, gJ_m = vmap(oracle.suff_stats)(X_mh)
            mu_m = flatten_stats(jnp.mean(gh_m, axis=0), jnp.mean(gJ_m, axis=0))
            grad_mh = grad_from_mu(gh_obs, gJ_obs, mu_m)
            e_mh.append(float(jnp.linalg.norm(grad_mh - grad_true) / norm_true))

            # Stein-BQ (design = Gibbs samples)
            F = vmap(lambda x: flatten_stats(*oracle.suff_stats(x)))(X_gibbs)  # (n,p)
            mu_bq, info = tune_and_estimate_stein_bq(oracle, X_gibbs, F)
            grad_bq = grad_from_mu(gh_obs, gJ_obs, mu_bq)
            e_bq.append(float(jnp.linalg.norm(grad_bq - grad_true) / norm_true))

        errs_gibbs.append((np.mean(e_gibbs), np.std(e_gibbs)))
        errs_mh.append((np.mean(e_mh), np.std(e_mh)))
        errs_bq.append((np.mean(e_bq), np.std(e_bq)))

        print(f"n={n:4d} | Gibbs: {np.mean(e_gibbs):.3e} ± {np.std(e_gibbs):.3e} | "
              f"MH: {np.mean(e_mh):.3e} ± {np.std(e_mh):.3e} | "
              f"BQ: {np.mean(e_bq):.3e} ± {np.std(e_bq):.3e}")

    # plot
    ns_np = np.array(ns, dtype=float)
    def mean_std(arr): return np.array([m for (m, s) in arr]), np.array([s for (m, s) in arr])
    mg, sg = mean_std(errs_gibbs)
    mm, sm = mean_std(errs_mh)
    mb, sb = mean_std(errs_bq)

    plt.figure(figsize=(8,5))
    plt.loglog(ns_np, mg, '-o', label='Gibbs MC')
    plt.fill_between(ns_np, np.maximum(1e-12, mg - 1.96*sg), mg + 1.96*sg, alpha=0.15)
    plt.loglog(ns_np, mm, '-s', label='MH MC')
    plt.fill_between(ns_np, np.maximum(1e-12, mm - 1.96*sm), mm + 1.96*sm, alpha=0.15)
    plt.loglog(ns_np, mb, '-D', label='Stein-BQ (KDSD)')
    plt.fill_between(ns_np, np.maximum(1e-12, mb - 1.96*sb), mb + 1.96*sb, alpha=0.15)
    plt.xlabel('n (samples / design size)')
    plt.ylabel('Relative ℓ₂ error ‖∇θℓ̂ - ∇θℓ‖ / ‖∇θℓ‖')
    plt.title(f'Potts (L={L}, q={q}, β={beta})')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------

def main():
    # quick single-run sanity check (optional)
    L, q, beta = 4, 3, 1.0/2.269
    run_sweep_and_plot(L=L, q=q, beta=beta, seeds=10,
                       ns=(25, 50, 100, 200, 400),
                       burnin=500, thinning=5)

if __name__ == "__main__":
    main()