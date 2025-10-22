# potts_cd_raw_to_eff.py
import os
os.environ["JAX_ENABLE_X64"] = "1"   # set before importing jax

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, lax, jit
import jax.scipy.linalg as sla
from scipy.optimize import minimize_scalar

# ---------------------------
# True data generator (i.i.d. categorical)
# ---------------------------
def generate_categorical_sequence(key, p, d, q=3):
    assert q == 3, "this toy generator uses [p, p, 1-2p] for q=3."
    probs  = jnp.array([p, p, 1 - 2*p], dtype=jnp.float64)
    keys   = jrand.split(key, d)
    seq    = vmap(lambda k: jrand.choice(k, q, p=probs))(keys)
    return seq.astype(jnp.int32)

def sample_true_sequences(key, p, d, n_samples, q=3):
    keys = jrand.split(key, n_samples)
    return vmap(lambda k: generate_categorical_sequence(k, p, d, q))(keys)  # (n_samples, d)

# ---------------------------
# Structure mask: 1-D chain on d sites (edges i–i+1)
# ---------------------------
def chain_mask(d: int) -> jnp.ndarray:
    M = jnp.zeros((d, d, 1, 1), dtype=jnp.float64)
    idx = jnp.arange(d - 1)
    M = M.at[idx,   idx+1, 0, 0].set(1.0)
    M = M.at[idx+1, idx,   0, 0].set(1.0)
    return M  # diagonal implicitly 0

def make_J_eff(J_raw: jnp.ndarray, M: jnp.ndarray,
               symmetrize: bool = True, zero_diag: bool = True) -> jnp.ndarray:
    J = 0.5 * (J_raw + jnp.transpose(J_raw, (1, 0, 3, 2))) if symmetrize else J_raw
    J = J * M
    if zero_diag:
        d = J.shape[0]
        J = J.at[jnp.arange(d), jnp.arange(d), :, :].set(0.0)
    return J

# ---------------------------
# Potts energy (labels path, used for training)
# ---------------------------
def energy_param(h: jnp.ndarray, J_raw: jnp.ndarray, q: int, x: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    J_eff = make_J_eff(J_raw, M)
    x_oh  = jax.nn.one_hot(x, q, dtype=jnp.float64)  # (d,q)
    e_h   = jnp.einsum('iq,iq->', x_oh, h)
    e_J   = 0.5 * jnp.einsum('iq,ijab,jb->', x_oh, J_eff, x_oh)
    return e_h + e_J

def beta_times_energy(h, J_raw, q, beta, x, M):
    return beta * energy_param(h, J_raw, q, x, M)

# β·∂E/∂θ at a single x (θ = (h, J_raw)); autodiff flows through make_J_eff
def grad_U_theta(h, J_raw, q, beta, x, M):
    def U(h_, J_):  # scalar potential
        return beta_times_energy(h_, J_, q, beta, x, M)
    return jax.grad(U, argnums=(0, 1))(h, J_raw)  # returns (∂U/∂h, ∂U/∂J_raw)

def mean_grad_U_over(h, J_raw, q, beta, X, M):
    gh, gJ = vmap(lambda x: grad_U_theta(h, J_raw, q, beta, x, M))(X)
    return jnp.mean(gh, axis=0), jnp.mean(gJ, axis=0)

# ---------------------------
# Gibbs sampler (uses a *fixed* J_eff for this step)
# ---------------------------
def cond_probs_fixed(h, J_eff, q, beta, x, i):
    base = h[i]                  # (q,)
    Ji   = J_eff[i]              # (d,q,q)
    xoh  = jax.nn.one_hot(x, q, dtype=jnp.float64)
    xoh  = xoh.at[i].set(0.0)
    add  = jnp.einsum('jkq,jq->k', Ji, xoh)   # (q,)
    logits = -beta * (base + add)
    return jax.nn.softmax(logits)

def gibbs_sweep_scan(h, J_eff, q, beta, key, x):
    """A single Gibbs sweep using jax.lax.scan for robustness."""
    d = x.shape[0]
    keys = jrand.split(key, d)
    
    def body(x_carry, i):
        key_i = keys[i]
        probs = cond_probs_fixed(h, J_eff, q, beta, x_carry, i)
        # Use jnp.log for numerical stability with categorical
        xi = jrand.categorical(key_i, jnp.log(probs))
        x_new = x_carry.at[i].set(xi)
        return x_new, None # new state, no output per step
        
    x_final, _ = lax.scan(body, x, jnp.arange(d))
    return x_final

def gibbs_k_steps_scan(h, J_eff, q, beta, key, x0, k_steps):
    """Runs k Gibbs sweeps, also using scan."""
    keys = jrand.split(key, k_steps)
    
    def body(x_carry, i):
        key_i = keys[i]
        x_new = gibbs_sweep_scan(h, J_eff, q, beta, key_i, x_carry)
        return x_new, None
        
    x_final, _ = lax.scan(body, x0, jnp.arange(k_steps))
    return x_final

# ---------------------------
# MC gradient of NLL (descent):  ∇NLL = (neg - pos) + reg
# ---------------------------
def grad_nll_mc(h, J_raw, q, beta, X_data, X_model, M, lambda_h=0.0, lambda_J=0.0):
    gh_pos, gJ_pos = mean_grad_U_over(h, J_raw, q, beta, X_data,  M)
    gh_neg, gJ_neg = mean_grad_U_over(h, J_raw, q, beta, X_model, M)
    grad_h = (gh_neg - gh_pos) + lambda_h * h
    grad_J = (gJ_neg - gJ_pos) + lambda_J * J_raw
    grad_J = grad_J * M
    return grad_h, grad_J

# ===========================================================
# Stein KSD kernel on integer labels (two-sided Stein kernel)
# (This is used for Stein-kernel BQ rank-1 c tuning.)
# ===========================================================
def rbf_kernel_int(x, y, gamma):
    # Hamming RBF on labels
    h = jnp.sum(x != y)
    return jnp.exp(-2.0 * gamma * h)

@partial(jit, static_argnames=('q',))
def energy_params_for_kernel(h, J_eff, q, beta, x):
    # Scores for the Stein kernel should use the *effective* couplings (prob model)
    x_oh = jax.nn.one_hot(x, q, dtype=jnp.float64)
    e_h = jnp.einsum('iq,iq->', x_oh, h)
    e_J = 0.5 * jnp.einsum('iq,ijab,jb->', x_oh, J_eff, x_oh)
    return e_h + e_J

@jit
def neighbors_cyclic(x, q, shift):
    d = x.shape[0]
    x_shift = (x + shift) % q
    X = jnp.tile(x, (d, 1))
    return X.at[jnp.arange(d), jnp.arange(d)].set(x_shift)

@partial(jit, static_argnames=('q',))
def cyclic_score_params_eff(h, J_eff, q, beta, x):
    nbh = neighbors_cyclic(x, q, shift=+1)
    Ex  = energy_params_for_kernel(h, J_eff, q, beta, x)
    Enb = vmap(lambda z: energy_params_for_kernel(h, J_eff, q, beta, z))(nbh)
    return 1.0 - jnp.exp(-beta * (Enb - Ex))   # (d,)

@partial(jit, static_argnames=('q',))
def stein_kernel_pair_params_eff(h, J_eff, q, beta, x, y, gamma: float):
    d = x.shape[0]
    sx = cyclic_score_params_eff(h, J_eff, q, beta, x)   # (d,)
    sy = cyclic_score_params_eff(h, J_eff, q, beta, y)   # (d,)
    k_xy = rbf_kernel_int(x, y, gamma)

    x_inv = neighbors_cyclic(x, q, shift=-1)
    y_inv = neighbors_cyclic(y, q, shift=-1)

    k_x_yinv    = vmap(lambda yr: rbf_kernel_int(x,  yr, gamma))(y_inv)
    k_xinv_y    = vmap(lambda xr: rbf_kernel_int(xr, y,  gamma))(x_inv)
    K_xinv_yinv = vmap(lambda xr: vmap(lambda yr: rbf_kernel_int(xr, yr, gamma))(y_inv))(x_inv)

    trace_term = jnp.trace(K_xinv_yinv) - jnp.sum(k_x_yinv) - jnp.sum(k_xinv_y) + d * k_xy
    t1 = jnp.dot(sx, sy) * k_xy
    t2 = - jnp.dot(sx, (k_xy - k_x_yinv))
    t3 = - jnp.dot((k_xy - k_xinv_y), sy)
    return t1 + t2 + t3 + trace_term

def stein_kernel_matrix_params(h, J_eff, q, beta, X, gamma: float):
    pair = lambda xi, xj: stein_kernel_pair_params_eff(h, J_eff, q, beta, xi, xj, gamma)
    return vmap(vmap(pair, in_axes=(None, 0)), in_axes=(0, None))(X, X)   # (n,n)

def median_gamma_on_int(X, q, n_subset=500, key=None):
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
    return float(1.0 / jnp.clip(med_sq, 1e-12, None))

# -----------------------------------------------------------
# Rank-1 exact tuning of c (Sherman–Morrison) with fixed γ
# (Your original function; kept intact.)
# -----------------------------------------------------------
def tune_c_and_mu_given_gamma(h, J_eff, q, beta, X, fvals, gamma, jitter=1e-8):
    """Exact rank-1 optimization over c given fixed gamma; returns (mu_hat, c_opt)."""
    n, p = fvals.shape
    # Build Ks(γ) once
    Ks = stein_kernel_matrix_params(h, J_eff, q, beta, X, gamma)
    Ks = Ks + jitter * jnp.eye(n)
    Ls, lower = sla.cho_factor(Ks, lower=True)
    solve = lambda B: sla.cho_solve((Ls, lower), B)

    ones = jnp.ones((n, 1), dtype=jnp.float64)
    v    = solve(ones)                     # Ks^{-1} 1
    t    = float((ones.T @ v)[0, 0])       # 1^T Ks^{-1} 1
    A    = solve(fvals)                     # Ks^{-1} F   (n,p)
    uTA  = (ones.T @ A).reshape(1, p)      # 1^T Ks^{-1} F  (1,p)
    logdet_Ks = 2.0 * jnp.sum(jnp.log(jnp.diag(Ls)))

    # Define exact NLL(c) via Sherman–Morrison; optimize in log-space for stability
    def nll_logc(logc):
        c = jnp.exp(logc)
        denom = 1.0 + c * t
        KinvF = A - (c / denom) * (v @ uTA)         # (n,p)
        trace_term = jnp.sum(fvals * KinvF)          # tr(F^T K^{-1} F)
        logdetK = logdet_Ks + jnp.log(denom)
        return 0.5 * trace_term + 0.5 * p * logdetK + 0.5 * n * p * jnp.log(2 * jnp.pi)

    # Wrap for SciPy (float in/out)
    def fun_np(ell): return float(nll_logc(jnp.array(ell)))
    res = minimize_scalar(fun_np, bounds=(-20.0, 20.0), method='bounded')
    logc_opt = res.x
    c_opt = float(np.exp(logc_opt))

    denom = 1.0 + c_opt * t
    mu_hat = (c_opt / denom) * uTA.ravel()         # μ̂ = (c/denom) * (1^T Ks^{-1} F)
    return mu_hat, c_opt

# -----------------------------------------------------------
# Build F matrix from RAW parameter grads (targets for BQ)
# -----------------------------------------------------------
def build_Fmat_beta_grad_raw(h, J_raw, q, beta, X_design, J_mask):
    """
    Rows: f(x_i) = β ∇_{(h,J_raw)} E(x_i).
    Uses autodiff through make_J_eff so updates match RAW parameterization.
    Returns Fmat: (n, p) with p = d*q + d*d*q*q
    """
    gh_all, gJ_all = vmap(lambda x: grad_U_theta(h, J_raw, q, beta, x, J_mask))(X_design)
    d = h.shape[0]
    def row(i):
        gh_i = gh_all[i].reshape(-1)          # (d*q,)
        gJ_i = gJ_all[i].reshape(-1)          # (d*d*q*q,)
        return jnp.concatenate([gh_i, gJ_i], axis=0)
    return vmap(row)(jnp.arange(X_design.shape[0]))

def unpack_mu_hat(mu_hat, d, q):
    n_h = d * q
    gh_neg = mu_hat[:n_h].reshape(d, q)
    gJ_neg = mu_hat[n_h:].reshape(d, d, q, q)
    return gh_neg, gJ_neg

# =========================
# Exact logZ and dataset NLL
# =========================
def _idx_to_states_np(start, end, q, d):
    """Vectorized base-q unranking: [start,end) -> integer states of shape (n,d)."""
    import numpy as _np
    idx = _np.arange(start, end, dtype=_np.int64)
    out = _np.empty((idx.size, d), dtype=_np.int32)
    x = idx.copy()
    for k in range(d - 1, -1, -1):
        out[:, k] = (x % q).astype(_np.int32)
        x //= q
    return out

def exact_logZ_batched(h, J_raw, M, beta, q, d, batch_size=200_000):
    """
    Exact log-partition via enumeration; fine for small q^d (e.g., 3^4=81).
    Uses energy_param (which builds J_eff from J_raw & M).
    """
    import numpy as _np
    total = q ** d
    Mmax = -_np.inf
    sum_shift = 0.0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        X = jnp.array(_idx_to_states_np(start, end, q, d), dtype=jnp.int32)   # (n,d)
        E = vmap(lambda x: energy_param(h, J_raw, q, x, M))(X)                # (n,)
        logw = -beta * E
        m = float(jnp.max(logw))
        if m > Mmax:
            # rescale running sum for log-sum-exp stability
            sum_shift *= _np.exp(Mmax - m)
            Mmax = m
        sum_shift += float(jnp.sum(jnp.exp(logw - Mmax)))
    return Mmax + _np.log(sum_shift)

def nll_dataset(h, J_raw, M, beta, q, d, X_data, logZ=None):
    """
    Mean negative log-likelihood on X_data under current params.
    """
    if logZ is None:
        logZ = exact_logZ_batched(h, J_raw, M, beta, q, d)
    Es = vmap(lambda x: energy_param(h, J_raw, q, x, M))(X_data)              # (N,)
    logp = -beta * Es - logZ
    nll  = -float(jnp.mean(logp))
    return nll, float(logZ)

# ---------------------------
# Training loop (PCD-k) with optional Stein–kernel BQ negative phase
# ---------------------------
def train_potts_cd_raw_to_eff(
    seed=0,
    d=4, q=3, beta=1.0/2.269,
    steps=300, learning_rate=1e-2,
    cd_k=1, num_chains=400,
    x_data=None,                 # (N,d)
    lambda_h=1e-4, lambda_J=1e-3,
    use_bq=False,                # <<< turn on Stein-kernel BQ negative phase
    design_n=400):

    key = jrand.PRNGKey(seed)
    key, k_h, k_J, k_chains, k_gamma = jrand.split(key, 5)

    # parameters
    h     = 0.01  * jrand.normal(k_h, (d, q))
    J_raw = 0.01 * jrand.normal(k_J, (d, d, q, q))

    # structure mask & chains
    M = chain_mask(d)  # (d,d,1,1)
    X_chains = jrand.randint(k_chains, (num_chains, d), 0, q)

    # data (positive phase)
    if x_data is None:
        raise ValueError("x_data must be provided for positive phase in this script.")
    X_data = x_data

    bq_gamma = None

    for t in range(steps):
        # 1) Build J_eff once for this step (used by sampler and Stein kernel scores)
        J_eff = make_J_eff(J_raw, M)

        # 2) Roll persistent chains with *fixed* J_eff (we'll reuse for design &/or MC)
        key, k_neg = jrand.split(key)
        keys = jrand.split(k_neg, num_chains)
        X_chains = vmap(lambda kk, x0: gibbs_k_steps_scan(h, J_eff, q, beta, kk, x0, cd_k))(keys, X_chains)

        # 3) Positive phase from data
        gh_pos, gJ_pos = mean_grad_U_over(h, J_raw, q, beta, X_data,  M)

        # 4) Negative phase
        if use_bq:
            # Build/refresh design from chains (fast)
            X_design = X_chains[:design_n]
            # gamma: median heuristic on one-hot labels (once or occasionally)
            bq_gamma = median_gamma_on_int(X_design, q, key=k_gamma)

            # Ks with EFFECTIVE params; F with RAW grads
            Ks = stein_kernel_matrix_params(h, J_eff, q, beta, X_design, bq_gamma)
            Fmat = build_Fmat_beta_grad_raw(h, J_raw, q, beta, X_design, M)
            mu_hat, c_opt = tune_c_and_mu_given_gamma(h, J_eff, q, beta, X_design, Fmat, bq_gamma)
            gh_neg, gJ_neg = unpack_mu_hat(mu_hat, d, q)
        else:
            # Standard MC/PCD negative phase
            gh_neg, gJ_neg = mean_grad_U_over(h, J_raw, q, beta, X_chains, M)

        # 5) Gradients of NLL wrt RAW params
        grad_h = -(gh_neg - gh_pos) + lambda_h * h
        grad_J = -(gJ_neg - gJ_pos) + lambda_J * J_raw
        grad_J = grad_J * M  # keep on-mask

        # 6) Update
        h     = h     - learning_rate * grad_h
        J_raw = J_raw - learning_rate * grad_J

        # 7) Gauge: per-site zero-mean over states
        # h = h - h.mean(axis=1, keepdims=True)

        # --- MODIFIED LOGGING BLOCK ---
        if (t % 100) == 0 or (t == steps - 1):
            # Calculate NLL using the current parameters
            # WARNING: This is slow for large d as it computes the exact logZ
            nll, logZ = nll_dataset(h, J_raw, M, beta, q, d, x_data)
            
            J_eff_log = make_J_eff(J_raw, M)
            neg_name = "SteinBQ" if use_bq else "CD"
            
            # Add nll to the print statement
            print(f"step {t:4d} | NLL={nll:.4f} | logZ={logZ:.4f} | "
                  f"||h||={float(jnp.linalg.norm(h)):.4f} | "
                  f"||J||={float(jnp.linalg.norm(J_eff_log)):.4f} | [neg={neg_name}]")

    return h, J_raw, M, make_J_eff(J_raw, M)
   

# ---------------------------
# KSD utilities (ONE-HOT path) — optional eval
# ---------------------------
def median_heuristic_gamma(samples, n_subset=500, key=None):
    n_total = samples.shape[0]
    if (key is not None) and (n_total > n_subset):
        idx = jax.random.choice(key, n_total, shape=(n_subset,), replace=False)
        samples_sub = samples[idx]
    else:
        samples_sub = samples if n_total <= n_subset else samples[:n_subset]
    n = samples_sub.shape[0]
    flat = samples_sub.reshape(n, -1)
    norms = jnp.sum(flat**2, axis=1)
    d2 = norms[:, None] - 2.0 * (flat @ flat.T) + norms[None, :]
    tri = jnp.triu_indices(n, 1)
    med_sq = jnp.median(d2[tri])
    return 1.0 / (2.0 * jnp.clip(med_sq, 1e-8, None))

@partial(jit, static_argnames=['shift'])
def get_neighbor(x_i, shift):
    return jnp.roll(x_i, shift=shift, axis=-1)

@partial(jit, static_argnames=['shift'])
def get_all_neighbors(x, shift):
    L, q = x.shape
    eye = jnp.eye(L, dtype=x.dtype)[:, :, None]
    x_perm_rows = vmap(get_neighbor, in_axes=(0, None))(x, shift)
    x_tiled = jnp.tile(x, (L, 1, 1))
    x_perm_tiled = jnp.tile(x_perm_rows, (L, 1, 1))
    return x_tiled * (1 - eye) + x_perm_tiled * eye

@jit
def energy_onehot(x_oh: jnp.ndarray, h: jnp.ndarray,
                  J_raw: jnp.ndarray, J_mask: jnp.ndarray) -> jnp.ndarray:
    J_eff = make_J_eff(J_raw, J_mask)
    e_h   = jnp.einsum('iq,iq->', x_oh, h)
    e_J   = 0.5 * jnp.einsum('iq,ijab,jb->', x_oh, J_eff, x_oh)
    return e_h + e_J

@jit
def discrete_model_score(x, h, J, J_mask, beta):
    nbrs = get_all_neighbors(x, shift=1)
    E_x  = energy_onehot(x, h, J, J_mask)
    E_n  = vmap(energy_onehot, in_axes=(0, None, None, None))(nbrs, h, J, J_mask)
    return 1.0 - jnp.exp(-beta * (E_n - E_x))

@jit
def base_rbf_kernel(x_flat, y_flat, gamma):
    return jnp.exp(-gamma * jnp.sum((x_flat - y_flat) ** 2))

def discrete_ksd_u_p_term(x, y, h, J, J_mask, beta, gamma):
    L, q = x.shape
    sx = discrete_model_score(x, h, J, J_mask, beta)
    sy = discrete_model_score(y, h, J, J_mask, beta)
    xn = get_all_neighbors(x, shift=-1)
    yn = get_all_neighbors(y, shift=-1)
    xf, yf = x.ravel(), y.ravel()
    vflat = vmap(jnp.ravel)
    xnf, ynf = vflat(xn), vflat(yn)
    k_xy     = base_rbf_kernel(xf, yf, gamma)
    k_x_y_n  = vmap(base_rbf_kernel, in_axes=(None, 0, None))(xf, ynf, gamma)
    k_x_n_y  = vmap(base_rbf_kernel, in_axes=(0, None, None))(xnf, yf, gamma)
    k_nn     = vmap(vmap(base_rbf_kernel, in_axes=(0, None, None)), in_axes=(None, 0, None))(xnf, ynf, gamma)
    trace_term = (jnp.trace(k_nn) - jnp.sum(k_x_y_n) - jnp.sum(k_x_n_y) + L * k_xy)
    return (
        jnp.dot(sx, sy) * k_xy
        - jnp.dot(sx, (k_xy - k_x_y_n))
        - jnp.dot((k_xy - k_x_n_y), sy)
        + trace_term
    )

@partial(jit, static_argnames=['n_samples'])
def compute_discrete_ksd(samples, h, J, J_mask, beta, gamma, n_samples=None):
    if n_samples is None:
        n_samples = samples.shape[0]
    u = vmap(
        vmap(discrete_ksd_u_p_term, in_axes=(None, 0, None, None, None, None, None)),
        in_axes=(0, None, None, None, None, None, None)
    )
    M = u(samples[:n_samples], samples[:n_samples], h, J, J_mask, beta, gamma)
    total = jnp.sum(M) - jnp.sum(jnp.diag(M))
    return total / (n_samples * (n_samples - 1))

# ---------------------------
# Demo
# ---------------------------
if __name__ == "__main__":
    SEED = 0
    d, q = 4, 3          # try d=4 or 8; this is 1-D chain with d sites
    beta = 1.0 / 2.269
    N    = 2048
    TRUE_P = 0.2

    key = jrand.PRNGKey(SEED)
    X_data = sample_true_sequences(key, p=TRUE_P, d=d, n_samples=N, q=q)

    # === Train with Stein-kernel BQ negative phase ===
    h_bq, Jraw_bq, M, Jeff_bq = train_potts_cd_raw_to_eff(
        seed=SEED, d=d, q=q, beta=beta,
        steps=800, learning_rate=3e-2,
        cd_k=100, num_chains=10,
        x_data=X_data,
        lambda_h=0, lambda_J=0,
        use_bq=False,            # <<< turn on Stein-kernel BQ
        design_n=50
    )

    # Sanity
    offmask_norm = float(jnp.linalg.norm(Jeff_bq * (1.0 - M)))
    print("off-mask ||J_eff|| (should be ~0):", offmask_norm)
    print("shapes: h", h_bq.shape, "J_raw", Jraw_bq.shape, "J_eff", Jeff_bq.shape)

    # Optional: KSD on observed TRUE data (one-hot) with learned params
    X_data_oh = jax.nn.one_hot(X_data, num_classes=q, dtype=jnp.float64)  # (N,d,q)
    gamma_ksd = float(median_heuristic_gamma(X_data_oh))
    ksd2 = compute_discrete_ksd(
        X_data_oh, h_bq, Jraw_bq, M, beta, gamma_ksd,
        n_samples=min(256, X_data_oh.shape[0])
    )
    print(f"KSD^2 ≈ {float(ksd2):.6e}")

    nll, logZ = nll_dataset(h_bq, Jraw_bq, M, beta, q, d, X_data)
    print(f"exact logZ = {logZ:.6f}   NLL(data) = {nll:.6f}")
