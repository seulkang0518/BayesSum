import optax
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import jax.random as jrand
import jax.scipy.linalg as sla
import matplotlib.pyplot as plt

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

jax.config.update("jax_enable_x64", True)

# -----------------
# (Optional) Discrete KSD evaluation
# -----------------
def median_heuristic_lambda_hamming(samples_onehot, n_subset=500, key=None):
    X = samples_onehot
    n_total, L, q = X.shape
    if (key is not None) and (n_total > n_subset):
        idx = jax.random.choice(key, n_total, shape=(n_subset,), replace=False)
        X = X[idx]
    n = X.shape[0]
    Dmatch = jnp.einsum('ilq,jlq->ij', X, X)  # #matches
    H = L - Dmatch                            # Hamming distance (unnormalized)
    tri = jnp.triu_indices(n, 1)
    med = jnp.median(H[tri])
    return jnp.log(2.0) / jnp.clip(med, 1e-8, None)

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
def discrete_model_score(x, h, J, J_mask, beta):
    nbrs = get_all_neighbors(x, shift=1)
    E_x = energy(x, h, J, J_mask)
    E_n = vmap(energy, in_axes=(0, None, None, None))(nbrs, h, J, J_mask)
    return 1.0 - jnp.exp(-beta * (E_n - E_x))

def base_exp_hamming_kernel_ksd(x_flat, y_flat, lam, L):
    # x_flat,y_flat are flattened one-hots of shape (L*q,)
    dot = jnp.dot(x_flat, y_flat)   # == #matches for one-hot per site
    return jnp.exp(-lam * L + lam * dot)

def discrete_ksd_u_p_term(x, y, h, J, J_mask, beta, lam):
    L, q = x.shape
    sx = discrete_model_score(x, h, J, J_mask, beta)
    sy = discrete_model_score(y, h, J, J_mask, beta)

    xn = get_all_neighbors(x, shift=-1)
    yn = get_all_neighbors(y, shift=-1)

    xf, yf = x.ravel(), y.ravel()
    vflat = vmap(jnp.ravel)
    xnf, ynf = vflat(xn), vflat(yn)

    k_xy    = base_exp_hamming_kernel_ksd(xf, yf, lam, L)
    k_x_y_n = vmap(base_exp_hamming_kernel_ksd, in_axes=(None, 0, None, None))(xf, ynf, lam, L)
    k_x_n_y = vmap(base_exp_hamming_kernel_ksd, in_axes=(0, None, None, None))(xnf, yf, lam, L)
    k_nn    = vmap(
                  vmap(base_exp_hamming_kernel_ksd, in_axes=(0, None, None, None)),
                  in_axes=(None, 0, None, None)
              )(xnf, ynf, lam, L)

    trace_term = (jnp.trace(k_nn) - jnp.sum(k_x_y_n) - jnp.sum(k_x_n_y) + L * k_xy)
    return jnp.dot(sx, sy) * k_xy - jnp.dot(sx, (k_xy - k_x_y_n)) - jnp.dot((k_xy - k_x_n_y), sy) + trace_term

@partial(jit, static_argnames=['n_samples'])
def compute_discrete_ksd(samples, h, J, J_mask, beta, lam, n_samples=None):
    if n_samples is None:
        n_samples = samples.shape[0]
    u = vmap(vmap(discrete_ksd_u_p_term, in_axes=(None, 0, None, None, None, None, None)),
             in_axes=(0, None, None, None, None, None, None))
    M = u(samples[:n_samples], samples[:n_samples], h, J, J_mask, beta, lam)
    total = jnp.sum(M) - jnp.sum(jnp.diag(M))
    return total / (n_samples * (n_samples - 1))

# -----------------
# Data generation
# -----------------
def generate_categorical_sequence(key, p, L, q):
    probs = jnp.array([p, p, 1 - 2 * p], dtype=jnp.float64)
    keys = jax.random.split(key, L)
    seq = [jax.random.choice(k, q, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)

def data_preprocess(p, q, L, num_sequences, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_sequences)
    seqs = jnp.stack([generate_categorical_sequence(k, p, L, q=q) for k in keys])  # (N,L)
    data_onehot = jax.nn.one_hot(seqs, num_classes=q, dtype=jnp.float64)          # (N,L,q)
    return seqs, data_onehot

def _ints_to_base_q(ints: jnp.ndarray, q: int, L: int) -> jnp.ndarray:
    def one_num_to_digits(n):
        digs = []
        x = n
        for _ in range(L):
            digs.append(x % q)
            x = x // q
        return jnp.array(digs[::-1], dtype=jnp.int32)  # length-L
    return vmap(one_num_to_digits)(ints)

def logZ_exact_enumeration(h, J, J_mask, beta, L, q, batch = 200_000):
    total = int(q ** L)
    logZ = -jnp.inf  # running log-sum-exp accumulator
    for start in range(0, total, batch):
        m = min(batch, total - start)
        idx = jnp.arange(start, start + m, dtype=jnp.int64)        # (m,)
        digits = _ints_to_base_q(idx, q, L)                        # (m, L)
        X_chunk = jax.nn.one_hot(digits, num_classes=q, dtype=jnp.float64)  # (m, L, q)

        E = vmap(energy, in_axes=(0, None, None, None))(X_chunk, h, J, J_mask)  # (m,)
        logf = -beta * E

        mshift = jnp.max(logf)
        chunk_lse = mshift + jnp.log(jnp.clip(jnp.sum(jnp.exp(logf - mshift)), 1e-300))
        logZ = jnp.logaddexp(logZ, chunk_lse)
    return float(logZ)

def evaluate_logZ_nll_exact(params, X, beta, J_mask):
    h, J = params
    L, q = X.shape[1], X.shape[2]
    E_data = vmap(energy, in_axes=(0, None, None, None))(X, h, J, J_mask)
    data_term = jnp.mean(beta * E_data)
    logZ = logZ_exact_enumeration(h, J, J_mask, beta, L, q)
    return float(logZ), float(data_term + logZ)

# -----------------
# Mask utilities
# -----------------
def apply_update(h, J, gh, gJ, opt_state, J_mask, tx):
    updates, opt_state = tx.update((gh, gJ), opt_state, params=(h, J))
    h, J = optax.apply_updates((h, J), updates)
    J = project_J(J, J_mask)
    return h, J, opt_state

def adjacency_mask(L):
    M = jnp.zeros((L, L), dtype=jnp.float64)
    idx = jnp.arange(L - 1)
    M = M.at[idx, idx + 1].set(1.0)
    M = M.at[idx + 1, idx].set(1.0)
    return M[:, :, None, None]

@jit
def project_J(J, J_mask):
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    return Jsym * J_mask

# ------------------------
# Energy & Energy Gradient
# ------------------------
@jit
def energy(x, h, J, J_mask):
    Jeff = project_J(J, J_mask)
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    return -(e_J + e_h)   # p(x) ∝ exp(-β U(x)) with U = -(h+J) ⇒ p ∝ exp(β(h+J))

@jit
def grad_E_theta(x, h, J, J_mask):
    def E_params(h_, J_):
        return energy(x, h_, J_, J_mask)
    return jax.grad(E_params, argnums=(0, 1))(h, J)

@jit
def mean_grad_U_over(X, h, J, J_mask):
    dEh, dEJ = vmap(lambda x: grad_E_theta(x, h, J, J_mask))(X)
    return jnp.mean(dEh, axis=0), jnp.mean(dEJ, axis=0)

# ------------------------
# Helpers for polynomial base (flattened one-hot)
# ------------------------
@partial(jit, static_argnames=['q'])
def onehot_flat_from_int(x_int, q):
    return jax.nn.one_hot(x_int, q, dtype=jnp.float64).reshape(-1)

@jit
def poly2_kernel_flat(xf, yf):
    return (jnp.dot(xf, yf) + 1.0) ** 2

def base_exp_hamming_kernel(xf, yf, d):
    dot = jnp.dot(xf, yf)
    return jnp.exp(-0.5 * d + 0.5 * dot)

# ------------------------
# Stein BQ (model-side moment estimate) with polynomial base
# ------------------------
@jit
def energy_int(x_int, h, J, J_mask):
    q = h.shape[1]
    xoh = jax.nn.one_hot(x_int, q, dtype=jnp.float64)
    return energy(xoh, h, J, J_mask)

@partial(jit, static_argnames=['q', 'shift'])
def neighbors_cyclic(x, q, shift):
    d = x.shape[0]
    x_shift = (x + shift) % q
    X = jnp.tile(x, (d, 1))
    return X.at[jnp.arange(d), jnp.arange(d)].set(x_shift)

@jit
def cyclic_score_params(h, J, J_mask, beta, x):
    q = h.shape[1]
    nbh = neighbors_cyclic(x, q, shift=+1)
    Ex  = energy_int(x, h, J, J_mask)
    Enb = vmap(lambda z: energy_int(z, h, J, J_mask))(nbh)
    return 1.0 - jnp.exp(-beta * (Enb - Ex))   # (d,)

@jit
def stein_kernel_pair_params(h, J, J_mask, beta, x_int, y_int, _unused=None):
    d = x_int.shape[0]
    q = h.shape[1]

    sx = cyclic_score_params(h, J, J_mask, beta, x_int)  # (d,)
    sy = cyclic_score_params(h, J, J_mask, beta, y_int)  # (d,)

    xf = onehot_flat_from_int(x_int, q)  # (d*q,)
    yf = onehot_flat_from_int(y_int, q)  # (d*q,)
    k_xy = base_exp_hamming_kernel(xf, yf, d)

    x_inv = neighbors_cyclic(x_int, q, shift=-1)
    y_inv = neighbors_cyclic(y_int, q, shift=-1)
    xinv_flat = vmap(lambda xx: onehot_flat_from_int(xx, q))(x_inv)
    yinv_flat = vmap(lambda yy: onehot_flat_from_int(yy, q))(y_inv)

    k_x_yinv    = vmap(lambda yf_: base_exp_hamming_kernel(xf, yf_, d))(yinv_flat)
    k_xinv_y    = vmap(lambda xf_: base_exp_hamming_kernel(xf_, yf, d))(xinv_flat)
    K_xinv_yinv = vmap(lambda xf_: vmap(lambda yf_: base_exp_hamming_kernel(xf_, yf_, d))(yinv_flat))(xinv_flat)

    trace_term = jnp.trace(K_xinv_yinv) - jnp.sum(k_x_yinv) - jnp.sum(k_xinv_y) + d * k_xy
    t1 = jnp.dot(sx, sy) * k_xy
    t2 = - jnp.dot(sx, (k_xy - k_x_yinv))
    t3 = - jnp.dot((k_xy - k_xinv_y), sy)
    return t1 + t2 + t3 + trace_term

# @jit
# def stein_kernel_pair_params(h, J, J_mask, beta, x_int, y_int, _unused=None):
#     d = x_int.shape[0]
#     q = h.shape[1]

#     sx = cyclic_score_params(h, J, J_mask, beta, x_int)  # (d,)
#     sy = cyclic_score_params(h, J, J_mask, beta, y_int)  # (d,)

#     xf = onehot_flat_from_int(x_int, q)  # (d*q,)
#     yf = onehot_flat_from_int(y_int, q)  # (d*q,)
#     k_xy = poly2_kernel_flat(xf, yf)

#     x_inv = neighbors_cyclic(x_int, q, shift=-1)
#     y_inv = neighbors_cyclic(y_int, q, shift=-1)
#     xinv_flat = vmap(lambda xx: onehot_flat_from_int(xx, q))(x_inv)
#     yinv_flat = vmap(lambda yy: onehot_flat_from_int(yy, q))(y_inv)

#     k_x_yinv    = vmap(lambda yf_: poly2_kernel_flat(xf, yf_))(yinv_flat)
#     k_xinv_y    = vmap(lambda xf_: poly2_kernel_flat(xf_, yf))(xinv_flat)
#     K_xinv_yinv = vmap(lambda xf_: vmap(lambda yf_: poly2_kernel_flat(xf_, yf_))(yinv_flat))(xinv_flat)

#     trace_term = jnp.trace(K_xinv_yinv) - jnp.sum(k_x_yinv) - jnp.sum(k_xinv_y) + d * k_xy
#     t1 = jnp.dot(sx, sy) * k_xy
#     t2 = - jnp.dot(sx, (k_xy - k_x_yinv))
#     t3 = - jnp.dot((k_xy - k_xinv_y), sy)
#     return t1 + t2 + t3 + trace_term

def stein_kernel_matrix_params(h, J, J_mask, beta, X_int, _unused=None):
    pair = lambda xi, xj: stein_kernel_pair_params(h, J, J_mask, beta, xi, xj, None)
    return vmap(vmap(pair, in_axes=(None, 0)), in_axes=(0, None))(X_int, X_int)

def stein_bq_poly2_mu(h, J, J_mask, beta, X_int, Fmat, ridge=1e-3, jitter=1e-8):
    Ks = stein_kernel_matrix_params(h, J, J_mask, beta, X_int, None)
    n = Ks.shape[0]
    Ksr = Ks + (ridge + jitter) * jnp.eye(n)

    Ls, lower = sla.cho_factor(Ksr, lower=True)
    solve = lambda B: sla.cho_solve((Ls, lower), B)

    ones = jnp.ones((n, 1), dtype=jnp.float64)
    v = solve(ones)                          # K^{-1} 1
    denom = (ones.T @ v)[0, 0] + 1e-12
    w = (v / denom).reshape(n)               # (n,)

    mu_hat = w @ Fmat                        # (p,)
    return mu_hat

def build_Fmat_beta_grad_raw(h, J, J_mask, X_bq_onehot):
    gh_all, gJ_all = vmap(lambda x: grad_E_theta(x, h, J, J_mask))(X_bq_onehot)
    d, q = h.shape
    def row(i):
        gh_i = gh_all[i].reshape(-1)
        gJ_i = gJ_all[i].reshape(-1)
        return jnp.concatenate([gh_i, gJ_i], axis=0)
    return vmap(row)(jnp.arange(X_bq_onehot.shape[0]))

def unpack_mu_hat(mu_hat, d, q):
    n_h = d * q
    dEh_bq = mu_hat[:n_h].reshape(d, q)
    dEJ_bq = mu_hat[n_h:].reshape(d, d, q, q)
    return dEh_bq, dEJ_bq

# ------------------------
# Conditional logits & NPLL (early stopping metric)
# ------------------------
def cond_logits_all(h, J, beta, x_int):
    d, q = h.shape
    xoh = jax.nn.one_hot(x_int, q, dtype=jnp.float64)  # (d, q)

    def one_site(_, i):
        xoh_wo = xoh.at[i].set(0.0)
        Ji = J[i]                             # (d, q, q)
        Ji_kjl = jnp.transpose(Ji, (1, 0, 2)).reshape(q, d*q)   # (q, d*q)
        msg_i  = Ji_kjl @ xoh_wo.reshape(d*q)                   # (q,)
        logits_i = beta * (h[i] + msg_i)
        return None, logits_i

    _, logits = lax.scan(one_site, None, jnp.arange(d))
    return logits  # (d, q)

@jit
def npll_batch(h, J, beta, X_int):
    def npll_one(x_int):
        logits = cond_logits_all(h, J, beta, x_int)            # (d,q)
        logp = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        idx = jnp.arange(x_int.shape[0])
        ll  = logp[idx, x_int].sum()
        return -ll
    vals = vmap(npll_one)(X_int)
    return vals.mean() / h.shape[0]  # per-site

def l2norm_tree(tree):
    return jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(tree)]))

# ------------------------
# Gibbs sampler (compiled)
# ------------------------
def gibbs_sweep(h, J, q, beta, key, x):
    d = x.shape[0]
    def body(i, carry):
        xx, kk = carry
        kk, sub = jrand.split(kk)

        xoh = jax.nn.one_hot(xx, q, dtype=jnp.float64)
        xoh = xoh.at[i].set(0.0)

        Ji = J[i]                                           # (d, q, q)
        Ji_kjl = jnp.transpose(Ji, (1, 0, 2)).reshape(q, d*q)
        msg_i = Ji_kjl @ xoh.reshape(d*q)                   # (q,)

        logits = beta * (h[i] + msg_i)
        xi = jrand.categorical(sub, logits).astype(jnp.int32)
        xx = xx.at[i].set(xi)
        return (xx, kk)
    (x_new, _) = lax.fori_loop(0, d, body, (x, key))
    return x_new

def gibbs_sample_compiled(h, J_eff, q, beta, key, n_samples, burnin=300, thinning=2, x0=None):
    d = h.shape[0]
    if x0 is None:
        key, kx = jrand.split(key)
        x0 = jrand.randint(kx, (d,), 0, q)
    total_sweeps = burnin + n_samples * thinning
    def one_sweep(carry, _):
        x, k = carry
        k, sub = jrand.split(k)
        x = gibbs_sweep(h, J_eff, q, beta, sub, x)
        return (x, k), x
    (_, _), traj = lax.scan(one_sweep, (x0, key), None, length=total_sweeps)
    kept = traj[burnin::thinning]            # (n_samples, d)
    return kept

# ------------------------
# CD gradient (with optional Stein-BQ model moment)
# ------------------------
def cd_main(X_true_oh, X_model_oh, X_model_int, J, J_mask, h, beta, lambda_J, lambda_h, run_bq):
    d, q = h.shape
    dEh_data, dEJ_data = mean_grad_U_over(X_true_oh, h, J, J_mask)

    if run_bq:
        Fmat = build_Fmat_beta_grad_raw(h, J, J_mask, X_model_oh)  # (n, d*q + d*d*q*q)
        mu_hat = stein_bq_poly2_mu(h, J, J_mask, beta, X_model_int, Fmat, ridge=1e-3)
        dEh_model, dEJ_model = unpack_mu_hat(mu_hat, d, q)
    else:
        dEh_model, dEJ_model = mean_grad_U_over(X_model_oh, h, J, J_mask)

    grad_h = beta * (dEh_data - dEh_model) + lambda_h * h
    grad_J = beta * (dEJ_data - dEJ_model) + lambda_J * J
    grad_J = project_J(grad_J, J_mask)
    return grad_h, grad_J

# ------------------------
# Deterministic initializers (shared across MC vs BQ)
# ------------------------
def init_params_from_seed(seed, d, q, scale=0.01):
    key = jrand.PRNGKey(seed)
    key, k_h, k_J = jrand.split(key, 3)
    h0 = scale * jrand.normal(k_h, (d, q))
    J0 = scale * jrand.normal(k_J, (d, d, q, q))
    return h0, J0

def init_gibbs_state_from_seed(seed, d, q, tag=0):
    # Optional identical initial Gibbs state for fairness; tag lets you vary by n_model etc.
    key = jrand.PRNGKey(seed ^ (12345 + 17 * int(tag)))
    return jrand.randint(key, (d,), 0, q)

# ------------------------
# Training
# ------------------------
def cd_training(seed, d, q, p, beta, steps, l_r, n_model, n_true, lambda_h, lambda_J,
                run_bq, patience=20, ksd_every=200, val_frac=0.2,
                h0=None, J0=None, gibbs_x0=None):
    # NOTE: h0/J0 allow MC & BQ runs to share identical initials.
    # gibbs_x0 optionally fixes the initial Gibbs state for negatives.
    key = jrand.PRNGKey(seed)

    # === Initial params (shared if provided) ===
    if (h0 is not None) and (J0 is not None):
        h = h0
        J = J0
    else:
        key, k_h, k_J = jrand.split(key, 3)
        h = 0.01 * jrand.normal(k_h, (d, q))
        J = 0.01 * jrand.normal(k_J, (d, d, q, q))

    J_mask = adjacency_mask(d)
    J = project_J(J, J_mask)

    # === Optimizer setup ===
    clip = 1.0
    lr   = l_r
    tx = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    opt_state = tx.init((h, J))

    X_true_int, X_true_oh = data_preprocess(p, q, d, n_true, seed)
    n_val = max(1, int(val_frac * n_true))
    X_val_int   = X_true_int[-n_val:]
    X_train_int = X_true_int[:-n_val]
    X_train_oh  = X_true_oh[:-n_val]

    best_npll = jnp.inf
    best = {'h': h, 'J': J}
    patience_ctr = 0
    prev_h, prev_J = h, J

    for t in range(steps):
        J = project_J(J, J_mask)

        key, k_gibbs = jrand.split(key, 2)
        # --- Gibbs negatives; optionally fixed initial state for fairness ---
        x0 = gibbs_x0
        X_model_int = gibbs_sample_compiled(h, J, q, beta, k_gibbs, n_model,
                                            burnin=200, thinning=2, x0=x0)
        X_model_oh  = jax.nn.one_hot(X_model_int, num_classes=q, dtype=jnp.float64)

        grad_h, grad_J = cd_main(X_train_oh, X_model_oh, X_model_int, J, J_mask, h,
                                 beta, lambda_J, lambda_h, run_bq)
        h, J, opt_state = apply_update(h, J, grad_h, grad_J, opt_state, J_mask, tx)

        npll_val = float(npll_batch(h, J, beta, X_val_int))
        g_norm   = float(l2norm_tree({'h': grad_h, 'J': grad_J}))
        dtheta   = float(l2norm_tree({'h': h - prev_h, 'J': J - prev_J}))
        prev_h, prev_J = h, J

        if npll_val + 1e-6 < float(best_npll):
            best_npll = npll_val
            best = {'h': h, 'J': J}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (t % 50) == 0 or (t == steps - 1):
            print(f"step {t:4d} | NPLL_val/site={npll_val:.4f} | ||grad||={g_norm:.3e} | ||Δθ||={dtheta:.3e} | pat={patience_ctr}")

        if ksd_every and (t % ksd_every == 0):
            X_monitor = X_train_oh[:min(256, X_train_oh.shape[0])]
            lam_monitor = float(median_heuristic_lambda_hamming(X_monitor))
            ksd2_small = float(compute_discrete_ksd(X_monitor, h, J, J_mask, beta, lam_monitor,
                                                    n_samples=min(128, X_monitor.shape[0])))
            print(f"   KSD^2 (mini) = {ksd2_small:.3e}")

        if patience_ctr >= patience:
            print(f"Early stopping at step {t} (no val NPLL improvement for {patience} steps).")
            break

    h_best, J_best = best['h'], best['J']
    params = (h_best, J_best)

    lam = median_heuristic_lambda_hamming(X_train_oh, key=jax.random.PRNGKey(0))
    ksd2 = compute_discrete_ksd(X_train_oh, h_best, J_best, J_mask, beta, lam, n_samples=min(256, X_train_oh.shape[0]))
    logZ_final, nll_final = evaluate_logZ_nll_exact(params, X_train_oh, beta, J_mask)

    return h_best, J_best, J_mask, abs(float(ksd2)), nll_final

# ------------------------
# Experiment runner with shared initials
# ------------------------
def run_experiments(
    seeds, n_models, d, q, p, beta, steps, l_r, n_true,
    lambda_h, lambda_J, run_bq, patience=100, ksd_every=200,
    init_provider=None,           # function: seed -> (h0,J0)
    gibbs_x0_provider=None        # optional: (seed, n_model) -> x0
):
    results = []
    for seed in seeds:
        # Prepare shared initials (if requested)
        h0, J0 = (init_provider(seed) if init_provider is not None else (None, None))
        # print(h0)
        # print(J0)
        for n_model in n_models:
            print("="*60)
            print(f"Seed={seed}, n_model={n_model}")

            h_init = None if h0 is None else jnp.array(h0, copy=True)
            J_init = None if J0 is None else jnp.array(J0, copy=True)

            x0 = gibbs_x0_provider(seed, n_model) if gibbs_x0_provider is not None else None

            h_final, J_final, J_mask, ksd2, nll_final = cd_training(
                seed, d, q, p, beta, steps, l_r,
                n_model, n_true, lambda_h, lambda_J,
                run_bq, patience=patience, ksd_every=ksd_every,
                h0=h_init, J0=J_init, gibbs_x0=x0
            )
            results.append({
                "seed": seed,
                "n_model": n_model,
                "ksd2": float(ksd2),
                "nll": float(nll_final)
            })
            print(f"Done: seed={seed}, n_model={n_model}, KSD^2={ksd2:.3e}, NLL={nll_final:2e}")
    return results

def _true_nll_nats(L, p):
    return float(L * (-(2*p*np.log(p) + (1-2*p)*np.log(1-2*p))))

def plot_ksd_nll(ksd_mc, ksd_bq, ksd_bq_eh, n_models, d, p):
    # true_nll = _true_nll_nats(d, p)
    fig, ax1 = plt.subplots(figsize=(10.0, 6.0))

    ax1.plot(n_models, ksd_mc, 'k-o', label='KSD$^2$ (MC)')
    ax1.plot(n_models, ksd_bq, 'b-s', label='KSD$^2$ (BayesSum - Polynomial (2nd))')
    ax1.plot(n_models, ksd_bq_eh, 'b-^', linestyle='--', label='KSD$^2$ (BayesSum - Exp Hamming)')
    

    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('KSD$^2$')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

    # ax2 = ax1.twinx()
    # ax2.plot(n_models, nll_mc, 'k--o', label='NLL (MC)')
    # ax2.plot(n_models, nll_bq, 'b--s', label='NLL (BayesSum)')
    # ax2.axhline(true_nll, color='gray', linestyle=':', label='Optimal NLL')  # True NLL
    # ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
    # ax2.set_ylim(15.5, 16.5)
    # ax2.set_ylabel('NLL')

    handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # handles = handles1 + handles2
    # labels = labels1 + labels2
    ax1.legend(handles1, labels1, loc='best', fontsize=26)

    plt.tight_layout()
    plt.savefig('ksd_nll_cd.pdf')
    plt.close(fig)

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    d, q, p = 15, 3, 0.4
    beta = 1.0 / 2.269
    steps = 2000
    l_r = 5e-3
    lambda_h, lambda_J = 5e-3, 5e-3

    seeds = [0, 1, 2, 3, 4]
    n_models = [50, 100, 500, 1000]

    def init_provider(seed):
        return init_params_from_seed(seed, d, q)

    bq_results = run_experiments(
        seeds, n_models,
        d, q, p, beta, steps, l_r, n_true=2000,
        lambda_h=lambda_h, lambda_J=lambda_J,
        run_bq=True, patience=200, ksd_every=200,
        init_provider=init_provider,
        gibbs_x0_provider=None  # or gibbs_x0_provider if you want to fix Gibbs x0
    )

    # mc_results = run_experiments(
    #     seeds, n_models,
    #     d, q, p, beta, steps, l_r, n_true=2000,
    #     lambda_h=lambda_h, lambda_J=lambda_J,
    #     run_bq=False, patience=200, ksd_every=200,
    #     init_provider=init_provider,
    #     gibbs_x0_provider=None  # must match the BQ choice for fairness
    # )

    # Aggregate means over seeds
    # ksd_mc_means = []
    # nll_mc_means = []

    ksd_bq_means = []
    nll_bq_means = []

    for n in n_models:
        # ksd_mc_vals = [r["ksd2"] for r in mc_results if r["n_model"] == n]
        ksd_bq_vals = [r["ksd2"] for r in bq_results if r["n_model"] == n]
        # nll_mc_vals = [r["nll"]  for r in mc_results if r["n_model"] == n]
        nll_bq_vals = [r["nll"]  for r in bq_results if r["n_model"] == n]
        # ksd_mc_means.append(np.mean(ksd_mc_vals))
        ksd_bq_means.append(np.mean(ksd_bq_vals))
        # nll_mc_means.append(np.mean(nll_mc_vals))
        nll_bq_means.append(np.mean(nll_bq_vals))

    # ksd_mc = jnp.array(ksd_mc_means)
    ksd_bq = jnp.array(ksd_bq_means)
    # nll_mc = jnp.array(nll_mc_means)
    nll_bq = jnp.array(nll_bq_means)

    print("Final NLL")
    # print("MC:", nll_mc)
    print("BQ:", nll_bq)

    print("Final KSD^2")
    # print("MC:", ksd_mc)
    print("BQ:", ksd_bq)

    # nll_bq = [15.8004, 15.7926138, 15.78620551, 15.78573711]
    # nll_mc = [15.78866686, 15.78656951, 15.78532849, 15.78725057]

    # ksd_bq = [0.00205008, 0.00174232, 0.00131944, 0.00114112]
    # ksd_mc = [0.00226057, 0.00189264, 0.00156035, 0.00134715]

    # # exp hamming kernel
    # ksd_bq_eh = [0.00140546, 0.00182478, 0.00143684, 0.00122525] 
    # nll_bq_eh = [15.792111, 15.7871861, 15.78452858, 15.784539]

    # # Optional plotting (disabled true-NLL line)
    # plot_ksd_nll(ksd_mc, ksd_bq, ksd_bq_eh, n_models, d, p)
