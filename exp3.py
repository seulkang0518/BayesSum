import numpy as np
from functools import partial
import os
import itertools

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.special import logsumexp
import optax
import matplotlib.pyplot as plt
from scipy.optimize import minimize  # for tuning (c, sigma^2, eta)

jax.config.update("jax_enable_x64", True)

# ----------------- Matplotlib style -----------------
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 26
plt.rc('text', usetex=False)
plt.rc('axes', titlesize=26, labelsize=26, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=20, frameon=False)
plt.rc('xtick', labelsize=20, direction='in')
plt.rc('ytick', labelsize=20, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

# =============================================================================
# Data (i.i.d. categorical; not Potts-correlated)
# =============================================================================
def generate_categorical_sequence(key, p, d):
    probs = jnp.array([p, p, 1 - 2 * p], dtype=jnp.float64)
    keys = jax.random.split(key, d)
    seq = [jax.random.choice(k, 3, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)

def data_preprocess(p, q, d, num_sequences):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_sequences)
    sequences_true = [generate_categorical_sequence(k, p, d) for k in keys]
    sigma_onehot_synthetic = jax.nn.one_hot(
        jnp.stack(sequences_true), num_classes=q, dtype=jnp.float64
    )
    return sequences_true, sigma_onehot_synthetic, d

# =============================================================================
# Lattice mask and energy
# =============================================================================
def lattice_J_mask(Lside: int) -> jnp.ndarray:
    d = Lside * Lside
    idx = jnp.arange(d).reshape(Lside, Lside)
    rights = jnp.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    downs  = jnp.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    edges = jnp.concatenate([rights, downs], axis=0)
    mask = jnp.zeros((d, d, 1, 1), dtype=jnp.float64)
    mask = mask.at[edges[:, 0], edges[:, 1], 0, 0].set(1.0)
    mask = mask.at[edges[:, 1], edges[:, 0], 0, 0].set(1.0)
    return mask

@partial(jax.jit)
def energy(x, h, J, J_mask):
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    return e_J + e_h

# =============================================================================
# Kernel (exp(dot) on one-hots)
# =============================================================================
@partial(jax.jit)
def kernel_embedding(lambda_, d, q):
    return ((1.0 + (q - 1.0) * jnp.exp(-lambda_)) / q) ** d

@partial(jax.jit)
def gram_matrix(sigma_batch, lambda_):
    n, d, q = sigma_batch.shape
    sigma_flat = sigma_batch.reshape(n, -1)
    def k(x1, x2):
        dot = jnp.dot(x1, x2)
        return jnp.exp(-lambda_ * d + lambda_ * dot)
    return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

# =============================================================================
# MC & BQ logZ
# =============================================================================
# logZ_from_pool(..., n_mc) -> n_mc is arg index 6
@partial(jax.jit, static_argnums=(6,))
def logZ_from_pool(h, J, beta, Xi_onehot, key, J_mask, n_mc):
    M = Xi_onehot.shape[0]
    perm = jax.random.permutation(key, M)
    idx = perm[:n_mc]
    Xi_sub = Xi_onehot[idx]
    E_vals = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_sub, h, J, J_mask)
    logf = -beta * E_vals
    m = jnp.max(logf)
    Z_hat = jnp.mean(jnp.exp(logf - m))
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

@partial(jax.jit)
def logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask):
    E_vals = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_onehot, h, J, J_mask)
    logf = -beta * E_vals
    m = jnp.max(logf)
    f_shift = jnp.exp(logf - m)[:, None]
    Z_hat = (w_bc.T @ f_shift)[0, 0]
    # Stability: if Z_hat <= 0, project weights to nonnegative & renormalize
    def fix(_):
        w_pos = jnp.maximum(w_bc, 0.0)
        s = jnp.sum(w_pos)
        w_pos = jnp.where(s > 1e-16, w_pos / s, w_pos)
        return (w_pos.T @ f_shift)[0, 0]
    Z_hat = jax.lax.cond(Z_hat > 1e-300, lambda z: z, fix, Z_hat)
    return jnp.log(jnp.clip(Z_hat, 1e-300, None)) + m

# =============================================================================
# Loss, NLL, Update (MC/BQ)
# =============================================================================
def loss_and_aux(params, Sigma_batch, key, Xi_onehot, w_bc, lambda_dummy, beta,
                 weight_decay, run_bq, J_mask, n_mc):
    h, J = params
    energies = jax.vmap(energy, in_axes=(0, None, None, None))(Sigma_batch, h, J, J_mask)
    if run_bq and (w_bc is not None):
        log_Z = logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask)
    else:
        key, zkey = jax.random.split(key)
        log_Z = logZ_from_pool(h, J, beta, Xi_onehot, zkey, J_mask, n_mc=n_mc)
    nll = jnp.mean(beta * energies + log_Z)
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    l2 = jnp.sum(h**2) + jnp.sum(Jeff**2)
    loss = nll + weight_decay * l2
    return loss, {"log_Z": log_Z}

# calculate_nll(..., n_mc, ..., run_bq, ...)
# n_mc index 3, run_bq index 5
@partial(jax.jit, static_argnums=(3, 5))
def calculate_nll(params, all_data, Xi_onehot, n_mc, J_mask, run_bq, w_bc=None):
    h, J = params
    beta = 1.0
    key = jax.random.PRNGKey(0)
    if run_bq and (w_bc is not None):
        log_Z = logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask)
    else:
        log_Z = logZ_from_pool(h, J, beta, Xi_onehot, key, J_mask, n_mc=n_mc)
    energies = jax.vmap(energy, in_axes=(0, None, None, None))(all_data, h, J, J_mask)
    return jnp.mean(beta * energies + log_Z)

def make_update_fn(optimizer, Xi_onehot, w_bc_ref, J_mask):
    # update_step(..., n_samples, n_mc, ..., run_bq, step)
    # indices: n_samples=4, n_mc=5, run_bq=9
    @partial(jax.jit, static_argnums=(4, 5, 9))
    def update_step(params, opt_state, Sigma, key,
                    n_samples, n_mc,
                    lambda_dummy, beta, weight_decay, run_bq, step):
        num_total = Sigma.shape[0]
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, num_total)
        idx = perm[:n_samples]
        Sigma_batch = Sigma[idx]

        w_bc = w_bc_ref  # captured; can be replaced outside JIT if you retune

        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, Sigma_batch, key, Xi_onehot, w_bc, lambda_dummy, beta,
            weight_decay, run_bq, J_mask, n_mc
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux["log_Z"]
    return update_step

# =============================================================================
# Discrete KSD utilities
# =============================================================================
def median_heuristic_gamma(samples, n_subset=500, key=None):
    n_total = samples.shape[0]
    if (key is not None) and (n_total > n_subset):
        idx = jax.random.choice(key, n_total, shape=(n_subset,), replace=False)
        samples_sub = samples[idx]
    else:
        samples_sub = samples if n_total <= n_subset else samples[:n_subset]
    n = samples_sub.shape[0]
    samples_flat = samples_sub.reshape(n, -1)
    norms_sq = jnp.sum(samples_flat**2, axis=1)
    dist_sq = norms_sq[:, None] - 2.0 * (samples_flat @ samples_flat.T) + norms_sq[None, :]
    tri = jnp.triu_indices(n, k=1)
    med_sq = jnp.median(dist_sq[tri])
    return 1.0 / (2.0 * jnp.clip(med_sq, 1e-8, None))

@partial(jax.jit, static_argnums=(1,))
def get_neighbor(x_i, shift):
    return jnp.roll(x_i, shift=shift, axis=-1)

@partial(jax.jit, static_argnums=(1,))
def get_all_neighbors(x, shift):
    d, q = x.shape
    eye = jnp.eye(d, dtype=x.dtype)[:, :, None]
    x_permuted_rows = vmap(get_neighbor, in_axes=(0, None))(x, shift)
    x_tiled = jnp.tile(x, (d, 1, 1))
    x_permuted_tiled = jnp.tile(x_permuted_rows, (d, 1, 1))
    neighbors = x_tiled * (1 - eye) + x_permuted_tiled * eye
    return neighbors

@partial(jax.jit)
def discrete_model_score(x, h, J, J_mask, beta):
    neighbors = get_all_neighbors(x, shift=1)
    E_x = energy(x, h, J, J_mask)
    E_neighbors = vmap(energy, in_axes=(0, None, None, None))(neighbors, h, J, J_mask)
    score = 1.0 - jnp.exp(-beta * (E_neighbors - E_x))
    return score

@partial(jax.jit)
def base_rbf_kernel(x_flat, y_flat, gamma):
    return jnp.exp(-gamma * jnp.sum((x_flat - y_flat) ** 2))

def discrete_ksd_u_p_term(x, y, h, J, J_mask, beta, gamma):
    d, q = x.shape
    score_x = discrete_model_score(x, h, J, J_mask, beta)
    score_y = discrete_model_score(y, h, J, J_mask, beta)
    x_neighbors_inv = get_all_neighbors(x, shift=-1)
    y_neighbors_inv = get_all_neighbors(y, shift=-1)
    x_flat = x.ravel(); y_flat = y.ravel()
    vmap_flatten = vmap(jnp.ravel)
    x_neighbors_inv_flat = vmap_flatten(x_neighbors_inv)
    y_neighbors_inv_flat = vmap_flatten(y_neighbors_inv)
    k_xy = base_rbf_kernel(x_flat, y_flat, gamma)
    k_x_y_neighbors = vmap(base_rbf_kernel, in_axes=(None, 0, None))(x_flat, y_neighbors_inv_flat, gamma)
    delta_y_k = k_xy - k_x_y_neighbors
    k_x_neighbors_y = vmap(base_rbf_kernel, in_axes=(0, None, None))(x_neighbors_inv_flat, y_flat, gamma)
    delta_x_k = k_xy - k_x_neighbors_y
    k_neighbors_neighbors = vmap(vmap(base_rbf_kernel, in_axes=(0, None, None)),
                                 in_axes=(None, 0, None))(x_neighbors_inv_flat, y_neighbors_inv_flat, gamma)
    trace_term = (jnp.trace(k_neighbors_neighbors)
                  - jnp.sum(k_x_y_neighbors)
                  - jnp.sum(k_x_neighbors_y)
                  + d * k_xy)
    t1 = jnp.dot(score_x, score_y) * k_xy
    t2 = -jnp.dot(score_x, delta_y_k)
    t3 = -jnp.dot(delta_x_k, score_y)
    t4 = trace_term
    return t1 + t2 + t3 + t4

# compute_discrete_ksd_subsample(..., n_samples=None) -> n_samples is index 6
@partial(jax.jit, static_argnums=(6,))
def compute_discrete_ksd_subsample(samples, h, J, J_mask, beta, gamma, n_samples=None):
    N = samples.shape[0]
    if n_samples is None or n_samples > N:
        n_samples = N
    S = samples[:n_samples]
    u_p_vmapped = vmap(
        vmap(discrete_ksd_u_p_term, in_axes=(None, 0, None, None, None, None, None)),
        in_axes=(0, None, None, None, None, None, None)
    )
    u_p_matrix = u_p_vmapped(S, S, h, J, J_mask, beta, gamma)
    total_sum = jnp.sum(u_p_matrix) - jnp.sum(jnp.diag(u_p_matrix))
    ksd_sq = total_sum / (n_samples * (n_samples - 1))
    return ksd_sq

def compute_discrete_ksd_linear(samples, h, J, J_mask, beta, gamma, m_pairs=50_000, key=jax.random.PRNGKey(0)):
    N = samples.shape[0]
    key, ki, kj = jax.random.split(key, 3)
    i = jax.random.randint(ki, (m_pairs,), 0, N)
    j = jax.random.randint(kj, (m_pairs,), 0, N)
    def up(idx):
        ii, jj = idx
        return discrete_ksd_u_p_term(samples[ii], samples[jj], h, J, J_mask, beta, gamma)
    vals = vmap(up)(jnp.stack([i, j], axis=1))
    return jnp.mean(vals)

# =============================================================================
# ======== Robust zero-mean GP tuning on log-integrand with amp & nugget ======
# =============================================================================
def pool_targets_for_tuning(h, J, beta, Xi_onehot, J_mask):
    # y_log = (log f) - mean(log f)
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_onehot, h, J, J_mask)
    logf = -beta * E
    y_log = (logf - jnp.mean(logf))[:, None]
    return y_log

def lml_zero_mean(K, y):
    L, lower = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve((L, lower), y, check_finite=False)
    quad = float((y.T @ alpha)[0, 0])
    logdet = 2.0 * float(jnp.sum(jnp.log(jnp.diag(L))))
    n = y.shape[0]
    return -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)

def tune_hypers_minimize(Xi_onehot, y_log,
                         c0=1.0, sig0=1.0, eta0=1e-4,
                         bounds_c=(0.2, 3.0), bounds_sig=(1e-3, 1e3), bounds_eta=(1e-6, 1e-2),
                         maxiter=80):
    n, d, q = Xi_onehot.shape
    def objective(theta_arr):
        c, sig2, eta = float(theta_arr[0]), float(theta_arr[1]), float(theta_arr[2])
        lam_eff = c / d
        Kb = gram_matrix(Xi_onehot, lam_eff)
        K = sig2 * Kb + eta * jnp.eye(n, dtype=jnp.float64)
        return float(-lml_zero_mean(K, y_log))
    x0 = np.array([c0, sig0, eta0], dtype=float)
    bounds = [bounds_c, bounds_sig, bounds_eta]
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": maxiter})
    c_star, sig2_star, eta_star = map(float, res.x)
    return c_star, sig2_star, eta_star

def precompute_bc_weights_with_hypers(Xi_onehot, c_star, sig2_star, eta_star):
    n, d, q = Xi_onehot.shape
    lam_eff = c_star / d
    Kb = gram_matrix(Xi_onehot, lam_eff)
    K = sig2_star * Kb + eta_star * jnp.eye(n, dtype=jnp.float64)
    z_base = jnp.full((n, 1), kernel_embedding(lam_eff, d, q), dtype=jnp.float64)
    z = sig2_star * z_base
    Lc, lower = cho_factor(K, lower=True, check_finite=False)
    w = cho_solve((Lc, lower), z, check_finite=False)
    return w, lam_eff

def bq_diagnostics(Xi_onehot, lam_eff, w):
    K = gram_matrix(Xi_onehot, lam_eff)
    evals = jnp.linalg.eigvalsh((K + K.T) / 2.0)
    cond = float(jnp.max(evals) / jnp.maximum(jnp.min(evals), 1e-16))
    neg = int((w < 0).sum())
    l1 = float(jnp.sum(jnp.abs(w)))
    print(f"[BQ diag] cond(K)~{cond:.2e} | eig(min,max)=({float(evals.min()):.2e},{float(evals.max()):.2e}) | neg_w={neg}/{w.shape[0]} | ||w||1={l1:.2e}")

# =============================================================================
# Main experiment (MC/BQ with robust tuning & optional re-tune)
# =============================================================================
def run_single_experiment(seed, n_bq, n_mc, run_bq,
                          RETUNE_EVERY=200):
    print(f"\n--- Running Experiment for Seed {seed} | run_bq={run_bq} | n={n_bq} ---")
    assert n_mc <= n_bq, "n_mc must be <= n_bq"

    # Problem setup
    p = 0.4; q = 3; Lside = 4; num_sequences = 3000
    d = Lside * Lside

    # Data
    _, sigma_onehot_synthetic, _ = data_preprocess(p, q, d, num_sequences)

    # Pool Xi
    key = jax.random.PRNGKey(seed)
    key, int_key = jax.random.split(key)
    Xi = jax.random.randint(int_key, (n_bq, d), minval=0, maxval=q)
    Xi_onehot = jax.nn.one_hot(Xi, num_classes=q, dtype=jnp.float64)

    # Lattice mask
    J_mask = lattice_J_mask(Lside)

    # Init params
    key, h_key, j_key = jax.random.split(key, 3)
    h_model = jax.random.normal(h_key, shape=(d, q), dtype=jnp.float64) * 0.01
    J_model = jax.random.normal(j_key, shape=(d, d, q, q), dtype=jnp.float64) * 0.01
    params_model = (h_model, J_model)

    # Training config
    learning_rate = 1e-3
    weight_decay = 0.0
    num_steps = 2000
    beta = 1/2.269
    lambda_dummy = 0.0

    optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params_model)

    # ---- BQ weights (robust tuning) ----
    w_bc = None
    lam_eff = None
    if run_bq:
        y_log = pool_targets_for_tuning(h_model, J_model, beta, Xi_onehot, J_mask)
        # Warm start c0 around d*0.1 if you liked lambda≈0.1; with d=16 => c0≈1.6
        c_star, sig2_star, eta_star = tune_hypers_minimize(Xi_onehot, y_log,
                                                           c0=max(0.5, d*0.1),
                                                           bounds_c=(0.3, 3.0),
                                                           bounds_sig=(1e-3, 1e3),
                                                           bounds_eta=(1e-6, 1e-2),
                                                           maxiter=80)
        w_bc, lam_eff = precompute_bc_weights_with_hypers(Xi_onehot, c_star, sig2_star, eta_star)
        print(f"[BQ tuned] c*={c_star:.3f} → λ_eff={lam_eff:.3f}, σ²={sig2_star:.2e}, η={eta_star:.2e}")
        bq_diagnostics(Xi_onehot, lam_eff, w_bc)

    # Retune hook (outside JIT)
    def maybe_retune(step, params):
        nonlocal w_bc, lam_eff
        if (not run_bq) or (RETUNE_EVERY <= 0):
            return
        if step > 0 and (step % RETUNE_EVERY == 0):
            h_cur, J_cur = params
            y_log = pool_targets_for_tuning(h_cur, J_cur, beta, Xi_onehot, J_mask)
            c_star, sig2_star, eta_star = tune_hypers_minimize(Xi_onehot, y_log,
                                                               c0=1.0, bounds_c=(0.3, 3.0),
                                                               bounds_sig=(1e-3, 1e3),
                                                               bounds_eta=(1e-6, 1e-2),
                                                               maxiter=60)
            w_bc, lam_eff = precompute_bc_weights_with_hypers(Xi_onehot, c_star, sig2_star, eta_star)
            print(f"[BQ re-tuned@{step}] c*={c_star:.3f} → λ_eff={lam_eff:.3f}, σ²={sig2_star:.2e}, η={eta_star:.2e}")
            bq_diagnostics(Xi_onehot, lam_eff, w_bc)

    update_step = make_update_fn(optimizer, Xi_onehot, w_bc, J_mask)

    params = params_model
    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        params, opt_state, loss, log_Z = update_step(
            params, opt_state, sigma_onehot_synthetic, subkey,
            n_samples=500, n_mc=int(n_mc),
            lambda_dummy=lambda_dummy, beta=beta,
            weight_decay=weight_decay, run_bq=run_bq, step=step
        )
        if step % 200 == 0:
            print(f"Step {step:4d}/{num_steps-1} | Loss: {loss:.6f} | logZ: {float(log_Z):.6f}")
        maybe_retune(step, params)

    h_final, J_final = params
    final_nll = calculate_nll(
        params, sigma_onehot_synthetic, Xi_onehot,
        n_mc=int(n_mc), J_mask=J_mask, run_bq=run_bq, w_bc=w_bc
    )
    print(f"Final NLL: {final_nll:.6f}")

    gamma = median_heuristic_gamma(sigma_onehot_synthetic)
    ksd_value_discrete = compute_discrete_ksd_linear(
        sigma_onehot_synthetic, h_final, J_final, J_mask, beta, gamma
    )
    print(f"Final KSD^2: {ksd_value_discrete:.6f}")

    return float(np.asarray(ksd_value_discrete)), float(np.asarray(final_nll))

# =============================================================================
# Sweeps + plotting
# =============================================================================
def run_sweep(n_list, num_seeds, run_bq):
    ksd_means, ksd_stds = [], []
    nll_means, nll_stds = [], []
    for n in n_list:
        ksd_vals, nll_vals = [], []
        for seed in range(num_seeds):
            ksd_v, nll_v = run_single_experiment(seed, n_bq=n, n_mc=n, run_bq=run_bq,
                                                 RETUNE_EVERY=200 if run_bq else 0)
            ksd_vals.append(ksd_v); nll_vals.append(nll_v)
        ksd_vals = np.asarray(ksd_vals, dtype=np.float64)
        nll_vals = np.asarray(nll_vals, dtype=np.float64)
        ksd_means.append(ksd_vals.mean()); nll_means.append(nll_vals.mean())
        ksd_stds.append(ksd_vals.std(ddof=1) if len(ksd_vals) > 1 else 0.0)
        nll_stds.append(nll_vals.std(ddof=1) if len(nll_vals) > 1 else 0.0)
    return np.array(ksd_means), np.array(ksd_stds), np.array(nll_means), np.array(nll_stds)


# =============================================================================
# Main
# =============================================================================
def main():
    # Sanity: tiny exact demo (comment out if not needed)
    # /_ = run_exact_demo(seed=0, Lside=2, q=3, num_sequences=500, steps=300, lr=5e-3)

    # MC vs Robust BQ (auto-tuned on log-integrand, with amp & nugget)
    n_list = [2000, 3000, 4000]   # pool size
    num_seeds = 2

    print("\n=== Running BQ sweep (robust) ===")
    bq_ksd_mean, bq_ksd_std, bq_nll_mean, bq_nll_std = run_sweep(n_list, num_seeds, run_bq=True)
    print("BQ KSD mean:", bq_ksd_mean)
    print("BQ NLL mean:", bq_nll_mean)

    print("\n=== Running MC sweep ===")
    mc_ksd_mean, mc_ksd_std, mc_nll_mean, mc_nll_std = run_sweep(n_list, num_seeds, run_bq=False)
    print("MC KSD mean:", mc_ksd_mean)
    print("MC NLL mean:", mc_nll_mean)

    # Optional: plot
    # plot_dual_axis_abs_log(n_list, bq_ksd_mean, bq_nll_mean, mc_ksd_mean, mc_nll_mean)

if __name__ == "__main__":
    main()
