import time
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import optax
import matplotlib.pyplot as plt
import os

jax.config.update("jax_enable_x64", True)

# ----------------- Matplotlib style (set usetex=False if no LaTeX) -----------------
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.rc('axes', titlesize=12, labelsize=12, grid=True)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=12, frameon=False)
plt.rc('xtick', labelsize=12, direction='in')
plt.rc('ytick', labelsize=12, direction='in')
plt.rc('figure', figsize=(6, 4), dpi=100)

# -------------
#  Data generator
# -------------
def generate_categorical_sequence(key, p, L):
    probs = jnp.array([p, p, 1 - 2 * p], dtype=jnp.float64)
    keys = jax.random.split(key, L)
    seq = [jax.random.choice(k, 3, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)

def data_preprocess(p, q, L, num_sequences):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_sequences)
    sequences_true = [generate_categorical_sequence(k, p, L) for k in keys]
    sigma_onehot_synthetic = jax.nn.one_hot(
        jnp.stack(sequences_true), num_classes=q, dtype=jnp.float64
    )  # (N, L, q)
    return sequences_true, sigma_onehot_synthetic, L

# -------------
#  Energy
# -------------
@jit
def energy(x, h, J, J_mask):
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))  # symmetrize across (i,j)
    Jeff = Jsym * J_mask
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    return e_J + e_h

# -------------
#  Exponential dot-product kernel on one-hots & kernel mean
# -------------
@jit
def kernel_embedding(lambda_, L, q):
    return ((1.0 + (q - 1.0) * jnp.exp(-lambda_)) / q) ** L

@jit
def gram_matrix(sigma_batch, lambda_):
    n, L, q = sigma_batch.shape
    sigma_flat = sigma_batch.reshape(n, -1)
    def k(x1, x2):
        dot = jnp.dot(x1, x2)
        return jnp.exp(-lambda_ * L + lambda_ * dot)
    return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

# ====================
#  Pool-based Z estimators (MC and BQ)
# ====================
@partial(jit, static_argnums=(6,))  # <-- make n_mc static to avoid dynamic slice
def logZ_from_pool(h, J, beta, Xi_onehot, key, J_mask, n_mc):
    M = Xi_onehot.shape[0]
    # IMPORTANT: assume n_mc <= M (we assert outside before calling)
    perm = jax.random.permutation(key, M)
    idx = perm[:n_mc]                  # static stop -> OK under jit
    Xi_sub = Xi_onehot[idx]            # (n_mc, L, q)
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_sub, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    Z_hat = jnp.mean(jnp.exp(logf - m))
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

def precompute_bc_weights(Xi_onehot, lambda_):
    K = gram_matrix(Xi_onehot, lambda_) + 1e-8 * jnp.eye(Xi_onehot.shape[0], dtype=jnp.float64)
    Lc, lower = cho_factor(K, lower=True)
    L, q = Xi_onehot.shape[1], Xi_onehot.shape[2]
    z = jnp.full((Xi_onehot.shape[0], 1), kernel_embedding(lambda_, L, q), dtype=jnp.float64)
    w = cho_solve((Lc, lower), z)  # (M,1)
    return w  # float64

@jit
def logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask):
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_onehot, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    f_shift = jnp.exp(logf - m)[:, None]  # (M,1)
    Z_hat = (w_bc.T @ f_shift)[0, 0]
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

@jit
def logZ_mc_full_pool(h, J, beta, Xi_onehot, J_mask):
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_onehot, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    Z_hat = jnp.mean(jnp.exp(logf - m))
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

# ====================
#  Loss and update
# ====================
def loss_and_aux(params, Sigma_batch, key, Xi_onehot, w_bc, lambda_, beta,
                 weight_decay, run_bq, J_mask, n_mc):
    h, J = params
    energies = jax.vmap(energy, in_axes=(0, None, None, None))(Sigma_batch, h, J, J_mask)

    if run_bq and (w_bc is not None):
        log_Z = logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask)
    else:
        key, zkey = jax.random.split(key)
        log_Z = logZ_from_pool(h, J, beta, Xi_onehot, zkey, J_mask, n_mc)

    nll = -jnp.mean(-beta * energies - log_Z)

    # L2 reg (on effective parameters)
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    l2 = jnp.sum(h**2) + jnp.sum(Jeff**2)
    loss = nll + weight_decay * l2
    return loss, {"log_Z": log_Z}

def make_update_fn(optimizer, Xi_onehot, w_bc, J_mask):
    @partial(jax.jit, static_argnums=(4, 5, 9))  # n_samples, n_mc, run_bq are static
    def update_step(params, opt_state, Sigma, key,
                    n_samples, n_mc,
                    lambda_, beta, weight_decay, run_bq):
        num_total = Sigma.shape[0]
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, num_total, shape=(n_samples,), replace=False)
        Sigma_batch = Sigma[idx]

        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, Sigma_batch, key, Xi_onehot, w_bc, lambda_, beta,
            weight_decay, run_bq, J_mask, n_mc
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux["log_Z"]
    return update_step

# ====================
#  Discrete KSD utilities
# ====================
@partial(jit, static_argnames=['n_subset'])
def median_heuristic_gamma(samples, n_subset=500):
    n_total = samples.shape[0]
    if n_total > n_subset:
        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, n_total, shape=(n_subset,), replace=False)
        samples_sub = samples[idx]
    else:
        samples_sub = samples
    n = samples_sub.shape[0]
    samples_flat = samples_sub.reshape(n, -1)
    norms_sq = jnp.sum(samples_flat**2, axis=1)
    dot_prods = samples_flat @ samples_flat.T
    dist_sq_matrix = norms_sq[:, None] - 2 * dot_prods + norms_sq[None, :]
    triu_indices = jnp.triu_indices(n, k=1)
    distances_sq = dist_sq_matrix[triu_indices]
    median_sq = jnp.median(distances_sq)
    gamma = 1.0 / (2.0 * jnp.clip(median_sq, 1e-12, None))
    return gamma

@partial(jit, static_argnames=['shift'])
def get_neighbor(x_i, shift):
    return jnp.roll(x_i, shift=shift, axis=-1)

@partial(jit, static_argnames=['shift'])
def get_all_neighbors(x, shift):
    L, q = x.shape
    eye = jnp.eye(L, dtype=x.dtype)[:, :, None]
    x_permuted_rows = vmap(get_neighbor, in_axes=(0, None))(x, shift)  # (L, q)
    x_tiled = jnp.tile(x, (L, 1, 1))
    x_permuted_tiled = jnp.tile(x_permuted_rows, (L, 1, 1))
    neighbors = x_tiled * (1 - eye) + x_permuted_tiled * eye
    return neighbors  # (L, L, q)

@jit
def discrete_model_score(x, h, J, J_mask, beta):
    neighbors = get_all_neighbors(x, shift=1)
    E_x = energy(x, h, J, J_mask)
    E_neighbors = vmap(energy, in_axes=(0, None, None, None))(neighbors, h, J, J_mask)
    score = 1.0 - jnp.exp(-beta * (E_neighbors - E_x))
    return score  # (L,)

@jit
def base_rbf_kernel(x_flat, y_flat, gamma):
    return jnp.exp(-gamma * jnp.sum((x_flat - y_flat) ** 2))

def discrete_ksd_u_p_term(x, y, h, J, J_mask, beta, gamma):
    L, q = x.shape
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
    trace_term = jnp.sum(jnp.diag(k_neighbors_neighbors) - k_x_y_neighbors - k_x_neighbors_y + k_xy)

    t1 = jnp.dot(score_x, score_y) * k_xy
    t2 = -jnp.dot(score_x, delta_y_k)
    t3 = -jnp.dot(delta_x_k, score_y)
    t4 = trace_term
    return t1 + t2 + t3 + t4

@partial(jit, static_argnames=['n_samples'])
def compute_discrete_ksd(samples, h, J, J_mask, beta, gamma, n_samples=None):
    if n_samples is None:
        n_samples = samples.shape[0]
    u_p_vmapped = vmap(
        vmap(discrete_ksd_u_p_term, in_axes=(None, 0, None, None, None, None, None)),
        in_axes=(0, None, None, None, None, None, None)
    )
    u_p_matrix = u_p_vmapped(samples[:n_samples], samples[:n_samples], h, J, J_mask, beta, gamma)
    total_sum = jnp.sum(u_p_matrix) - jnp.sum(jnp.diag(u_p_matrix))
    ksd_sq = total_sum / (n_samples * (n_samples - 1))
    return ksd_sq

# ====================
#  Main experiment (per seed)
# ====================
def run_single_experiment(seed, n_bq, n_mc, run_bq):
    print(f"\n--- Running Experiment for Seed {seed} | run_bq={run_bq} | n={n_bq} ---")
    assert n_mc <= n_bq, "n_mc must be <= n_bq (or switch to with-replacement MC)."

    # --- Data ---
    p = 0.4; q = 3; L = 4; num_sequences = 500
    _, sigma_onehot_synthetic, L = data_preprocess(p, q, L, num_sequences)  # (N, L, q)

    # --- Integration pool ---
    key = jax.random.PRNGKey(seed)
    key, int_key = jax.random.split(key)
    Xi = jax.random.randint(int_key, (n_bq, L), minval=0, maxval=q)
    Xi_onehot = jax.nn.one_hot(Xi, num_classes=q, dtype=jnp.float64)  # (M, L, q)

    # --- Mask ---
    J_mask = (1.0 - jnp.eye(L, dtype=jnp.float64))[:, :, None, None]  # (L,L,1,1)

    # --- Init params ---
    key, h_key, j_key = jax.random.split(key, 3)
    h_model = jax.random.normal(h_key, shape=(L, q), dtype=jnp.float64) * 0.01
    J_model = jax.random.normal(j_key, shape=(L, L, q, q), dtype=jnp.float64) * 0.01
    params_model = (h_model, J_model)

    # --- Training config ---
    learning_rate = 1e-3
    weight_decay = 0.0
    num_steps = 2000
    beta = 1.0
    lambda_ = 0.01

    # --- Optimizer ---
    optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params_model)

    # --- BQ weights if needed ---
    w_bc = precompute_bc_weights(Xi_onehot, lambda_) if run_bq else None
    update_step = make_update_fn(optimizer, Xi_onehot, w_bc, J_mask)

    # --- Train ---
    params = params_model
    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        params, opt_state, loss, log_Z = update_step(
            params, opt_state, sigma_onehot_synthetic, subkey,
            n_samples=300, n_mc=int(n_mc),
            lambda_=lambda_, beta=beta,
            weight_decay=weight_decay, run_bq=run_bq,
        )
        if step % 20 == 0:
            print(f"Step {step:4d}/{num_steps-1} | Loss: {loss:.6f} | Log_Z: {log_Z:.6f}")

    h_final, J_final = params

    # --- KDSD ---
    gamma = median_heuristic_gamma(sigma_onehot_synthetic)
    ksd_value_discrete = compute_discrete_ksd(
        sigma_onehot_synthetic, h_final, J_final, J_mask, beta, gamma
    )
    print(f"Final KDSD^2: {ksd_value_discrete:.6f}")
    return float(np.asarray(ksd_value_discrete))

# ====================
#  Sweeps + plotting
# ====================
def run_sweep(n_list, num_seeds, run_bq):
    means, stds = [], []
    for n in n_list:
        vals = []
        for seed in range(num_seeds):
            v = run_single_experiment(seed, n_bq=n, n_mc=n, run_bq=run_bq)
            vals.append(v)
        vals = np.asarray(vals, dtype=np.float64)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)

def plot_kdsd_means_with_errorbars(n_list, bq_mean, bq_std, mc_mean, mc_std,
                                   save_dir="results",
                                   fname_png="kdsd_curves.png",
                                   fname_pdf="kdsd_curves.pdf",
                                   legend_only_pdf="kdsd_legend.pdf"):
    os.makedirs(save_dir, exist_ok=True)
    x = np.asarray(n_list, dtype=float)

    plt.figure(figsize=(6.5, 4.5))
    hbq = plt.errorbar(x, bq_mean, yerr=bq_std, fmt='o-', capsize=3, label="BQ")
    hmc = plt.errorbar(x, mc_mean, yerr=mc_std, fmt='s--', capsize=3, label="MC")

    plt.xlabel("Pool size n (also MC subsample)")
    plt.ylabel(r"KDSD$^2$")
    plt.yscale('log')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    out_png = os.path.join(save_dir, fname_png)
    out_pdf = os.path.join(save_dir, fname_pdf)
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    # separate legend-only figure (optional)
    handles = [hbq.lines[0], hmc.lines[0]]
    labels = ["BQ", "MC"]
    fig_leg, ax_leg = plt.subplots(figsize=(4.5, 1.1))
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, ncol=2, loc='center', fontsize=12, frameon=False)
    plt.savefig(os.path.join(save_dir, legend_only_pdf), bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved {out_png} and {out_pdf}")

# ====================
#  Main
# ====================
def main():
    n_list = [100, 200, 500, 1000]
    num_seeds = 5

    print("\n=== Running BQ sweep ===")
    bq_mean, bq_std = run_sweep(n_list, num_seeds, run_bq=True)

    print("\n=== Running MC sweep ===")
    mc_mean, mc_std = run_sweep(n_list, num_seeds, run_bq=False)

    os.makedirs("results", exist_ok=True)
    np.savez("results/ksd_sweeps.npz",
             n=np.array(n_list),
             bq_mean=bq_mean, bq_std=bq_std,
             mc_mean=mc_mean, mc_std=mc_std)

    print(bq_mean)
    print(mc_mean)

    # plot_kdsd_means_with_errorbars(
    #     n_list, bq_mean, bq_std, mc_mean, mc_std,
    #     save_dir="results",
    #     fname_png="kdsd_curves.png",
    #     fname_pdf="kdsd_curves.pdf",
    #     legend_only_pdf="kdsd_legend.pdf"
    # )

if __name__ == "__main__":
    main()
