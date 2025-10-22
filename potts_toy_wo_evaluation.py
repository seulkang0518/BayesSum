import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import optax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

jax.config.update("jax_enable_x64", True)

# -----------------
# Deterministic parameter initializer (shared across MC/BQ)
# -----------------
def init_params_from_seed(seed, L, q, scale=0.01):
    key = jax.random.PRNGKey(seed)
    key, k_h, k_J = jax.random.split(key, 3)
    h0 = scale * jax.random.normal(k_h, (L, q), dtype=jnp.float64)
    J0 = scale * jax.random.normal(k_J, (L, L, q, q), dtype=jnp.float64)
    return h0, J0

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

def base_exp_hamming_kernel(x_flat, y_flat, lam, L):
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

    k_xy    = base_exp_hamming_kernel(xf, yf, lam, L)
    k_x_y_n = vmap(base_exp_hamming_kernel, in_axes=(None, 0, None, None))(xf, ynf, lam, L)
    k_x_n_y = vmap(base_exp_hamming_kernel, in_axes=(0, None, None, None))(xnf, yf, lam, L)
    k_nn    = vmap(
                  vmap(base_exp_hamming_kernel, in_axes=(0, None, None, None)),
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

def _ints_to_base_q(ints: jnp.ndarray, q: int, L: int) -> jnp.ndarray:
    def one_num_to_digits(n):
        digs = []
        x = n
        for _ in range(L):
            digs.append(x % q)
            x = x // q
        return jnp.array(digs[::-1], dtype=jnp.int32)  # length-L
    return vmap(one_num_to_digits)(ints)

def logZ_exact_enumeration(h, J, J_mask, beta, L: int, q: int, chunk: int = 200_000) -> float:
    total = int(q ** L)
    logZ = -jnp.inf  # running log-sum-exp accumulator
    for start in range(0, total, chunk):
        m = min(chunk, total - start)
        idx = jnp.arange(start, start + m, dtype=jnp.int64)        # (m,)
        digits = _ints_to_base_q(idx, q, L)                        # (m, L)
        X_chunk = jax.nn.one_hot(digits, num_classes=q, dtype=jnp.float64)  # (m, L, q)

        E = vmap(energy, in_axes=(0, None, None, None))(X_chunk, h, J, J_mask)  # (m,)
        logf = -beta * E
        mshift = jnp.max(logf)
        chunk_lse = mshift + jnp.log(jnp.clip(jnp.sum(jnp.exp(logf - mshift)), 1e-300))  # logsumexp(logf)
        logZ = jnp.logaddexp(logZ, chunk_lse)  # merge with running total
    return float(logZ)

def evaluate_logZ_nll_exact(params, X_eval, beta, J_mask):
    h, J = params
    L, q = X_eval.shape[1], X_eval.shape[2]
    E_data = vmap(energy, in_axes=(0, None, None, None))(X_eval, h, J, J_mask)
    data_term = jnp.mean(beta * E_data)
    logZ = logZ_exact_enumeration(h, J, J_mask, beta, L, q)
    return float(logZ), float(data_term + logZ)

def evaluate_logZ_nll(params, X_eval, beta, X_pool, J_mask, w_bc, run_bq):
    h, J = params
    E_data = vmap(energy, in_axes=(0, None, None, None))(X_eval, h, J, J_mask)
    if run_bq:
        logZ = logZ_bc_on_pool(h, J, beta, X_pool, w_bc, J_mask, X_pool.shape[1], X_pool.shape[2])
    else:
        logZ = logZ_mc_full_pool(h, J, beta, X_pool, J_mask, X_pool.shape[1], X_pool.shape[2])
    return logZ, float(jnp.mean(beta * E_data + logZ))

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
    sequences = [generate_categorical_sequence(k, p, L, q=q) for k in keys]
    data_onehot = jax.nn.one_hot(jnp.stack(sequences), num_classes=q, dtype=jnp.float64)  # (N,L,q)
    return sequences, data_onehot

# -----------------
# Mask utilities
# -----------------
def adjacency_mask_1d(L):
    M = jnp.zeros((L, L))
    idx = jnp.arange(L - 1)
    M = M.at[idx, idx + 1].set(1.0)
    M = M.at[idx + 1, idx].set(1.0)
    return M[:, :, None, None]

@jit
def project_J(J, J_mask):
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    return Jsym * J_mask

# -----------------
# Energy
# -----------------
@jit
def energy(x, h, J, J_mask):
    Jeff = project_J(J, J_mask)
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    return -(e_J + e_h)

@jit
def f_single(x, J, J_mask, h, beta):
    return jnp.exp(-beta * energy(x, h, J, J_mask))

@jit
def f_batch(X, J, J_mask, h, beta):
    return vmap(lambda x: f_single(x, J, J_mask, h, beta))(X)

# -----------------
# log Z estimators (MC and BQ)
# -----------------

# MC
@jit
def logZ_mc_full_pool(h, J, beta, X_mc, J_mask, L, q):
    E = vmap(energy, in_axes=(0, None, None, None))(X_mc, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    log_Eu = jnp.log(jnp.clip(jnp.mean(jnp.exp(logf - m)), 1e-300)) + m
    return log_Eu + L * jnp.log(q)

# BQ
@jit
def kernel_embedding(lambda_, L, q):
    return ((1.0 + (q - 1.0) * jnp.exp(-lambda_)) / q) ** L

@jit
def gram_matrix(X_bq, lambda_):
    n, L, q = X_bq.shape
    X_flat = X_bq.reshape(n, -1)
    def k(x1, x2):
        dot = jnp.dot(x1, x2)
        return jnp.exp(-lambda_ * L + lambda_ * dot)
    return vmap(lambda x: vmap(lambda y: k(x, y))(X_flat))(X_flat)

def precompute_bc_weights(X_bq, lambda_):
    K = gram_matrix(X_bq, lambda_) + 1e-6 * jnp.eye(X_bq.shape[0], dtype=jnp.float64)
    Lc, lower = cho_factor(K, lower=True)
    z = jnp.full((X_bq.shape[0], 1), kernel_embedding(lambda_, X_bq.shape[1], X_bq.shape[2]), dtype=jnp.float64)
    w = cho_solve((Lc, lower), z)  # (n,1)
    return jnp.ravel(w)            # <- return (n,) to avoid reshaping later

@jit
def logZ_bc_on_pool(h, J, beta, X_bq, w_bc, J_mask, L, q):
    E     = vmap(energy, in_axes=(0, None, None, None))(X_bq, h, J, J_mask)
    logf  = -beta * E
    m     = jnp.max(logf)
    fshift= jnp.exp(logf - m)

    w = jnp.ravel(w_bc)
    w_pos = jnp.maximum(w, 0.0)            # positive parts
    w_neg = jnp.maximum(-w, 0.0)           # abs(negative parts)

    sum_pos = jnp.sum(w_pos * fshift)
    sum_neg = jnp.sum(w_neg * fshift)
    s = sum_pos - sum_neg                  # signed sum

    s = jnp.clip(s, 1e-300, None)
    return jnp.log(s) + m + L * jnp.log(q)

# -----------------
# Loss & training step
# -----------------
def loss_and_aux(params, X_gd, key,
                 X, beta, weight_decay, J_mask,
                 run_bq, w_bc, L, q):
    h, J = params
    E_data = vmap(energy, in_axes=(0, None, None, None))(X_gd, h, J, J_mask)
    if run_bq:
        logZ = logZ_bc_on_pool(h, J, beta, X, w_bc, J_mask, L, q)
    else:
        logZ = logZ_mc_full_pool(h, J, beta, X, J_mask, L, q)
    nll = jnp.mean(beta * E_data + logZ)
    Jeff = project_J(J, J_mask)
    l2 = jnp.sum(h**2) + jnp.sum(Jeff**2)
    loss = nll + weight_decay * l2
    return loss, {"logZ": logZ, "nll": nll}

def make_update_fn(X, J_mask, optimizer, run_bq, w_bc, L, q):
    @partial(jax.jit, static_argnums=(4,))
    def update_step(params, opt_state, X_gd, key, beta, weight_decay):
        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, X_gd, key,
            X, beta, weight_decay, J_mask,
            run_bq, w_bc, L, q
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        h_new, J_new = optax.apply_updates(params, updates)
        J_new = project_J(J_new, J_mask)
        params = (h_new, J_new)
        return params, opt_state, loss, aux
    return update_step

# -----------------
# Single experiment (now accepts shared h0,J0)
# -----------------
def run_single_experiment(seed, L, q, p, n_gd, n, run_bq, beta, lr, weight_decay, num_steps,
                          h0=None, J0=None):
    print(f"\n--- Seed {seed} | L={L}, q={q}, n(BQ or MC)={n},  n(GD) = {n_gd}, run_bq={run_bq} ---")

    _, X_gd = data_preprocess(p, q, L, n_gd, seed=seed)

    key = jax.random.PRNGKey(seed)
    key, k_pool, k_init = jax.random.split(key, 3)

    # Pool used for MC/BQ logZ
    X_int = jax.random.randint(k_pool, (n, L), minval=0, maxval=q)
    X = jax.nn.one_hot(X_int, num_classes=q, dtype=jnp.float64)

    # If BQ: precompute weights on the SAME pool
    if run_bq:
        lambda_ = 0.3
        w_bc = precompute_bc_weights(X, lambda_)
    else:
        w_bc = None

    # Mask
    J_mask = adjacency_mask_1d(L)

    # ===== Params (shared if provided) =====
    if (h0 is not None) and (J0 is not None):
        h = jnp.array(h0, copy=True)
        J = jnp.array(J0, copy=True)
    else:
        kh, kJ = jax.random.split(k_init)
        h = jax.random.normal(kh, (L, q), dtype=jnp.float64) * 0.01
        J = jax.random.normal(kJ, (L, L, q, q), dtype=jnp.float64) * 0.01

    J = project_J(J, J_mask)
    params = (h, J)

    # Optimizer & step fn
    optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
    opt_state = optimizer.init(params)
    update_step = make_update_fn(X, J_mask, optimizer, run_bq, w_bc, L, q)

    # Training loop
    for step in range(num_steps):
        key, kstep = jax.random.split(key)
        params, opt_state, loss, aux = update_step(params, opt_state, X_gd, kstep, beta, weight_decay)
        if step % 20 == 0:
            h_curr, J_curr = params
            h_norm = jnp.linalg.norm(h_curr)
            J_norm = jnp.linalg.norm(J_curr)
            Jeff_norm = jnp.linalg.norm(project_J(J_curr, J_mask))
            print(f"step {step:4d} | loss {loss:.6f} | nll {aux['nll']:.6f} | logZ {aux['logZ']:.6f} | "
                  f"‖h‖={h_norm:.6f} | ‖J‖={J_norm:.6f} | ‖Jeff‖={Jeff_norm:.6f}")

    h_final, J_final = params

    # Diagnostics - KSD^2
    lam = median_heuristic_lambda_hamming(X_gd, key=jax.random.PRNGKey(0))
    ksd2 = compute_discrete_ksd(X_gd, h_final, J_final, J_mask, beta, lam, n_samples=min(256, X_gd.shape[0]))

    # Diagnostics - NLL & logZ (exact enumeration)
    logZ_final, nll_final = evaluate_logZ_nll_exact(params, X_gd, beta, J_mask)

    return float(np.asarray(jnp.abs(ksd2))), logZ_final, nll_final, params, J_mask, aux

# -----------------
# Multi-run driver (now shares h0,J0 between MC & BQ)
# -----------------
def run_many(n_list, seeds, L, q, p, n_gd, beta, lr, weight_decay, num_steps):
    summary = {("MC", n): {"nll": [], "logZ": [], "ksd": []} for n in n_list}
    summary.update({("BQ", n): {"nll": [], "logZ": [], "ksd": []} for n in n_list})

    for n in n_list:
        for s in seeds:
            # --- Shared initials for this seed ---
            h0, J0 = init_params_from_seed(s, L, q, scale=0.01)

            # MC (uses shared h0,J0)
            ksd_mc, logZ_final_mc, nll_final_mc, _, _, _ = run_single_experiment(
                seed=s, L=L, q=q, p=p, n_gd=n_gd, n=n, run_bq=False,
                beta=beta, lr=lr, weight_decay=weight_decay, num_steps=num_steps,
                h0=h0, J0=J0
            )
            summary[("MC", n)]["nll"].append(float(nll_final_mc))
            summary[("MC", n)]["logZ"].append(float(logZ_final_mc))
            summary[("MC", n)]["ksd"].append(abs(float(ksd_mc)))

            # BQ (same seed and SAME h0,J0)
            ksd_bq, logZ_final_bq, nll_final_bq, _, _, _ = run_single_experiment(
                seed=s, L=L, q=q, p=p, n_gd=n_gd, n=n, run_bq=True,
                beta=beta, lr=lr, weight_decay=weight_decay, num_steps=num_steps,
                h0=h0, J0=J0
            )
            summary[("BQ", n)]["nll"].append(float(nll_final_bq))
            summary[("BQ", n)]["logZ"].append(float(logZ_final_bq))
            summary[("BQ", n)]["ksd"].append(abs(float(ksd_bq)))

    # print aggregated results
    print("\n=== Multi-seed summary (mean ± std) ===")
    for n in n_list:
        for meth in ["MC", "BQ"]:
            nll = np.array(summary[(meth, n)]["nll"])
            logZ = np.array(summary[(meth, n)]["logZ"])
            ksd = np.array(summary[(meth, n)]["ksd"])
            print(f"{meth} | n={n:4d} | "
                  f"NLL={nll.mean():.4f}±{nll.std(ddof=1):.4f} | "
                  f"logZ={logZ.mean():.4f}±{logZ.std(ddof=1):.4f} | "
                  f"KSD={ksd.mean():.4e}±{ksd.std(ddof=1):.4e}")
    return summary

def _true_nll_nats(L, p):
    return float(L * (-(2*p*np.log(p) + (1-2*p)*np.log(1-2*p))))

def plot_ksd_nll(summary, n_list, L, p,
                 main_path="ksd_nll_main.pdf",
                 legend_path="ksd_nll_legend.pdf"):

    n = np.array(n_list, dtype=float)

    def collect(meth, metric):
        return np.array([np.mean(summary[(meth, int(nn))][metric]) for nn in n_list])

    ksd_mc = collect("MC", "ksd")
    ksd_bq = collect("BQ", "ksd")
    nll_mc = collect("MC", "nll")
    nll_bq = collect("BQ", "nll")

    true_nll = _true_nll_nats(L, p)

    # -------- main figure (no legend) --------
    fig, ax1 = plt.subplots(figsize=(10.0, 6.0))

    # Left: KSD^2 (solid)
    ax1.plot(n, ksd_mc, 'k-o', label='KSD$^2$ (MC)')      # MC KSD^2
    ax1.plot(n, ksd_bq, 'b-s', label='KSD$^2$ (BayesSum)')        # BQ KSD^2
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('KSD$^2$')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

    # Right: NLL (dashed)
    ax2 = ax1.twinx()
    ax2.plot(n, nll_mc, 'k--o', label='NLL (MC)')       # MC NLL
    ax2.plot(n, nll_bq, 'b--s', label='NLL (BayesSum)')       # BQ NLL
    ax2.axhline(true_nll, color='gray', linestyle=':', label='Optimal NLL')  # True NLL
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('NLL')

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, loc='best', fontsize=26)

    plt.tight_layout()
    plt.savefig('ksd_nll.pdf')
    plt.close(fig)

if __name__ == "__main__":
    n_list = [10, 50, 100, 250, 500, 1000]
    seeds  = list(range(5))            # run 5 seeds
    L = 15
    q = 3
    p = 0.4
    n_gd = 1000
    beta = 1/2.269
    lr = 1e-3
    weight_decay = 5e-3
    num_steps = 1000

    summary = run_many(n_list, seeds, L, q, p, n_gd, beta, lr, weight_decay, num_steps)
    plot_ksd_nll(summary, n_list, L, p, main_path="ksd_nll_ksd.pdf", legend_path="ksd_nll_ksd_legend.pdf")
