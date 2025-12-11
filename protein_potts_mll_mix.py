import os, time
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import optax
import matplotlib.pyplot as plt

# ---------- Matplotlib style ----------
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

# ================================================================
# Utilities: params, masks, energy, data
# ================================================================
def init_params_from_seed(seed, L, q, scale=0.01): 
    key = jax.random.PRNGKey(seed)
    k_h, k_J = jax.random.split(key)
    h0 = scale * jax.random.normal(k_h, (L, q), dtype=jnp.float64)
    J0 = scale * jax.random.normal(k_J, (L, L, q, q), dtype=jnp.float64)
    return h0, J0

## Modifying J such that we allow nearest-neighbor couplings on this 1D chain
def adjacency_mask_1d(L):
    M = jnp.zeros((L, L), dtype=jnp.float64)
    idx = jnp.arange(L - 1)
    M = M.at[idx, idx + 1].set(1.0)
    M = M.at[idx + 1, idx].set(1.0)
    return M[:, :, None, None]  # (L,L,1,1)

@jit
def project_J(J, J_mask):
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    return Jsym * J_mask

@jit
def energy(x, h, J, J_mask):
    Jeff = project_J(J, J_mask)
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)  # pairwise - 0.5 is multiplied to avoid double counting
    e_h = jnp.einsum('ik,ik->', x, h)                  # field
    return -(e_J + e_h)

def generate_categorical_sequence(key, p, L, q):
    if q != 3:
        raise ValueError("generate_categorical_sequence demo expects q=3.")
    probs = jnp.array([p, p, 1 - 2 * p], dtype=jnp.float64)
    keys = jax.random.split(key, L)
    seq = [jax.random.choice(k, q, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)

# Generating sequences and their onehot encodings
def data_preprocess(p, q, L, num_sequences, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_sequences)
    sequences = [generate_categorical_sequence(k, p, L, q=q) for k in keys]
    data_onehot = jax.nn.one_hot(jnp.stack(sequences), num_classes=q, dtype=jnp.float64)  # (N,L,q)
    return sequences, data_onehot

# =========================================================================
# KSD² diagnostic with exp-Hamming Kernel with fixed X_eval + global lambda
# =========================================================================
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

def base_exp_hamming_kernel_flat(x_flat, y_flat, lam, L):
    dot = jnp.dot(x_flat, y_flat)
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
    k_xy    = base_exp_hamming_kernel_flat(xf, yf, lam, L)
    k_x_y_n = vmap(base_exp_hamming_kernel_flat, in_axes=(None, 0, None, None))(xf, ynf, lam, L)
    k_x_n_y = vmap(base_exp_hamming_kernel_flat, in_axes=(0, None, None, None))(xnf, yf, lam, L)
    k_nn    = vmap(vmap(base_exp_hamming_kernel_flat, in_axes=(0, None, None, None)),
                   in_axes=(None, 0, None, None))(xnf, ynf, lam, L)
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

def median_hamming_lambda(X_onehot): # Find lambda value for ksd evaluation
    Lh = X_onehot.shape[1]
    Dmatch = jnp.einsum("ilq,jlq->ij", X_onehot, X_onehot)
    H = Lh - Dmatch # Calculate hamming distance for X_onehot
    tri = jnp.triu_indices(H.shape[0], 1)
    med = jnp.median(H[tri]) # Find median value
    return float(jnp.log(2.0) / jnp.clip(med, 1e-8, None)) #To make median distance to have K(x,y) = 0.5 

# ================================================================
# Mixture proposal Q_MIX (for ISMC / BQ-importance)
# ================================================================
@partial(jit, static_argnames=['q'])
def pseudo_product_pi(h, J, ref_config, beta, q):
    ref_onehot = jax.nn.one_hot(ref_config, num_classes=q, dtype=jnp.float64)
    J_term = jnp.einsum('l j a b, j b -> l a', J, ref_onehot)
    logits = beta * (h + J_term)
    logits = logits - jnp.max(logits, axis=1, keepdims=True) #For numerical stability
    return jax.nn.softmax(logits, axis=1)

@partial(jit, static_argnames=['n_samples'])
def sample_from_product_pi(key, pi, n_samples): #Based on pseudo_product_pi, we sample randomly using p = probs for n_samples
    L, q = pi.shape
    keys = jax.random.split(key, L)
    def draw(k, probs):
        return jax.random.choice(k, q, shape=(n_samples,), p=probs).astype(jnp.int32)
    return jnp.array([draw(k, pi[l]) for l,k in enumerate(keys)]).T

@jit
def log_q_single(x_int, pi):
    L = x_int.shape[0]
    return jnp.sum(jnp.log(jnp.clip(pi[jnp.arange(L), x_int], 1e-30, None)))

@jit
def log_q_mix(x_int, all_pis):
    M = all_pis.shape[0]
    log_probs_per_component = vmap(log_q_single, in_axes=(None, 0))(x_int, all_pis)
    return jax.scipy.special.logsumexp(log_probs_per_component) - jnp.log(M)

v_log_q_mix = vmap(log_q_mix, in_axes=(0, None))

@jit
def mu_k_exp_hamming_under_Q_single(y_int, pi, lam): # Closed form KME for pseudo-like model for each Q
    L = y_int.shape[0]
    gather = pi[jnp.arange(L), y_int]
    term = gather * (jnp.exp(lam) - 1.0) + 1.0
    return jnp.exp(-lam * L) * jnp.prod(term)

@jit
def z_q_mix(y_int, all_pis, lam): # Closed form KME ofor the Q_mix
    z_per_component = vmap(mu_k_exp_hamming_under_Q_single, in_axes=(None, 0, None))(y_int, all_pis, lam)
    return jnp.mean(z_per_component)

v_z_q_mix = vmap(z_q_mix, in_axes=(0, None, None))

def gram_exp_hamming_on_ints(X_int, lam):
    n, L = X_int.shape
    def k_ij(xi, xj):
        matches = jnp.sum(xi == xj)
        return jnp.exp(-lam * L + lam * matches)
    return vmap(lambda xi: vmap(lambda xj: k_ij(xi, xj))(X_int))(X_int)

def precompute_bc_importance_MIX(X_nodes_int, all_pis, lam):
    K = gram_exp_hamming_on_ints(X_nodes_int, lam)
    K = K + 1e-4 * jnp.eye(K.shape[0], dtype=jnp.float64)
    cholK, _ = cho_factor(K, lower=True)
    z_Q = v_z_q_mix(X_nodes_int, all_pis, lam)
    return cholK, z_Q

# ================================================================
# Importance-BayesSum and IS-MC
# ================================================================
@partial(jit, static_argnames=['L','q'])
def logZ_bayesum_importance(h, J, beta, X_nodes_int, log_q_X, cholK, z_Q, J_mask, L, q, lam):
    X_nodes = jax.nn.one_hot(X_nodes_int, num_classes=q, dtype=jnp.float64)
    E = vmap(energy, in_axes=(0, None, None, None))(X_nodes, h, J, J_mask)

    # New function g
    logf = -beta * E
    logw = -L * jnp.log(q) - log_q_X
    g = jnp.exp(logf + logw)

    # BQ mean
    alpha = cho_solve((cholK, True), g)
    I_hat = jnp.dot(z_Q, alpha)
    I_hat = jnp.clip(I_hat, 1e-300, None)

    return jnp.log(I_hat) + L * jnp.log(q)

@partial(jit, static_argnames=['L','q'])
def logZ_is_mc_on_Q(h, J, beta, X_nodes_int, log_q_X, J_mask, L, q):
    X_nodes = jax.nn.one_hot(X_nodes_int, num_classes=q, dtype=jnp.float64)
    E    = vmap(energy, in_axes=(0, None, None, None))(X_nodes, h, J, J_mask)

    #Importance Sampling Trick
    logf = -beta * E
    logw = -L * jnp.log(q) - log_q_X
    logg = logw + logf

    #Logsum Trick
    m = jnp.max(logg)
    log_mean_g = jnp.log(jnp.clip(jnp.mean(jnp.exp(logg - m)), 1e-300)) + m
    logZ = log_mean_g + L * jnp.log(q)

    return logZ

# ================================================================
# Exact enumeration (optional diagnostics)
# ================================================================
def _ints_to_base_q(ints: jnp.ndarray, q: int, L: int) -> jnp.ndarray:
    def one_num_to_digits(n):
        digs, x = [], n
        for _ in range(L):
            digs.append(x % q); x = x // q
        return jnp.array(digs[::-1], dtype=jnp.int32)
    return vmap(one_num_to_digits)(ints)

def logZ_exact_enumeration(h, J, J_mask, beta, L: int, q: int, chunk: int = 200_000) -> float:
    total = int(q ** L)
    logZ = -jnp.inf
    for start in range(0, total, chunk):
        m = min(chunk, total - start)
        idx = jnp.arange(start, start + m, dtype=jnp.int64)
        digits = _ints_to_base_q(idx, q, L)
        X_chunk = jax.nn.one_hot(digits, num_classes=q, dtype=jnp.float64)
        E = vmap(energy, in_axes=(0, None, None, None))(X_chunk, h, J, J_mask)
        logf = -beta * E
        mshift = jnp.max(logf)
        chunk_lse = mshift + jnp.log(jnp.clip(jnp.sum(jnp.exp(logf - mshift)), 1e-300))
        logZ = jnp.logaddexp(logZ, chunk_lse)
    return float(logZ)

def evaluate_logZ_nll_exact(params, X_eval, beta, J_mask): # NLL = -LL = data_term + logZ
    h, J = params
    L, q = X_eval.shape[1], X_eval.shape[2]
    E_data = vmap(energy, in_axes=(0, None, None, None))(X_eval, h, J, J_mask)
    data_term = jnp.mean(beta * E_data) 
    logZ = logZ_exact_enumeration(h, J, J_mask, beta, L, q)
    return float(logZ), float(data_term + logZ)

# ================================================================
# Loss / training step (choose estimator per-call)
# ================================================================
def loss_and_aux(params, X_gd, *,
                 estimator,
                 ismcp_pack,
                 bq_pack,
                 lam, beta, weight_decay, J_mask, L, q):
    h, J = params
    E_data = vmap(energy, in_axes=(0, None, None, None))(X_gd, h, J, J_mask)

    if estimator == 'ISMC':
        X_nodes_int, log_q_X = ismcp_pack
        logZ = logZ_is_mc_on_Q(h, J, beta, X_nodes_int, log_q_X, J_mask, L, q)
    elif estimator == 'BQ':
        cholK, z_Q, X_nodes_int, log_q_X = bq_pack
        logZ = logZ_bayesum_importance(h, J, beta, X_nodes_int, log_q_X, cholK, z_Q, J_mask, L, q, lam)
    else:
        raise ValueError("Unknown estimator")

    nll = jnp.mean(beta * E_data + logZ)
    Jeff = project_J(J, J_mask)
    l2 = jnp.sum(h**2) + jnp.sum(Jeff**2)
    loss = nll + weight_decay * l2
    return loss, {"logZ": logZ, "nll": nll}

def make_update_fn(optimizer, L, q):
    @partial(jax.jit, static_argnames=['estimator'])
    def update_step(params, opt_state, X_gd, *,
                    estimator, ismcp_pack, bq_pack,
                    lam, beta, weight_decay, J_mask):
        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, X_gd,
            estimator=estimator,
            ismcp_pack=ismcp_pack,
            bq_pack=bq_pack,
            lam=lam, beta=beta, weight_decay=weight_decay, J_mask=J_mask, L=L, q=q
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        h_new, J_new = optax.apply_updates(params, updates)
        J_new = project_J(J_new, J_mask)
        params = (h_new, J_new)
        return params, opt_state, loss, aux
    return update_step

# ================================================================
# Single experiment (with fixed eval + global KSD lambda)
# ================================================================
def run_single_experiment(seed, *,
                          L, q, p, n_gd,
                          n_nodes,
                          estimator,
                          beta, lam, ksd_lam,
                          lr, weight_decay, num_steps,
                          X_eval_fixed,             # <-- fixed evaluation set
                          h0, J0,
                          n_mix_components):
    print(f"\n--- Seed {seed} | L={L}, q={q}, n_nodes={n_nodes}, estimator={estimator} ---")

    # Data
    _, X_gd = data_preprocess(p, q, L, n_gd, seed=seed)
    X_gd_int = jnp.argmax(X_gd, axis=-1).astype(jnp.int32)
    J_mask = adjacency_mask_1d(L)

    # Params
    h = jnp.array(h0, copy=True)
    J = jnp.array(J0, copy=True)
    J = project_J(J, J_mask)
    params = (h, J)

    key = jax.random.PRNGKey(seed)
    k_pool_init, k_ref = jax.random.split(key, 2)

    ismcp_pack   = None
    bq_pack      = None
    current_lam = lam
    assert n_nodes % n_mix_components == 0, "n_nodes must be a multiple of n_mix_components."

    # How many samples do we want to draw from each mix?
    n_per_mix = max(1, n_nodes // n_mix_components)

    # Choosing random reference points to create pseudo-like model
    ref_indices = jax.random.choice(k_ref, n_gd, shape=(n_mix_components,), replace=False)
    ref_configs = X_gd_int[ref_indices]

    v_pseudo = vmap(pseudo_product_pi, in_axes=(None, None, 0, None, None))
    all_pis = v_pseudo(h0, project_J(J0, J_mask), ref_configs, beta, q)

    keys_for_sampling = jax.random.split(k_pool_init, n_mix_components)
    sample_fn = partial(sample_from_product_pi, n_samples=n_per_mix)
    X_nodes_int_list = vmap(sample_fn)(keys_for_sampling, all_pis)
    X_nodes_int = jnp.reshape(X_nodes_int_list, (n_nodes, L))

    log_q_X = v_log_q_mix(X_nodes_int, all_pis)
    if estimator == 'ISMC':
        ismcp_pack = (X_nodes_int, log_q_X)
    else:
        cholK, z_Q = precompute_bc_importance_MIX(X_nodes_int, all_pis, current_lam)
        bq_pack = (cholK, z_Q, X_nodes_int, log_q_X)

    # Optimizer & update
    optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
    opt_state = optimizer.init(params)
    update_step = make_update_fn(optimizer, L, q)

    start_time = time.perf_counter()

    for step in range(num_steps):

        if (step > 0) and (step % 100 == 0):
            # refresh X_nodes for every 100 steps
            k_ref_step = jax.random.PRNGKey(seed * 10_000 + step)
            k_sam_step = jax.random.PRNGKey(seed * 20_000 + step)
            ref_indices = jax.random.choice(k_ref_step, n_gd, shape=(n_mix_components,), replace=False)
            ref_configs = X_gd_int[ref_indices]

            v_pseudo = vmap(pseudo_product_pi, in_axes=(None, None, 0, None, None))
            h_curr, J_curr = params
            all_pis = v_pseudo(h_curr, project_J(J_curr, J_mask), ref_configs, beta, q)
            keys_for_sampling = jax.random.split(k_sam_step, n_mix_components)
            sample_fn = partial(sample_from_product_pi, n_samples=n_per_mix)

            X_nodes_int_list = vmap(sample_fn)(keys_for_sampling, all_pis)
            X_nodes_int = jnp.reshape(X_nodes_int_list, (n_nodes, L))

            log_q_X = v_log_q_mix(X_nodes_int, all_pis)

            if estimator == 'ISMC':
                ismcp_pack = (X_nodes_int, log_q_X)
            else:
                cholK, z_Q = precompute_bc_importance_MIX(X_nodes_int, all_pis, current_lam)
                bq_pack = (cholK, z_Q, X_nodes_int, log_q_X)

        params, opt_state, loss, aux = update_step(
            params, opt_state, X_gd,
            estimator=estimator,
            ismcp_pack=ismcp_pack,
            bq_pack=bq_pack,
            lam=current_lam,
            beta=beta, weight_decay=weight_decay, J_mask=J_mask
        )

        if step % 50 == 0:
            h_curr, J_curr = params
            print(f"step {step:4d} | loss {float(loss):.6f} | nll {float(aux['nll']):.6f} | logZ {float(aux['logZ']):.6f} | "
                  f"‖h‖={float(jnp.linalg.norm(h_curr)):.4f} | ‖J‖={float(jnp.linalg.norm(J_curr)):.4f} | lam={current_lam:.4f}")

    elapsed = time.perf_counter() - start_time
    h_final, J_final = params

    # KSD: fixed X_eval set + global lam
    ksd2 = compute_discrete_ksd(
        X_eval_fixed, h_final, J_final, J_mask, beta, ksd_lam,
        n_samples=X_eval_fixed.shape[0]
    )

    logZ_exact, nll_exact = evaluate_logZ_nll_exact((h_final, J_final), X_eval_fixed, beta, J_mask)
    return float(np.asarray(jnp.abs(ksd2))), float(logZ_exact), float(nll_exact), float(elapsed)

# ================================================================
# Multi-run drivers (separate n_lists for ISMC vs BQ)
# ================================================================
def _run_many_for_method(method, n_list, seeds, *,
                         L, q, p, n_gd, beta, lam, ksd_lam, lr, weight_decay, num_steps,
                         n_mix_components, X_eval_fixed):
    summary = {(method, int(n)): {"nll": [], "logZ": [], "ksd": [], "time": []} for n in n_list}
    for n in n_list:
        for s in seeds:
            h0, J0 = init_params_from_seed(s, L, q, scale=0.01)
            ksd, logZ, nll, t = run_single_experiment(
                s, L=L, q=q, p=p, n_gd=n_gd, n_nodes=n,
                estimator=('ISMC' if method == 'ISMC' else 'BQ'),
                beta=beta, lam=lam, ksd_lam=ksd_lam,
                lr=lr, weight_decay=weight_decay, num_steps=num_steps,
                X_eval_fixed=X_eval_fixed,
                h0=h0, J0=J0, n_mix_components=n_mix_components
            )
            summary[(method, int(n))]["nll"].append(float(nll))
            summary[(method, int(n))]["logZ"].append(float(logZ))
            summary[(method, int(n))]["ksd"].append(abs(float(ksd)))
            summary[(method, int(n))]["time"].append(float(t))
    return summary

def run_many_separate(
    n_list_ismc, n_list_bq, seeds, *,
    L, q, p, n_gd, beta, lam, ksd_lam, lr, weight_decay, num_steps, n_mix_components,
    X_eval_fixed
):
    summary_ismc = {}
    summary_bq   = {}

    if n_list_ismc is not None and len(n_list_ismc) > 0:
        summary_ismc = _run_many_for_method(
            "ISMC", n_list_ismc, seeds,
            L=L, q=q, p=p, n_gd=n_gd, beta=beta, lam=lam, ksd_lam=ksd_lam,
            lr=lr, weight_decay=weight_decay, num_steps=num_steps,
            n_mix_components=n_mix_components, X_eval_fixed=X_eval_fixed
        )

    if n_list_bq is not None and len(n_list_bq) > 0:
        summary_bq = _run_many_for_method(
            "BQ", n_list_bq, seeds,
            L=L, q=q, p=p, n_gd=n_gd, beta=beta, lam=lam, ksd_lam=ksd_lam,
            lr=lr, weight_decay=weight_decay, num_steps=num_steps,
            n_mix_components=n_mix_components, X_eval_fixed=X_eval_fixed
        )

    return summary_ismc, summary_bq

# ================================================================
# Save / Load (per-method)
# ================================================================
def save_summary_simple(filename, summary, n_list, method):
    n_list = np.asarray(n_list, dtype=int)
    metrics = ("nll", "logZ", "ksd", "time")
    seeds_count = len(summary[(method, int(n_list[0]))][metrics[0]])
    out = {"n_list": n_list, "seeds_count": np.array([seeds_count], dtype=int), "method": np.array([method])}
    for metric in metrics:
        mat = np.zeros((len(n_list), seeds_count), dtype=float)
        for i, n in enumerate(n_list):
            vals = summary[(method, int(n))][metric]
            if len(vals) != seeds_count:
                raise ValueError(f"Unequal seeds for {method}, n={n}: got {len(vals)}, expected {seeds_count}")
            mat[i, :] = np.asarray(vals, dtype=float)
        out[f"{method}_{metric}"] = mat
    np.savez(filename, **out)
    print(f"Saved {filename}")

def load_summary_simple(filename):
    Z = np.load(filename, allow_pickle=True)
    n_list = Z["n_list"].astype(int)
    seeds_count = int(Z["seeds_count"][0])
    method = str(Z["method"][0])
    metrics = ("nll", "logZ", "ksd", "time")
    summary = {}
    for i, n in enumerate(n_list):
        summary[(method, int(n))] = {}
        for metric in metrics:
            key = f"{method}_{metric}"
            if key not in Z.files:
                raise ValueError(f"Missing {key} in {filename}")
            mat = Z[key]
            summary[(method, int(n))][metric] = mat[i, :].tolist()
    return summary, n_list, method

# ================================================================
# Plotting (tighter bars + no time filter)
# ================================================================
def _matrix(summary, n_list, method, metric):
    return np.array([summary[(method, int(n))][metric] for n in n_list], dtype=float)

def plot_ksd_nll_vs_time_sorted(
        n_list_bq, n_list_mc, summary_bq, summary_mc,
        methods=("ISMC","BQ"),
        true_value=15.823802519792164,
        ksd_metric="ksd", err_metric="nll",
        save_path="results", fname="Potts_ksd_nll_vs_time_sorted.pdf",
        styles=None, smooth=False, ci_level=95):
    os.makedirs(save_path, exist_ok=True)
    if styles is None:
        styles = {
            "ISMC": {"color": "g", "marker": "^", "label": "MC"},
            "BQ":   {"color": "b", "marker": "s", "label": "BayesSum"},
        }
    z = 1.0 if ci_level == 68 else 1.96

    def _matrix_local(summary, n_list, method, metric):
        return np.array([summary[(method, int(n))][metric] for n in n_list], dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 3))
    for meth in methods:
        if meth == "ISMC":
            n_list = n_list_mc
            summary = summary_mc
        else:
            n_list = n_list_bq
            summary = summary_bq
        if n_list is None or len(n_list) == 0:
            continue

        M_time = _matrix_local(summary, n_list, meth, "time")
        M_ksd = _matrix_local(summary, n_list, meth, ksd_metric)
        M_nll = _matrix_local(summary, n_list, meth, err_metric)

        x_mean = M_time.mean(axis=1)  # seconds
        order = np.argsort(x_mean)
        x = x_mean[order]
        y1_mean = M_ksd.mean(axis=1)[order]
        y1_se   = (M_ksd.std(axis=1, ddof=1) / np.sqrt(M_ksd.shape[1]))[order]

        st = styles[meth]
        ax1.errorbar(
            x, y1_mean, yerr=z*y1_se,
            fmt='-', color=st["color"], marker=st["marker"], markersize=7,
            elinewidth=1.5, capsize=4, label=rf"KSD$^2$ ({st['label']})",
        )

    ax1.set_xlabel("Time", fontsize=32)
    ax1.set_ylabel(r"KSD$^2$", color="black", fontsize=32)
    ax1.set_yscale("log")
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    h1, l1 = ax1.get_legend_handles_labels()
    # ax1.legend(h1, l1, loc="best", fontsize=32)
    plt.tight_layout()
    out = os.path.join(save_path, fname)
    plt.savefig(out, format="pdf")
    print(f"Saved figure to {out}")

    h1, l1 = ax1.get_legend_handles_labels()

    fig_legend, ax_legend = plt.subplots(figsize=(6, 1))  # adjust size as needed
    ax_legend.axis('off')  # no axes
    ax_legend.legend(h1, l1, ncol=2, loc='center', fontsize=32, frameon=False)

    plt.savefig(os.path.join(save_path, "potts_mll_legends.pdf"), bbox_inches='tight')
    plt.close(fig_legend)


def plot_ksd_vs_n(
        n_list_bq, n_list_mc, summary_bq, summary_mc,
        methods=("ISMC","BQ"),
        save_path="results", fname="Potts_ksd_vs_n.pdf",
        styles=None, ci_level=95):
    """
    Plots mean KSD^2 ± z*SE across seeds vs number of nodes n, for each method present.
    Relies on summary structure: summary[(method, int(n))]["ksd"] = list over seeds.
    """
    os.makedirs(save_path, exist_ok=True)
    if styles is None:
        styles = {
            "ISMC": {"color": "g", "marker": "^", "label": "IS-MC"},
            "BQ":   {"color": "b", "marker": "s", "label": "BayesSum"},
            "MC":   {"color": "k", "marker": "o", "label": "MC"},
        }
    z = 1.0 if ci_level == 68 else 1.96

    def _matrix_local(summary, n_list, method, metric):
        return np.array([summary[(method, int(n))][metric] for n in n_list], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 3))

    for meth in methods:
        if meth == "ISMC":
            n_list = n_list_mc
            summ   = summary_mc
        else:
            n_list = n_list_bq
            summ   = summary_bq



        M_ksd   = _matrix_local(summ, n_list, meth, "ksd")  # shape: (len(n_list), n_seeds)
        y_mean  = M_ksd.mean(axis=1)
        y_se    = M_ksd.std(axis=1, ddof=1) / np.sqrt(M_ksd.shape[1])

        st = styles[meth]
        ax.errorbar(
            n_list, y_mean, yerr=z*y_se,
            fmt='-', marker=st["marker"], color=st["color"], capsize=4,
            elinewidth=1.5, markersize=7, label=rf"KSD$^2$ ({st['label']})"
        )

    ax.set_xlabel("Number of Points", fontsize=32)
    ax.set_ylabel(r"KSD$^2$", fontsize=32)
    ax.set_yscale("log")
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    # ax.legend(loc="best", fontsize=28)
    plt.tight_layout()
    out = os.path.join(save_path, fname)
    plt.savefig(out, format="pdf")
    print(f"Saved figure to {out}")


# ================================================================
# Main (scouting run with tighter KSD bars)
# ================================================================
if __name__ == "__main__":
    # Experiment settings
    L = 15
    q = 3
    p = 0.4
    n_gd = 2000
    beta = 1/2.269
    lr = 1e-3
    weight_decay = 5e-3
    num_steps = 1000
    n_mix_components = 10  

    # Different n_lists per method - n values must be divisible by n_mix_components!
    n_list_ismc =  [20, 50, 80, 100,  200,  500, 1000] 
    n_list_bq   = [50, 100, 120, 160, 200]


    # # Fixed evaluation set for KSD across ALL runs/seeds
    # _, X_eval_global = data_preprocess(p, q, L, 2000, seed=123)
    # lam_ksd_global = median_hamming_lambda(X_eval_global)

    # seeds = list(range(30))  

    # summary_ismc, summary_bq = run_many_separate(
    #     n_list_ismc, n_list_bq, seeds,
    #     L=L, q=q, p=p, n_gd=n_gd, beta=beta, lam=0.1104, ksd_lam=0.1104,
    #     lr=lr, weight_decay=weight_decay, num_steps=num_steps,
    #     n_mix_components=n_mix_components, X_eval_fixed=X_eval_global
    # )
    # save_summary_simple("ismc_results.npz", summary_ismc, n_list_ismc, method="ISMC")
    # save_summary_simple("bq_results.npz",   summary_bq,   n_list_bq,   method="BQ")

    summary_ismc, n_list_ismc_loaded, _ = load_summary_simple("ismc_results.npz")
    summary_bq,   n_list_bq_loaded,   _ = load_summary_simple("bq_results.npz")


    os.makedirs("results", exist_ok=True)
    plot_ksd_nll_vs_time_sorted(
        n_list_bq=n_list_bq_loaded,
        n_list_mc=n_list_ismc_loaded[1:],
        summary_bq=summary_bq,
        summary_mc=summary_ismc,
        methods=("ISMC","BQ"),
        true_value=15.823802519792164,
        save_path="results",
        fname="Potts_ksd&NLL_vs_time_TIGHT.pdf",
        ci_level=95,  # 68% default (shorter bars)
    )

    summary_ismc, n_list_ismc_loaded, _ = load_summary_simple("ismc_results_full.npz")
    plot_ksd_vs_n(
	    n_list_bq=n_list_bq_loaded,
	    n_list_mc=n_list_ismc_loaded,
	    summary_bq=summary_bq,
	    summary_mc=summary_ismc,
	    methods=("ISMC","BQ"),
	    save_path="results",
	    fname="Potts_ksd_vs_n_TIGHT.pdf",
	    ci_level=95
	)
