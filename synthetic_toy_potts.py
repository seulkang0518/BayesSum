import time
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import optax
jax.config.update("jax_enable_x64", True)


# -------------
#  Data generator
# -------------
def generate_categorical_sequence(key, p, L):
    probs = jnp.array([p, p, 1 - 2 * p])
    keys = jax.random.split(key, L)
    seq = [jax.random.choice(k, 3, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)


def data_preprocess(p, q, L, num_sequences):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_sequences)
    sequences_true = [generate_categorical_sequence(k, p, L) for k in keys]
    sigma_onehot_synthetic = jax.nn.one_hot(jnp.stack(sequences_true), num_classes=q)
    return sequences_true, sigma_onehot_synthetic, L

#  BQ Set-up

@jit
def energy(x, h, J, J_mask):

    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)
    e_h = jnp.einsum('ik,ik->', x, h)

    return e_J + e_h

@jit
def kernel_embedding(lambda_, L, q):
    return ((1.0 + (q - 1.0) * jnp.exp(-lambda_)) / q) ** L


@jit
def gram_matrix(sigma_batch, lambda_):
    n, d, q = sigma_batch.shape
    sigma_flat = sigma_batch.reshape(n, -1)  # (n, d*q)
    def k(x1, x2):
        dot = jnp.dot(x1, x2)
        return jnp.exp(-lambda_ * d + lambda_ * dot)
    return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

# ====================
#  Pool-based Z estimators (MC and BQ)
# ====================

@jit
def logZ_from_pool(h, J, beta, Xi_onehot, key, J_mask, n_mc):
    """Monte Carlo estimate of log Z using a separate integration pool.
    Subsamples M_SUB rows each call via permutation (JAX-friendly).
    """
    M = Xi_onehot.shape[0]
    perm = jax.random.permutation(key, M)
    idx = perm[:n_mc]
    Xi_sub = Xi_onehot[idx]

    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_sub, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    Z_hat = jnp.mean(jnp.exp(logf - m))
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m


def precompute_bc_weights(Xi_onehot, lambda_):
    """Precompute BQ weights once on the pool using the correct kernel mean."""
    K = gram_matrix(Xi_onehot, lambda_) + 1e-8 * jnp.eye(Xi_onehot.shape[0])
    Lc, lower = cho_factor(K, lower=True)
    L, q = Xi_onehot.shape[1], Xi_onehot.shape[2]
    z = jnp.full((Xi_onehot.shape[0], 1), kernel_embedding(lambda_, L, q))
    w = cho_solve((Lc, lower), z)  # (M,1)
    return w



@jit
def logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask):
    """Bayesian cubature estimate of log Z using precomputed weights on pool."""
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_onehot, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    f_shift = jnp.exp(logf - m)[:, None]  # (M,1)
    Z_hat = (w_bc.T @ f_shift)[0, 0]
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

# ---- Sanity-check helper: MC on the *entire* pool (no subsampling) ----
@jit
def logZ_mc_full_pool(h, J, beta, Xi_onehot, J_mask):
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_onehot, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    Z_hat = jnp.mean(jnp.exp(logf - m))
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

# ====================
#  Loss and update (uses separate pool for Z)
# ====================

def loss_and_aux(params, Sigma_batch, key, Xi_onehot, w_bc, lambda_, beta, weight_decay, run_bq, J_mask):
    h, J = params

    # Positive/data term on the minibatch
    energies = jax.vmap(energy, in_axes=(0, None, None, None))(Sigma_batch, h, J, J_mask)

    # Partition function term on the independent pool
    if run_bq and (w_bc is not None):
        log_Z = logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask)
    else:
        key, zkey = jax.random.split(key)
        log_Z = logZ_from_pool(h, J, beta, Xi_onehot, zkey, J_mask)

    # NLL
    nll = -jnp.mean(-beta * energies - log_Z)

    # L2 on EFFECTIVE parameters only (optional; set weight_decay=0.0 to disable)
    # Recompute Jeff for regularization consistency
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    l2 = jnp.sum(h**2) + jnp.sum(Jeff**2)

    loss = nll + weight_decay * l2
    return loss, {"log_Z": log_Z}


def make_update_fn(optimizer, Xi_onehot, w_bc, J_mask):
    @partial(jax.jit, static_argnums=(4, 8))  # n_samples, run_bq
    def update_step(params, opt_state, Sigma, key, n_samples, lambda_, beta, weight_decay, run_bq):
        num_total = Sigma.shape[0]

        # Data minibatch
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, num_total, shape=(n_samples,), replace=False)
        Sigma_batch = Sigma[idx]

        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, Sigma_batch, key, Xi_onehot, w_bc, lambda_, beta, weight_decay, run_bq, J_mask
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux["log_Z"]

    return update_step


# KDSD 

def median_heuristic_gamma(samples, n_subset=500, key=None):
    n_total = samples.shape[0]
    if (key is not None) and (n_total > n_subset):
        idx = jax.random.choice(key, n_total, shape=(n_subset,), replace=False)
        samples_sub = samples[idx]
    else:
        samples_sub = samples if n_total <= n_subset else samples[:n_subset]  # deterministic

    n = samples_sub.shape[0]
    samples_flat = samples_sub.reshape(n, -1)
    norms_sq = jnp.sum(samples_flat**2, axis=1)
    dist_sq = norms_sq[:, None] - 2.0 * (samples_flat @ samples_flat.T) + norms_sq[None, :]
    tri = jnp.triu_indices(n, k=1)
    med_sq = jnp.median(dist_sq[tri])
    return 1.0 / (2.0 * jnp.clip(med_sq, 1e-8, None))

@partial(jit, static_argnames=['shift'])
def get_neighbor(x_i, shift):
    """Cyclically permutes a single one-hot vector."""
    return jnp.roll(x_i, shift=shift, axis=-1)

# Vectorize to get all L neighbors of a full sequence x
# For each position i in L, we create a neighbor by permuting only that position.
@partial(jit, static_argnames=['shift'])
def get_all_neighbors(x, shift):
    """Generates all L neighbors of a sequence x by cyclically permuting one position at a time."""
    # x has shape (L, q). We want to create L different versions of x.
    # The identity matrix will help us select one position at a time.
    L, q = x.shape
    eye = jnp.eye(L, dtype=x.dtype)[:, :, None] # (L, L, 1)

    # Create the permuted versions for each position
    x_permuted_rows = vmap(get_neighbor, in_axes=(0, None))(x, shift) # (L, q)

    # Use the identity matrix to select the original row or the permuted row
    # tile x to be (L, L, q) so we can mask it
    x_tiled = jnp.tile(x, (L, 1, 1)) # (L, L, q)

    # Create a tiled version of the permuted rows
    x_permuted_tiled = jnp.tile(x_permuted_rows, (L, 1, 1)) # (L, L, q)

    # When eye is 1, use the permuted version. When eye is 0, use the original.
    neighbors = x_tiled * (1-eye) + x_permuted_tiled * eye
    return neighbors

@jit
def discrete_model_score(x, h, J, J_mask, beta):
    """
    Computes the discrete difference score function s_p(x).
    s_p(x)_i = 1 - p(¬i x) / p(x) = 1 - exp(-beta * (E(¬i x) - E(x)))
    """
    # Create neighbors by cyclic permutation (0->1->2->0)
    # This corresponds to the operator ¬ in the paper.
    neighbors = get_all_neighbors(x, shift=1) # Shape (L, L, q)

    # Compute energy of original sample and all neighbors
    E_x = energy(x, h, J, J_mask)
    E_neighbors = vmap(energy, in_axes=(0, None, None, None))(neighbors, h, J, J_mask) # Shape (L,)

    # Compute the score vector
    score = 1.0 - jnp.exp(-beta * (E_neighbors - E_x))
    return score # Shape (L,)

@jit
def base_rbf_kernel(x_flat, y_flat, gamma):
    """A simple RBF kernel on flattened inputs."""
    # The squared Euclidean distance on one-hot vectors is proportional to the Hamming
    # distance, so this is consistent with the paper's recommendation.
    return jnp.exp(-gamma * jnp.sum((x_flat - y_flat)**2))

def discrete_ksd_u_p_term(x, y, h, J, J_mask, beta, gamma):
    """
    Computes the discrete KSD kernel κ_p(x, y) for a single pair of samples.
    This directly implements Equation 12 from Yang et al. (2018).
    """
    L, q = x.shape

    # --- 1. Get scores and neighbors ---
    score_x = discrete_model_score(x, h, J, J_mask, beta)
    score_y = discrete_model_score(y, h, J, J_mask, beta)

    # The adjoint operator Δ* uses the inverse permutation (0<-1<-2<-0).
    # This is achieved with a negative shift in jnp.roll.
    x_neighbors_inv = get_all_neighbors(x, shift=-1) # (L, L, q)
    y_neighbors_inv = get_all_neighbors(y, shift=-1) # (L, L, q)

    # --- 2. Flatten all inputs for the kernel ---
    x_flat = x.flatten()
    y_flat = y.flatten()
    # Flatten neighbors for vectorized kernel computation
    vmap_flatten = vmap(jnp.ravel) # <--- CORRECTED LINE
    x_neighbors_inv_flat = vmap_flatten(x_neighbors_inv) # (L, L*q)
    y_neighbors_inv_flat = vmap_flatten(y_neighbors_inv) # (L, L*q)

    # --- 3. Compute kernel terms ---
    k_xy = base_rbf_kernel(x_flat, y_flat, gamma)

    # Δ*_x' k(x, x') = k(x, x') - k(x, ¬x')
    k_x_y_neighbors = vmap(base_rbf_kernel, in_axes=(None, 0, None))(x_flat, y_neighbors_inv_flat, gamma)
    delta_y_k = k_xy - k_x_y_neighbors # Shape (L,)

    # Δ*_x k(x, x') = k(x, x') - k(¬x, x')
    k_x_neighbors_y = vmap(base_rbf_kernel, in_axes=(0, None, None))(x_neighbors_inv_flat, y_flat, gamma)
    delta_x_k = k_xy - k_x_neighbors_y # Shape (L,)

    # Δ*_{x_i, x'_j} k(x, x') = k(x,x') - k(x,¬j x') - k(¬i x,x') + k(¬i x,¬j x')
    # We need the trace, so we only compute for i=j.
    k_neighbors_neighbors = vmap(vmap(base_rbf_kernel, in_axes=(0, None, None)), in_axes=(None, 0, None))(
        x_neighbors_inv_flat, y_neighbors_inv_flat, gamma
    ) # Shape (L, L)

    L = x.shape[0]
    trace_term = (jnp.trace(k_neighbors_neighbors) - jnp.sum(k_x_y_neighbors) - jnp.sum(k_x_neighbors_y) + L * k_xy)

    # --- 4. Assemble the KSD kernel from Equation 12 ---
    t1 = jnp.dot(score_x, score_y) * k_xy
    t2 = -jnp.dot(score_x, delta_y_k)
    t3 = -jnp.dot(delta_x_k, score_y)
    t4 = trace_term

    return t1 + t2 + t3 + t4

@partial(jit, static_argnames=['n_samples'])
def compute_discrete_ksd(samples, h, J, J_mask, beta, gamma, n_samples=None):
    if n_samples is None:
        n_samples = samples.shape[0]
    u_p_vmapped = vmap(vmap(discrete_ksd_u_p_term, in_axes=(None, 0, None, None, None, None, None)),
                           in_axes=(0, None, None, None, None, None, None))
    u_p_matrix = u_p_vmapped(samples, samples, h, J, J_mask, beta, gamma)
    total_sum = jnp.sum(u_p_matrix) - jnp.sum(jnp.diag(u_p_matrix))
    ksd_sq = total_sum / (n_samples * (n_samples - 1))
    return ksd_sq

# ====================
#  Main
# ====================

def run_single_experiment(seed, n_bq, n_mc):
    """Runs the entire training and evaluation for a single random seed."""
    print(f"\n--- Running Experiment for Seed {seed} ---")

    # --- Data ---
    global beta
    p = 0.4
    q = 3
    max_len = 4
    num_sequences = 500
    sequences_true, sigma_onehot_synthetic, L = data_preprocess(p, q, max_len, num_sequences)

    # --- Integration pool (independent of data) ---
    key = jax.random.PRNGKey(seed)
    key, int_key = jax.random.split(key)
    Xi = jax.random.randint(int_key, (n_bq, L), minval=0, maxval=q)
    Xi_onehot = jax.nn.one_hot(Xi, num_classes=q)

    # --- Build mask for Setup A (constraints inside energy) ---
    J_mask = (1.0 - jnp.eye(L))[:, :, None, None]  # (L,L,1,1) 0 on diag, 1 elsewhere

    # --- Init params ---
    key, h_key, j_key = jax.random.split(key, 3)
    h_model = jax.random.normal(h_key, shape=(L, q)) * 0.01
    J_model = jax.random.normal(j_key, shape=(L, L, q, q)) * 0.01
    params_model = (h_model, J_model)  # raw params; constraints applied in energy only

    # --- Training config ---
    learning_rate = 1e-3
    weight_decay = 0 # set 0.0 to disable L2
    num_steps = 2000
    beta = 1.0
    lambda_ = 0.01

    # --- Optimizer ---
    optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params_model)
    w_bc = precompute_bc_weights(Xi_onehot, lambda_)
    update_step = make_update_fn(optimizer, Xi_onehot, w_bc, J_mask)

    # --- Train ---
    print("--- Starting Training ---")
    params = params_model
    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        params, opt_state, loss, log_Z = update_step(
            params, opt_state, sigma_onehot_synthetic, subkey,
            n_samples=300, lambda_=lambda_, beta=beta,
            weight_decay=weight_decay, run_bq=False,
        )
        if step % 20 == 0:
            print(f"Step {step:4d}/{num_steps-1} | Loss: {loss:.4f} | Log_Z: {log_Z:.4f}")

    print("--- Training Complete ---\n")
    h_final, J_final = params

    # KDSD EVALUATION 
    print("\n--- Starting KSD Evaluation (Native Discrete Method from Paper) ---")
    gamma = median_heuristic_gamma(sigma_onehot_synthetic)
    ksd_value_discrete = compute_discrete_ksd(
        sigma_onehot_synthetic, h_final, J_final, J_mask, beta, gamma
    )
    print("\n" + "="*55)
    print(f"✅ Final KDSD^2 (Native Discrete): {ksd_value_discrete:.6f}")
    print("="*55)
    return ksd_value_discrete

def main():
    
    num_seeds = 5
    kdsd_results = []

    for seed in range(num_seeds):
        final_kdsd = run_single_experiment(seed)
        kdsd_results.append(final_kdsd)

    # --- Final Report ---
    kdsd_results = np.array(kdsd_results)
    print("\n" + "="*55)
    print("--- Final Results Across All Seeds ---")
    print(f"Individual KDSD^2 scores: {kdsd_results}")
    print(f"Mean KDSD^2: {np.mean(kdsd_results):.6f}")
    print(f"Std Dev of KDSD^2: {np.std(kdsd_results):.6f}")
    print("="*55)



if __name__ == "__main__":
    main()