import time
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import optax
jax.config.update("jax_enable_x64", True)

# ====================
#  Global knobs
# ====================
M_POOL = 4000        # size of the big pool to draw from
M_SUB  = 500         # MC subsample size per iteration (must be <= M_POOL)
M_BQ   = 500         # BQ subsample size per iteration (must be <= M_POOL)
N_SAMPLES_DATA = 300 # data minibatch size (does NOT affect Z subsampling)

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

# -------------
#  Model / kernel
# -------------
@jit
def energy(x, h, J, J_mask):
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    return e_J + e_h

@jit
def kernel_embedding(lambda_, L, q):
    # closed-form mean embedding of product discrete Brownian kernel
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
#  Z estimators
# ====================
@jit
def logZ_from_pool(h, J, beta, Xi_onehot, key, J_mask):
    """MC estimate: subsample M_SUB points from the large pool every call."""
    M = Xi_onehot.shape[0]
    perm = jax.random.permutation(key, M)
    idx = perm[:M_SUB]
    Xi_sub = Xi_onehot[idx]
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_sub, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    Z_hat = jnp.mean(jnp.exp(logf - m))
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

@partial(jit, static_argnums=(5,))  # M_BQ static for JIT
def logZ_bc_from_pool_resample(h, J, beta, Xi_onehot, J_mask, M_BQ, key, lambda_):
    """BQ estimate: subsample M_BQ points from the large pool every call, then compute BQ weights on that subset."""
    M = Xi_onehot.shape[0]
    perm = jax.random.permutation(key, M)
    idx = perm[:M_BQ]
    Xi_sub = Xi_onehot[idx]  # (M_BQ, L, q)

    # Compute K and weights on-the-fly for the sampled subset (50x50 is cheap)
    K = gram_matrix(Xi_sub, lambda_) + 1e-8 * jnp.eye(M_BQ)
    Lc, lower = cho_factor(K, lower=True)
    L, q = Xi_sub.shape[1], Xi_sub.shape[2]
    z = jnp.full((M_BQ, 1), kernel_embedding(lambda_, L, q))
    w = cho_solve((Lc, lower), z)  # (M_BQ, 1)

    # Weighted integral of exp(-beta * E)
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_sub, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    f_shift = jnp.exp(logf - m)[:, None]  # (M_BQ, 1)
    Z_hat = (w.T @ f_shift)[0, 0]
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

# ====================
#  Loss / training step
# ====================
def loss_and_aux(params, Sigma_batch, key, Xi_onehot, _w_bc_unused, lambda_, beta, weight_decay, run_bq, J_mask):
    h, J = params

    # Data term on the minibatch
    energies = jax.vmap(energy, in_axes=(0, None, None, None))(Sigma_batch, h, J, J_mask)

    # Z term: both MC and BQ resample from the large pool each iteration
    key, zkey = jax.random.split(key)
    if run_bq:
        log_Z = logZ_bc_from_pool_resample(h, J, beta, Xi_onehot, J_mask, M_BQ, zkey, lambda_)
    else:
        log_Z = logZ_from_pool(h, J, beta, Xi_onehot, zkey, J_mask)

    # NLL + L2 on effective params
    nll = -jnp.mean(-beta * energies - log_Z)
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    l2 = jnp.sum(h**2) + jnp.sum(Jeff**2)
    loss = nll + weight_decay * l2
    return loss, {"log_Z": log_Z}

def make_update_fn(optimizer, Xi_onehot, w_bc_unused, J_mask):
    @partial(jax.jit, static_argnums=(4, 8))  # n_samples, run_bq
    def update_step(params, opt_state, Sigma, key, n_samples, lambda_, beta, weight_decay, run_bq):
        num_total = Sigma.shape[0]
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, num_total, shape=(n_samples,), replace=False)
        Sigma_batch = Sigma[idx]

        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, Sigma_batch, key, Xi_onehot, w_bc_unused, lambda_, beta, weight_decay, run_bq, J_mask
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux["log_Z"]
    return update_step

# ====================
#  Diagnostics (optional)
# ====================
M_STAT = 1024
@jit
def mean_energy(h, J, batch, J_mask):
    return jax.vmap(energy, in_axes=(0, None, None, None))(batch, h, J, J_mask).mean()

def energy_stats(h, J, data_batch, Xi_onehot, key, J_mask):
    E_data = mean_energy(h, J, data_batch, J_mask)
    M = Xi_onehot.shape[0]
    perm = jax.random.permutation(key, M)
    idx = perm[:min(M_STAT, M)]
    E_pool = mean_energy(h, J, Xi_onehot[idx], J_mask)
    return E_data, E_pool, E_pool - E_data

# ====================
#  Main
# ====================
def main():
    global M_POOL, M_SUB, M_BQ

    # Safety checks
    assert M_SUB <= M_POOL, "M_SUB must be <= M_POOL"
    assert M_BQ  <= M_POOL, "M_BQ must be <= M_POOL"

    # --- Data ---
    p = 0.4
    q = 3
    L = 4
    num_sequences = 500
    sequences_true, sigma_onehot_synthetic, L = data_preprocess(p, q, L, num_sequences)

    # --- Integration pool (independent of data) ---
    key = jax.random.PRNGKey(0)
    key, int_key = jax.random.split(key)
    Xi = jax.random.randint(int_key, (M_POOL, L), minval=0, maxval=q)
    Xi_onehot = jax.nn.one_hot(Xi, num_classes=q)

    # --- Mask (no self-couplings) ---
    J_mask = (1.0 - jnp.eye(L))[:, :, None, None]

    # --- Init params ---
    key, h_key, j_key = jax.random.split(key, 3)
    h_model = jax.random.normal(h_key, shape=(L, q)) * 0.01
    J_model = jax.random.normal(j_key, shape=(L, L, q, q)) * 0.01
    params_model = (h_model, J_model)

    # --- Training config ---
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_steps = 2000
    beta = 1.0
    lambda_ = 0.01

    # --- Optimizer ---
    optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params_model)

    # --- Build update function (BQ weights are computed on-the-fly each iter) ---
    w_bc_unused = None
    update_step = make_update_fn(optimizer, Xi_onehot, w_bc_unused, J_mask)

    # --- Train ---
    params = params_model
    for step in range(num_steps):
        key, subkey, statkey1, statkey2 = jax.random.split(key, 4)
        params, opt_state, loss, log_Z = update_step(
            params,
            opt_state,
            sigma_onehot_synthetic,
            subkey,
            n_samples=N_SAMPLES_DATA,   # data minibatch size (does not affect Z subsampling)
            lambda_=lambda_,
            beta=beta,
            weight_decay=weight_decay,
            run_bq=False,                 # True => BQ(resampled-50); False => MC(resampled-50)
        )

        if step % 20 == 0:
            print(f"Step {step:4d}/{num_steps-1} | Loss: {loss:.4f} | Log_Z: {log_Z:.4f}")

            # # Optional diagnostics (commented to keep stdout light)
            # h_chk, J_chk = params
            # n_tot = sigma_onehot_synthetic.shape[0]
            # d_idx = jax.random.permutation(statkey1, n_tot)[:256]
            # data_stat = sigma_onehot_synthetic[d_idx]
            # E_d, E_p, gap = energy_stats(h_chk, J_chk, data_stat, Xi_onehot, statkey2, J_mask)
            # print(f"E_data: {float(E_d):+.3f} | E_pool: {float(E_p):+.3f} | gap: {float(gap):+.3f}")

    h_final, J_final = params

    # # Final quick check (optional)
    # key, mc_key, bq_key = jax.random.split(key, 3)
    # mc_resamp_50 = logZ_from_pool(h_final, J_final, beta, Xi_onehot, mc_key, J_mask)
    # bq_resamp_50 = logZ_bc_from_pool_resample(h_final, J_final, beta, Xi_onehot, J_mask, M_BQ, bq_key, lambda_)
    # print(f"FINAL | MC(50, resample): {mc_resamp_50:.6f} | BQ(50, resample): {bq_resamp_50:.6f}")

if __name__ == "__main__":
    main()
