# import time
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from Bio import AlignIO
# from Bio.PDB import PDBParser
# from functools import partial

# import jax
# import jax.numpy as jnp
# from jax import jit, vmap
# from jax.scipy.linalg import cho_factor, cho_solve
# import optax
# from scipy.stats import spearmanr

# # Use 64-bit for stability
# jax.config.update("jax_enable_x64", True)

# def generate_true_sequence(key, p, max_len=50):
#     ### here the function generates a single sequence from the paper's "true" p distribution.
#     ### Define the probabilities for A, B, STOP
#     probs = jnp.array([p, p, 1 - 2*p])
    
#     sequence = []
#     current_char = -1
    
#     while current_char != 2 and len(sequence) < max_len:
#         key, subkey = jax.random.split(key)
#         current_char = jax.random.choice(subkey, jnp.arange(3), p=probs)
        
#         if current_char != 2:
#             sequence.append(current_char)
            
#     return jnp.array(sequence, dtype=jnp.int32)


# def data_preprocess(p, q, max_len, num_sequences):   
#     key = jax.random.PRNGKey(42)
#     keys = jax.random.split(key, num_sequences)
#     sequences_true = [generate_true_sequence(k, p) for k in keys]

#     # --- Preprocess data for the model (padding and one-hot encoding) ---
#     max_len = max(len(s) for s in sequences_true if len(s) > 0)
#     padded_sequences = jnp.full((num_sequences, max_len), 2, dtype=jnp.int32)
#     for i, seq in enumerate(sequences_true):
#         if len(seq) > 0:
#             padded_sequences = padded_sequences.at[i, :len(seq)].set(seq)

#     sigma_onehot_synthetic = jax.nn.one_hot(padded_sequences, num_classes=q)
#     return sequences_true, sigma_onehot_synthetic, max_len

# def symmetrize_J(J):
#     return 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))

# def first_stop_index(x_arg, stop_token=2):
#     # x_arg: (L,) int labels per site
#     # returns index of first STOP or L if none
#     has_stop = jnp.any(x_arg == stop_token)
#     idx = jnp.argmax(x_arg == stop_token)  # 0 if first element is STOP, else first True
#     return jnp.where(has_stop, idx, x_arg.shape[0])

# def masked_onehot(x_onehot, stop_idx):
#     # zero out positions strictly after the first STOP (keep the STOP position itself)
#     L, q = x_onehot.shape
#     mask = (jnp.arange(L) <= stop_idx).astype(x_onehot.dtype)  # (L,)
#     return x_onehot * mask[:, None]

# def energy_masked(x_onehot, h, J, stop_token=2):
#     # score only up to and including the first STOP; ignore trailing tokens
#     Jsym = symmetrize_J(J)
#     x_arg = jnp.argmax(x_onehot, axis=1)
#     sidx  = first_stop_index(x_arg, stop_token=stop_token)
#     xm    = masked_onehot(x_onehot, sidx)           # (L,q)

#     e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', xm, Jsym, xm)
#     e_h = jnp.einsum('ik,ik->', xm, h)
#     return e_J + e_h

# def stop_violation_penalty(x_onehot, stop_token=2):
#     # penalize any non-STOP after the first STOP
#     x_arg = jnp.argmax(x_onehot, axis=1)
#     sidx  = first_stop_index(x_arg, stop_token=stop_token)
#     after = jnp.arange(x_arg.shape[0]) > sidx
#     bad   = jnp.logical_and(after, x_arg != stop_token)
#     return jnp.sum(bad.astype(jnp.float32))

# def energy(x, h, J):
#     Jsym = symmetrize_J(J)
#     e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jsym, x)
#     e_h = jnp.einsum('ik,ik->', x, h)
#     return e_J + e_h

# @jit
# def kernel_embedding(lambda_, d_q):
#     return ((1 + 2 * jnp.exp(-lambda_)) / 3) ** (d_q / 3)

# @jit
# def double_integral(lambda_, d_q):
#     return kernel_embedding(lambda_, d_q)

# @jit
# def gram_matrix(sigma_batch, lambda_):
#     n, d, q = sigma_batch.shape
#     sigma_flat = sigma_batch.reshape(n, -1)  # Shape (n, d*q)

#     def k(x1, x2):
#         dot = jnp.dot(x1, x2)  
#         return jnp.exp(-lambda_ * d + lambda_ * dot) 

#     return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

# @jit
# def bayesian_cubature(sigma_batch, f_vals, lambda_):
#     n = sigma_batch.shape[0]
#     sigma_flat = sigma_batch.reshape(n, -1)
#     d_q = sigma_flat.shape[1]

#     K = gram_matrix(sigma_batch, lambda_) + 1e-8 * jnp.eye(n)

#     try:
#         L, lower = cho_factor(K, lower=True)
#     except:
#         raise ValueError("Cholesky decomposition failed. The kernel matrix is likely not positive definite.")

#     z = jnp.full((n, 1), kernel_embedding(lambda_, d_q))
#     f_vals = f_vals.reshape(n, 1)

#     K_inv_z = cho_solve((L, lower), z)
#     K_inv_f = cho_solve((L, lower), f_vals)

#     mean_integral = (z.T @ K_inv_f)[0, 0]
#     variance = double_integral(lambda_, d_q) - (z.T @ K_inv_z)[0, 0]

#     return mean_integral, variance

# # def loss_and_aux(params, Sigma_batch, key, lambda_, beta, weight_decay, run_bq):
# #     h, J = params
# #     energies = jax.vmap(energy, in_axes=(0, None, None))(Sigma_batch, h, J)
# #     log_f = -beta * energies
# #     m = jnp.max(log_f)
# #     f_shift = jnp.exp(log_f - m)

# #     if run_bq:
# #         Z_shift, _ = bayesian_cubature(Sigma_batch, f_shift, lambda_)
# #     else:
# #         Z_shift = jnp.mean(f_shift)

# #     log_Z = jnp.log(jnp.clip(Z_shift, 1e-20, None)) + m
# #     nll = -jnp.mean(-beta * energies - log_Z)

# #     # optional L2
# #     h, J = params
# #     l2 = jnp.sum(h**2) + jnp.sum(J**2)
# #     loss = nll + weight_decay * l2

# #     return loss, {"log_Z": log_Z}

# def loss_and_aux(params, Sigma_batch, key, lambda_, beta, weight_decay, run_bq,
#                  stop_token=2, pen_after_stop=5.0, wd_h=1e-3, wd_J=5e-3,
#                  wd_J_stop=5e-2,        # NEW: strong shrinkage for STOP couplings
#                  wd_stop_smooth=1e-2):  # NEW: smooth h[:, STOP] across positions
#     h, J = params

#     # --- masked energy, BQ, NLL (unchanged) ---
#     energies = jax.vmap(energy_masked, in_axes=(0, None, None, None))(
#         Sigma_batch, h, J, stop_token
#     )
#     log_f = -beta * energies
#     m = jnp.max(log_f)
#     f_shift = jnp.exp(log_f - m)
#     Z_shift, _ = bayesian_cubature(Sigma_batch, f_shift, lambda_) if run_bq else (jnp.mean(f_shift), None)
#     log_Z = jnp.log(jnp.clip(Z_shift, 1e-20, None)) + m
#     nll   = -jnp.mean(-beta * energies - log_Z)

#     # --- penalties (unchanged) ---
#     pen = jnp.mean(jax.vmap(stop_violation_penalty, in_axes=(0, None))(Sigma_batch, stop_token))

#     # --- NEW: zero out STOP couplings in J ---
#     # mask STOP on either side of the pair (i,j,a,b)
#     # shape: (L, L, q, q)
#     q = h.shape[1]
#     stop_mask_a = jax.nn.one_hot(stop_token, q)  # (q,)
#     Ma = stop_mask_a[None, None, :, None]        # (1,1,q,1)
#     Mb = stop_mask_a[None, None, None, :]        # (1,1,1,q)
#     Mstop = jnp.clip(Ma + Mb, 0, 1)              # 1 where a==STOP or b==STOP
#     reg_J_stop = jnp.sum((J * Mstop) ** 2)

#     # --- NEW: encourage nearly constant h[:, STOP] across positions ---
#     h_stop = h[:, stop_token]               # (L,)
#     reg_stop_smooth = jnp.sum((h_stop[1:] - h_stop[:-1])**2)

#     # --- usual L2, but keep it mild now that we have targeted priors ---
#     reg_base = wd_h * jnp.sum(h**2) + wd_J * jnp.sum(J**2)

#     loss = nll + pen_after_stop * pen + reg_base \
#            + wd_J_stop * reg_J_stop + wd_stop_smooth * reg_stop_smooth

#     return loss, {"log_Z": log_Z}

# # 2) Make a compiled update function that CAPTURES `optimizer` instead of receiving it
# def make_update_fn(optimizer):
#     @partial(jax.jit, static_argnums=(4, 8))
#     def update_step(params, opt_state, Sigma, key, n_samples, lambda_, beta, weight_decay, run_bq):
#         num_total = Sigma.shape[0]
#         key, sub = jax.random.split(key)
#         idx = jax.random.choice(sub, num_total, shape=(n_samples,), replace=False)
#         Sigma_batch = Sigma[idx]

#         (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
#             params, Sigma_batch, key, lambda_, beta, weight_decay, run_bq
#         )
#         updates, new_opt_state = optimizer.update(grads, opt_state, params)
#         new_params = optax.apply_updates(params, updates)
#         return new_params, new_opt_state, loss, aux["log_Z"]
#     return update_step

# # def negative_log_likelihood(params, Sigma, key, n_samples, lambda_, beta, weight_decay, run_bq):
# #     h, J = params

# #     num_total = Sigma.shape[0]
# #     indices = jnp.arange(num_total)
# #     shuffled_indices = jax.random.permutation(key, indices)
# #     sample_indices = shuffled_indices[:n_samples]
# #     Sigma_samples = Sigma[sample_indices]
    
# #     energies = jax.vmap(energy, in_axes=(0, None, None))(Sigma_samples, h, J)
# #     log_f_values = -beta * energies
# #     m = jnp.max(log_f_values)
# #     safe_integrand_values = jnp.exp(log_f_values - m)
    
# #     if run_bq:
# #         Z_shifted, _ = bayesian_cubature(Sigma_samples, safe_integrand_values, lambda_)
# #     else:
# #         Z_shifted = jnp.mean(safe_integrand_values)

# #     log_Z = jnp.log(jnp.clip(Z_shifted, 1e-20, None)) + m
    
# #     nll = -jnp.mean(-beta * energies - log_Z)
    
# #     return nll


# # def update(params, opt_state, optimizer, Sigma, key, n_samples, lambda_, beta, weight_decay, run_bq):
# #     loss, grads = jax.value_and_grad(negative_log_likelihood)(params, Sigma, key, n_samples, lambda_, beta, weight_decay, run_bq)
    
# #     updates, new_opt_state = optimizer.update(grads, opt_state, params)
# #     new_params = optax.apply_updates(params, updates)

# #     num_total = Sigma.shape[0]
# #     indices = jnp.arange(num_total)
# #     shuffled_indices = jax.random.permutation(key, indices)
# #     sample_indices = shuffled_indices[:n_samples]
# #     Sigma_samples = Sigma[sample_indices]

# #     h, J = new_params
# #     energies = jax.vmap(energy, in_axes=(0, None, None))(Sigma_samples, h, J) 
# #     log_f_values = -beta * energies
# #     m = jnp.max(log_f_values)
# #     safe_integrand_values = jnp.exp(log_f_values - m)

# #     if run_bq:
# #         Z_shifted, _ = bayesian_cubature(Sigma_samples, safe_integrand_values, lambda_)
# #     else:
# #         Z_shifted = jnp.mean(safe_integrand_values)

# #     log_Z = jnp.log(jnp.clip(Z_shifted, 1e-20, None)) + m

# #     return new_params, new_opt_state, loss, log_Z

# # update_jitted = jax.jit(update, static_argnames=['optimizer', 'n_samples', 'run_bq'])

# def sample_from_trained_model(key, L, q, h, J, n_samples, n_steps=50, burn_in=200, thin=20, stop_token=2):
#     Jsym = symmetrize_J(J)

#     def enforce_absorbing(x):
#         # once STOP appears at k, force x[k+1:] = STOP
#         idxs = jnp.where(x == stop_token)[0]
#         return jnp.where(
#             idxs.size > 0,
#             x.at[idxs[0]+1:].set(stop_token),
#             x,
#         )

#     samples = []
#     x = jax.random.randint(key, (L,), 0, q, dtype=jnp.int32)

#     # burn-in
#     for _ in range(burn_in):
#         key, s1, s2, s3 = jax.random.split(key, 4)
#         pos = jax.random.randint(s1, (), 0, L)
#         old_aa = x[pos]
#         prop   = jax.random.randint(s2, (), 0, q, dtype=jnp.int32)

#         delta_E_h = h[pos, prop] - h[pos, old_aa]
#         j_idx = jnp.arange(L)
#         # include both halves via Jsym
#         delta_E_J = jnp.sum(Jsym[pos, j_idx, prop, x[j_idx]] - Jsym[pos, j_idx, old_aa, x[j_idx]])
#         delta_E = delta_E_h + delta_E_J
#         accept = jnp.minimum(1.0, jnp.exp(-beta * delta_E))
#         x = jnp.where(jax.random.uniform(s3) < accept, x.at[pos].set(prop), x)
#         x = enforce_absorbing(x)

#     # draw/thin
#     for _ in range(n_samples * thin):
#         key, s1, s2, s3 = jax.random.split(key, 4)
#         pos = jax.random.randint(s1, (), 0, L)
#         old_aa = x[pos]
#         prop   = jax.random.randint(s2, (), 0, q, dtype=jnp.int32)

#         delta_E_h = h[pos, prop] - h[pos, old_aa]
#         j_idx = jnp.arange(L)
#         delta_E_J = jnp.sum(Jsym[pos, j_idx, prop, x[j_idx]] - Jsym[pos, j_idx, old_aa, x[j_idx]])
#         delta_E = delta_E_h + delta_E_J
#         accept = jnp.minimum(1.0, jnp.exp(-beta * delta_E))
#         x = jnp.where(jax.random.uniform(s3) < accept, x.at[pos].set(prop), x)
#         x = enforce_absorbing(x)

#         if ((_ + 1) % thin) == 0:
#             samples.append(x)

#     return jnp.stack(samples)

# # --- Compare the most important statistic: sequence length distribution ---
# def get_lengths(padded_seqs, stop_token=2):
#     # Find the first occurrence of the stop token in each row
#     lengths = jnp.argmax(padded_seqs == stop_token, axis=1)
#     # If a sequence has no stop token, its length is the max length
#     no_stop_token = jnp.all(padded_seqs != stop_token, axis=1)
#     lengths = jnp.where(no_stop_token, padded_seqs.shape[1], lengths)
#     return lengths

# def main():
#     global beta
#     # ---------------------DATA GENERATING--------------------------
#     p = 0.4
#     q = 3
#     max_len = 5
#     num_sequences = 3000

#     sequences_true, sigma_onehot_synthetic, L = data_preprocess(p, q, max_len, num_sequences)

#     # ----------- Initialize model parameters randomly -------------
#     key = jax.random.PRNGKey(0)
#     key, h_model_key, j_model_key = jax.random.split(key, 3)
#     h_model = jax.random.normal(h_model_key, shape=(L, q)) * 0.01
#     J_model = jax.random.normal(j_model_key, shape=(L, L, q, q)) * 0.01
#     params_model = (h_model, J_model)

#     # --------------- Set up stable optimizer ----------------------
#     learning_rate = 5e-4
#     weight_decay  = 0.0        # we now pass wd_h, wd_J to loss instead
#     num_steps     = 2000
#     beta          = 1.0
#     lambda_       = 0.01

#     # optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
#     # opt_state = optimizer.init(params_model)

#     # # --------------------- Training loop ---------------------------
#     # params = params_model
#     # for step in range(num_steps):
#     #     key, subkey = jax.random.split(key) 
#     #     params, opt_state, loss, log_Z = update_jitted(
#     #         params, 
#     #         opt_state, 
#     #         optimizer, 
#     #         sigma_onehot_synthetic, 
#     #         subkey,
#     #         n_samples=300, 
#     #         lambda_=lambda_, 
#     #         beta=beta, 
#     #         weight_decay=weight_decay, 
#     #         run_bq=True
#     #     )
#     #     if step % 20 == 0:
#     #         print(f"Step {step:4d}/{num_steps-1} | Loss: {loss:.4f} | Log_Z: {log_Z:.4f} ")

#     optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
#     opt_state = optimizer.init(params_model)
#     update_step = make_update_fn(optimizer)   # <-- build the jitted step with optimizer closed over

#     params = params_model
#     for step in range(num_steps):
#         key, subkey = jax.random.split(key)
#         params, opt_state, loss, log_Z = update_step(params, opt_state, sigma_onehot_synthetic, subkey,
#                                                     n_samples=400,         # a bit larger stabilizes BQ
#                                                     lambda_=lambda_, beta=beta,
#                                                     weight_decay=0.0,      # unused now
#                                                     run_bq=True)
#         if step % 20 == 0:
#             print(f"Step {step:4d}/{num_steps-1} | Loss: {loss:.4f} | Log_Z: {log_Z:.4f}")

#     h_final, J_final = params
#     key, sample_key = jax.random.split(key)
#     sequences_model_padded = sample_from_trained_model(sample_key, L, q, h_final, J_final, num_sequences)

#     lengths_true = [len(s) for s in sequences_true]
#     lengths_model = get_lengths(sequences_model_padded)

#     # --- Plot the distributions ---
#     plt.figure(figsize=(10, 6))
#     plt.hist(lengths_true, bins=np.arange(0, max_len + 1), alpha=0.7, label='True Data', density=True)
#     plt.hist(lengths_model, bins=np.arange(0, max_len + 1), alpha=0.7, label='Model-Generated Data', density=True)
#     plt.title("Distribution of Sequence Lengths")
#     plt.xlabel("Sequence Length")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid(True, axis='y')
#     plt.show()


# if __name__ == "__main__":
#     main()



import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import optax

jax.config.update("jax_enable_x64", True)

# ====================
#  Knobs & switches
# ====================
RUN_BQ = True              # True => use BQ; False => use MC
BQ_RESAMPLE = False        # If RUN_BQ=True: False => fixed pool + precomputed weights; True => resample subset + on-the-fly weights
M_POOL = 2000              # size of the big pool to draw from (integration proposals)
M_SUB  = 300               # MC subsample size per iteration (must be <= M_POOL)
M_BQ   = 300               # BQ subsample size per iteration for resample mode (must be <= M_POOL)
N_SAMPLES_DATA = 300       # data minibatch size (does NOT affect Z subsampling)

# ====================
#  Data generator (true data is categorical with probs [p,p,1-2p])
# ====================
def generate_categorical_sequence(key, p, L):
    probs = jnp.array([p, p, 1.0 - 2.0 * p])
    keys = jax.random.split(key, L)
    seq = [jax.random.choice(k, 3, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)

def data_preprocess(p, q, L, num_sequences):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_sequences)
    sequences_true = [generate_categorical_sequence(k, p, L) for k in keys]
    sigma_onehot_synthetic = jax.nn.one_hot(jnp.stack(sequences_true), num_classes=q)  # (N, L, q)
    return sequences_true, sigma_onehot_synthetic, L

# ====================
#  Model / kernel
# ====================
@jit
def energy(x, h, J, J_mask):
    # x: (L, q) one-hot
    # h: (L, q)
    # J: (L, L, q, q)
    # J_mask: (L, L, 1, 1) 0 on diag, 1 elsewhere (no self-couplings)
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, Jeff, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    return e_J + e_h

@jit
def kernel_embedding(lambda_, L, q):
    # mean embedding for the *uniform* product measure over {0..q-1}^L
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
#  Log Z estimators
# ====================
@jit
def logZ_from_pool(h, J, beta, Xi_onehot, key, J_mask):
    """MC: resample M_SUB from the pool each call."""
    M = Xi_onehot.shape[0]
    idx = jax.random.permutation(key, M)[:M_SUB]
    Xi_sub = Xi_onehot[idx]
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_sub, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    Z_hat = jnp.mean(jnp.exp(logf - m))
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

def precompute_bc_weights(Xi_onehot, lambda_):
    """BQ weights for the whole pool (fixed)."""
    K = gram_matrix(Xi_onehot, lambda_) + 1e-8 * jnp.eye(Xi_onehot.shape[0])
    Lc, lower = cho_factor(K, lower=True)
    L, q = Xi_onehot.shape[1], Xi_onehot.shape[2]
    z = jnp.full((Xi_onehot.shape[0], 1), kernel_embedding(lambda_, L, q))
    w = cho_solve((Lc, lower), z)  # (M,1)
    return w  # (M,1)

@jit
def logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask):
    """BQ: fixed pool, precomputed weights."""
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_onehot, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    f_shift = jnp.exp(logf - m)[:, None]  # (M,1)
    Z_hat = (w_bc.T @ f_shift)[0, 0]
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

@partial(jit, static_argnums=(5,))  # M_BQ is static
def logZ_bc_from_pool_resample(h, J, beta, Xi_onehot, J_mask, M_BQ, key, lambda_):
    """BQ: resample M_BQ each call; recompute weights on that subset."""
    M = Xi_onehot.shape[0]
    idx = jax.random.permutation(key, M)[:M_BQ]
    Xi_sub = Xi_onehot[idx]  # (M_BQ, L, q)
    # kernel + weights
    K = gram_matrix(Xi_sub, lambda_) + 1e-8 * jnp.eye(M_BQ)
    Lc, lower = cho_factor(K, lower=True)
    L, q = Xi_sub.shape[1], Xi_sub.shape[2]
    z = jnp.full((M_BQ, 1), kernel_embedding(lambda_, L, q))
    w = cho_solve((Lc, lower), z)  # (M_BQ,1)
    # integrate exp(-beta E)
    E = jax.vmap(energy, in_axes=(0, None, None, None))(Xi_sub, h, J, J_mask)
    logf = -beta * E
    m = jnp.max(logf)
    f_shift = jnp.exp(logf - m)[:, None]
    Z_hat = (w.T @ f_shift)[0, 0]
    return jnp.log(jnp.clip(Z_hat, 1e-20, None)) + m

# ====================
#  Loss & update (robust branching)
# ====================
def loss_and_aux(params, Sigma_batch, key, Xi_onehot, w_bc, lambda_, beta, weight_decay, use_bq, use_bq_resample, J_mask):
    h, J = params
    # positive term
    energies = jax.vmap(energy, in_axes=(0, None, None, None))(Sigma_batch, h, J, J_mask)
    # negative term (log Z)
    key, zkey = jax.random.split(key)

    def bq_branch(_):
        return jax.lax.cond(
            use_bq_resample,
            lambda __: logZ_bc_from_pool_resample(h, J, beta, Xi_onehot, J_mask, M_BQ, zkey, lambda_),
            lambda __: logZ_bc_on_pool(h, J, beta, Xi_onehot, w_bc, J_mask),
            operand=None
        )

    log_Z = jax.lax.cond(
        use_bq,
        bq_branch,
        lambda _: logZ_from_pool(h, J, beta, Xi_onehot, zkey, J_mask),
        operand=None
    )

    nll = -jnp.mean(-beta * energies - log_Z)  # (minus) log-likelihood on batch
    # L2 on effective params
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    Jeff = Jsym * J_mask
    l2 = jnp.sum(h**2) + jnp.sum(Jeff**2)
    loss = nll + weight_decay * l2
    return loss, {"log_Z": log_Z, "nll": nll}

def make_update_fn(optimizer, Xi_onehot, w_bc, J_mask):
    @partial(jax.jit, static_argnums=(4, 8, 9))  # n_samples, use_bq, use_bq_resample are static
    def update_step(params, opt_state, Sigma, key, n_samples, lambda_, beta, weight_decay, use_bq, use_bq_resample):
        num_total = Sigma.shape[0]
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, num_total, shape=(n_samples,), replace=False)
        Sigma_batch = Sigma[idx]

        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
            params, Sigma_batch, key, Xi_onehot, w_bc, lambda_, beta, weight_decay, use_bq, use_bq_resample, J_mask
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux
    return update_step

# ====================
#  Main
# ====================
def main():
    # --- Data ---
    p = 0.4         # P(x=0)=p, P(x=1)=p, P(x=2)=1-2p
    q = 3
    L = 4
    num_sequences = 500
    _, sigma_onehot_synthetic, _ = data_preprocess(p, q, L, num_sequences)

    # --- Integration pool (uniform over {0..q-1}^L) ---
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
    params = (h_model, J_model)

    # --- Training config ---
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_steps = 2000
    beta = 1.0
    lambda_ = 0.01

    # --- Optimizer ---
    optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params)

    # --- BQ weights (only if using fixed-pool BQ) ---
    w_bc = None
    if RUN_BQ and not BQ_RESAMPLE:
        w_bc = precompute_bc_weights(Xi_onehot, lambda_)  # guaranteed non-None if we will use BQ fixed

    # --- Build update function ---
    update_step = make_update_fn(optimizer, Xi_onehot, w_bc, J_mask)

    # --- Train ---
    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        params, opt_state, loss, aux = update_step(
            params, opt_state, sigma_onehot_synthetic, subkey,
            n_samples=N_SAMPLES_DATA,
            lambda_=lambda_, beta=beta, weight_decay=weight_decay,
            use_bq=RUN_BQ, use_bq_resample=BQ_RESAMPLE
        )

        if step % 20 == 0:
            print(f"Step {step:4d}/{num_steps-1} | Loss: {aux['nll'] + weight_decay * 0:.4f} | "
                  f"NLL: {aux['nll']:.4f} | Log_Z: {aux['log_Z']:.4f}")

    h_final, J_final = params
    print("Training done.")

if __name__ == "__main__":
    main()
