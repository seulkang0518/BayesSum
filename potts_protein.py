import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO
from Bio.PDB import PDBParser

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import optax
from scipy.stats import spearmanr

jax.config.update("jax_enable_x64", True)


def energy(x, h, J):
    e_J = 0.5 * jnp.einsum('ik,ijkl,jl->', x, J, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    return e_J + e_h

@jit
def kernel_embedding(lambda_, d_q):
    return ((1 + 20 * jnp.exp(-lambda_)) / 21) ** (d_q / 21)

@jit
def double_integral(lambda_, d_q):
    return kernel_embedding(lambda_, d_q)

@jit
def gram_matrix(sigma_batch, lambda_):
    n, d, q = sigma_batch.shape
    sigma_flat = sigma_batch.reshape(n, -1)  # Shape (n, d*q)

    def k(x1, x2):
        dot = jnp.dot(x1, x2)  # Number of matching one-hot elements
        return jnp.exp(-lambda_ * d + lambda_ * dot)

    return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

@jit
def bayesian_cubature(sigma_batch, f_vals, lambda_):
    n = sigma_batch.shape[0]
    sigma_flat = sigma_batch.reshape(n, -1)
    d_q = sigma_flat.shape[1]

    K = gram_matrix(sigma_batch, lambda_) + 1e-8 * jnp.eye(n)

    try:
        L, lower = cho_factor(K, lower=True)
    except:
        raise ValueError("Cholesky decomposition failed. The kernel matrix is likely not positive definite.")

    z = jnp.full((n, 1), kernel_embedding(lambda_, d_q))
    f_vals = f_vals.reshape(n, 1)

    K_inv_z = cho_solve((L, lower), z)
    K_inv_f = cho_solve((L, lower), f_vals)

    mean_integral = (z.T @ K_inv_f)[0, 0]
    variance = double_integral(lambda_, d_q) - (z.T @ K_inv_z)[0, 0]

    return mean_integral, variance


# ---------- DATA HANDLING & PREPROCESSING --------------

def encoding(filename):
    return AlignIO.read(filename, "stockholm")

def amino_mapping(alignment):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
    aa_int = {amino_acid: i for i, amino_acid in enumerate(amino_acids)}
    total_aa = len(amino_acids)

    encoded = [[aa_int.get(residue, total_aa - 1) for residue in str(record.seq)] for record in alignment]

    sigma_int = jnp.array(encoded, dtype=jnp.int32)
    sigma_onehot = jax.nn.one_hot(sigma_int, num_classes=total_aa, dtype=jnp.float64)
    return sigma_int, sigma_onehot

# ---------- TRAINING --------------

def negative_log_likelihood(params, Sigma, lambda_, beta, weight_decay, run_bq):
    """ Here we are using the full dataset """
    h, J = params
    energies = jax.vmap(energy, in_axes=(0, None, None))(Sigma, h, J) 
    
    log_f_values = -beta * energies
    m = jnp.max(log_f_values)
    safe_integrand_values = jnp.exp(log_f_values - m)

    if run_bq:
        Z_shifted, _ = bayesian_cubature(Sigma, safe_integrand_values, lambda_)
    else:
        Z_shifted = jnp.mean(jnp.exp(log_f_values))

    log_Z = jnp.log(jnp.clip(Z_shifted, 1e-20, None)) + m
    
    nll = -jnp.mean(-beta * energies - log_Z)

    # --- L2 Regularization Term ---
    l2_h = jnp.sum(h**2)
    l2_J = jnp.sum(J**2)
    l2_penalty = weight_decay * (l2_h + l2_J)

    return nll + l2_penalty

def update(params, opt_state, optimizer, Sigma, lambda_, beta, weight_decay, run_bq):
   
    loss, grads = jax.value_and_grad(negative_log_likelihood)(params, Sigma, lambda_, beta, weight_decay, run_bq)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Re-compute log_Z outside the grad context for monitoring
    h, J = new_params
    energies = jax.vmap(energy, in_axes=(0, None, None))(Sigma, h, J) 

    log_f_values = -beta * energies
    m = jnp.max(log_f_values)
    safe_integrand_values = jnp.exp(log_f_values - m)

    if run_bq:
        Z_shifted, _ = bayesian_cubature(Sigma, safe_integrand_values, lambda_)
    else:
        Z_shifted = jnp.mean(jnp.exp(log_f_values))

    log_Z = jnp.log(jnp.clip(Z_shifted, 1e-20, None)) + m

    return new_params, new_opt_state, loss, log_Z

update_jitted = jax.jit(update, static_argnames=['optimizer',  'run_bq'])

# ---------- EVALUATION --------------

def sample_potts_mcmc(key, h, J, beta, n_samples=10000, n_steps=100, initial_seq=None):
    """
    Samples from a Potts model using Metropolis-Hastings MCMC.
    `n_steps` acts as the thinning factor between samples.
    """
    L, q = h.shape
    samples = []
    
    if initial_seq is None:
        key, subkey = jax.random.split(key)
        x = jax.random.randint(subkey, (L,), 0, q, dtype=jnp.int32)
    else:
        x = initial_seq

    j_indices = jnp.arange(L)
    for step in range(n_steps * n_samples):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        
        pos = jax.random.randint(subkey1, (), 0, L)
        old_aa = x[pos]
        proposal = (old_aa + jax.random.randint(subkey2, (), 1, q, dtype=jnp.int32)) % q
        
        delta_E_h = h[pos, proposal] - h[pos, old_aa]
        
        J_interactions_new = J[pos, j_indices, proposal, x[j_indices]]
        J_interactions_old = J[pos, j_indices, old_aa, x[j_indices]]
        delta_E_J = jnp.sum(J_interactions_new - J_interactions_old)
        
        delta_E = delta_E_h + delta_E_J
        
        accept_prob = jnp.minimum(1.0, jnp.exp(-beta * delta_E))
        accept = jax.random.uniform(subkey3) < accept_prob
        
        x = x.at[pos].set(jnp.where(accept, proposal, old_aa))
        
        if (step + 1) % n_steps == 0:
            samples.append(x)
            
    samples_int = jnp.stack(samples)
    return jax.nn.one_hot(samples_int, num_classes=q, dtype=jnp.float64)

def compute_first_order(sigma):
    return jnp.mean(sigma, axis=0)

def compute_second_order(sigma):
    """Computes second-order statistics (pairwise frequencies)."""
    return jnp.einsum('nik,njl->ijkl', sigma, sigma) / sigma.shape[0]

def spearman_corr(x, y):
    """Computes Spearman correlation between two matrices."""
    corr, _ = spearmanr(np.ravel(x), np.ravel(y))
    return corr

def main():
    # --- Hyperparameters ---
    lambda_param = 0.01
    beta = 1.0           
    learning_rate = 0.001         
    num_steps = 1000              
    weight_decay = 0.1         
    run_bq = True
    # --- Data Loading ---
    alignment = encoding("PF00041_seed_short.sto")
    # alignment = encoding("PF00041.alignment.seed")    
    sigma_int, sigma_onehot = amino_mapping(alignment)
    len_protein = sigma_onehot.shape[1]
    len_aa = sigma_onehot.shape[2]

    # --- Parameter Initialization ---
    key = jax.random.PRNGKey(0)
    key, h_key, j_key = jax.random.split(key, 3)

    h = 0.01 * jax.random.normal(h_key, shape=(len_protein, len_aa), dtype=jnp.float64)
    J_init = jax.random.normal(j_key, shape=(len_protein, len_protein, len_aa, len_aa), dtype=jnp.float64)
    J_init = jnp.abs(J_init)
    J = (J_init + jnp.transpose(J_init, (1, 0, 3, 2))) / 2.0
    J = J.at[jnp.arange(len_protein), jnp.arange(len_protein)].set(0.0)
    J = J / 40000.0 # Scaling

    params = (h, J)

    # --- Optimizer Setup ---
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.95
    )
    optimizer = optax.adam(schedule)
    
    # schedule = optax.exponential_decay(
    # init_value=learning_rate,      # Start at 0.001
    # transition_steps=1000,         # How often to decay the LR
    # decay_rate=0.9,                # The rate of decay
    # end_value=0.00001)              # You can optionally specify a final LR

    # # Your chain is already well-structured
    # optimizer = optax.chain(
    #     optax.clip(1.0),               # Essential for preventing explosions
    #     optax.adam(schedule))

    opt_state = optimizer.init(params)

    # --- Training Loop ---
    print("Starting training...")
    for step in range(num_steps):
        params, opt_state, loss, log_Z = update_jitted(params, opt_state, optimizer, sigma_onehot, lambda_param, beta, weight_decay, run_bq)

        if step % 50 == 0 or step == num_steps - 1:
            if jnp.isnan(loss):
                print(f"Stopping at step {step} due to NaN loss.")
                break
            print(f"Step {step:4d}/{num_steps-1} | Loss: {loss:.4f} | Est. log(Z): {log_Z:.4f}")


    print("Training finished.")

    # --- Evaluation ---
    h_final, J_final = params

    # --- First-order correlation analysis ---
    print("\nStarting MCMC sampling to generate sequences from the model...")
    key, subkey = jax.random.split(key)
    sigma_sampled = sample_potts_mcmc(subkey, h_final, J_final, beta, n_samples=10000, n_steps=100, initial_seq=sigma_int[0])
    print("MCMC sampling finished.")

    print("\nComputing first-order correlations...")
    f_train = compute_first_order(sigma_onehot)
    f_sampled = compute_first_order(sigma_sampled)
    rho_1st = spearman_corr(f_train, f_sampled)
    print(f"--> First-order Spearman correlation: {rho_1st:.4f}")

    # ... (Second-order and other analyses would go here) ...


if __name__ == "__main__":
    main()
