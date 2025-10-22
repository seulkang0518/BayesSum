from Bio import AlignIO
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import time
import optax
import matplotlib.pyplot as plt
import seaborn as sns
from jax.numpy import sqrt
jax.config.update("jax_enable_x64", True)

learning_rate = 0.05
optimizer = optax.adam(learning_rate)

@jit
def f_single(sigma_onehot, J, h, beta):
    e_h = 0.5 * jnp.einsum('ik,ik->', sigma_onehot, h)  
    e_J = jnp.einsum('ik,ijkl,jl->', sigma_onehot, J, sigma_onehot)
    return -beta * (e_h + e_J)

def f_batch(sigma_batch, J, h, beta):
    return vmap(lambda x: f_single(x, J, h, beta))(sigma_batch)

@jit
def kernel_embedding(lambda_, d_q):
    """Analytic integral of the kernel over the entire state space."""
    return ((1 + 20 * jnp.exp(-lambda_)) / 21) ** (d_q / 21)

@jit
def double_integral(lambda_, d_q):
    return kernel_embedding(lambda_, d_q)

@jit
def gram_matrix(sigma_batch, lambda_):
    """Computes the Gram matrix K(X, X) using an RBF-like kernel."""
    n, d, q = sigma_batch.shape
    sigma_flat = sigma_batch.reshape(n, -1)  # Shape (n, d*q)

    # Kernel k(x1, x2) = exp(-lambda * HammingDistance(x1, x2))
    def k(x1, x2):
        dot = jnp.dot(x1, x2)  # Number of matching one-hot elements
        return jnp.exp(-lambda_ * d + lambda_ * dot)

    return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

# @jit
def bayesian_cubature(sigma_batch, f_vals, lambda_):
    """Approximates the integral of f using Bayesian Quadrature."""
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

def run_experiment(n_vals, lambda_, d, J, h, seed, beta, X, run_mc, run_bmc):
    key = jax.random.PRNGKey(seed)
    bmc_means, mc_means, times = [], [], []

    start = time.time()
    if run_bmc:
        y = f_batch(X, J, h, beta)
        m = jnp.max(y)
        y_shifted = jnp.exp(y - m)
        Z_shifted, _ = bayesian_cubature(X, y_shifted, lambda_)
        log_Z = jnp.log(jnp.clip(Z_shifted, 1e-20, None)) + m
        jax.block_until_ready(log_Z)
        bmc_means.append(log_Z)
    else:
        bmc_means.append(jnp.nan)

    if run_mc:
        key, subkey = jax.random.split(key)
        y = f_batch(X, J, h, beta)
        m = jnp.max(y)
        y_shifted = jnp.exp(y - m)
        log_Z = jnp.log(jnp.mean(y_shifted)) + m
        jax.block_until_ready(log_Z)
        mc_means.append(log_Z)
    else:
        mc_means.append(jnp.nan)
    times.append(time.time() - start)

    return {
        "bmc_means": jnp.array(bmc_means),
        "mc_means": jnp.array(mc_means),
        "times": jnp.array(times)
    }

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

def main():
    global beta
    lambda_ = 0.001
    beta = 0.5

    alignment = encoding("PF00041.alignment.seed")
    sigma_int, sigma = amino_mapping(alignment)
    len_protein = sigma.shape[1]
    len_aa = 21

    # key = jax.random.PRNGKey(0)
    # J = jax.random.normal(key, shape=(len_protein, len_protein, len_aa, len_aa))
    # J = jnp.abs(J)
    # J = (J + jnp.transpose(J, (1, 0, 3, 2))) / 2.0
    # J = J.at[jnp.arange(len_protein), jnp.arange(len_protein)].set(0.0)
    # J = J/40000.0

    # key, subkey = jax.random.split(key)
    # h = jax.random.normal(subkey, shape=(len_protein, len_aa)) * 0.01
    key = jax.random.PRNGKey(0)
    key, h_key, j_key = jax.random.split(key, 3)

    h = 0.01 * jax.random.normal(h_key, shape=(len_protein, len_aa), dtype=jnp.float64)
    J_init = jax.random.normal(j_key, shape=(len_protein, len_protein, len_aa, len_aa), dtype=jnp.float64)
    J_init = jnp.abs(J_init)
    # Symmetrize J and remove diagonal
    J = (J_init + jnp.transpose(J_init, (1, 0, 3, 2))) / 2.0
    J = J.at[jnp.arange(len_protein), jnp.arange(len_protein)].set(0.0)
    J = J / 40000.0 # Scaling

    # n_vals = jnp.array([5, 20, 50, 98])
    n_vals = jnp.array([98])
    results = run_experiment(n_vals, lambda_, len_protein, J, h, 0, beta, sigma, True, True)
    print("BMC mean:", results["bmc_means"])
    print("MC mean:", results["mc_means"])
    
    print("done")

if __name__ == "__main__":
    main()