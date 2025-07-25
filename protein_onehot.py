from Bio import AlignIO
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve
import time
jax.config.update("jax_enable_x64", True)

@jit
def f_single(sigma_onehot, J, beta):
    energy = jnp.einsum('ik,ijkl,jl->', sigma_onehot, J, sigma_onehot)
    return jnp.exp(-beta * energy)

def f_batch(sigma_batch, J, beta):
    return vmap(lambda x: f_single(x, J, beta))(sigma_batch)

@jit
def kernel_embedding(lambda_, d_q):
    return ((1 + 20 * jnp.exp(-lambda_)) / 21) ** (d_q / 21)

@jit
def double_integral(lambda_, d_q):
    return kernel_embedding(lambda_, d_q)

@jit
def gram_matrix(sigma_batch, lambda_):
    n, d, q = sigma_batch.shape
    sigma_flat = sigma_batch.reshape(n, -1)
    d_q = sigma_flat.shape[1]
    def k(x1, x2):
        dot = jnp.dot(x1, x2)
        return jnp.exp(-lambda_ * jnp.sum(x1 != x2))
    return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

@jit
def kernel_vec(sigma_onehot, sigma_batch, lambda_):
    x = sigma_onehot.reshape(-1)
    X = sigma_batch.reshape(sigma_batch.shape[0], -1)
    d_q = x.shape[0]
    dots = X @ x
    return jnp.exp(-lambda_ * d_q + lambda_ * dots).reshape(-1, 1)

# @jit
def bayesian_cubature(sigma_batch, f_vals, lambda_):
    n = sigma_batch.shape[0]
    sigma_flat = sigma_batch.reshape(n, -1)
    d_q = sigma_flat.shape[1]

    K = gram_matrix(sigma_batch, lambda_) + 1e-8 * jnp.eye(n)
    # # Diagnostic: condition number
    # condK = jnp.linalg.cond(K)
    # print("Condition number of K (n={0}):".format(n), condK)

    try:
        L, lower = cho_factor(K)
    except:
        raise ValueError("Cholesky failed: kernel matrix not positive definite.")
    z = jnp.full((n, 1), kernel_embedding(lambda_, d_q))
    f_vals = f_vals.reshape(n, 1)
    K_inv_z = cho_solve((L, lower), z)
    K_inv_f = cho_solve((L, lower), f_vals)
    mean = (z.T @ K_inv_f)[0, 0]
    var = double_integral(lambda_, d_q) - (z.T @ K_inv_z)[0, 0]
    return mean, var

def sample_states(key, d, n):
    raw = jax.random.randint(key, shape=(n, d), minval=0, maxval=21)
    return jax.nn.one_hot(raw, num_classes=21).astype(jnp.float32)

def run_experiment(n_vals, lambda_, d, J, seed, beta, run_mc, run_bmc):
    key = jax.random.PRNGKey(seed)
    bmc_means, mc_means, times = [], [], []
    for n in n_vals:
        start = time.time()
        if run_bmc:
            key, subkey = jax.random.split(key)
            X = sample_states(subkey, d, n)

            # # Diagnostics: energy statistics
            # energies = vmap(lambda x: jnp.einsum('ik,ijkl,jl->', x, J, x))(X)
            # print(f"n={n}, energy mean: {jnp.mean(energies):.4f}, min: {jnp.min(energies):.4f}, max: {jnp.max(energies):.4f}")
        
            y = f_batch(X, J, beta)

            # # Diagnostics: mean and variance of f(sigma)
            # print(f"n={n}, f_batch mean: {jnp.mean(y):.4f}, var: {jnp.var(y):.4f}")

            mu_bmc, _ = bayesian_cubature(X, y, lambda_)
            jax.block_until_ready(mu_bmc)
            bmc_means.append(mu_bmc)
        else:
            bmc_means.append(jnp.nan)
        if run_mc:
            key, subkey = jax.random.split(key)
            X_mc = sample_states(subkey, d, n)
            X_mc_int = jnp.argmax(X_mc, axis=-1)
            energy = jnp.einsum('ik,ijkl,jl->', X_mc[0], J, X_mc[0])
            n, d = X_mc_int.shape
            f_vals_mc = f_batch(X_mc, J, beta)
            mu_mc = jnp.mean(f_vals_mc)
            jax.block_until_ready(mu_mc)
            mc_means.append(mu_mc)
        else:
            mc_means.append(jnp.nan)
        times.append(time.time() - start)
    return {
        "bmc_means": jnp.array(bmc_means),
        "mc_means": jnp.array(mc_means),
        "times": jnp.array(times)
    }

def run_multiple_seeds(n_vals, lambda_, d, J, num_seeds, beta, run_mc, run_bmc):
    bmc_all, mc_all, times_all = [], [], []
    for seed in range(num_seeds):
        result = run_experiment(n_vals, lambda_, d, J, seed, beta, run_mc, run_bmc)
        bmc_all.append(result["bmc_means"])
        mc_all.append(result["mc_means"])
        times_all.append(result["times"])
    return {
        "bmc_mean_error": jnp.mean(jnp.abs(jnp.stack(bmc_all)), axis=0),
        "mc_mean_error": jnp.mean(jnp.abs(jnp.stack(mc_all)), axis=0),
        "times_mean": jnp.mean(jnp.stack(times_all))
    }

def encoding(filename):
    return AlignIO.read(filename, "stockholm")

def amino_mapping(alignment):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
    aa_int = {amino_acid: i + 1 for i, amino_acid in enumerate(amino_acids)}
    total_aa = len(amino_acids)
    encoded = [[aa_int.get(residue, total_aa - 1) for residue in str(record.seq)] for record in alignment]
    sigma_int = jnp.array(encoded)
    sigma = jax.nn.one_hot(sigma_int, num_classes=total_aa)
    return encoded, sigma


@jax.jit
def log_prob(sigma_onehot, h, J, log_Z):
    return -jnp.einsum('ik,ijkl,jl->', sigma_onehot, J, sigma_onehot) - log_Z

# Negative log-likelihood over all sequences
@jax.jit
def negative_log_likelihood(params, Sigma, log_Z):
    h, J = params
    log_probs = jax.vmap(lambda sigma: log_prob(sigma, h, J, log_Z))(Sigma)
    return -jnp.mean(log_probs)

@jax.jit
def update(params, opt_state, Sigma, log_Z):
    loss, grads = jax.value_and_grad(negative_log_likelihood)(params, Sigma, log_Z)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def main():
    k_b = 1
    T_c = 5*10000
    beta = 1 / (k_b * T_c)
    lambda_ = 1.0/(276*21)
    alignment = encoding("PF01738_seed.sto")
    _, sigma = amino_mapping(alignment)
    len_protein = sigma.shape[1]
    len_aa = 21
    key = jax.random.PRNGKey(0)
    J = jax.random.normal(key, shape=(len_protein, len_protein, len_aa, len_aa)) 
    J = jnp.abs(J)
    J = (J + jnp.transpose(J, (1, 0, 3, 2))) / 2
    J = J.at[jnp.arange(len_protein), jnp.arange(len_protein)].set(0.0)
    n_vals = jnp.array([10, 50, 100, 200, 300])

    results = run_multiple_seeds(n_vals, lambda_, len_protein, J, 10, beta, True, True)
    print("BMC mean:", results["bmc_mean_error"])
    print("MC mean:", results["mc_mean_error"])
    # for lambda_ in [0.05, 0.052, 0.054, 0.056, 0.058, 0.06]:
    #     results = run_multiple_seeds(n_vals, lambda_, len_protein, J, 10, beta, True, True)
    #     print("BMC mean:", results["bmc_mean_error"])
    #     print("MC mean:", results["mc_mean_error"])

    # log_Z = approximate_log_Z(lambda_)
    # num_steps = 100

    # for step in range(num_steps):
    #     params, opt_state, loss = update(params, opt_state, Sigma, log_Z)
    #     if step % 10 == 0:
    #         print(f"Step {step}, Negative log-likelihood: {loss:.4f}")


if __name__ == "__main__":
    main()