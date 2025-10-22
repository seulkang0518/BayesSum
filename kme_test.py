import jax
import jax.numpy as jnp
from jax import vmap, jit, lax, random
from jax.scipy.special import gammaln, gammaincc, betainc, gammainc
from jax.scipy.stats import poisson, nbinom
import scipy.stats as sp_stats
from jax import config

config.update("jax_enable_x64", True)

# --- Kernel Definitions ---
def poly1_kernel(x, y, c):
    return x * y + c

def brownian_kernel(x, y):
    return jnp.minimum(x, y)

# Your efficient, vectorized kernels for the Ising model
def kernel_pointwise_vals(X, y, lam, d):
    diff = X - y
    sq_dist = jnp.sum(diff * diff, axis=1)
    return jnp.exp(-lam * sq_dist / 2.0)

def kernel_tanimoto_vals(X, y):
    dot = X @ y
    union = X.sum(axis=1) + y.sum() - dot
    return jnp.where(union > 0, dot / union, 1.0)

# --- Closed-Form Expressions ---
def poisson_poly1(lam, y, c):
    return y * lam + c

@jit
def stable_comb_log(n, k):
    """Computes log(n choose k) using gammaln for stability."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

@jit
def ising_pointwise_cf(lam, d):
    return jnp.exp(-lam * d / 2.0) * (jnp.cosh(lam / 2.0)**d)

@jit
def ising_tanimoto_cf(y, d):
    """Robust closed-form for Tanimoto kernel embedding."""
    t = jnp.sum(y)
    def outer_loop(s, outer_val):
        a_min = jnp.maximum(0, s - d + t)
        a_max = jnp.minimum(s, t)
        def inner_loop(a, inner_val):
            log_term = stable_comb_log(t, a) + stable_comb_log(d - t, s - a)
            prob_term = jnp.exp(log_term)
            kernel_val = a / (s + t - a)
            return inner_val + jnp.where(s + t - a > 0, kernel_val * prob_term, 0.0)
        inner_sum = lax.fori_loop(a_min, a_max + 1, inner_loop, 0.0)
        return outer_val + inner_sum
    total_sum = lax.fori_loop(0, d + 1, outer_loop, 0.0)
    return total_sum / (2.0**d)

# --- Distribution Samplers ---
def sample_ising_pm1(key, d, n_samples):
    b = random.bernoulli(key, 0.5, shape=(n_samples, d))
    return (2 * b.astype(jnp.int32) - 1).astype(jnp.float64)

def sample_ising_01(key, d, n_samples):
    int_samples = random.randint(key, shape=(n_samples, d), minval=0, maxval=2, dtype=jnp.int32)
    return int_samples.astype(jnp.float64)

# --- Vectorize 1D Closed-Form Functions ---
poisson_poly1_vec = vmap(poisson_poly1, in_axes=(None, 0, None))

# --- CORRECTED: TWO SPECIALIZED EMPIRICAL FUNCTIONS ---

def empirical_embedding_exact(k_fn, x_vals, probs, y_vals):
    """For 1D exact summation using the probability vector."""
    K_matrix = vmap(lambda y: vmap(k_fn, in_axes=(0, None))(x_vals, y))(y_vals)
    return K_matrix @ probs

def empirical_embedding_mc(kernel_fn_vec, samples, y_point):
    """For high-D Monte Carlo estimation using a vectorized kernel."""
    kernel_vals = kernel_fn_vec(samples, y_point)
    return kernel_vals.mean()

# --- Test and compare ---
def compare_embeddings():
    # --- Setup for 1D Distributions ---
    xmax = 500
    x_grid = jnp.arange(0, xmax + 1)
    c = 1.0
    lam_p = 10.0
    poisson_pmf = poisson.pmf(x_grid, mu=lam_p)

    print("--- Verifying 1D Discrete Distributions ---")
    # Call the exact summation function
    emp1 = empirical_embedding_exact(lambda x, y: poly1_kernel(x, y, c), x_grid, poisson_pmf, x_grid)
    cf1 = poisson_poly1_vec(lam_p, x_grid, c)
    print(f"Poisson + Poly1 max|emp - cf| = {float(jnp.max(jnp.abs(emp1 - cf1))):.6f}")

    # --- Setup for Ising Models ---
    d = 20
    n_samples = 4_000_000
    key = random.PRNGKey(0)

    print(f"\n--- Verifying Ising Model (d={d}, N={n_samples}) ---")

    # 1. Test Pointwise (RBF) Kernel
    lam_i = 0.5
    key, subkey = random.split(key)
    samples_pm1 = sample_ising_pm1(subkey, d, n_samples)
    y_pm1 = random.rademacher(random.PRNGKey(1), shape=(d,), dtype=jnp.float64)

    cf_ising1 = ising_pointwise_cf(lam_i, d)
    # Define the kernel function with parameters
    kernel_fn_pointwise = lambda X, y: kernel_pointwise_vals(X, y, lam=lam_i, d=d)
    # Call the Monte Carlo function
    mc1 = empirical_embedding_mc(kernel_fn_pointwise, samples_pm1, y_pm1)
    print(f"Ising {{-1,+1}} + RBF | closed-form - MC | = {jnp.abs(cf_ising1 - mc1):.6f}")

    # 2. Test Tanimoto Kernel
    key, subkey = random.split(key)
    samples_01 = sample_ising_01(subkey, d, n_samples)
    y_01 = (random.uniform(random.PRNGKey(2), shape=(d,)) > 0.5).astype(jnp.float64)
    
    cf_ising2 = ising_tanimoto_cf(y_01, d)
    # Call the Monte Carlo function with the Tanimoto kernel
    mc2 = empirical_embedding_mc(kernel_tanimoto_vals, samples_01, y_01)
    print(f"Ising {{0,1}} + Tanimoto | closed-form - MC | = {jnp.abs(cf_ising2 - mc2):.6f}")

if __name__ == "__main__":
    compare_embeddings()