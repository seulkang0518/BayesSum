import jax
import jax.numpy as jnp
from jax import vmap, random
from jax.scipy.special import gammaln, gammaincc, betainc, gammainc
from jax.scipy.stats import poisson, nbinom
import scipy.stats as sp_stats
from jax import config

config.update("jax_enable_x64", True)

def poly1_kernel(x, y, c):
    return x * y + c

def poly2_kernel(x, y, c):
    return (x * y + c) ** 2

def brownian_kernel(x, y):
    return jnp.minimum(x, y)

def poisson_poly1(lam, y, c):
    return y * lam + c

def poisson_poly2(lam, y, c):
    return y**2 * (lam**2 + lam) + 2 * c * y * lam + c**2

def nb_poly1(r, p, y, c):
    return y * r * (1 - p) / p + c

def nb_poly2(r, p, y, c):
    term1 = (r * (1 - p) / p) ** 2
    term2 = r * (1 - p) / p**2
    return y**2 * (term1 + term2) + 2 * c * y * r * (1 - p) / p + c**2

def skellam_poly1(lam1, lam2, y, c):
    return y * (lam1 - lam2) + c

def skellam_poly2(lam1, lam2, y, c):
    term1 = (lam1 + lam2)
    term2 = (lam1 - lam2) ** 2
    return y**2 * (term1 + term2) + 2 * c * y * (lam1 - lam2) + c**2

def kernel_pointwise_vals(x, y, lam):
    diff = x - y
    sq_dist = jnp.sum(diff * diff, axis=1)
    return jnp.exp(-lam * sq_dist / 2.0)

def kernel_tanimoto_vals(X, y01):
    dot = X @ y01
    den = X.sum(axis=1) + y01.sum() - dot
    return jnp.where(den > 0, dot / den, 0.0)

def poisson_brownian(lam, y, xmax):
    x_vals = jnp.arange(0, xmax + 1)
    pmf = poisson.pmf(x_vals, mu=lam)
    mask = x_vals <= y
    sum_part = jnp.sum(jnp.where(mask, x_vals * pmf, 0.0))
    gamma_part = y * gammainc(y + 1, lam)
    return sum_part + gamma_part

def nb_brownian_algebraic(r, p, y, nmax):
    n_vals = jnp.arange(0, nmax)
    mask = n_vals < y
    cdf_vals = betainc(r, n_vals + 1.0, p)
    sum_of_cdfs = jnp.sum(jnp.where(mask, cdf_vals, 0.0))
    return y - sum_of_cdfs

def log_brownian_algebraic(p, y, kmax):
    k = jnp.arange(1.0, kmax)
    mask = k < y
    sum_terms = (y - k) * (p**k) / k
    coeff = 1.0 / jnp.log1p(-p)
    return y + coeff * jnp.sum(jnp.where(mask, sum_terms, 0.0))

def closed_ising_pointwise(d, lam):
    return jnp.exp(-lam * d / 2.0) * jnp.cosh(lam / 2.0) ** d

def _binom(n, k):
    k = jnp.minimum(k, n)
    return jnp.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

def closed_ising_tanimoto(d, t):
    s_vals = jnp.arange(0, d + 1)
    result = 0.0
    for s in s_vals:
        a_min = jnp.maximum(0, s - d + t)
        a_max = jnp.minimum(s, t)
        a_vals = jnp.arange(a_min, a_max + 1)
        a_term = a_vals / (s + t - a_vals)
        bin1 = _binom(t, a_vals)
        bin2 = _binom(d - t, s - a_vals)
        weight = bin1 * bin2
        result += jnp.sum(a_term * weight)
    return result / 2**d

def ising_pm1(key, d, n):
    b = random.bernoulli(key, 0.5, shape=(n, d))
    return (2 * b.astype(jnp.int32) - 1)

def ising_0_1(key, d, n):
    int_samples = random.randint(key, shape=(n, d), minval=0, maxval=2, dtype=jnp.int32)
    return int_samples.astype(jnp.float64)

# --- Vectorize functions (NO JIT) ---
poisson_poly1_vec = vmap(poisson_poly1, in_axes=(None, 0, None))
poisson_poly2_vec = vmap(poisson_poly2, in_axes=(None, 0, None))
nb_poly1_vec = vmap(nb_poly1, in_axes=(None, None, 0, None))
nb_poly2_vec = vmap(nb_poly2, in_axes=(None, None, 0, None))
skellam_poly1_vec = vmap(skellam_poly1, in_axes=(None, None, 0, None))
skellam_poly2_vec = vmap(skellam_poly2, in_axes=(None, None, 0, None))
poisson_brownian_vec = vmap(poisson_brownian, in_axes=(None, 0, None))
nb_brownian_vec_algebraic = vmap(nb_brownian_algebraic, in_axes=(None, None, 0, None))
log_brownian_vec_algebraic = vmap(log_brownian_algebraic, in_axes=(None, 0, None))

def empirical_embedding_exact(k_fn, x_vals, probs, y_vals):
    K_matrix = vmap(lambda y: vmap(k_fn, in_axes=(0, None))(x_vals, y))(y_vals)
    return K_matrix @ probs

# --- Test and compare ---
def compare_embeddings():
    xmax = 500
    x_grid = jnp.arange(0, xmax + 1)
    c = 1.0

    lam = 10.0
    r, p = 20.0, 0.5
    lam1, lam2 = 8.0, 6.0
    p_log = 0.8

    poisson_pmf = poisson.pmf(x_grid, mu=lam)
    nb_pmf = nbinom.pmf(x_grid, n=r, p=p)
    log_support = jnp.arange(1, xmax + 1)
    log_pmf = sp_stats.logser.pmf(log_support, p_log)
    skellam_support = jnp.arange(-xmax, xmax + 1)
    skellam_pmf = sp_stats.skellam.pmf(skellam_support, lam1, lam2)

    print("--- Polynomial Kernels ---")
    emp1 = empirical_embedding_exact(lambda x, y: poly1_kernel(x, y, c), x_grid, poisson_pmf, x_grid)
    cf1 = poisson_poly1_vec(lam, x_grid, c)
    print(f"Poisson + Poly1 max|emp - cf| = {float(jnp.max(jnp.abs(emp1 - cf1))):.6f}")

    emp2 = empirical_embedding_exact(lambda x, y: poly2_kernel(x, y, c), x_grid, poisson_pmf, x_grid)
    cf2 = poisson_poly2_vec(lam, x_grid, c)
    print(f"Poisson + Poly2 max|emp - cf| = {float(jnp.max(jnp.abs(emp2 - cf2))):.6f}")

    emp3 = empirical_embedding_exact(lambda x, y: poly1_kernel(x, y, c), x_grid, nb_pmf, x_grid)
    cf3 = nb_poly1_vec(r, p, x_grid, c)
    print(f"NegBin + Poly1 max|emp - cf|  = {float(jnp.max(jnp.abs(emp3 - cf3))):.6f}")

    emp4 = empirical_embedding_exact(lambda x, y: poly2_kernel(x, y, c), x_grid, nb_pmf, x_grid)
    cf4 = nb_poly2_vec(r, p, x_grid, c)
    print(f"NegBin + Poly2 max|emp - cf|  = {float(jnp.max(jnp.abs(emp4 - cf4))):.6f}")

    emp5 = empirical_embedding_exact(lambda x, y: poly1_kernel(x, y, c), skellam_support, skellam_pmf, skellam_support)
    cf5 = skellam_poly1_vec(lam1, lam2, skellam_support, c)
    print(f"Skellam + Poly1 max|emp - cf|= {float(jnp.max(jnp.abs(emp5 - cf5))):.6f}")

    emp6 = empirical_embedding_exact(lambda x, y: poly2_kernel(x, y, c), skellam_support, skellam_pmf, skellam_support)
    cf6 = skellam_poly2_vec(lam1, lam2, skellam_support, c)
    print(f"Skellam + Poly2 max|emp - cf|= {float(jnp.max(jnp.abs(emp6 - cf6))):.6f}")

    print("\n--- Brownian & Other Kernels ---")
    emp_pb = empirical_embedding_exact(brownian_kernel, x_grid, poisson_pmf, x_grid)
    cf_pb = poisson_brownian_vec(lam, x_grid, xmax)
    print(f"Poisson + Brownian max|emp - cf| = {float(jnp.max(jnp.abs(emp_pb - cf_pb))):.6f}")

    emp_nb = empirical_embedding_exact(brownian_kernel, x_grid, nb_pmf, x_grid)
    cf_nb_alg = nb_brownian_vec_algebraic(r, p, x_grid, xmax)
    print(f"NegBin + Brownian max|emp - cf|  = {float(jnp.max(jnp.abs(emp_nb - cf_nb_alg))):.6f}")

    emp_log = empirical_embedding_exact(brownian_kernel, log_support, log_pmf, log_support)
    cf_log_alg = log_brownian_vec_algebraic(p_log, log_support, xmax)
    print(f"Log + Brownian max|emp - cf|    = {float(jnp.max(jnp.abs(emp_log - cf_log_alg))):.6f}")

    print("\n--- Ising Model Kernels ---")
    n = 400000
    key = random.PRNGKey(0)
    d, lam_pw = 12, 1.3

    key, ky, kX = random.split(key, 3)
    yspin = random.choice(ky, jnp.array([-1, 1]), shape=(d,))
    Xspin = ising_pm1(kX, d, n)

    emp_ising_pw = kernel_pointwise_vals(Xspin, yspin, lam_pw / 2.0).mean()
    cf_ising_pw = closed_ising_pointwise(d, lam_pw)
    print(f"Ising Pointwise |emp - cf|      = {float(jnp.abs(emp_ising_pw - cf_ising_pw)):.6f}")

    d_tan, t = 14, 6
    key, ky, kX = random.split(key, 3)
    y01 = jnp.concatenate([jnp.ones(t, dtype=jnp.int32), jnp.zeros(d_tan - t, dtype=jnp.int32)])
    y01 = random.permutation(ky, y01)
    X01 = ising_0_1(kX, d_tan, n)
    emp_tanimoto = kernel_tanimoto_vals(X01, y01).mean()
    cf_tanimoto = closed_ising_tanimoto(d_tan, t)
    print(f"Ising Tanimoto |emp - cf|       = {float(jnp.abs(emp_tanimoto - cf_tanimoto)):.6f}")


if __name__ == "__main__":
    compare_embeddings()