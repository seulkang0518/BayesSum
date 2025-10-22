import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import gammaln, gammainc, betainc

jax.config.update("jax_enable_x64", True)

# ----------------------- RNG -----------------------
def make_key(seed=0):
    return random.PRNGKey(seed)

# ----------------------- Helper Functions -----------------------

def poisson(key, lam, n):
    return random.poisson(key, lam=lam, shape=(n,))

def negative_binomial(key, r, p, n):
    key_g, key_p = random.split(key)
    scale = (1.0 - p) / p
    lam = random.gamma(key_g, a=r, shape=(n,)) * scale
    return random.poisson(key_p, lam=lam)

def logarithm(key, p, n, Kmax=20000):
    ks = jnp.arange(1, Kmax + 1, dtype=jnp.float64)
    pmf = (-1.0 / jnp.log1p(-p)) * (p ** ks) / ks
    pmf = pmf / pmf.sum()
    cdf = jnp.cumsum(pmf)
    u = random.uniform(key, shape=(n,))
    idx = jnp.searchsorted(cdf, u, side='right')
    return (idx + 1).astype(jnp.int32)


def ising_pm1(key, d, n):
    b = random.bernoulli(key, 0.5, shape=(n, d))
    return (2 * b.astype(jnp.int32) - 1)

def ising_0_1(key, d, n):
    return random.bernoulli(key, 0.5, shape=(n, d)).astype(jnp.int32)

# ------------ Kernels -------------
def brownian(x, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return jnp.minimum(x[:, None], y[None, :])

def kernel_pointwise_vals(X, y, lam):
    diff = X - y
    return jnp.exp(-lam * jnp.sum(diff * diff, axis=1) / 2.0)

def kernel_tanimoto_vals(X, y01):
    dot = X @ y01
    den = X.sum(axis=1) + y01.sum() - dot
    return jnp.where(den > 0, dot / den, 0.0)


# ------------ Empirical KME -------------
def empirical_kme(samples, y_grid, kernel_fn):
    K = kernel_fn(samples, y_grid)
    return K.mean(axis=0)

# ------------ Closed-form KMEs -------------
@jax.jit
def poisson_brownian_single(y, lam, xmax=200):

    x_vals = jnp.arange(0, xmax + 1)
    log_pmf = x_vals * jnp.log(lam) - lam - gammaln(x_vals + 1)
    pmf = jnp.exp(log_pmf)
    mask = x_vals <= y
    term1 = jnp.sum(jnp.where(mask, x_vals * pmf, 0.0))
    term2 = y * gammainc(y + 1, lam)
    
    return term1 + term2

def nb_brownian_single(y, r, p, nmax=100):
    n_vals = jnp.arange(0, nmax)
    cdf_vals = betainc(r, n_vals + 1.0, p)
    mask = n_vals < y
    return y - jnp.sum(jnp.where(mask, cdf_vals, 0.0))

@jax.jit
def log_brownian_single(y, p, kmax=20000):
    k = jnp.arange(1, kmax + 1)
    coeff = -1.0 / jnp.log1p(-p)
    pmf = coeff * (p ** k) / k
    mask = k <= y
    term1 = jnp.sum(jnp.where(mask, k * pmf, 0.0))
    cdf_y = jnp.sum(jnp.where(mask, pmf, 0.0))
    term2 = y * (1.0 - cdf_y)
    return term1 + term2

def closed_ising_pointwise(d, lam):
    return jnp.exp(-lam * d / 2.0) * jnp.cosh(lam / 2.0) ** d

@jax.jit
def _binom(n, k):
    k = jnp.minimum(k, n)
    return jnp.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

def closed_ising_tanimoto(d, t):
    # Exact expression for expected tanimoto similarity between random vector and fixed vector with ||y|| = t
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


poisson_brownian_vec = vmap(poisson_brownian_single, in_axes=(0, None))
nb_brownian_vec = vmap(nb_brownian_single, in_axes=(0, None, None))
log_brownian_vec = vmap(log_brownian_single, in_axes=(0, None))


# ----------------------- Demo -----------------------
if __name__ == "__main__":
    key = make_key(0)
    n = 400_000
    y_grid = jnp.arange(0, 30, dtype=jnp.int32)

    print("--- Brownian Kernels ---")
    lam = 3.7
    key, k1 = random.split(key)
    Xp = poisson(k1, lam, n)
    emp = empirical_kme(Xp, y_grid, brownian)
    cf = poisson_brownian_vec(y_grid, lam)
    print(f"Poisson + Brownian max|emp - cf| = {float(jnp.max(jnp.abs(emp - cf))):.6f}")


    r, p = 6.0, 0.4
    key, k2 = random.split(key)
    Xnb = negative_binomial(k2, r, p, n)
    emp = empirical_kme(Xnb, y_grid, brownian)
    cf = nb_brownian_vec(y_grid, r, p)
    print(f"NegBin + Brownian max|emp - cf| = {float(jnp.max(jnp.abs(emp - cf))):.6f}")

    p_log = 0.5
    key, k3 = random.split(key)
    Xlog = logarithm(k3, p_log, n)
    emp = empirical_kme(Xlog, y_grid, brownian)
    cf = log_brownian_vec(y_grid, p_log)
    print(f"Log + Brownian max|emp - cf| = {float(jnp.max(jnp.abs(emp - cf))):.6f}")

    print("\n--- Ising Kernels ---")
    d, lam_pw = 12, 1.3
    key, ky, kX = random.split(key, 3)
    yspin = random.choice(ky, jnp.array([-1, 1]), shape=(d,))
    Xspin = ising_pm1(kX, d, n)
    emp = kernel_pointwise_vals(Xspin, yspin, lam_pw).mean()
    cf = closed_ising_pointwise(d, lam_pw)
    print(f"Ising pointwise |emp - cf| = {float(jnp.abs(emp - cf)):.6f}")

    d, t = 14, 6
    key, ky, kX = random.split(key, 3)
    y01 = jnp.concatenate([jnp.ones(t, dtype=jnp.int32), jnp.zeros(d - t, dtype=jnp.int32)])
    y01 = random.permutation(ky, y01)
    X01 = ising_0_1(kX, d, n)
    emp = kernel_tanimoto_vals(X01, y01).mean()
    cf = closed_ising_tanimoto(d, t)
    print(f"Ising Tanimoto |emp - cf| = {float(jnp.abs(emp - cf)):.6f}")


