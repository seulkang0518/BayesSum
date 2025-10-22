import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
import jax.scipy.linalg as sla
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats

jax.config.update("jax_enable_x64", True)

# ----------------- Matplotlib style -----------------
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

# -----------------------------
# 0) Lattice nearest-neighbor mask
# -----------------------------
def adjacency_mask(d):

    M = jnp.zeros((d, d))
    idx = jnp.arange(d - 1)
    M = M.at[idx, idx + 1].set(1.0)
    M = M.at[idx + 1, idx].set(1.0)

    return M[:, :, None, None]

@partial(jit, static_argnums=(0,1))
def h_chain(d, q):
    a = jnp.arange(q, dtype=jnp.float64)   
    row = 0.1 * a                     
    return jnp.tile(row[None, :], (d, 1))  

@partial(jit, static_argnums=(0, 1))
def J_chain(d, q):
    M = jnp.full((q, q), 0.0, dtype=jnp.float64).at[jnp.arange(q), jnp.arange(q)].set(1.0)
    J = jnp.zeros((d, d, q, q), dtype=jnp.float64)
    idx = jnp.arange(d - 1)
    J = J.at[idx,   idx+1, :, :].set(M)
    J = J.at[idx+1, idx,   :, :].set(M)
    return 0.1 * J
# -----------------------------
# 1) Potts oracle (Z-free)
# -----------------------------
class PottsOracle:
    def __init__(self, d, q, beta):
        self.d, self.q = int(d), int(q)
        self.h = h_chain(self.d, self.q)
        self.J = J_chain(self.d, self.q)
        self.beta = beta

    def energy(self, x):
        xoh = jax.nn.one_hot(x, self.q, dtype=jnp.float64)
        e_h = jnp.einsum('iq,iq->', xoh, self.h)
        e_J = 0.5 * jnp.einsum('iq,ijqr,jr->', xoh, self.J, xoh)
        return -(e_h + e_J)

    def cond_probs(self, x, i):
        base = self.h[i]                   # (q,)
        Ji   = self.J[i]                   # (d, q, q)
        xoh  = jax.nn.one_hot(x, self.q, dtype=jnp.float64)  # (d, q)
        xoh  = xoh.at[i].set(0.0)          # exclude self
        add  = jnp.einsum('jkq,jq->k', Ji, xoh)        # (q,)
        logits = -self.beta * (base + add)
        return jax.nn.softmax(logits)

    @staticmethod
    def neighbors_cyclic(x, q, shift):
        d = x.shape[0]
        x_shift = (x + shift) % q
        X = jnp.tile(x, (d, 1))
        return X.at[jnp.arange(d), jnp.arange(d)].set(x_shift)

    def cyclic_score(self, x):
        nbh = self.neighbors_cyclic(x, self.q, shift=+1)
        Ex = self.energy(x)
        Enb = vmap(self.energy)(nbh)
        return 1.0 - jnp.exp(-self.beta * (Enb - Ex))

# --------------------------------
# 2) Exact ground truth (batched enumeration)
# --------------------------------
def _idx_to_states_np(start, end, q, d):
    idx = np.arange(start, end, dtype=np.int64)
    out = np.empty((idx.size, d), dtype=np.int32)
    x = idx.copy()
    for k in range(d - 1, -1, -1):
        out[:, k] = (x % q).astype(np.int32)
        x //= q
    return out

def exact_expectation_batched(oracle: PottsOracle, f, batch_size: int = 200_000):
    d, q = oracle.d, oracle.q
    total_states = q ** d

    M = -np.inf           # running max of log-weights
    sum_w_shift = 0.0     # sum exp(logw - M)
    sum_fw_shift = 0.0    # sum f * exp(logw - M)

    for start in range(0, total_states, batch_size):
        end = min(start + batch_size, total_states)

        states_np = _idx_to_states_np(start, end, q, d)    # (B, d)
        X = jnp.array(states_np, dtype=jnp.int32)

        E = vmap(oracle.energy)(X)                         # (B,)
        fvals = vmap(f)(X)                                 # (B,)
        logw = -oracle.beta * E                            # (B,)

        m = float(jnp.max(logw))                           # batch max
        if m > M:                                          # rescale previous sums
            sum_w_shift *= np.exp(M - m)
            sum_fw_shift *= np.exp(M - m)
            M = m

        w_shift = jnp.exp(logw - M)
        sum_w_shift += float(jnp.sum(w_shift))
        sum_fw_shift += float(jnp.sum(w_shift * fvals))

    return jnp.array(sum_fw_shift / sum_w_shift, dtype=jnp.float64)

def exact_expectation_auto(oracle: PottsOracle, f, max_states=3**16, batch_size=200_000):
    total_states = oracle.q ** oracle.d
    if total_states <= max_states:
        return exact_expectation_batched(oracle, f, batch_size=batch_size)
    raise RuntimeError(f"q^d={total_states} exceeds max_states={max_states}; "
                       f"increase threshold or add a fallback.")

# -------------------------
# 3) MCMC Samplers
# -------------------------
def gibbs_sample(oracle: PottsOracle, key, n, burnin=300, thinning=2, x0=None):
    d, q = oracle.d, oracle.q
    if x0 is None:
        x0 = jax.random.randint(key, (d,), 0, q)
    total = burnin + n * thinning
    x = jnp.array(x0)
    out = []
    k = key
    for t in range(total):
        for i in range(d):
            k, sub = jax.random.split(k)
            probs = oracle.cond_probs(x, i)
            xi = int(jax.random.categorical(sub, jnp.log(probs)))
            x = x.at[i].set(xi)
        if t >= burnin and ((t - burnin) % thinning == 0):
            out.append(x)
    return jnp.stack(out, axis=0)

def metropolis_hastings_sample(oracle: PottsOracle, key, n, burnin=300, thinning=2, x0=None):
    d, q = oracle.d, oracle.q
    if x0 is None:
        key, subkey = jax.random.split(key)
        x0 = jax.random.randint(subkey, (d,), 0, q)

    total = burnin + n * thinning
    x_current = jnp.array(x0)
    E_current = oracle.energy(x_current)
    out = []
    for t in range(total):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        flip_idx = jax.random.randint(subkey1, shape=(), minval=0, maxval=d)
        a_old = int(x_current[flip_idx])
        cands = jnp.delete(jnp.arange(q), a_old)  # all states except the current one
        new_val = jax.random.choice(subkey2, cands)
        x_proposal = x_current.at[flip_idx].set(new_val)

        E_proposal = oracle.energy(x_proposal)
        delta_E = E_proposal - E_current
        acc = jnp.minimum(1.0, jnp.exp(-oracle.beta * delta_E))
        u = jax.random.uniform(subkey3)
        if u < acc:
            x_current = x_proposal
            E_current = E_proposal
        if t >= burnin and ((t - burnin) % thinning == 0):
            out.append(x_current)
    return jnp.stack(out, axis=0)

# --------------------------------------------
# 4) Cyclic Stein kernel & (plain) BQ posterior mean
# --------------------------------------------
def onehot_flat(x, q):
    return jax.nn.one_hot(x, q, dtype=jnp.float64).reshape(-1)

def rbf_kernel(x, y, gamma):
    h = jnp.sum(x != y)              # Hamming distance
    return jnp.exp(-2.0 * gamma * h) # equals exp(-gamma * ||onehot(x)-onehot(y)||^2)

@partial(jit, static_argnames='oracle')
def stein_kernel_pair(oracle: PottsOracle, x, y, gamma):
    d, q = oracle.d, oracle.q
    sx = oracle.cyclic_score(x)                         # (d,)
    sy = oracle.cyclic_score(y)                         # (d,)
    k_xy = rbf_kernel(x, y, gamma)                      # scalar

    x_inv = oracle.neighbors_cyclic(x, q, shift=-1)     # (d,d)
    y_inv = oracle.neighbors_cyclic(y, q, shift=-1)     # (d,d)

    k_x_yinv = vmap(lambda yr: rbf_kernel(x, yr, gamma))(y_inv)   # (d,)
    k_xinv_y = vmap(lambda xr: rbf_kernel(xr, y, gamma))(x_inv)   # (d,)

    K_xinv_yinv = vmap(lambda xr: vmap(lambda yr: rbf_kernel(xr, yr, gamma))(y_inv))(x_inv)  # (d,d)
    trace_term = jnp.trace(K_xinv_yinv) - jnp.sum(k_x_yinv) - jnp.sum(k_xinv_y) + d * k_xy

    t1 = jnp.dot(sx, sy) * k_xy
    t2 = - jnp.dot(sx, (k_xy - k_x_yinv))
    t3 = - jnp.dot((k_xy - k_xinv_y), sy)
    return t1 + t2 + t3 + trace_term

def stein_kernel_matrix(oracle: PottsOracle, X, gamma):
    pair = lambda xi, xj: stein_kernel_pair(oracle, xi, xj, gamma)
    return vmap(vmap(pair, in_axes=(None, 0)), in_axes=(0, None))(X, X)

def median_gamma_on_int(X, q, n_subset=500, key=None):
    n = X.shape[0]
    if key is not None and n > n_subset:
        idx = jax.random.choice(key, n, shape=(n_subset,), replace=False)
        Xs = X[idx]
    else:
        Xs = X if n <= n_subset else X[:n_subset]
    Z = jax.nn.one_hot(Xs, q, dtype=jnp.float64).reshape(Xs.shape[0], -1)
    norms = jnp.sum(Z ** 2, axis=1, keepdims=True)
    D2 = norms + norms.T - 2.0 * (Z @ Z.T)
    iu = jnp.triu_indices(Z.shape[0], k=1)
    med_sq = jnp.median(D2[iu])
    return float(1.0 / jnp.clip(med_sq, 1e-12, None))


def neg_log_marginal_likelihood(params, X, y, oracle, jitter=1e-8):
    log_gamma, log_c = params
    gamma = jnp.exp(log_gamma)
    c = jnp.exp(log_c)

    n = X.shape[0]
    Ks = stein_kernel_matrix(oracle, X, gamma)        # (n,n)
    K = Ks + c * jnp.ones((n, n)) + jitter * jnp.eye(n)

    L, lower = sla.cho_factor(K, lower=True)
    alpha = sla.cho_solve((L, lower), y)
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    nll = 0.5 * jnp.dot(y, alpha) + 0.5 * log_det_K + 0.5 * n * jnp.log(2 * jnp.pi)
    return nll

def tune_and_estimate_stein_bq(oracle, X, fvals, key, jitter=1e-8):
    gamma_init = median_gamma_on_int(X, oracle.q, key=key)
    c_init = jnp.maximum(jnp.var(fvals), 1e-6)
    init = jnp.log(jnp.array([gamma_init, c_init]))

    nll_fun = lambda p: neg_log_marginal_likelihood(p, X, fvals, oracle, jitter=jitter)
    grad_nll = jax.jit(jax.grad(nll_fun))

    def fun_np(p_np): return float(nll_fun(jnp.array(p_np)))
    def jac_np(p_np): return np.array(grad_nll(jnp.array(p_np)), dtype=np.float64)

    res = minimize(fun=fun_np, x0=np.array(init, dtype=np.float64),
                   method='L-BFGS-B', jac=jac_np)

    log_gamma_opt, log_c_opt = res.x
    gamma_opt = float(np.exp(log_gamma_opt))
    c_opt = float(np.exp(log_c_opt))

    n = X.shape[0]
    Ks_opt = stein_kernel_matrix(oracle, X, gamma_opt)
    K_opt = Ks_opt + c_opt * jnp.ones((n, n)) + jitter * jnp.eye(n)
    L, lower = sla.cho_factor(K_opt, lower=True)
    alpha = sla.cho_solve((L, lower), fvals)               # K^{-1} y
    ones = jnp.ones_like(fvals)
    I_bq_plain = float(c_opt * jnp.dot(ones, alpha))       # c * 1^T K^{-1} y

    return I_bq_plain, {'gamma': gamma_opt, 'c': c_opt, 'opt_result': res}

# -------------------------
# 5) Plot
# -------------------------
def plot_results(sample_sizes, all_errors):
    plt.figure()
    styles = {
        'Gibbs':   {'color': 'black',   'marker': 'o', 'label': 'Gibbs', 'linestyle': '--'},
        'Metropolis Hastings':      {'color': 'black',  'marker': 's', 'label': 'Metropolis Hastings', 'linestyle': '-'},
        'BayesSum':   {'color': 'blue',    'marker': 'D', 'label': 'BayesSum', 'linestyle': '-'},
    }
    for name, error_data in all_errors.items():
        errors_matrix = np.array(error_data).T   # shape: (seeds, len(sample_sizes))^T
        mean_err = np.mean(errors_matrix, axis=0)
        se_err = scipy.stats.sem(errors_matrix, axis=0)
        y_low = np.clip(mean_err - 1.96 * se_err, 1e-12, None)
        y_high = mean_err + 1.96 * se_err
        st = styles[name]
        plt.plot(sample_sizes, np.clip(mean_err, 1e-12, None),
                 linestyle=st['linestyle'], color=st['color'], marker=st['marker'],
                 label=st['label'], lw=2, markersize=8)
        plt.fill_between(sample_sizes, y_low, y_high, color=st['color'], alpha=0.15)

    plt.yscale('log')
    plt.title("Unnormalized Potts")
    plt.xlabel("Number of Points")
    plt.ylabel("Absolute Error")
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()
    filename = "stein_kernel_abs_error.pdf"
    plt.savefig(filename, format='pdf')
    print(f"\nPlot saved as {filename}")
    plt.close()

# -----------------------------
# 6) Experiment driver
# -----------------------------
if __name__ == "__main__":
    # --------- Choose lattice size here ---------                   
    d, q = 15, 3
    beta = 1 / 2.269
    oracle = PottsOracle(d, q, beta)

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Test function
    f = lambda x: 1.0 / (1.0 + jnp.exp(-jnp.sum(x).astype(jnp.float64) / d))

    # Exact expectation (batched full enumeration if q^d <= 3**16)
    E_true = exact_expectation_auto(oracle, f, max_states=3**16, batch_size=200_000)
    print(f"Ground-truth E[sigmoid(sum(x)/d)]: {float(E_true):.6f}\n")

    sample_sizes = [100, 200, 500, 1000]
    seeds = 10
    burnin, thinning = 500, 5

    all_errors = {'Gibbs': [], 'Metropolis Hastings': [], 'BayesSum': []}

    for n in sample_sizes:
        gibbs_errs_n, mh_errs_n, stein_errs_n = [], [], []
        for r in range(seeds):
            key_gibbs = jax.random.fold_in(k3, r)
            key_mh    = jax.random.fold_in(k4, r)

            # Gibbs MC
            X_gibbs = gibbs_sample(oracle, key_gibbs, n, burnin=burnin, thinning=thinning)
            print(X_gibbs)
    #         fvals_gibbs = vmap(f)(X_gibbs)
    #         E_gibbs_mc = float(jnp.mean(fvals_gibbs))
    #         gibbs_errs_n.append(abs(E_gibbs_mc - E_true))

    #         # MH MC
    #         X_mh = metropolis_hastings_sample(oracle, key_mh, n, burnin=burnin, thinning=thinning)
    #         fvals_mh = vmap(f)(X_mh)
    #         E_mh_mc = float(jnp.mean(fvals_mh))
    #         mh_errs_n.append(abs(E_mh_mc - E_true))

    #         # Stein-BQ (plain posterior mean) using Gibbs design
    #         I_bq, _ = tune_and_estimate_stein_bq(oracle, X_gibbs, fvals_gibbs, key_gibbs)
    #         stein_errs_n.append(abs(I_bq - E_true))

    #     print(f"n={n:4d} | Gibbs-MC err       : {np.mean(gibbs_errs_n):.4e} (±{np.std(gibbs_errs_n):.2e})")
    #     print(f"       | MH-MC err          : {np.mean(mh_errs_n):.4e} (±{np.std(mh_errs_n):.2e})")
    #     print(f"       | Stein-BQ err       : {np.mean(stein_errs_n):.4e} (±{np.std(stein_errs_n):.2e})")

    #     all_errors['Gibbs'].append(gibbs_errs_n)
    #     all_errors['Metropolis Hastings'].append(mh_errs_n)
    #     all_errors['BayesSum'].append(stein_errs_n)

    # plot_results(sample_sizes, all_errors)
