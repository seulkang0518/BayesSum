import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
import jax.scipy.linalg as sla
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats
import math
from jax.scipy.special import erfinv

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
    def __init__(self, d, q, beta, lam):
        self.d, self.q = int(d), int(q)
        self.h = h_chain(self.d, self.q)
        self.J = J_chain(self.d, self.q)
        self.beta = beta
        self.lam = lam

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
@partial(jit, static_argnames=['q'])
def onehot_flat(x, q):
    return jax.nn.one_hot(x, q, dtype=jnp.float64).reshape(-1)

def rbf_kernel(x, y, gamma):
    h = jnp.sum(x != y)              # Hamming distance
    return jnp.exp(-2.0 * gamma * h) # equals exp(-gamma * ||onehot(x)-onehot(y)||^2)

@jit
def poly2_kernel_flat(xf, yf):
    return (jnp.dot(xf, yf) + 1.0) ** 2

def base_exp_hamming_kernel(xf, yf, lam, d):
    dot = jnp.dot(xf, yf)   
    return jnp.exp(-lam * d + lam * dot)

# @partial(jit, static_argnames='oracle')
# def stein_kernel_pair(oracle: PottsOracle, x, y):
#     d, q = oracle.d, oracle.q
#     sx = oracle.cyclic_score(x)                         
#     sy = oracle.cyclic_score(y)

#     x_oh = onehot_flat(x, q)
#     y_oh = onehot_flat(y, q)
#     k_xy = poly2_kernel_flat(x_oh, y_oh)

#     x_inv = oracle.neighbors_cyclic(x, q, shift=-1)
#     y_inv = oracle.neighbors_cyclic(y, q, shift=-1)
#     xinv_oh = vmap(lambda xx: onehot_flat(xx, q))(x_inv)
#     yinv_oh = vmap(lambda yy: onehot_flat(yy, q))(y_inv)

#     k_x_yinv    = vmap(lambda yf_: poly2_kernel_flat(x_oh, yf_))(yinv_oh)
#     k_xinv_y    = vmap(lambda xf_: poly2_kernel_flat(xf_, y_oh))(xinv_oh)
#     K_xinv_yinv = vmap(lambda xf_: vmap(lambda yf_: poly2_kernel_flat(xf_, yf_))(yinv_oh))(xinv_oh)

#     trace_term = jnp.trace(K_xinv_yinv) - jnp.sum(k_x_yinv) - jnp.sum(k_xinv_y) + d * k_xy
#     t1 = jnp.dot(sx, sy) * k_xy
#     t2 = - jnp.dot(sx, (k_xy - k_x_yinv))
#     t3 = - jnp.dot((k_xy - k_xinv_y), sy)
#     return t1 + t2 + t3 + trace_term

@partial(jit, static_argnames='oracle')
def stein_kernel_pair(oracle: PottsOracle, x, y):
    d, q = oracle.d, oracle.q
    lam = oracle.lam

    sx = oracle.cyclic_score(x)                         
    sy = oracle.cyclic_score(y)

    x_oh = onehot_flat(x, q)
    y_oh = onehot_flat(y, q)
    k_xy = base_exp_hamming_kernel(x_oh, y_oh, lam, d)

    x_inv = oracle.neighbors_cyclic(x, q, shift=-1)
    y_inv = oracle.neighbors_cyclic(y, q, shift=-1)
    xinv_oh = vmap(lambda xx: onehot_flat(xx, q))(x_inv)
    yinv_oh = vmap(lambda yy: onehot_flat(yy, q))(y_inv)

    k_x_yinv    = vmap(lambda yf_: base_exp_hamming_kernel(x_oh, yf_, lam, d))(yinv_oh)
    k_xinv_y    = vmap(lambda xf_: base_exp_hamming_kernel(xf_, y_oh, lam, d))(xinv_oh)
    K_xinv_yinv = vmap(lambda xf_: vmap(lambda yf_: base_exp_hamming_kernel(xf_, yf_, lam, d))(yinv_oh))(xinv_oh)

    trace_term = jnp.trace(K_xinv_yinv) - jnp.sum(k_x_yinv) - jnp.sum(k_xinv_y) + d * k_xy
    t1 = jnp.dot(sx, sy) * k_xy
    t2 = - jnp.dot(sx, (k_xy - k_x_yinv))
    t3 = - jnp.dot((k_xy - k_xinv_y), sy)
    return t1 + t2 + t3 + trace_term

def stein_kernel_matrix(oracle: PottsOracle, X):
    pair = lambda xi, xj: stein_kernel_pair(oracle, xi, xj)
    return vmap(vmap(pair, in_axes=(None, 0)), in_axes=(0, None))(X, X)


def bayes_sard(oracle: PottsOracle, X, fvals, ridge=1e-3, jitter=1e-8):

    Ks = stein_kernel_matrix(oracle, X)
    n = Ks.shape[0]
    Ksr = Ks + (ridge + jitter) * jnp.eye(n)

    Ls, lower = sla.cho_factor(Ksr, lower=True)
    solve = lambda B: sla.cho_solve((Ls, lower), B)

    ones = jnp.ones((n, 1), dtype=jnp.float64)
    v = solve(ones)                          # K^{-1} 1
    denom = (ones.T @ v)[0, 0] + 1e-12
    w = (v / denom).reshape(n)               # (n,)

    mu_hat = w @ fvals                        # (p,)
    var_hat = jnp.maximum(0.0, w @ (Ks @ w))

    return mu_hat, var_hat


def save_errors(filename, sample_sizes, all_errors, all_vars):
    out = {"sample_sizes": np.asarray(sample_sizes, dtype=int)}
    for method, per_N_lists in all_errors.items():
        # per_N_lists: list over sample_sizes, each is list over seeds
        arr = np.asarray([np.asarray(lst, dtype=float) for lst in per_N_lists], dtype=float)
        out[f"{method}_errors"] = arr  # shape (num_sample_sizes, seeds)
    np.savez(filename, **out)

    for method, per_N_lists in all_vars.items():
        # per_N_lists: list over sample_sizes, each is list over seeds
        arr = np.asarray([np.asarray(lst, dtype=float) for lst in per_N_lists], dtype=float)
        out[f"{method}_vars"] = arr  # shape (num_sample_sizes, seeds)
    np.savez(filename + "_vars", **out)

    print(f"Saved {filename} (portable)")

def load_errors(filename1, filename2):
    data1 = np.load(filename1)
    data2 = np.load(filename2)
    sample_sizes = data1["sample_sizes"]
    all_errors = {}
    all_vars = {}

    for key in data1.files:
        if key == "sample_sizes":
            continue
        if key.endswith("_errors"):
            method = key[:-len("_errors")]  # strip suffix
            arr = data1[key]  # shape (num_sample_sizes, seeds)
            all_errors[method] = [row.tolist() for row in arr]

    for key in data2.files:
        if key == "sample_sizes":
            continue
        if key.endswith("_vars"):
            method = key[:-len("_vars")]  # strip suffix
            arr = data2[key]  # shape (num_sample_sizes, seeds)
            all_vars[method] = [row.tolist() for row in arr]

    return sample_sizes, all_errors, all_vars

# -------------------------
# 5) Plot
# -------------------------
def calibration(n_calibration, seeds, lambda_, d, q, beta, t_e, key_seed, burnin, thinning):
    mus, vars_ = [], []

    key = jax.random.PRNGKey(key_seed)

    stein_g_mus_n, stein_mh_mus_n = [], []
    stein_g_vars_n, stein_mh_vars_n = [], []

    for r in range(seeds):
        key_gibbs = jax.random.fold_in(key, r)
        key_mh    = jax.random.fold_in(key, r+5)

        # Gibbs MC
        X_gibbs = gibbs_sample(oracle, key_gibbs, n_calibration, burnin, thinning)
        fvals_gibbs = vmap(f)(X_gibbs)

        # MH MC
        X_mh = metropolis_hastings_sample(oracle, key_mh, n_calibration, burnin, thinning)
        fvals_mh = vmap(f)(X_mh)

        # Stein-BQ using Gibbs design
        bsc_g_mu, bsc_g_var = bayes_sard(oracle, X_gibbs, fvals_gibbs)
        stein_g_mus_n.append(bsc_g_mu)
        stein_g_vars_n.append(bsc_g_var)

        # Stein-BQ using MH design
        bsc_mh_mu, bsc_mh_var = bayes_sard(oracle, X_mh, fvals_mh)
        stein_mh_mus_n.append(bsc_mh_mu)
        stein_mh_vars_n.append(bsc_mh_var)

    stein_g_mus_n  = jnp.array(stein_g_mus_n)
    stein_g_vars_n = jnp.maximum(jnp.array(stein_g_vars_n), 1e-30)

    stein_mh_mus_n  = jnp.array(stein_mh_mus_n)
    stein_mh_vars_n = jnp.maximum(jnp.array(stein_mh_vars_n), 1e-30)

    C_nom = jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.05, 0.99, 20)])
    z = jnp.sqrt(2.0) * erfinv(C_nom) 

    half_g = z[:,None] * jnp.sqrt(stein_g_vars_n)[None,:]
    inside_g = (t_e >= stein_g_mus_n[None,:] - half_g) & (t_e <= stein_g_mus_n[None,:] + half_g)
    emp_cov_g = jnp.mean(inside_g, axis=1)

    half_mh = z[:,None] * jnp.sqrt(stein_mh_vars_n)[None,:]
    inside_mh = (t_e >= stein_mh_mus_n[None,:] - half_mh) & (t_e <= stein_mh_mus_n[None,:] + half_mh)
    emp_cov_mh = jnp.mean(inside_mh, axis=1)

    return float(t_e), C_nom, emp_cov_g, emp_cov_mh


def estimate_alpha(N, MAE, min_index=1):
    N_fit = N[min_index:]
    mae_fit = MAE[min_index:]

    x = np.log(N_fit)
    y = np.log(mae_fit)

    a, b = np.polyfit(x, y, 1)
    alpha = -a
    return alpha, a, b

def format_log_label(n):

    if n <= 0:
        return r"$0$"
    e = math.floor(math.log10(n))
    m = n / (10**e)
    
    m_str = "{:.1f}".format(m)
    if m_str == "1.0":
        return r"$10^{{{}}}$".format(e)
    else:
        return r"${} \times 10^{{{}}}$".format(m_str, e)

def plot_results(sample_sizes, all_errors):
    plt.figure()
    styles = {
        # 'Gibbs':                 {'color': 'black', 'marker': 'o', 'label': 'Gibbs',                 'linestyle': '--'},
        'Metropolis Hastings':   {'color': 'black', 'marker': 's', 'label': 'Metropolis Hastings',   'linestyle': '-'},
        # 'BayesSum (Gibbs)':      {'color': 'blue',  'marker': 'D', 'label': 'Gibbs',                 'linestyle': '--'},
        'BayesSum (MH)':         {'color': 'blue',  'marker': '^', 'label': 'Metropolis Hastings',   'linestyle': '-'},
    }

    eps = 1e-12
    for name, error_data in all_errors.items():
        # error_data: list over sample_sizes, each is list over seeds
        # -> matrix shape (seeds, num_sample_sizes)
        errors_matrix = np.array(error_data).T

        # Robust center and spread across seeds
        median_err = np.median(errors_matrix, axis=0)
        q25 = np.quantile(errors_matrix, 0.25, axis=0)
        q75 = np.quantile(errors_matrix, 0.75, axis=0)

        st = styles[name]

        # Plot median line
        plt.plot(
            sample_sizes, np.clip(median_err, eps, None),
            linestyle=st['linestyle'], color=st['color'], marker=st['marker'],
            label=None, lw=2, markersize=8
        )

        # Shade IQR (25–75%)
        y_low  = np.clip(q25, eps, None)
        y_high = np.clip(q75, eps, None)
        plt.fill_between(sample_sizes, y_low, y_high, color=st['color'], alpha=0.15)

    plt.xscale('log')
    plt.yscale('log')
    plt.title("Potts Model", fontsize=32)
    plt.xlabel("Number of Points", fontsize=32)
    plt.ylabel("Absolute Error", fontsize=32)

    log_labels = [format_log_label(n) for n in sample_sizes]
    
    plt.xticks(sample_sizes, log_labels, fontsize=28) 
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(sample_sizes))
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    # # Manual legend just for the sampler families (as you had)
    # plt.legend(
    #     handles=[
    #         # plt.Line2D([], [], color='black', marker='o', linestyle='--', label='Gibbs'),
    #         plt.Line2D([], [], color='black', marker='s', linestyle='-',  label='Metropolis Hastings'),
    #     ],
    #     fontsize=32, loc='best'
    # )

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
    lam = 0.01
    oracle = PottsOracle(d, q, beta, lam)

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Test function
    f = lambda x: 1.0 / (1.0 + jnp.exp(-jnp.sum(x).astype(jnp.float64) / d))

    # # Exact expectation (batched full enumeration if q^d <= 3**16)
    # E_true = exact_expectation_auto(oracle, f, max_states=3**16, batch_size=200_000)
    # print(f"Ground-truth E[sigmoid(sum(x)/d)]: {float(E_true):.6f}\n")

    # sample_sizes = [1000]
    sample_sizes = [100, 178, 316, 562, 1000]
    seeds = 50
    burnin, thinning = 500, 5

    # all_errors = {'Gibbs': [], 'Metropolis Hastings': [], 'BayesSum (Gibbs)': [], 'BayesSum (MH)': []}
    # all_vars = {'BayesSum (Gibbs)': [], 'BayesSum (MH)': []}
    # bsc_mus = []
    # bsc_vars = []

    # for n in sample_sizes:
    #     gibbs_errs_n, mh_errs_n, stein_g_errs_n, stein_mh_errs_n = [], [], [], []
    #     gibbs_vars_n, mh_vars_n, stein_g_vars_n, stein_mh_vars_n = [], [], [], []

    #     for r in range(seeds):
    #         key_gibbs = jax.random.fold_in(k3, r)
    #         key_mh    = jax.random.fold_in(k4, r)

    #         # Gibbs MC
            # X_gibbs = gibbs_sample(oracle, key_gibbs, n, burnin=burnin, thinning=thinning)
            # print(X_gibbs)
            # fvals_gibbs = vmap(f)(X_gibbs)
            # E_gibbs_mc = float(jnp.mean(fvals_gibbs))
            # gibbs_errs_n.append(abs(E_gibbs_mc - E_true))

        #     # MH MC
        #     X_mh = metropolis_hastings_sample(oracle, key_mh, n, burnin=burnin, thinning=thinning)
        #     fvals_mh = vmap(f)(X_mh)
        #     E_mh_mc = float(jnp.mean(fvals_mh))
        #     mh_errs_n.append(abs(E_mh_mc - E_true))

        #     # Stein-BQ using Gibbs design
        #     bsc_g_mu, bsc_g_var = bayes_sard(oracle, X_gibbs, fvals_gibbs)
        #     # bsc_mus.append(bsc_mu); bsc_vars.append(bsc_var)
        #     stein_g_errs_n.append(abs(bsc_g_mu - E_true))
        #     stein_g_vars_n.append(bsc_g_var)

        #     # Stein-BQ using MH design
        #     bsc_mh_mu, bsc_mh_var = bayes_sard(oracle, X_mh, fvals_mh)
        #     # bsc_mus.append(bsc_mu); bsc_vars.append(bsc_var)
        #     stein_mh_errs_n.append(abs(bsc_mh_mu - E_true))
        #     stein_mh_vars_n.append(bsc_mh_var)

        # # print(f"n={n:4d} | Gibbs-MC err       : {np.mean(gibbs_errs_n):.4e} (±{np.std(gibbs_errs_n):.2e})")
        # # print(f"       | MH-MC err          : {np.mean(mh_errs_n):.4e} (±{np.std(mh_errs_n):.2e})")
        # # print(f"       | Stein-BQ (Gibbs) err       : {np.mean(stein_g_errs_n):.4e} (±{np.std(stein_g_errs_n):.2e})")
        # # print(f"       | Stein-BQ (MG) err       : {np.mean(stein_mh_errs_n):.4e} (±{np.std(stein_mh_errs_n):.2e})")

    #     all_errors['Gibbs'].append(gibbs_errs_n)
    #     all_errors['Metropolis Hastings'].append(mh_errs_n)
    #     all_errors['BayesSum (Gibbs)'].append(stein_g_errs_n)
    #     all_errors['BayesSum (MH)'].append(stein_mh_errs_n)

    #     all_vars['BayesSum (Gibbs)'].append(stein_g_vars_n)
    #     all_vars['BayesSum (MH)'].append(stein_mh_vars_n)

    # save_errors("unnormalised_potts_1.npz", sample_sizes, all_errors, all_vars)
    # print("unnormalised_potts_1.npz")

    sample_sizes, all_errors, all_vars = load_errors("unnormalised_potts_1.npz", "unnormalised_potts_vars_1.npz")
    # print(all_errors)
    all_errors.pop("Gibbs")
    all_errors.pop("BayesSum (Gibbs)")
    methods = ["Gibbs", "Metropolis Hastings", "BayesSum (Gibbs)", "BayesSum (MH)"]

    # def compute_means(results_dict):
    #     means = {}
    #     for method, arrays in results_dict.items():
    #         # arrays is a list of lists → convert each to numpy and take mean
    #         method_means = [np.mean(np.array(arr)) for arr in arrays]
    #         means[method] = method_means
    #     return means

    # means_dict = compute_means(all_errors)

    # for m in methods:
    #     alpha, a, b = estimate_alpha(sample_sizes, means_dict[m], min_index=1)
    #     print(f"{m}: α ≈ {alpha}")

    # print(all_errors)
    # print(all_vars)
    plot_results(sample_sizes, all_errors)

    # # calibrations = {}
    # # calibration_seeds = 50
    # # n_calibration = 60
    # # key_seed = 5
    # # t_e, C_nom, emp_cov_g, emp_cov_mh = calibration(n_calibration, calibration_seeds, lam, d, q, beta, E_true, key_seed, burnin, thinning)
    # # jnp.savez("results/unnormalised_potts_g_calibration_results.npz",
    # #          t_true=t_e, C_nom=jnp.array(C_nom),emp_cov=jnp.array(emp_cov_g))
    # # jnp.savez("results/unnormalised_potts_mh_calibration_results.npz",
    # #          t_true=t_e, C_nom=jnp.array(C_nom),emp_cov = jnp.array(emp_cov_mh))
















