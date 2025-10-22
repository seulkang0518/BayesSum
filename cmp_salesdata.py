# # cmp_mc_is_stable.py
# import json
# from functools import partial

# import jax
# jax.config.update("jax_enable_x64", True)

# import jax.numpy as jnp
# from jax import jit, value_and_grad, random
# from jax.scipy.special import gammaln
# import jax.nn as jnn

# import numpy as np
# import optax
# import matplotlib.pyplot as plt

# # -----------------
# # Data I/O
# # -----------------
# def data_loading(filename="sales_hist.json"):
#     with open(filename) as f:
#         data = json.load(f)
#     return {int(k): int(v) for k, v in data.items()}

# def params_init(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     fs = np.array([sales_hist[k] for k in xs])
#     n  = fs.sum()
#     mean = (xs * fs).sum() / n
#     var  = (xs**2 * fs).sum() / n - mean**2
#     # crude moment-based init (reasonable starting point)
#     nu0  = max(1e-3, mean / max(var, 1e-8))
#     lam0 = (mean + (nu0 - 1.0) / (2.0 * nu0)) ** nu0
#     lam0 = float(np.clip(lam0, 1e-6, 1e6))
#     return float(nu0), float(lam0)

# # -----------------
# # Sufficient stats
# # -----------------
# def cmp_suff_stats(sales_hist):
#     ys = jnp.array(list(sales_hist.keys()), dtype=jnp.float64)
#     cs = jnp.array(list(sales_hist.values()), dtype=jnp.float64)
#     n  = cs.sum()
#     s1 = jnp.sum(ys * cs)
#     s2 = jnp.sum(cs * gammaln(ys + 1.0))
#     return n, s1, s2


# # -----------------
# # Truncation "truth" for diagnostics
# # -----------------
# def logZ_trunc(nu, lam, y_max=8000):
#     xs    = jnp.arange(0, y_max + 1, dtype=jnp.float64)
#     log_p = -lam + xs * jnp.log(jnp.maximum(lam, 1e-300)) - gammaln(xs + 1.0)
#     log_f = (1.0 - nu) * gammaln(xs + 1.0)
#     m     = jnp.max(log_f + log_p)
#     logE  = jnp.log(jnp.sum(jnp.exp(log_f + log_p - m))) + m
#     return float(logE + lam)


# # -----------------
# # Fixed-base IS estimator: log Z(lam,nu)
# # -----------------
# @jit
# def logZ_mc_fixed_base_from_samples(Y0, nu, lam, lam0):
#     """
#     Y0   : fixed samples ~ Pois(lam0), shape (M,)
#     lam0 : fixed base rate
#     lam  : current λ
#     nu   : current ν

#     log Z(λ,ν) = lam0 + log E_{λ0}[(Y!)^(1-ν) * (λ/λ0)^Y]
#     """
#     Y0f   = Y0.astype(jnp.float64)
#     # log[(Y!)^(1-ν) (λ/λ0)^Y] = (1-ν)log(Y!) + Y*(log λ - log λ0)
#     log_f = (1.0 - nu) * gammaln(Y0f + 1.0) \
#             + Y0f * (jnp.log(jnp.maximum(lam, 1e-300)) - jnp.log(lam0))

#     m     = jnp.max(log_f)
#     log_E = jnp.log(jnp.mean(jnp.exp(log_f - m))) + m
#     return log_E + lam0


# # -----------------
# # Training (MC + IS, with ν-floor and base refresh)
# # -----------------
# def train_cmp_mc_is(
#     sales_hist,
#     seed=0,
#     mc_n=300_000,           # big batch: stabilizes gradients
#     lr=1e-3,                # gentle learning rate
#     steps=1500,
#     nu_floor=1.0,           # prevents ν→0 collapse
#     refresh_ratio=0.25      # refresh base when |λ/λ0 − 1| > 25%
# ):
#     key = random.PRNGKey(seed)

#     # # Poisson-ish start is safest
#     # ys = jnp.asarray(sorted(sales_hist.keys()), dtype=jnp.float64)
#     # cs = jnp.asarray([sales_hist[int(k)] for k in ys], dtype=jnp.float64)
#     # mean = float((ys * cs).sum() / jnp.maximum(cs.sum(), 1.0))
#     # nu0  = 1.0
#     # lam0 = float(jnp.clip(mean, 1e-6, 1e12))
#     nu0, lam0 = 1.0, 1.2
#     # Raw params (softplus later)
#     def inv_softplus(y):
#         y = np.asarray(y, dtype=np.float64)
#         return np.where(y > 20.0, y, np.log(np.expm1(y)))
#     raw = jnp.array([inv_softplus(nu0), inv_softplus(lam0)], dtype=jnp.float64)
#     # raw = jnp.array([inv_softplus(nu0 - nu_floor), inv_softplus(lam0)], dtype=jnp.float64)

#     # Sufficient stats once
#     n, s1, s2 = cmp_suff_stats(sales_hist)

#     # Fixed-base samples
#     key, k0 = random.split(key)
#     Y0 = random.poisson(k0, lam=lam0, shape=(mc_n,))

#     # Loss (closes over stats). Y0/lam0 provided as args for JIT.
#     @jit
#     def loss_and_aux(raw_params, Y0_arg, lam0_arg):
#         lam = jnn.softplus(raw_params[1]) + 1e-12
#         nu  = nu_floor + jnn.softplus(raw_params[0])  # floor
#         logZ = logZ_mc_fixed_base_from_samples(Y0_arg, nu, lam, lam0_arg)
#         nll = n * logZ - s1 * jnp.log(lam) + nu * s2
#         return nll, {"logZ": logZ, "nu": nu, "lam": lam}

#     # Optimizer
#     opt = optax.chain(optax.clip(1.0), optax.adam(lr))
#     opt_state = opt.init(raw)

#     @jit
#     def step(raw_params, opt_state, Y0_arg, lam0_arg):
#         (loss, aux), grads = value_and_grad(loss_and_aux, has_aux=True)(raw_params, Y0_arg, lam0_arg)
#         updates, opt_state = opt.update(grads, opt_state, raw_params)
#         raw_params = optax.apply_updates(raw_params, updates)
#         return raw_params, opt_state, loss, aux

#     # Train loop with occasional base refresh
#     for t in range(steps):
#         raw, opt_state, loss, aux = step(raw, opt_state, Y0, jnp.asarray(lam0))
#         if (t % 50) == 0:
#             print(f"step {t:04d} | NLL={float(loss):.6f} | logZ={float(aux['logZ']):.6f} "
#                   f"| nu={float(aux['nu']):.6f} | lambda={float(aux['lam']):.6f}")

#         lam_now = float(aux["lam"])
#         if abs(lam_now / lam0 - 1.0) > refresh_ratio:
#             # Refresh the base distribution
#             lam0 = lam_now
#             key, k0 = random.split(key)
#             Y0 = random.poisson(k0, lam=lam0, shape=(mc_n,))

#     lam_hat = float(jnn.softplus(raw[1]))
#     nu_hat  = float(nu_floor + jnn.softplus(raw[0]))
#     return nu_hat, lam_hat


# # -----------------
# # Plot helpers
# # -----------------
# def cmp_logpmf(y, nu, lam, logZ):
#     y = jnp.asarray(y, dtype=jnp.float64)
#     return y * jnp.log(lam) - nu * gammaln(y + 1.0) - logZ

# def empirical_pmf_from_hist(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     cs = np.array([sales_hist[k] for k in xs], dtype=float)
#     p  = cs / cs.sum()
#     return xs, p


# # -----------------
# # Main
# # -----------------
# if __name__ == "__main__":
#     # Load
#     sales_hist = data_loading("sales_hist.json")

#     # Train (MC+IS, stabilized)
#     nu_hat, lam_hat = train_cmp_mc_is(
#         sales_hist,
#         seed=0,
#         mc_n=200,
#         lr=1e-3,
#         steps=1500,
#         nu_floor=0.2,
#         refresh_ratio=0.25
#     )
#     print("Final params:", {"nu": nu_hat, "lambda": lam_hat})

#     # Diagnostics vs truncation
#     logZ_true = logZ_trunc(nu_hat, lam_hat, y_max=8000)
#     print(f"logZ (trunc) ≈ {logZ_true:.6f}")

#     # Plot empirical vs CMP model
#     xs_emp, p_emp = empirical_pmf_from_hist(sales_hist)
#     y_max_plot = max(xs_emp.max(), int(lam_hat + 8*np.sqrt(max(lam_hat, 1e-8))), 60)
#     ys = jnp.arange(0, y_max_plot + 1)

#     logp_model = cmp_logpmf(ys, jnp.asarray(nu_hat), jnp.asarray(lam_hat), jnp.asarray(logZ_true))
#     pmf_model  = np.asarray(jnp.exp(logp_model))

#     plt.figure(figsize=(8, 4.6))
#     plt.bar(xs_emp, p_emp, width=0.9, alpha=0.5, label="data (empirical pmf)")
#     plt.plot(np.asarray(ys), pmf_model, marker="o", lw=1.8,
#              label=f"CMP model (ν={nu_hat:.3f}, λ={lam_hat:.3f})")
#     plt.xlabel("y")
#     plt.ylabel("Probability")
#     plt.title("Empirical vs CMP model (MC-IS stabilized)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # =================== CMP training: BQ (stable) + MC-IS (stabilized) ===================
# import json
# from functools import partial

# import jax
# jax.config.update("jax_enable_x64", True)

# import jax.numpy as jnp
# from jax import jit, vmap, random, value_and_grad
# from jax.scipy.linalg import cho_solve, cho_factor
# from jax.scipy.special import gammaln, gammainc
# import jax.nn as jnn

# import numpy as np
# import optax
# from scipy.stats import poisson as sp_poiss
# import matplotlib.pyplot as plt

# # plt.rcParams['axes.grid'] = True
# # plt.rcParams['font.family'] = 'DeJavu Serif'
# # plt.rcParams['font.serif'] = ['Times New Roman']
# # plt.rcParams['axes.labelsize'] = 26
# # plt.rc('text', usetex=True)
# # plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
# # # plt.rc('font', family='Arial', size=12)
# # plt.rc('axes', titlesize=26, labelsize=26, grid=True)
# # plt.rc('lines', linewidth=2)
# # plt.rc('legend', fontsize=26, frameon=False)
# # plt.rc('xtick', labelsize=26, direction='in')
# # plt.rc('ytick', labelsize=26, direction='in')
# # plt.rc('figure', figsize=(6, 4), dpi=100)

# # -----------------
# # I/O + init utils
# # -----------------
# def data_loading(filename="sales_hist.json"):
#     with open(filename) as f:
#         data = json.load(f)
#     # keys come as strings -> to ints
#     return {int(k): int(v) for k, v in data.items()}

# def params_init(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     fs = np.array([sales_hist[k] for k in xs])
#     n  = fs.sum()
#     mean = (xs * fs).sum() / n
#     var  = (xs**2 * fs).sum() / n - mean**2
#     # crude moment-based init (reasonable starting point)
#     nu0  = max(1e-3, mean / max(var, 1e-8))
#     lam0 = (mean + (nu0 - 1.0) / (2.0 * nu0)) ** nu0
#     lam0 = float(np.clip(lam0, 1e-6, 1e6))
#     return float(nu0), float(lam0)

# def inv_softplus(y):
#     y = np.asarray(y, dtype=np.float64)
#     # numerically stable inverse of softplus
#     return np.where(y > 20.0, y, np.log(np.expm1(y)))

# # -----------------
# # Design (fixed generous grid) for BQ
# # -----------------
# def build_design(lam, tail_eps=1e-10, min_cap=100, max_cap=5000):
#     q = int(sp_poiss.ppf(1.0 - tail_eps, lam))
#     xmax = int(np.clip(q, min_cap, max_cap))
#     X = jnp.arange(0, xmax + 1, dtype=jnp.int32)
#     return X, xmax

# # -----------------
# # Kernels / embeddings for BQ
# # -----------------
# def brownian_kernel(X1, X2):
#     X1 = jnp.atleast_1d(X1).reshape(-1, 1)
#     X2 = jnp.atleast_1d(X2).reshape(-1, 1)
#     return jnp.minimum(X1, X2.T)

# @partial(jit, static_argnames=('xmax',))
# def kernel_embedding_poisson_fixed(lambda_, xi, xmax):
#     xmax   = int(xmax)
#     x_vals = jnp.arange(0, xmax + 1)

#     log_pmf = -lambda_ + x_vals * jnp.log(jnp.maximum(lambda_, 1e-300)) - gammaln(x_vals + 1.0)
#     pmf     = jnp.exp(log_pmf)

#     term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals * pmf, 0.0))
#     tail  = gammainc(xi + 1.0, lambda_)
#     xi_f  = xi.astype(jnp.float64)
#     term2 = xi_f * tail
#     return term1 + term2

# @partial(jit, static_argnames=('xmax',))
# def kernel_embedding_poisson(lambda_, X, xmax):
#     X = jnp.atleast_1d(X)
#     return vmap(lambda xi: kernel_embedding_poisson_fixed(lambda_, xi, xmax))(X)

# def precompute_chol(X):
#     K = brownian_kernel(X, X) + 1e-6 * jnp.eye(X.shape[0], dtype=jnp.float64)
#     Lc, _ = cho_factor(K, lower=True)   # keep only the factor
#     return Lc

# @partial(jit, static_argnames=('xmax',))
# def logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax):
#     mu   = kernel_embedding_poisson(lam, X_grid, xmax).reshape(-1, 1)
#     w    = cho_solve((Lc, True), mu)              # (N,1)

#     logf = (1.0 - nu) * gammaln(X_grid + 1.0)     # (N,)
#     m    = jnp.max(logf)
#     fexp = jnp.exp(logf - m)
#     s    = jnp.dot(w.ravel(), fexp)               # scalar
#     s    = jnp.clip(s, 1e-300, None)
#     return jnp.log(s) + m + lam

# # -----------------
# # MC-IS: stabilized fixed-base estimator (with refresh)
# # -----------------
# @jit
# def logZ_mc_fixed_base_from_samples(Y0, nu, lam, lam0):
#     """
#     Y0 ~ Pois(lam0) fixed; lam0 is the current base rate.
#     log Z(λ,ν) = lam0 + log E_{λ0}[(Y!)^(1-ν) * (λ/λ0)^Y]
#     """
#     Y0f   = Y0.astype(jnp.float64)
#     log_f = (1.0 - nu) * gammaln(Y0f + 1.0) \
#             + Y0f * (jnp.log(jnp.maximum(lam, 1e-300)) - jnp.log(lam0))

#     m     = jnp.max(log_f)
#     log_E = jnp.log(jnp.mean(jnp.exp(log_f - m))) + m
#     return log_E + lam0

# # -----------------
# # Brute-force truncation (diagnostic truth proxy)
# # -----------------
# def logZ_trunc(nu, lam, y_max=8000):
#     xs = jnp.arange(0, y_max+1)
#     # p(y|lam)
#     log_p = -lam + xs * jnp.log(jnp.maximum(lam,1e-300)) - gammaln(xs+1.0)
#     # f_nu(y) = (y!)^{1-nu}
#     log_f = (1.0 - nu) * gammaln(xs + 1.0)
#     # log E[f(Y)] + lam
#     m = jnp.max(log_f + log_p)
#     logE = jnp.log(jnp.sum(jnp.exp(log_f + log_p - m))) + m
#     return float(logE + lam)

# def nll_true(nu, lam, sales_hist, y_max=8000):
#     if isinstance(sales_hist, dict):
#         ys = jnp.array(list(sales_hist.keys()))
#         cs = jnp.array(list(sales_hist.values()))
#         n  = float(cs.sum())
#         s1 = float(jnp.sum(ys * cs))
#         s2 = float(jnp.sum(cs * gammaln(ys + 1.0)))
#     else:
#         x = jnp.asarray(sales_hist)
#         n  = float(x.size)
#         s1 = float(jnp.sum(x))
#         s2 = float(jnp.sum(gammaln(x + 1.0)))
#     logZ = logZ_trunc(nu, lam, y_max=y_max)
#     return n * logZ - s1 * np.log(lam) + nu * s2

# # -----------------
# # Sufficient stats + unified loss wrappers
# # -----------------
# def cmp_suff_stats(sales_hist):
#     if isinstance(sales_hist, dict):
#         ys = jnp.array(list(sales_hist.keys()))
#         cs = jnp.array(list(sales_hist.values()))
#         n = cs.sum()
#         s1 = jnp.sum(ys * cs)
#         s2 = jnp.sum(cs * gammaln(ys + 1.0))
#     else:
#         x = jnp.asarray(sales_hist)
#         n = x.size
#         s1 = jnp.sum(x)
#         s2 = jnp.sum(gammaln(x + 1.0))
#     return n, s1, s2

# # ---- BQ loss (deterministic given grid) ----
# def make_bq_loss(sales_hist, X_grid, Lc, xmax):
#     n, s1, s2 = cmp_suff_stats(sales_hist)

#     @jit
#     def loss_and_aux_bq(raw_params):
#         lam = jnp.clip(jnn.softplus(raw_params[1]), 1e-12, 1e12)
#         nu  = jnp.clip(jnn.softplus(raw_params[0]), 1e-12, 1e12)
#         logZ = logZ_bc_on_grid(X_grid, nu, lam, Lc, int(xmax))
#         nll = n * logZ - s1 * jnp.log(lam) + nu * s2
#         return nll, {"logZ": logZ, "nu": nu, "lam": lam}
#     return loss_and_aux_bq

# # ---- MC-IS loss (with nu-floor; Y0/lam0 passed as args) ----
# def make_mc_is_loss(sales_hist, nu_floor):
#     n, s1, s2 = cmp_suff_stats(sales_hist)

#     @jit
#     def loss_and_aux_mc(raw_params, Y0_arg, lam0_arg):
#         lam = jnn.softplus(raw_params[1]) + 1e-12
#         nu  = nu_floor + jnn.softplus(raw_params[0])
#         logZ = logZ_mc_fixed_base_from_samples(Y0_arg, nu, lam, lam0_arg)
#         nll = n * logZ - s1 * jnp.log(lam) + nu * s2
#         return nll, {"logZ": logZ, "nu": nu, "lam": lam}
#     return loss_and_aux_mc

# # -----------------
# # Training loop (BQ or MC-IS stabilized)
# # -----------------
# def run_experiment(
#     seed,
#     sales_hist,
#     lr=1e-3,
#     num_steps=2000,
#     run_bq=True,
#     # BQ design
#     tail_eps=1e-12, min_cap=400, max_cap=5000, xmax_fixed=200,
#     # MC-IS stabilizers
#     mc_n=None, nu_floor=0.2, refresh_ratio=0.25,
#     # init
#     init_nu=1.0, init_lam=1.2
# ):
#     print(f"\n--- Seed {seed} | run_bq={run_bq} ---")
#     key = random.PRNGKey(seed)

#     # init params (you can swap to params_init(sales_hist) if you prefer)
#     nu0, lam0 = float(init_nu), float(init_lam)
#     raw_params = jnp.array([inv_softplus(nu0), inv_softplus(lam0)], dtype=jnp.float64)

#     # BQ design / precompute
#     if run_bq:
#         # fixed generous grid (you can also build via build_design(lam0, ...))
#         xmax = int(xmax_fixed)
#         X_grid = jnp.arange(0, xmax + 1, dtype=jnp.int32)
#         Lc = precompute_chol(X_grid)
#         loss_and_aux = make_bq_loss(sales_hist, X_grid, Lc, xmax)
#         # optimizer
#         opt = optax.chain(optax.clip(1.0), optax.adam(lr))
#         opt_state = opt.init(raw_params)

#         @jit
#         def step_bq(raw_params, opt_state):
#             (loss, aux), grads = value_and_grad(loss_and_aux, has_aux=True)(raw_params)
#             updates, opt_state = opt.update(grads, opt_state, raw_params)
#             raw_params = optax.apply_updates(raw_params, updates)
#             return raw_params, opt_state, loss, aux

#         for t in range(num_steps):
#             raw_params, opt_state, loss, aux = step_bq(raw_params, opt_state)
#             if (t % 50) == 0:
#                 print(f"step {t:04d} | NLL={float(loss):.6f} | "
#                       f"logZ={float(aux['logZ']):.6f} | "
#                       f"nu={float(aux['nu']):.6f} | lambda={float(aux['lam']):.6f}")

#         lam = float(jnn.softplus(raw_params[1]))
#         nu  = float(jnn.softplus(raw_params[0]))
#         return nu, lam, (X_grid, xmax, Lc), nu0, lam0

#     else:
#         # MC-IS: use stabilized fixed-base samples with refresh
#         # choose MC budget if not set (match BQ grid size heuristic)
#         if mc_n is None:
#             mc_n = int(xmax_fixed) + 1

#         key, k0 = random.split(key)
#         Y0 = random.poisson(k0, lam=lam0, shape=(mc_n,))

#         loss_and_aux = make_mc_is_loss(sales_hist, nu_floor)
#         opt = optax.chain(optax.clip(1.0), optax.adam(lr))
#         opt_state = opt.init(raw_params)

#         @jit
#         def step_mc(raw_params, opt_state, Y0_arg, lam0_arg):
#             (loss, aux), grads = value_and_grad(loss_and_aux, has_aux=True)(raw_params, Y0_arg, lam0_arg)
#             updates, opt_state = opt.update(grads, opt_state, raw_params)
#             raw_params = optax.apply_updates(raw_params, updates)
#             return raw_params, opt_state, loss, aux

#         for t in range(num_steps):
#             raw_params, opt_state, loss, aux = step_mc(raw_params, opt_state, Y0, jnp.asarray(lam0))
#             if (t % 50) == 0:
#                 print(f"step {t:04d} | NLL={float(loss):.6f} | "
#                       f"logZ={float(aux['logZ']):.6f} | "
#                       f"nu={float(aux['nu']):.6f} | lambda={float(aux['lam']):.6f}")

#             lam_now = float(aux["lam"])
#             # refresh the base if λ has drifted too far
#             if abs(lam_now / lam0 - 1.0) > refresh_ratio:
#                 lam0 = lam_now
#                 key, k0 = random.split(key)
#                 Y0 = random.poisson(k0, lam=lam0, shape=(mc_n,))

#         lam = float(jnn.softplus(raw_params[1]))
#         # note nu has the floor inside loss; replicate here:
#         nu  = float(nu_floor + jnn.softplus(raw_params[0]))
#         # return placeholders for BQ tuple to keep the signature consistent
#         return nu, lam, (None, None, None), nu0, lam0

# # ============== Plot helpers ==============
# def cmp_logpmf(y, nu, lam, logZ):
#     y = jnp.asarray(y)
#     return y * jnp.log(lam) - nu * gammaln(y + 1.0) - logZ

# def empirical_pmf_from_hist(sales_hist):
#     xs = np.array(sorted(sales_hist.keys()))
#     cs = np.array([sales_hist[k] for k in xs], dtype=float)
#     p  = cs / cs.sum()
#     return xs, p, int(cs.sum())

# # -----------------
# # Main + diagnostics
# # -----------------
# if __name__ == "__main__":

#     LR        = 5e-3
#     NUM_STEPS = 1500
#     SEED      = 0

#     # MC-IS knobs (ignored if RUN_BQ=True)
#     MC_N            = 201        # increase for smoother grads (e.g., 50_000+ in real runs)
#     NU_FLOOR        = 0.2          # keep away from heavy-tail collapse
#     REFRESH_RATIO   = 0.25         # refresh base when |λ/λ0 − 1| > 25%

#     # BQ grid knob
#     XMAX_FIXED      = 200

#     # Load data
#     sales_hist = data_loading("sales_hist.json")

#     # ys = jnp.asarray(sorted(sales_hist.keys()), dtype=jnp.float64)
#     # cs = jnp.asarray([sales_hist[int(k)] for k in ys], dtype=jnp.float64)
#     # mean = float((ys * cs).sum() / jnp.maximum(cs.sum(), 1.0))
#     nu_init  = 0.5
#     # # lam0 = float(jnp.clip(mean, 1e-6, 1e12))
#     lam_init = 1.2
#     # nu_init, lam_init = params_init(sales_hist)

#     # BQ Train
#     bq_nu_hat, bq_lam_hat, (bq_X_grid, bq_xmax, bq_Lc), bq_nu0, bq_lam0 = run_experiment(
#         SEED, sales_hist,
#         lr=LR, num_steps=NUM_STEPS, run_bq=True,
#         xmax_fixed=XMAX_FIXED,
#         mc_n=MC_N, nu_floor=NU_FLOOR, refresh_ratio=REFRESH_RATIO,
#         init_nu=nu_init, init_lam=lam_init
#     )

#     # MC Train
#     mc_nu_hat, mc_lam_hat, (mc_X_grid, mc_xmax, mc_Lc), mc_nu0, mc_lam0 = run_experiment(
#         SEED, sales_hist,
#         lr=LR, num_steps=NUM_STEPS, run_bq=False,
#         xmax_fixed=XMAX_FIXED,
#         mc_n=MC_N, nu_floor=NU_FLOOR, refresh_ratio=REFRESH_RATIO,
#         init_nu=nu_init, init_lam=lam_init
#     )
#     # For fair plotting, use truncation proxy at the final params
#     logZ_bq = logZ_trunc(bq_nu_hat, bq_lam_hat, y_max=8000)
#     logZ_mc = logZ_trunc(mc_nu_hat, mc_lam_hat, y_max=8000)
#     logZ_init  = logZ_trunc(nu_init, lam_init, y_max=8000)

#     # Compare true NLLs via truncation proxy
#     nll_bq = nll_true(bq_nu_hat, bq_lam_hat, sales_hist, y_max=8000)
#     nll_mc = nll_true(mc_nu_hat, mc_lam_hat, sales_hist, y_max=8000)
#     nll_init  = nll_true(nu_init, lam_init,  sales_hist, y_max=8000)
#     print(f"True NLL (trunc) at final BQ params = {nll_bq:.6f}")
#     print(f"True NLL (trunc) at final MC params = {nll_mc:.6f}")
#     print(f"True NLL (trunc) at init  params = {nll_init:.6f}")

#     # ---- Plot empirical vs model PMF ----
#     xs_emp, p_emp, n_emp = empirical_pmf_from_hist(sales_hist)
#     y_max_plot = max(xs_emp.max(), int(bq_lam_hat + 8*np.sqrt(max(bq_lam_hat, 1e-8))), 60)
#     ys = jnp.arange(0, y_max_plot + 1)

#     logp_bq = cmp_logpmf(ys, jnp.asarray(bq_nu_hat), jnp.asarray(bq_lam_hat), jnp.asarray(logZ_bq))
#     bq_pmf_model  = np.asarray(jnp.exp(logp_bq))

#     logp_mc = cmp_logpmf(ys, jnp.asarray(mc_nu_hat), jnp.asarray(mc_lam_hat), jnp.asarray(logZ_mc))
#     mc_pmf_model  = np.asarray(jnp.exp(logp_mc))

#     logp_model_init = cmp_logpmf(ys, jnp.asarray(nu_init), jnp.asarray(lam_init), jnp.asarray(logZ_init))
#     pmf_model_init  = np.asarray(jnp.exp(logp_model_init))

#     plt.figure(figsize=(8, 4.6))
#     plt.bar(xs_emp, p_emp, width=0.9, alpha=0.5, label="data (empirical pmf)")
#     plt.plot(np.asarray(ys), bq_pmf_model, marker="o", lw=1.8,
#              label=f"DBQ (ν={bq_nu_hat:.3f}, λ={bq_lam_hat:.3f})")
#     plt.plot(np.asarray(ys), mc_pmf_model, marker="^", lw=1.8,
#              label=f"MC (ν={mc_nu_hat:.3f}, λ={mc_lam_hat:.3f})")
#     plt.plot(np.asarray(ys), pmf_model_init, marker="s", lw=1.8,
#              label=f"Initial (ν={nu_init:.3f}, λ={lam_init:.3f})")
#     plt.xlabel("Count")
#     plt.ylabel("Probability")
#     plt.title("CMP Model with Sales Data")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# =================== CMP training: BQ (stable) + MC-IS (stabilized, multi-seed) ===================
import json
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap, random, value_and_grad
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import gammaln, gammainc
import jax.nn as jnn

import numpy as np
import optax
from scipy.stats import poisson as sp_poiss
import matplotlib.pyplot as plt

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
plt.rc('figure', figsize=(6, 4), dpi=100)

# -----------------
# I/O + init utils
# -----------------
def data_loading(filename="sales_hist.json"):
    with open(filename) as f:
        data = json.load(f)
    # keys come as strings -> to ints
    return {int(k): int(v) for k, v in data.items()}

def params_init(sales_hist):
    xs = np.array(sorted(sales_hist.keys()))
    fs = np.array([sales_hist[k] for k in xs])
    n  = fs.sum()
    mean = (xs * fs).sum() / n
    var  = (xs**2 * fs).sum() / n - mean**2
    # crude moment-based init (reasonable starting point)
    nu0  = max(1e-3, mean / max(var, 1e-8))
    lam0 = (mean + (nu0 - 1.0) / (2.0 * nu0)) ** nu0
    lam0 = float(np.clip(lam0, 1e-6, 1e6))
    return float(nu0), float(lam0)

def inv_softplus(y):
    y = np.asarray(y, dtype=np.float64)
    # numerically stable inverse of softplus
    return np.where(y > 20.0, y, np.log(np.expm1(y)))

# -----------------
# Design (fixed generous grid) for BQ
# -----------------
def build_design(lam, tail_eps=1e-10, min_cap=100, max_cap=5000):
    q = int(sp_poiss.ppf(1.0 - tail_eps, lam))
    xmax = int(np.clip(q, min_cap, max_cap))
    X = jnp.arange(0, xmax + 1, dtype=jnp.int32)
    return X, xmax

# -----------------
# Kernels / embeddings for BQ
# -----------------
def brownian_kernel(X1, X2):
    X1 = jnp.atleast_1d(X1).astype(jnp.float64).reshape(-1, 1)
    X2 = jnp.atleast_1d(X2).astype(jnp.float64).reshape(-1, 1)
    return jnp.minimum(X1, X2.T)

@partial(jit, static_argnames=('xmax',))
def kernel_embedding_poisson_fixed(lambda_, xi, xmax):
    x_vals = jnp.arange(0, xmax + 1, dtype=jnp.int32)
    x_vals_f = x_vals.astype(jnp.float64)

    log_pmf = -lambda_ + x_vals_f * jnp.log(jnp.maximum(lambda_, 1e-300)) - gammaln(x_vals_f + 1.0)
    pmf     = jnp.exp(log_pmf)

    term1 = jnp.sum(jnp.where(x_vals <= xi, x_vals_f * pmf, 0.0))
    tail  = gammainc(xi + 1.0, lambda_)  # P[X > xi]
    xi_f  = xi.astype(jnp.float64)
    term2 = xi_f * tail
    return term1 + term2

@partial(jit, static_argnames=('xmax',))
def kernel_embedding_poisson(lambda_, X, xmax):
    X = jnp.atleast_1d(X)
    return vmap(lambda xi: kernel_embedding_poisson_fixed(lambda_, xi, xmax))(X)

def precompute_chol(X):
    K = brownian_kernel(X, X) + 1e-6 * jnp.eye(X.shape[0], dtype=jnp.float64)
    Lc, _ = cho_factor(K, lower=True)   # keep only the factor
    return Lc

@partial(jit, static_argnames=('xmax',))
def logZ_bc_on_grid(X_grid, nu, lam, Lc, xmax):
    mu   = kernel_embedding_poisson(lam, X_grid, xmax).reshape(-1, 1)  # (N,1)
    w    = cho_solve((Lc, True), mu)                                   # (N,1)

    logf = (1.0 - nu) * gammaln(X_grid + 1.0)     # (N,)
    m    = jnp.max(logf)
    fexp = jnp.exp(logf - m)
    s    = jnp.dot(w.ravel(), fexp)               # scalar
    s    = jnp.clip(s, 1e-300, None)
    return jnp.log(s) + m + lam

# -----------------
# MC-IS: stabilized fixed-base estimator (with refresh)
# -----------------
@jit
def logZ_mc_fixed_base_from_samples(Y0, nu, lam, lam0):
    Y0f   = Y0.astype(jnp.float64)
    log_f = (1.0 - nu) * gammaln(Y0f + 1.0) \
            + Y0f * (jnp.log(jnp.maximum(lam, 1e-300)) - jnp.log(lam0))

    m     = jnp.max(log_f)
    # extra guard against log(0) when mc_n is tiny
    log_E = jnp.log(jnp.maximum(jnp.mean(jnp.exp(log_f - m)), 1e-300)) + m
    return log_E + lam0

# -----------------
# Brute-force truncation (diagnostic truth proxy)
# -----------------
def logZ_trunc(nu, lam, y_max=8000):
    xs = jnp.arange(0, y_max+1, dtype=jnp.int32)
    xsf = xs.astype(jnp.float64)
    # p(y|lam)
    log_p = -lam + xsf * jnp.log(jnp.maximum(lam,1e-300)) - gammaln(xsf+1.0)
    # f_nu(y) = (y!)^{1-nu}
    log_f = (1.0 - nu) * gammaln(xsf + 1.0)
    # log E[f(Y)] + lam
    m = jnp.max(log_f + log_p)
    logE = jnp.log(jnp.sum(jnp.exp(log_f + log_p - m))) + m
    return float(logE + lam)

def nll_true(nu, lam, sales_hist, y_max=8000):
    if isinstance(sales_hist, dict):
        ys = jnp.array(list(sales_hist.keys()))
        cs = jnp.array(list(sales_hist.values()))
        n  = float(cs.sum())
        s1 = float(jnp.sum(ys * cs))
        s2 = float(jnp.sum(cs * gammaln(ys + 1.0)))
    else:
        x = jnp.asarray(sales_hist)
        n  = float(x.size)
        s1 = float(jnp.sum(x))
        s2 = float(jnp.sum(gammaln(x + 1.0)))
    logZ = logZ_trunc(nu, lam, y_max=y_max)
    return n * logZ - s1 * np.log(lam) + nu * s2

# -----------------
# Sufficient stats + unified loss wrappers
# -----------------
def cmp_suff_stats(sales_hist):
    if isinstance(sales_hist, dict):
        ys = jnp.array(list(sales_hist.keys()))
        cs = jnp.array(list(sales_hist.values()))
        n = cs.sum()
        s1 = jnp.sum(ys * cs)
        s2 = jnp.sum(cs * gammaln(ys + 1.0))
    else:
        x = jnp.asarray(sales_hist)
        n = x.size
        s1 = jnp.sum(x)
        s2 = jnp.sum(gammaln(x + 1.0))
    return n, s1, s2

# ---- BQ loss (deterministic given grid) ----
def make_bq_loss(sales_hist, X_grid, Lc, xmax):
    n, s1, s2 = cmp_suff_stats(sales_hist)

    @jit
    def loss_and_aux_bq(raw_params):
        lam = jnp.clip(jnn.softplus(raw_params[1]), 1e-12, 1e12)
        nu  = jnp.clip(jnn.softplus(raw_params[0]), 1e-12, 1e12)
        logZ = logZ_bc_on_grid(X_grid, nu, lam, Lc, int(xmax))
        nll = n * logZ - s1 * jnp.log(lam) + nu * s2
        return nll, {"logZ": logZ, "nu": nu, "lam": lam}
    return loss_and_aux_bq

# ---- MC-IS loss (with nu-floor; Y0/lam0 passed as args) ----
def make_mc_is_loss(sales_hist, nu_floor):
    n, s1, s2 = cmp_suff_stats(sales_hist)

    @jit
    def loss_and_aux_mc(raw_params, Y0_arg, lam0_arg):
        lam = jnn.softplus(raw_params[1]) + 1e-12
        nu  = nu_floor + jnn.softplus(raw_params[0])
        logZ = logZ_mc_fixed_base_from_samples(Y0_arg, nu, lam, lam0_arg)
        nll = n * logZ - s1 * jnp.log(lam) + nu * s2
        return nll, {"logZ": logZ, "nu": nu, "lam": lam}
    return loss_and_aux_mc

# -----------------
# One training run (BQ or MC-IS)
# -----------------
def run_experiment(
    seed,
    sales_hist,
    lr=1e-3,
    num_steps=2000,
    run_bq=True,
    # BQ design
    tail_eps=1e-12, min_cap=400, max_cap=5000, xmax_fixed=200,
    # MC-IS stabilizers
    mc_n=None, nu_floor=0.2, refresh_ratio=0.25, refresh_check_every=10,
    # init
    init_nu=1.0, init_lam=1.2,
    mc_init_interpretation="absolute"  # "absolute" -> nu starts at init_nu; "offset" -> nu starts at nu_floor+init_nu
):
    print(f"\n--- Seed {seed} | run_bq={run_bq} ---")
    key = random.PRNGKey(seed)

    # init params
    if run_bq or (mc_init_interpretation == "offset"):
        raw_params = jnp.array([inv_softplus(init_nu), inv_softplus(init_lam)], dtype=jnp.float64)
    else:
        # MC path, "absolute": make ν(0) = init_nu (not nu_floor + init_nu)
        raw_params = jnp.array([inv_softplus(max(init_nu - (nu_floor if nu_floor is not None else 0.0), 1e-12)),
                                inv_softplus(init_lam)], dtype=jnp.float64)

    # BQ design / precompute
    if run_bq:
        xmax = int(xmax_fixed)
        X_grid = jnp.arange(0, xmax + 1, dtype=jnp.int32)
        Lc = precompute_chol(X_grid)
        loss_and_aux = make_bq_loss(sales_hist, X_grid, Lc, xmax)
        # optimizer
        opt = optax.chain(optax.clip(1.0), optax.adam(lr))
        opt_state = opt.init(raw_params)

        @jit
        def step_bq(raw_params, opt_state):
            (loss, aux), grads = value_and_grad(loss_and_aux, has_aux=True)(raw_params)
            updates, opt_state = opt.update(grads, opt_state, raw_params)
            raw_params = optax.apply_updates(raw_params, updates)
            return raw_params, opt_state, loss, aux

        for t in range(num_steps):
            raw_params, opt_state, loss, aux = step_bq(raw_params, opt_state)
            if (t % 50) == 0:
                print(f"step {t:04d} | NLL={float(loss):.6f} | "
                      f"logZ={float(aux['logZ']):.6f} | "
                      f"nu={float(aux['nu']):.6f} | lambda={float(aux['lam']):.6f}")

        lam = float(jnn.softplus(raw_params[1]))
        nu  = float(jnn.softplus(raw_params[0]))
        return nu, lam, (X_grid, xmax, Lc)

    else:
        # MC-IS: use stabilized fixed-base samples with refresh
        if mc_n is None:
            mc_n = int(xmax_fixed) + 1

        lam0 = float(init_lam)
        key, k0 = random.split(key)
        Y0 = random.poisson(k0, lam=lam0, shape=(mc_n,))

        loss_and_aux = make_mc_is_loss(sales_hist, nu_floor)
        opt = optax.chain(optax.clip(1.0), optax.adam(lr))
        opt_state = opt.init(raw_params)

        @jit
        def step_mc(raw_params, opt_state, Y0_arg, lam0_arg):
            (loss, aux), grads = value_and_grad(loss_and_aux, has_aux=True)(raw_params, Y0_arg, lam0_arg)
            updates, opt_state = opt.update(grads, opt_state, raw_params)
            raw_params = optax.apply_updates(raw_params, updates)
            return raw_params, opt_state, loss, aux

        for t in range(num_steps):
            raw_params, opt_state, loss, aux = step_mc(raw_params, opt_state, Y0, jnp.asarray(lam0))
            if (t % 50) == 0:
                print(f"step {t:04d} | NLL={float(loss):.6f} | "
                      f"logZ={float(aux['logZ']):.6f} | "
                      f"nu={float(aux['nu']):.6f} | lambda={float(aux['lam']):.6f}")

            if (t % refresh_check_every) == 0:
                lam_now = float(aux["lam"])
                ratio   = lam_now / lam0
                if (ratio > 1.0 + refresh_ratio) or (ratio < 1.0 / (1.0 + refresh_ratio)):
                    lam0 = lam_now
                    key, k0 = random.split(key)
                    Y0 = random.poisson(k0, lam=lam0, shape=(mc_n,))

        lam = float(jnn.softplus(raw_params[1]))
        nu  = float(nu_floor + jnn.softplus(raw_params[0]))
        return nu, lam, (None, None, None)

# ============== Plot + utilities ==============
def cmp_logpmf(y, nu, lam, logZ):
    y = jnp.asarray(y)
    return y * jnp.log(lam) - nu * gammaln(y + 1.0) - logZ

def empirical_pmf_from_hist(sales_hist):
    xs = np.array(sorted(sales_hist.keys()))
    cs = np.array([sales_hist[k] for k in xs], dtype=float)
    p  = cs / cs.sum()
    return xs, p, int(cs.sum())

def summarize(values):
    arr = np.asarray(values, dtype=float)
    return dict(mean=float(arr.mean()),
                std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                min=float(arr.min()),
                max=float(arr.max()))

def plot_results(sales_hist, bq_nu_hat, bq_lam_hat, mc_nu_mean, mc_lam_mean, nu_init, lam_init):
    xs_emp, p_emp, n_emp = empirical_pmf_from_hist(sales_hist)

    logZ_bq   = logZ_trunc(bq_nu_hat, bq_lam_hat, y_max=8000)
    logZ_mc_m = logZ_trunc(mc_nu_mean, mc_lam_mean, y_max=8000)
    logZ_init = logZ_trunc(nu_init, lam_init, y_max=8000)

    y_max_plot = max(
        xs_emp.max(),
        int(bq_lam_hat + 8*np.sqrt(max(bq_lam_hat, 1e-8))),
        int(mc_lam_mean + 8*np.sqrt(max(mc_lam_mean, 1e-8))),
        60,
    )
    ys = jnp.arange(0, y_max_plot + 1)

    logp_bq   = cmp_logpmf(ys, jnp.asarray(bq_nu_hat),     jnp.asarray(bq_lam_hat),  jnp.asarray(logZ_bq))
    logp_mc_m = cmp_logpmf(ys, jnp.asarray(mc_nu_mean),    jnp.asarray(mc_lam_mean), jnp.asarray(logZ_mc_m))
    logp_init = cmp_logpmf(ys, jnp.asarray(nu_init),       jnp.asarray(lam_init),    jnp.asarray(logZ_init))

    bq_pmf_model   = np.asarray(jnp.exp(logp_bq))
    mc_pmf_model_m = np.asarray(jnp.exp(logp_mc_m))
    pmf_model_init = np.asarray(jnp.exp(logp_init))

    plt.figure(figsize=(10, 6))
    plt.bar(xs_emp, p_emp, width=0.9, alpha=0.45, label="Empirical pmf (data)")
    plt.plot(np.asarray(ys), bq_pmf_model, color = 'b', marker="o", lw=1.8,
         label=fr"BayesSum ($\nu={bq_nu_hat:.3f},\ \lambda={bq_lam_hat:.3f}$)")
    plt.plot(np.asarray(ys), mc_pmf_model_m, color = 'k', marker="^", lw=1.8,
	         label=fr"MC ($\nu={mc_nu_mean:.3f},\ \lambda={mc_lam_mean:.3f}$)")
    plt.plot(np.asarray(ys), pmf_model_init, linestyle = '--', color = 'g', marker="s", lw=1.5, alpha=0.8,
	         label=fr"Initial ($\nu={nu_init:.3f},\ \lambda={lam_init:.3f}$)")
    plt.xlabel("Count")
    plt.ylabel("Probability")
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.title("CMP: BayesSum vs MC for  Sales Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig('cmp_sales_data.pdf')
    plt.clos()

# -----------------
# Main + diagnostics (multi-seed MC)
# -----------------
if __name__ == "__main__":
    # Hyperparams
    LR        = 5e-3
    NUM_STEPS = 1500

    # MC-IS knobs
    MC_N              = 201          # increase for smoother grads (e.g., 10_000+ for real runs)
    NU_FLOOR          = 0.2          # keep away from heavy-tail collapse
    REFRESH_RATIO     = 0.25         # refresh when |λ/λ0 − 1| > 25%
    REFRESH_EVERY     = 10

    # BQ grid knob
    XMAX_FIXED        = 200

    # Seeds
    MC_SEEDS = list(range(10))  # multi-seed MC
    BQ_SEED  = 0                # BQ is deterministic w.r.t. fixed grid; one run is enough

    # Load data
    sales_hist = data_loading("sales_hist.json")

    # Inits
    nu_init  = 0.5
    lam_init = 1.2
    # (Alternatively) nu_init, lam_init = params_init(sales_hist)

    # ===== BQ Train (single deterministic run) =====
    bq_nu_hat, bq_lam_hat, (bq_X_grid, bq_xmax, bq_Lc) = run_experiment(
        BQ_SEED, sales_hist,
        lr=LR, num_steps=NUM_STEPS, run_bq=True,
        xmax_fixed=XMAX_FIXED,
        mc_n=MC_N, nu_floor=NU_FLOOR, refresh_ratio=REFRESH_RATIO, refresh_check_every=REFRESH_EVERY,
        init_nu=nu_init, init_lam=lam_init
    )

    # Evaluate true NLL via trunc proxy
    logZ_bq  = logZ_trunc(bq_nu_hat, bq_lam_hat, y_max=8000)
    nll_bq   = nll_true(bq_nu_hat, bq_lam_hat, sales_hist, y_max=8000)

    print(f"\n[BQ] params: ν={bq_nu_hat:.6f}, λ={bq_lam_hat:.6f} | True NLL ≈ {nll_bq:.3f}")

    # ===== MC-IS Train (multi-seed) =====
    mc_records = []
    for seed in MC_SEEDS:
        mc_nu_hat, mc_lam_hat, _ = run_experiment(
            seed, sales_hist,
            lr=LR, num_steps=NUM_STEPS, run_bq=False,
            xmax_fixed=XMAX_FIXED,
            mc_n=MC_N, nu_floor=NU_FLOOR, refresh_ratio=REFRESH_RATIO, refresh_check_every=REFRESH_EVERY,
            init_nu=nu_init, init_lam=lam_init,
            mc_init_interpretation="absolute"  # set to "offset" to interpret init_nu as nu offset above floor
        )
        nll_mc = nll_true(mc_nu_hat, mc_lam_hat, sales_hist, y_max=8000)
        mc_records.append(dict(seed=seed, nu=mc_nu_hat, lam=mc_lam_hat, nll=nll_mc))
        print(f"[MC seed {seed:02d}] ν={mc_nu_hat:.6f}, λ={mc_lam_hat:.6f} | True NLL ≈ {nll_mc:.3f}")

    # ---- Summaries ----
    mc_nus  = np.array([r["nu"]  for r in mc_records], dtype=float)
    mc_lams = np.array([r["lam"] for r in mc_records], dtype=float)
    mc_nlls = np.array([r["nll"] for r in mc_records], dtype=float)

    s_nus  = summarize(mc_nus)
    s_lams = summarize(mc_lams)
    s_nlls = summarize(mc_nlls)
    print("\n[MC multi-seed summary]")
    print(f"ν:   mean={s_nus['mean']:.4f}  std={s_nus['std']:.4f}  min={s_nus['min']:.4f}  max={s_nus['max']:.4f}")
    print(f"λ:   mean={s_lams['mean']:.4f}  std={s_lams['std']:.4f}  min={s_lams['min']:.4f}  max={s_lams['max']:.4f}")
    print(f"NLL: mean={s_nlls['mean']:.2f}  std={s_nlls['std']:.2f}  min={s_nlls['min']:.2f}  max={s_nlls['max']:.2f}")

    # ---- Use MEAN(ν), MEAN(λ) for MC plotting ----
    mc_nu_mean = float(mc_nus.mean())
    mc_lam_mean = float(mc_lams.mean())
    print(f"\n[MC mean-params for plotting] ν̄={mc_nu_mean:.6f}, λ̄={mc_lam_mean:.6f}")


    plot_results(sales_hist, bq_nu_hat, bq_lam_hat, mc_nu_mean, mc_lam_mean, nu_init, lam_init)
    