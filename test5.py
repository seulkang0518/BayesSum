# potts_cd_factorized_when_J_small.py
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import vmap, lax

jax.config.update("jax_enable_x64", True)

# ============================== Data: i.i.d. categorical ==============================
def generate_categorical_sequence(key, p, d):
    probs = jnp.array([p, p, 1 - 2 * p], dtype=jnp.float64)
    keys = jrand.split(key, d)
    seq = [jrand.choice(k, 3, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)

def sample_true_sequences(key, p, d, n_samples):
    keys = jrand.split(key, n_samples)
    return jnp.stack([generate_categorical_sequence(k, p, d) for k in keys])  # (N,d)

# ============================== Lattice mask & helpers ==============================
def lattice_J_mask(Lside: int) -> jnp.ndarray:
    d = Lside * Lside
    idx = jnp.arange(d).reshape(Lside, Lside)
    rights = jnp.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    downs  = jnp.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    edges = jnp.concatenate([rights, downs], axis=0)
    mask = jnp.zeros((d, d, 1, 1), dtype=jnp.float64)
    mask = mask.at[edges[:, 0], edges[:, 1], 0, 0].set(1.0)
    mask = mask.at[edges[:, 1], edges[:, 0], 0, 0].set(1.0)
    diag = jnp.arange(d)
    mask = mask.at[diag, diag, 0, 0].set(0.0)
    return mask

def apply_mask_and_sym(J, J_mask):
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    if J_mask is not None:
        Jsym = Jsym * J_mask
    d = Jsym.shape[0]
    diag = jnp.arange(d)
    return Jsym.at[diag, diag, :, :].set(0.0)

# ============================== Potts oracle ==============================
class PottsOracle:
    def __init__(self, h, J, J_mask=None, beta: float = 1.0):
        self.h = jnp.array(h, dtype=jnp.float64)       # (d,q)
        self.beta = float(beta)
        d, q = self.h.shape
        self.d, self.q = int(d), int(q)
        self.J_mask = J_mask
        J = jnp.array(J, dtype=jnp.float64)            # (d,d,q,q)
        self.J = apply_mask_and_sym(J, J_mask)

    def energy(self, x):
        x_oh = jax.nn.one_hot(x, self.q, dtype=jnp.float64)
        e_h = jnp.einsum('iq,iq->', x_oh, self.h)
        e_J = 0.5 * jnp.einsum('iq,ijab,jb->', x_oh, self.J, x_oh)
        return e_h + e_J

    def cond_probs(self, x, i):
        base = self.h[i]                  # (q,)
        Ji   = self.J[i]                  # (d,q,q)
        xoh  = jax.nn.one_hot(x, self.q, dtype=jnp.float64)
        xoh  = xoh.at[i].set(0.0)
        add  = jnp.einsum('jkq,jq->k', Ji, xoh)        # (q,)
        logits = -self.beta * (base + add)
        return jax.nn.softmax(logits)

    def features(self, x):
        """Masked features (gh, gJ_live)."""
        x_oh = jax.nn.one_hot(x, self.q, dtype=jnp.float64)
        gh = self.beta * x_oh
        gJ = self.beta * jnp.einsum('ia,jb->ijab', x_oh, x_oh)
        d = self.d
        diag = jnp.arange(d)
        gJ = gJ.at[diag, diag, :, :].set(0.0)
        if self.J_mask is not None:
            gJ = gJ * jnp.broadcast_to(self.J_mask, gJ.shape)
        return gh, gJ

# ============================== Exact logZ (small L,q) ==============================
def _idx_to_states_np(start, end, q, d):
    import numpy as _np
    idx = _np.arange(start, end, dtype=_np.int64)
    out = _np.empty((idx.size, d), dtype=_np.int32)
    x = idx.copy()
    for k in range(d-1, -1, -1):
        out[:, k] = (x % q).astype(_np.int32); x //= q
    return out

def exact_logZ_batched(oracle: PottsOracle, L: int, q: int, batch_size: int = 200_000):
    d = L * L
    total = q ** d
    M = -np.inf; sum_shift = 0.0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        X = jnp.array(_idx_to_states_np(start, end, q, d), dtype=jnp.int32)
        E = vmap(oracle.energy)(X)
        logw = -oracle.beta * E
        m = float(jnp.max(logw))
        if m > M:
            sum_shift *= np.exp(M - m); M = m
        sum_shift += float(jnp.sum(jnp.exp(logw - M)))
    return M + np.log(sum_shift)

def nll_dataset(oracle: PottsOracle, X_data: jnp.ndarray, L: int, q: int, logZ: float | None = None):
    if logZ is None:
        logZ = exact_logZ_batched(oracle, L, q)
    Es = vmap(oracle.energy)(X_data)
    logp = -oracle.beta * Es - logZ
    return float(-jnp.mean(logp)), float(logZ)

# ============================== Gibbs sampler ==============================
def gibbs_sweep(oracle: PottsOracle, key, x):
    d = oracle.d
    def body(i, carry):
        x, k = carry
        k, sub = jrand.split(k)
        probs = oracle.cond_probs(x, i)
        xi = jrand.categorical(sub, jnp.log(probs))
        x = x.at[i].set(xi)
        return (x, k)
    return lax.fori_loop(0, d, body, (x, key))

def gibbs_k_steps(oracle: PottsOracle, key, x, k_steps: int):
    def body(_, carry): return gibbs_sweep(oracle, carry[1], carry[0])
    return lax.fori_loop(0, k_steps, body, (x, key))

# ============================== Training (CD; factorized negatives for h when J small) ==============================
def train_cd_with_factorized_h_when_J_small(
    X_data_int,                  # (N,d) ints
    L=4, q=3, beta=1/2.269,
    steps=2000, cd_k=400, num_chains=512,
    lr_h0=1e-2, lr_J0=1e-4,      # separate base LRs (J much smaller)
    lambda_h=1e-4, lambda_J=1e-2,
    tau_group=0.0,               # set >0 for block-shrink; 0 disables prox
    ema_decay=0.95,              # Polyak averaging on params
    J_small_thresh=1e-4,         # switch threshold on ||J||
    seed=0
):
    d = L * L
    key = jrand.PRNGKey(seed)
    key, k1, k2, kchains = jrand.split(key, 4)

    # init params (small J)
    h = 0.1   * jrand.normal(k1, (d, q))
    J = 0.005 * jrand.normal(k2, (d, d, q, q))
    J_mask = lattice_J_mask(L)
    oracle = PottsOracle(h, J, J_mask, beta=beta)

    # data + persistent chains
    X_data = jnp.array(X_data_int, dtype=jnp.int32)
    X_chains = jrand.randint(kchains, (num_chains, d), 0, q)
    Mfull = jnp.broadcast_to(J_mask, (d, d, q, q))

    def avg_features(oracle_local, X):
        gh, gJ = vmap(oracle_local.features)(X)  # gJ already masked
        return jnp.mean(gh, axis=0), jnp.mean(gJ, axis=0)

    for t in range(steps):
        # ----- schedules -----
        progress = t / max(1, steps - 1)
        # cosine decay (gentle)
        lr_h = lr_h0 * (0.5 * (1 + jnp.cos(jnp.pi * progress)))
        lr_J = lr_J0 * (0.5 * (1 + jnp.cos(jnp.pi * progress)))

        # ----- positive phase -----
        gh_pos, gJ_pos = avg_features(oracle, X_data)

        # ----- negative phase -----
        # decide whether to use factorized negatives for h (when J ~ 0)
        J_norm = float(jnp.linalg.norm(oracle.J * Mfull))
        use_factorized = J_norm < J_small_thresh

        if use_factorized:
            # exact factorized gh_neg when J ~ 0: p_model = softmax(beta * h)
            p_model = jax.nn.softmax(oracle.beta * oracle.h, axis=1)  # (d,q)
            gh_neg = oracle.beta * p_model
            # still compute gJ_neg from CD (very small anyway), or set to zeros
            gJ_neg = jnp.zeros_like(gJ_pos)
        else:
            key, kneg = jrand.split(key)
            keys = jrand.split(kneg, num_chains)
            def step_one(x, kk): return gibbs_k_steps(oracle, kk, x, cd_k)[0]
            X_chains = vmap(step_one)(X_chains, keys)
            gh_neg, gJ_neg = avg_features(oracle, X_chains)

        # ----- gradients (masked) -----
        gh_diff = gh_pos - gh_neg
        gJ_diff = (gJ_pos - gJ_neg) * Mfull

        grad_h = gh_diff + lambda_h * oracle.h
        grad_J = gJ_diff + lambda_J * oracle.J

        # ----- separate LR updates -----
        h_new = oracle.h - lr_h * grad_h
        J_new = oracle.J - lr_J * grad_J

        # ----- optional proximal group-lasso shrink on J_ij blocks -----
        if tau_group > 0.0:
            block_norms = jnp.linalg.norm(J_new, ord='fro', axis=(2, 3))  # (d,d)
            factors = jnp.maximum(0.0, 1.0 - tau_group / jnp.maximum(block_norms, 1e-12))
            J_new = J_new * factors[:, :, None, None]

        # ----- gauge + mask/sym -----
        h_new = h_new - h_new.mean(axis=1, keepdims=True)
        J_new = apply_mask_and_sym(J_new, J_mask)

        # ----- Polyak averaging (smooth dynamics) -----
        oracle.h = ema_decay * oracle.h + (1 - ema_decay) * h_new
        oracle.J = ema_decay * oracle.J + (1 - ema_decay) * J_new

        # ----- diagnostics -----
        if (t % 50) == 0 or t == steps - 1:
            nll_chk, _ = nll_dataset(oracle, X_data, L=L, q=q)
            # mean L1 marginals (data vs model) — good proxy when J≈0
            onehot = (X_data[..., None] == jnp.arange(q)[None, None, :]).astype(jnp.float64)
            p_data = onehot.mean(axis=0)                            # (d,q)
            p_model = jax.nn.softmax(oracle.beta * oracle.h, axis=1)
            mL1 = float(jnp.mean(jnp.abs(p_data - p_model)))
            EJ = float(jnp.linalg.norm(oracle.J * Mfull))
            print(f"[cd] t={t:4d}  NLL={nll_chk:.6f}  ||J||={EJ:.3e}  meanL1(p):{mL1:.3e}  "
                  f"use_fact={use_factorized}")

    # final exact NLL
    logZ_final = exact_logZ_batched(oracle, L, q)
    nll_final, _ = nll_dataset(oracle, X_data, L=L, q=q, logZ=logZ_final)
    print(f"[final] logZ={logZ_final:.6f}  NLL={nll_final:.6f}")
    return oracle

# ============================== Main ==============================
if __name__ == "__main__":
    # Problem: independent categorical
    L, q, p = 4, 3, 0.2
    d = L * L
    N = 4096
    beta = 1/2.269

    key = jrand.PRNGKey(123)
    X_data = sample_true_sequences(key, p, d, N)  # (N,d)

    oracle = train_cd_with_factorized_h_when_J_small(
        X_data_int=X_data, L=L, q=q, beta=beta,
        steps=2000, cd_k=400, num_chains=512,
        lr_h0=1e-2, lr_J0=1e-4,          # J updates gentler
        lambda_h=1e-4, lambda_J=1e-2,    # stronger shrink on J
        tau_group=0.0,                   # no hard prox (optional)
        ema_decay=0.95,
        J_small_thresh=1e-4,
        seed=0
    )

    # Baseline NLL = d * H([p,p,1-2p])
    pvec = jnp.array([p, p, 1 - 2 * p], dtype=jnp.float64)
    H = -(pvec * jnp.log(pvec)).sum()
    print(f"Baseline d*H = {float(d * H):.6f}")
