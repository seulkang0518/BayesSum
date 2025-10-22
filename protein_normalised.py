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
import pickle
jax.config.update("jax_enable_x64", True)

learning_rate = 0.05
optimizer = optax.adam(learning_rate)

@jit
def f_single(sigma_onehot, J, h, beta):
    e_h = jnp.einsum('ik,ik->', sigma_onehot, h)  
    e_J = jnp.einsum('ik,ijkl,jl->', sigma_onehot, J, sigma_onehot)
    return jnp.exp(-beta * (e_h + e_J))

def f_batch(sigma_batch, J, h, beta):
    return vmap(lambda x: f_single(x, J, h, beta))(sigma_batch)

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
        dot = jnp.dot(x1, x2)  # matches
        return jnp.exp(-lambda_ * d + lambda_ * dot)
    return vmap(lambda x: vmap(lambda y: k(x, y))(sigma_flat))(sigma_flat)

@jit
def empirical_kernel_embedding(X, lambda_):
    n = X.shape[0]
    gram = gram_matrix(X, lambda_)
    return jnp.mean(gram)

@jit
def kernel_vec(sigma_onehot, sigma_batch, lambda_):
    x = sigma_onehot.reshape(-1)
    X = sigma_batch.reshape(sigma_batch.shape[0], -1)
    dot = X @ x
    d = sigma_onehot.shape[0]
    return jnp.exp(-lambda_ * d + lambda_ * dot).reshape(-1, 1)

@jit
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

    ## diagnostics
    z_emp = jnp.mean(K, axis=0).reshape(-1, 1)
    ########

    var = double_integral(lambda_, d_q) - (z.T @ K_inv_z)[0, 0]
    return mean, var

def sample_states(key, d, n):
    raw = jax.random.randint(key, shape=(n, d), minval=0, maxval=21)
    return jax.nn.one_hot(raw, num_classes=21).astype(jnp.float32)

def run_experiment(n_vals, lambda_, d, J, h, seed, beta, run_mc, run_bmc):
    key = jax.random.PRNGKey(seed)
    bmc_means, mc_means, times = [], [], []
    for n in n_vals:
        start = time.time()
        if run_bmc:
            key, subkey = jax.random.split(key)
            X = sample_states(subkey, d, n)

            energies = vmap(lambda x: jnp.einsum('ik,ijkl,jl->', x, J, x))(X)
            mean_E = jnp.mean(energies)
            std_E = jnp.std(energies)
                
            y = jnp.exp(-beta * ((energies - mean_E) / std_E))

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
            f_vals_mc = f_batch(X_mc, J, h, beta)
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

def run_multiple_seeds(n_vals, lambda_, d, J, h, num_seeds, beta, run_mc, run_bmc):
    bmc_all, mc_all, times_all = [], [], []
    for seed in range(num_seeds):
        result = run_experiment(n_vals, lambda_, d, J, h, seed, beta, run_mc, run_bmc)
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
    aa_int = {amino_acid: i for i, amino_acid in enumerate(amino_acids)}
    total_aa = len(amino_acids)
    
    encoded = [[aa_int.get(residue, total_aa - 1) for residue in str(record.seq)] for record in alignment]
    
    sigma_int = jnp.array(encoded, dtype=jnp.int32)  
    sigma = jax.nn.one_hot(sigma_int, num_classes=total_aa)
    return sigma_int, sigma


###### Gradient Descent ########
@jax.jit
def log_prob_train(x, h, J, log_Z, mean_E, std_E):
    e_J = jnp.einsum('ik,ijkl,jl->', x, J, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    energy = e_J + e_h
    energy = (energy - mean_E) / std_E
    return -beta * energy - log_Z

@jax.jit
def log_prob_eval(x, h, J, log_Z):
    e_J = jnp.einsum('ik,ijkl,jl->', x, J, x)
    e_h = jnp.einsum('ik,ik->', x, h)
    energy = e_J + e_h
    return -beta * energy - log_Z

@jax.jit
def negative_log_likelihood(params, Sigma, log_Z, mean_E, std_E):
    h, J = params
    log_probs = jax.vmap(lambda sigma: log_prob_train(sigma, h, J, log_Z, mean_E, std_E))(Sigma)
    return -jnp.mean(log_probs)

@jax.jit
def update(params, opt_state, Sigma, log_Z):
    h, J = params
    energies = jax.vmap(lambda x: jnp.einsum('ik,ijkl,jl->', x, J, x))(Sigma)
    mean_E = jnp.mean(energies)
    std_E = jnp.std(energies)
    loss, grads = jax.value_and_grad(negative_log_likelihood)(params, Sigma, log_Z, mean_E, std_E)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, mean_E, std_E

#### Testing #######

def swap_halves(seq):
    mid = len(seq) // 2
    return np.concatenate([seq[mid:], seq[:mid]])

def generate_shuffled_msa(msa):
    shuffled = np.array(msa)  # convert to mutable NumPy array
    for i in range(shuffled.shape[1]):
        np.random.shuffle(shuffled[:, i])
    return jnp.array(shuffled)  # convert back to JAX array

def generate_log_likelihoods(msa_int, log_prob_fn):

    n = msa_int.shape[0]
    sample_size = min(n, 100)
    indices = np.random.choice(n, size=sample_size, replace=False)
    log_in_in, log_in_out1, log_in_out2 = [], [], []
    shuffled_msa = generate_shuffled_msa(msa_int)

    for idx in indices:
        x_in = jax.nn.one_hot(msa_int[idx], num_classes=21)
        x_out1 = jax.nn.one_hot(shuffled_msa[idx], num_classes=21)
        x_out2 = jax.nn.one_hot(swap_halves(msa_int[idx]), num_classes=21)

        log_in_in.append(log_prob_fn(x_in))
        log_in_out1.append(log_prob_fn(x_out1))
        log_in_out2.append(log_prob_fn(x_out2))

    return log_in_in, log_in_out1, log_in_out2


def generate_log_ratios(msa_int, log_prob_fn):
    n = msa_int.shape[0]
    num_samples = min(100, n)

    indices = np.random.choice(n, size=num_samples, replace=False)
    shuffled_msa = generate_shuffled_msa(msa_int)

    log_ratios_in_in, log_ratios_in_out1, log_ratios_in_out2 = [], [], []

    # Sanity check: Are log-probs changing at all?
    some_indices = np.random.choice(n, size=10, replace=False)
    log_probs = []

    for idx in some_indices:
        x = jax.nn.one_hot(msa_int[idx], num_classes=21)
        lp = log_prob_fn(x)
        log_probs.append(lp)

    log_probs = jnp.array(log_probs)
    print("Log probs of 10 MSA sequences:", log_probs)
    print("Mean:", jnp.mean(log_probs))
    print("Std dev:", jnp.std(log_probs))

    for _ in indices:
        i1, i2 = np.random.choice(n, size=2, replace=False)
        x1 = jax.nn.one_hot(msa_int[i1], num_classes=21)
        x2 = jax.nn.one_hot(msa_int[i2], num_classes=21)
        x_out1 = jax.nn.one_hot(shuffled_msa[i1], num_classes=21)
        x_out2 = jax.nn.one_hot(swap_halves(msa_int[i1]), num_classes=21)

        # Compute log-ratios
        log_ratios_in_in.append(log_prob_fn(x2) - log_prob_fn(x1))
        log_ratios_in_out1.append(log_prob_fn(x_out1) - log_prob_fn(x1))
        log_ratios_in_out2.append(log_prob_fn(x_out2) - log_prob_fn(x1))

    return (
        jnp.array(log_ratios_in_in),
        jnp.array(log_ratios_in_out1),
        jnp.array(log_ratios_in_out2)
    )

def sample_potts_mcmc(key, h, J, beta, n_samples=10000, n_steps=500):
    L, q = h.shape
    keys = jax.random.split(key, n_samples)
    x_init = jax.random.randint(key, shape=(n_samples, L), minval=0, maxval=q)

    def mcmc_step(x, key):
        key_pos, key_mut, key_accept = jax.random.split(key, 3)
        pos = jax.random.randint(key_pos, shape=(n_samples,), minval=0, maxval=L)
        proposal = (x[jnp.arange(n_samples), pos] + jax.random.randint(key_mut, (n_samples,), 1, q)) % q

        def single_energy(x_i, pos_i, aa_i):
            return h[pos_i, aa_i] + jnp.sum(J[pos_i, :, aa_i, x_i])

        e_old = vmap(single_energy)(x, pos, x[jnp.arange(n_samples), pos])
        x_new = x.at[jnp.arange(n_samples), pos].set(proposal)
        e_new = vmap(single_energy)(x_new, pos, proposal)

        delta_E = e_new - e_old
        accept_prob = jnp.exp(-beta * delta_E)
        accept = jax.random.uniform(key_accept, (n_samples,)) < accept_prob
        x_next = jnp.where(accept[:, None], x_new, x)
        return x_next

    x = x_init
    for i in range(n_steps):
        step_keys = jax.random.split(keys[i % len(keys)], n_samples)
        x = mcmc_step(x, step_keys)

    return jax.nn.one_hot(x, num_classes=q).astype(jnp.float32)

def compute_first_order(sigma):
    return jnp.mean(sigma, axis=0)

def compute_second_order(sigma):
    return jnp.einsum('nik,njl->ijkl', sigma, sigma) / sigma.shape[0]

def spearman_corr(x, y):
    from scipy.stats import spearmanr
    return spearmanr(np.ravel(x), np.ravel(y)).correlation

def coupling_frobenius(J):
    L = J.shape[0]
    J_norm = jnp.linalg.norm(J, axis=(2, 3))
    J_norm = J_norm.at[jnp.diag_indices(L)].set(0)
    return J_norm

def sequence_identity(seq1, seq2):
    return jnp.mean(seq1 == seq2)

def max_sequence_identity(samples, train):
    idents = []
    for s in samples:
        idents.append(jnp.max(jnp.array([sequence_identity(s, t) for t in train])))
    return jnp.array(idents)

def plot_log_ratios(log_in_in, log_out1, log_out2, title="Log-Ratio Distributions (Reproduced)"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 6))

    sns.kdeplot(log_in_in, label="In/In", color="blue", linewidth=2, bw_adjust=1.5)
    sns.kdeplot(log_out1, label="In/Out1", color="orange", linewidth=2, bw_adjust=1.5)
    sns.kdeplot(log_out2, label="In/Out2", color="green", linewidth=2, bw_adjust=1.5)

    plt.xlabel(r"$\log P(x_1) - \log P(x_2)$")
    plt.ylabel("Density")
    plt.title("Log-Ratio Distributions (Reproduced)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === MAIN ===
def main():
    global beta
    lambda_ = 1.0 / (276 * 21)
    beta = 0.5

    alignment = encoding("PF01738_seed.sto")
    sigma_int, sigma = amino_mapping(alignment)
    len_protein = sigma.shape[1]
    len_aa = 21

    key = jax.random.PRNGKey(0)
    J = jax.random.normal(key, shape=(len_protein, len_protein, len_aa, len_aa))
    J = jnp.abs(J)
    J = (J + jnp.transpose(J, (1, 0, 3, 2))) / 2
    J = J.at[jnp.arange(len_protein), jnp.arange(len_protein)].set(0.0)
    J = J / 40

    key, subkey = jax.random.split(key)
    h = 0.01 * jax.random.normal(subkey, shape=(len_protein, len_aa))

    ##### Bayesian Cubature (used for fixed log_Z) #####
    n_vals = jnp.array([300])
    results = run_multiple_seeds(n_vals, lambda_, len_protein, J, h, 10, beta, True, True)
    mu_bmc = results["bmc_mean_error"][0]
    log_Z = jnp.log(jnp.clip(mu_bmc, 1e-12, None))  # prevent log(0)

    ##### Optimizer #####
    learning_rate = 0.01
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.95
    )
    global optimizer
    optimizer = optax.adam(schedule)
    params = (h, J)
    opt_state = optimizer.init(params)

    ##### Training Loop #####
    num_steps = 100
    for step in range(num_steps):
        params, opt_state, loss, mean_E, std_E = update(params, opt_state, sigma, log_Z)
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step}, NLL: {loss:.4f}")

    print("mu_bmc:", mu_bmc)
    print("log_Z:", log_Z)

    h, J = params  # unpack learned parameters
    energies = jax.vmap(lambda x: jnp.einsum('ik,ijkl,jl->', x, J, x))(sigma)
    mean_E = jnp.mean(energies)
    std_E = jnp.std(energies)
    log_prob_fn = lambda x: log_prob_eval(x, h, J, log_Z)

    log_in_in, log_out1, log_out2 = generate_log_ratios(sigma_int, log_prob_fn)

    with open("log_ratios_normalized.pkl", "wb") as f:  # or _unnormalized.pkl
        pickle.dump((log_in_in, log_out1, log_out2), f)

    # plot_log_ratios(log_in_in, log_out1, log_out2)

    # === Evaluation: First-order and Second-order correlations ===
    key, subkey = jax.random.split(key)
    sigma_sampled = sample_potts_mcmc(subkey, h, J, beta, n_samples=10000, n_steps=500)

    f_train = compute_first_order(sigma)
    f_sampled = compute_first_order(sigma_sampled)
    rho_1st = spearman_corr(f_train, f_sampled)
    print(f"First-order Spearman correlation: {rho_1st:.4f}")

if __name__ == "__main__":
    main()
