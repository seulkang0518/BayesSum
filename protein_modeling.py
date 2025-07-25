# from Bio import AlignIO

# alignment = AlignIO.read("PF01738_seed.sto", "stockholm")
# print(f"Loaded {len(alignment)} sequences of length {alignment.get_alignment_length()}")

from Bio import AlignIO
import jax.numpy as jnp
import numpy as np
# alignment = AlignIO.read("PF01738_seed.sto", "stockholm")

# for record in alignment:
#     print(f">{record.id}")
#     print(record.seq)
k_b = 1
T_c = 5.0
beta = 1 / (k_b * T_c)

# Step 1: Define amino acid mapping
amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
aa_int = {amino_acid: i + 1 for i, amino_acid in enumerate(amino_acids)}  # 1-indexed
total_aa = len(amino_acids)

# Step 2: Load alignment
alignment = AlignIO.read("PF01738_seed.sto", "stockholm")

# Step 3: Encode sequences as integer indices
encoded = [[aa_int.get(residue, total_aa - 1) for residue in str(record.seq)] for record in alignment]
sigma_int = jnp.array(encoded)  # Shape: (B, N)

# Step 4: Confirm shape
sigma = jax.nn.one_hot(sigma_int, num_classes = total_aa)
num_protein, len_protein, total_aa = sigma.shape

# print("JAX-encoded Sigma shape:", Sigma.shape)
# # print("First sequence:", Sigma[0])
# print("First sequence:", Sigma)

# INITIALIZE PARAMETERS
key = jax.random.PRNGKey(0)
h = jax.random.normal(key, shape=(len_protein,))
J = jax.random.normal(key, shape=(len_protein, len_protein))
J = (J + J.T) / 2  # Make symmetric
J = J.at[jnp.diag_indices(len_protein)].set(0.0)  # Zero diagonal

# COMPUTE ENERGY (negative log unnormalized probability)
@jax.jit
def energy(sigma_onehot, h, J):
    field_term = jnp.sum(h * sigma_onehot)
    pairwise_term = 0.0
    for i in range(len_protein):
        for j in range(i + 1, len_protein):
            pairwise_term += sigma_onehot[i] @ J[i, j] @ sigma_onehot[j]
    return field_term + pairwise_term

# FAKE PARTITION FUNCTION (replace with kernel quadrature)
def approximate_log_Z(lambda_):
    # Replace with kernel quadrature over {1,...,q}^N
    return potts(lambda_)  # Placeholder

# Log-probability
@jax.jit
def log_prob(sigma_onehot, h, J, log_Z):
    return -energy(sigma_onehot, h, J) - log_Z

# Negative log-likelihood over all sequences
@jax.jit
def negative_log_likelihood(params, Sigma, log_Z):
    h, J = params
    log_probs = jax.vmap(lambda sigma: log_prob(sigma, h, J, log_Z))(Sigma)
    return -jnp.mean(log_probs)

# Training setup
params = (h, J)
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state, Sigma, log_Z):
    loss, grads = jax.value_and_grad(negative_log_likelihood)(params, Sigma, log_Z)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
lambda_ = 0.5
log_Z = approximate_log_Z(lambda_)
num_steps = 100

for step in range(num_steps):
    params, opt_state, loss = update(params, opt_state, Sigma, log_Z)
    if step % 10 == 0:
        print(f"Step {step}, Negative log-likelihood: {loss:.4f}")
