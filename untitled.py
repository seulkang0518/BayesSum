import jax
import jax.numpy as jnp

# -----------------
# Data generation
# -----------------
def generate_categorical_sequence(key, p, L, q=3):
    assert q == 3, "This generator is set for q=3."
    probs = jnp.array([p, p, 1 - 2 * p], dtype=jnp.float64)
    keys = jax.random.split(key, L)
    seq = [jax.random.choice(k, q, p=probs) for k in keys]
    return jnp.array(seq, dtype=jnp.int32)

def data_preprocess(p, q, L, num_sequences, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_sequences)
    sequences = [generate_categorical_sequence(k, p, L, q=q) for k in keys]
    data_onehot = jax.nn.one_hot(jnp.stack(sequences), num_classes=q, dtype=jnp.float64)  # (N,L,q)
    return sequences, data_onehot

# --- Assume these values from your experiment ---
L = 4
q = 3
# Let's assume p=0.4 for data generation consistency
p = 0.2
num_train_sequences = 1000
num_test_sequences = 500

# --- Placeholder Data Generation (use your actual data here) ---
# This just re-uses your data generation functions
key = jax.random.PRNGKey(42)
train_keys = jax.random.split(key, num_train_sequences)
train_sequences = [generate_categorical_sequence(k, p, L, q=q) for k in train_keys]
data_onehot = jax.nn.one_hot(jnp.stack(train_sequences), num_classes=q, dtype=jnp.float64)

key = jax.random.PRNGKey(99)
test_keys = jax.random.split(key, num_test_sequences)
test_sequences = [generate_categorical_sequence(k, p, L, q=q) for k in test_keys]
test_data_onehot = jax.nn.one_hot(jnp.stack(test_sequences), num_classes=q, dtype=jnp.float64)
# --- End of Placeholder Data ---


# 1. Calculate the probability of each state at each position from TRAINING data
p_independent = jnp.mean(data_onehot, axis=0)  # Shape (L, q)

# 2. Avoid log(0) by clipping probabilities
p_independent = jnp.clip(p_independent, 1e-9, None)

# 3. Calculate the NLL for the baseline model on the TEST data
# The log-likelihood for one sequence is the sum of log probabilities of the observed states
log_likelihood_per_sample = jnp.sum(test_data_onehot * jnp.log(p_independent), axis=(1, 2))
nll_baseline = -jnp.mean(log_likelihood_per_sample)


# --- Print Results ---
worst_case_nll = L * jnp.log(q)
print(f"Worst-case NLL (uniform random data): {worst_case_nll:.4f}")
print(f"Baseline NLL (independent sites model): {nll_baseline:.4f}")