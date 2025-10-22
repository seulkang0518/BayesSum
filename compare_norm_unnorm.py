import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp


def load_log_ratios(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # returns (log_in_in, log_out1, log_out2)


def compare_log_ratio_plots(normalized_data, unnormalized_data):
    in_in_norm, out1_norm, out2_norm = normalized_data
    in_in_unnorm, out1_unnorm, out2_unnorm = unnormalized_data

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(in_in_norm, label="In/In", color="blue", linewidth=2, bw_adjust=1.5)
    sns.kdeplot(out1_norm, label="In/Out1", color="orange", linewidth=2, bw_adjust=1.5)
    sns.kdeplot(out2_norm, label="In/Out2", color="green", linewidth=2, bw_adjust=1.5)
    plt.title("Normalized Model")
    plt.xlabel(r"$\log P(x_1) - \log P(x_2)$")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.kdeplot(in_in_unnorm, label="In/In", color="blue", linewidth=2, bw_adjust=1.5)
    sns.kdeplot(out1_unnorm, label="In/Out1", color="orange", linewidth=2, bw_adjust=1.5)
    sns.kdeplot(out2_unnorm, label="In/Out2", color="green", linewidth=2, bw_adjust=1.5)
    plt.title("Unnormalized Model")
    plt.xlabel(r"$\log P(x_1) - \log P(x_2)$")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    normalized_path = "log_ratios_normalized.pkl"
    unnormalized_path = "log_ratios_unnormalized.pkl"

    log_ratios_normalized = load_log_ratios(normalized_path)
    log_ratios_unnormalized = load_log_ratios(unnormalized_path)

    compare_log_ratio_plots(log_ratios_normalized, log_ratios_unnormalized)


if __name__ == "__main__":
    main()
