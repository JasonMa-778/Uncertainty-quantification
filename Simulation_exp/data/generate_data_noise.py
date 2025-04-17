import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def generate_data_with_noise(
    vi_dim,
    num_data,
    max_num_emb,
    v,
    seed,
    length_vary=True,
    sample_cut=10,
    len_cut=10,
    var_e=0.5,
    beta_var=0.15,
    plot=True,
    noise_mean=0.0,
    noise_var=1.0,
    noise_ratio=0.5  # e.g., 0.5 means 50% tokens are noise
):
    np.random.seed(seed)

    # Generate latent Vi
    Vi = np.random.randn(num_data, vi_dim) + v

    # Compute label probability
    beta = np.random.normal(loc=0, scale=beta_var, size=(vi_dim,))
    beta0 = np.mean(Vi @ beta.T)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    vib = -beta0 + Vi @ beta.T
    proba = sigmoid(vib)
    y = np.random.binomial(1, proba)

    # Train-test split
    X_train, X_test, y_train, y_test, train_vib, test_vib, train_vi, test_vi = train_test_split(
        Vi, y, vib, Vi, test_size=0.2, random_state=42
    )

    X_train = X_train[:sample_cut]
    y_train = y_train[:sample_cut]
    train_vib = train_vib[:sample_cut]
    train_vi = train_vi[:sample_cut]

    def create_sequence(V, total_len):
        ni = np.random.randint(1, max_num_emb + 1) if length_vary else max_num_emb
        ni = min(ni, total_len)  # Avoid out-of-bound

        n_clean = int(round(total_len * (1 - noise_ratio)))
        n_noise = total_len - n_clean

        clean = np.random.normal(loc=V, scale=var_e, size=(n_clean, vi_dim))
        noise = np.random.normal(loc=noise_mean, scale=noise_var, size=(n_noise, vi_dim))

        seq = np.vstack([clean, noise])
        np.random.shuffle(seq)  # shuffle clean + noise tokens
        return seq[:total_len]

    E_train = [create_sequence(V, len_cut) for V in X_train]
    E_test = [create_sequence(V, len_cut) for V in X_test]

    # Optional plots
    if plot:
        plt.hist(Vi @ beta.T, bins=10)
        plt.xlabel("vi * beta")
        plt.title("Distribution of Vi @ beta")
        plt.show()

        plt.hist(proba)
        plt.xlabel("Probability")
        plt.title("Distribution of predicted probabilities")
        plt.show()

    # Return standard outputs
    return E_train, E_test, y_train, y_test, beta, beta0, roc_auc_score(np.round(proba, 0), y), Vi, train_vi, test_vi, train_vib, test_vib
