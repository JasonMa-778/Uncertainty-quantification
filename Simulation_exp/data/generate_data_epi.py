import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def generate_data(vi_dim, num_data, max_num_emb, v, seed, length_vary=True, sample_cut=10, len_cut=10,var_e=0.5,beta_var=0.15,plot=True):
    
    """
    Data generation function.

    Args:
        vi_dim (int): Dimension of VI.
        num_data (int): Number of data samples.
        max_num_emb (int): Maximum number of embeddings.
        v (float): Value to add to the random data.
        seed (int): Random seed for reproducibility.
        length_vary (bool): Whether embedding lengths vary.
        sample_cut (int): Number of samples to retain after splitting.
        len_cut (int): Maximum embedding length to retain.

    Returns:
        list: Training and testing datasets along with other parameters.
    """
    np.random.seed(seed)
    
    # Generate Vi (feature vectors)
    Vi = np.random.randn(num_data, vi_dim) + v
    
    # Compute probability scores
    beta = np.random.normal(loc=0, scale=beta_var, size=(vi_dim,))
    beta0 = np.mean(Vi @ beta.T)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    vib = -beta0 + Vi @ beta.T
    proba = sigmoid(vib)
    y = np.random.binomial(1, proba)

    X_train, X_test, y_train, y_test, train_vib, test_vib, train_vi, test_vi = train_test_split(
        Vi, y, vib, Vi, test_size=0.2, random_state=42
    )

    X_train = X_train[:sample_cut]
    y_train = y_train[:sample_cut]
    train_vib = train_vib[:sample_cut]
    train_vi = train_vi[:sample_cut]

    E_train, E_test = [], []
    
    for V in X_train:
        ni = np.random.randint(1, max_num_emb + 1) if length_vary else max_num_emb
        ei = np.random.normal(V, var_e, (ni, vi_dim))
        E_train.append(ei[:len_cut, :])  # Apply len_cut AFTER train_test_split

    for V in X_test:
        ni = np.random.randint(1, max_num_emb + 1) if length_vary else max_num_emb
        ei = np.random.normal(V, var_e, (ni, vi_dim))
        E_test.append(ei[:len_cut, :])  # Apply len_cut AFTER train_test_split

    # Compute maximum AUC for reference
    true_y = np.round(proba, 0)
    if len(set(true_y)) > 1:
        maximum_auc = roc_auc_score(true_y, y)
    else:
        maximum_auc = None
    if plot:
        # Plot distributions
        plt.hist(Vi @ beta.T, bins=10)
        plt.xlabel("vi*beta")
        plt.show()

        plt.hist(proba)
        plt.xlabel("Probability of occurrence")
        plt.show()

    return E_train, E_test, y_train, y_test, beta, beta0, maximum_auc, Vi, train_vi, test_vi, train_vib, test_vib
