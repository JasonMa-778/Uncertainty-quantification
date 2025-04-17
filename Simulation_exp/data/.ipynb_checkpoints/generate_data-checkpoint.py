# data/generate_data.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def generate_data(vi_dim, num_data, max_num_emb, v, seed, length_vary=True,sample_cut=10,len_cut=10):
    """
    Data generation function.
    
    Args:
        vi_dim (int): Dimension of VI.
        num_data (int): Number of data samples.
        max_num_emb (int): Maximum number of embeddings.
        v (float): Value to add to the random data.
        seed (int): Random seed for reproducibility.
    
    Returns:
        list: Contains training and testing data splits along with other parameters.
    """
    np.random.seed(seed)
    Vi = np.random.randn(num_data, vi_dim) + v
    Vi = Vi[:sample_cut,:]
    beta = np.random.normal(loc=0, scale=0.15, size=(vi_dim,))
    beta0 = np.mean(Vi @ beta.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    vib = -beta0 + Vi @ beta.T
    proba = sigmoid(vib)
    y = np.random.binomial(1, proba)
    true_y = np.round(proba, 0)
    maximum_auc = roc_auc_score(true_y, y)
    
    E = []
    for V in Vi:
        if length_vary:
            ni = np.random.randint(1, max_num_emb + 1)
        else:
            ni = max_num_emb
        ei = np.random.normal(V, 0.5, (ni, vi_dim))
        E.append(ei[:len_cut,:])
    
    # Split the data
    X_train, X_test, y_train, y_test, train_vib, test_vib, train_vi, test_vi = train_test_split(
        E, y, vib, Vi, test_size=0.2, random_state=42
    )
    
    data = [X_train, X_test, y_train, y_test, beta, beta0, maximum_auc, Vi, train_vi, test_vi, train_vib, test_vib]
    
    # Plotting histograms
    plt.hist(Vi @ beta.T, bins=10)
    plt.xlabel("vi*beta")
    plt.show()
    
    plt.hist(proba)
    plt.xlabel("Probability of occurrence")
    plt.show()
    
    return data
