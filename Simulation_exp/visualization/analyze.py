# In[1]: 导入必要的库和自定义模块
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from data.generate_data import generate_data

from visualization.plot_utils import (
    compute_ece,
    calculate_auc,
    test_norm,
    logistic,
    uncertainty_auroc,
    compute_roc_prc,
    model_result_show,
    shannon_entropy,
    calculate_ause
)

def outcome_show(i,model_path,bay_path,uncertainty_method="mcdropout",plot=False,truncations=None,truncations_colors=None,models=None,poolings=None):
    """
    outcome_show(i):
    1) 生成数据
    2) 加载 MC dropout 结果文件
    3) 计算 Bayesian baseline (vib_miu_std, vib_var_std, prob_mean, prob_var 等)
    4) 调用 model_result_show 进行可视化
    5) 返回包含可视化指标的 DataFrame
    """
    sd=i+1-1
    data = generate_data(100, 15000, 50, 0, i)
    y = data[3]
    y = y.squeeze()
    beta = data[4].flatten()
    beta0 = data[5]
    if uncertainty_method=="mcdropout":
        with open(f'{model_path}/mcdropout_cls_seed{i}_.pkl','rb') as f:
            cls_result = pickle.load(f)
        with open(f'{model_path}/mcdropout_mean_seed{i}_.pkl','rb') as f:
            mp_result = pickle.load(f)
        with open(f'{model_path}/mcdropout_mean_no_pad_seed{i}_.pkl','rb') as f:
            np_mp_result = pickle.load(f)
        with open(f'{model_path}/mcdropout_mask_np_mp_seed{i}_.pkl','rb') as f:
            mk_np_mp_result = pickle.load(f)
    elif uncertainty_method=="ensemble":
        with open(f'{model_path}/ntrain_cls_4h1l_{i}.pkl','rb') as f:
            cls_result=pickle.load(f)
        with open(f'{model_path}/ntrain_mean_4h1l_{i}.pkl','rb') as f:
            mp_result=pickle.load(f)
        with open(f'{model_path}/ntrain_mean_no_pad_4h1l_{i}.pkl','rb') as f:
            np_mp_result=pickle.load(f)
        with open(f'{model_path}/ntrain_mask_np_mp_4h1l_{i}.pkl','rb') as f:
            mk_np_mp_result=pickle.load(f)
    elif uncertainty_method=="EDL":
        with open(f'{model_path}/EDL_cls_4h1l_{i}.pkl','rb') as f:
            cls_result=pickle.load(f)
        with open(f'{model_path}/EDL_mean_4h1l_{i}.pkl','rb') as f:
            mp_result=pickle.load(f)
        with open(f'{model_path}/EDL_mean_no_pad_4h1l_{i}.pkl','rb') as f:
            np_mp_result=pickle.load(f)
        with open(f'{model_path}/EDL_mask_np_mp_4h1l_{i}.pkl','rb') as f:
            mk_np_mp_result=pickle.load(f)
    else:
        print("uncertainty method not found!")
        
    if uncertainty_method!="EDL":
        prob_df_set_mp=cls_result+mp_result+np_mp_result+mk_np_mp_result
    elif uncertainty_method == "EDL":
        prob_df_set_mp=cls_result+mp_result+np_mp_result+mk_np_mp_result
        prob_df_mp=[]
        for i in range(len(prob_df_set_mp)):
            probability=prob_df_set_mp[i][0].mean(axis=1)
            uncertainty=prob_df_set_mp[i][1].mean(axis=1)
            evidence=prob_df_set_mp[i][2].mean(axis=1)
            prob_df_set_mp[i] = pd.DataFrame({
                "probability": probability,
                "evidence": evidence,
                "uncertainty": uncertainty
            })
    
            alpha_1 = prob_df_set_mp[i]["evidence"] + 1
            sum_alpha = alpha_1 / prob_df_set_mp[i]["probability"]
            second_alpha = sum_alpha - alpha_1
    
            prob_df_set_mp[i]["variance"] = alpha_1 * (sum_alpha - alpha_1 - 1) / (sum_alpha ** 2 * (sum_alpha + 1))
    
            alpha_params = np.stack([alpha_1, second_alpha], axis=1)
            data_points = np.array([np.random.dirichlet(alpha, size=100)[:, 0] for alpha in alpha_params])
    
            prob_df_mp.append(data_points)
        prob_df_set_mp=prob_df_mp
    print(sd)
    # Bayesian baseline 读取
    with open(f'{bay_path}/bay_beta{sd}.pkl', 'rb') as f:
        result = pickle.load(f)
    beta_samples = result[0]
    beta_0 = result[1]

    # 下方是 Bayesian baseline 的推断过程
    M = 1000
    sigma_e2 = 0.5**2
    sigma_v2 = 1.0**2
    num_sample = len(beta_samples)
    e_list = data[1]

    pred_probs = []
    vi = []
    sigma = []
    mu = []
    vib = []
    for e_i in e_list:
        n_i = e_i.shape[0]
        mu_i = np.sum(e_i, axis=0) / (n_i + (sigma_e2 / sigma_v2))
        variance = sigma_e2 / (n_i + (sigma_e2 / sigma_v2))

        v_i_samples = mu_i + np.random.normal(0, np.sqrt(variance), (M, mu_i.shape[0]))
        mu.append(mu_i)
        sigma.append(variance)
        vi.append(v_i_samples)
        #print(num_sample)
        idx_rand = np.random.choice(num_sample, M, replace=True)
        beta_samples_subset = beta_samples[idx_rand,:]
        beta_0_samples = beta_0[idx_rand]
        v_i_beta_products = np.sum(v_i_samples * beta_samples_subset, axis=1)
        vib.append(v_i_beta_products + beta_0_samples)

        probs = 1 / (1 + np.exp(-vib[-1]))
        pred_prob = np.mean(probs)
        pred_probs.append(pred_prob)

    vib_var_std = np.array([np.var(i, axis=0)/12 for i in vib])
    vib_miu_std = np.array([np.mean(i, axis=0) for i in vib])

    num_samples = 10000
    log_odds_samples = [
        np.random.normal(loc=mu, scale=np.sqrt(var), size=num_samples)
        for mu, var in zip(vib_miu_std, vib_var_std)
    ]
    prob_samples = [1 / (1 + np.exp(-log_odds)) for log_odds in log_odds_samples]
    prob_mean = np.array([np.mean(prob) for prob in prob_samples])
    prob_var = np.array([np.var(prob) for prob in prob_samples])
    n = [e.shape[0] for e in data[1]]

    df_names = [i for i in range(10000)]
    if uncertainty_method=="mcdropout":
        prob_df_set = [i[0] for i in prob_df_set_mp]
    elif uncertainty_method == "ensemble":
        prob_df_set= prob_df_set_mp
    else:
        prob_df_set= prob_df_set_mp

    LR_no_norm = model_result_show(prob_df_set, 
                                   y, 
                                   n, 
                                   vib_miu_std, 
                                   vib_var_std, 
                                   prob_mean,
                                   prob_var,
                                   poolings, 
                                   plot_results=plot,
                                   uncertainty_method=uncertainty_method,
                                   truncations=truncations,
                                   truncation_colors=truncations_colors,
                                   models=models)
    return LR_no_norm


import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import scipy.stats as stats

def baseline_ks(i, path,truncations):
    """
    baseline_ks(i, path):
    1) Generate data.
    2) Compute a Bayesian baseline and calculate various metrics including AUC, ECE, and uncertainty.
    3) Return all baseline metrics.
    
    Parameters:
    - i (int): Identifier or index for data generation and file loading.
    - path (str): Directory path where the Bayesian samples pickle file is stored.
    
    Returns:
    - baseline_values (dict): Dictionary containing all computed baseline metrics.
    """
    # Step 1: Generate data
    data = generate_data(100, 15000, 50, 0, i)
    y = data[3].squeeze()  # True labels, shape: [n_samples]
    beta = data[4].flatten()  # Coefficients, shape: [num_features]
    beta0 = data[5]  # Intercept, scalar or array depending on data generation
    
    M = 1000  # Number of Monte Carlo samples
    sigma_e2 = 0.5**2  # Variance of the error term
    sigma_v2 = 1.0**2  # Variance of beta
    
    # Step 2: Load Bayesian samples
    with open(f'{path}/bay_beta{i}.pkl', 'rb') as f:
        result = pickle.load(f)
    beta_samples = result[0]  # Shape: [num_samples, num_features]
    beta_0_samples = result[1]  # Shape: [num_samples]
    num_sample = len(beta_samples)
    
    e_list = data[1]  # List or array of embeddings, each shape: [n_i, num_features]
    
    Vi = data[9]  # Additional data for baseline computation, usage depends on context
    
    pred_probs = []  # List to store predicted probabilities
    vib = []  # List to store uncertainty measures
    n=[]
    # Iterate over each embedding to compute predictions and uncertainties
    for e_i in e_list:
        n_i = e_i.shape[0]  # Number of samples in the embedding
        n.append(n_i)
        mu_i = np.sum(e_i, axis=0) / (n_i + (sigma_e2 / sigma_v2))  # Mean
        variance = sigma_e2 / (n_i + (sigma_e2 / sigma_v2))  # Variance
    
        # Generate M samples from the posterior distribution
        v_i_samples = mu_i + np.random.normal(0, np.sqrt(variance), (M, mu_i.shape[0]))
        
        # Randomly select M samples of beta and beta0 with replacement
        idx_rand = np.random.choice(num_sample, M, replace=True)
        beta_samples_subset = beta_samples[idx_rand, :]  # Shape: [M, num_features]
        beta_0_samples_subset = beta_0_samples[idx_rand]  # Shape: [M]
        
        # Compute the linear combination of beta and embeddings
        v_i_beta_products = np.sum(v_i_samples * beta_samples_subset, axis=1)  # Shape: [M]
        vib_sample = v_i_beta_products + beta_0_samples_subset  # Shape: [M]
        vib.append(vib_sample)
        
        # Convert log-odds to probabilities using the sigmoid function
        probs = 1 / (1 + np.exp(-vib_sample))  # Shape: [M]
        pred_prob = np.mean(probs)  # Mean predicted probability for this embedding
        pred_probs.append(pred_prob)
    
    # Step 3: Calculate variance and mean for the Bayesian baseline
    vib_var_std = np.array([np.var(v, axis=0) for v in vib])  # Shape: [num_embeddings,]
    #vib_miu_std = np.array([np.sum(v * beta - beta0) for v in Vi])  # Shape: [num_embeddings,]
    vib_miu_std = np.array([np.mean(v, axis=0) for v in vib])
    print(f"Length of vib_var_std: {len(vib_var_std)}")
    print(f"Length of vib_miu_std: {len(vib_miu_std)}")
    #print(n)
    n=np.array(n)
    num_samples_uncertainty = 1000  # Number of samples for uncertainty estimation
    
    # Generate log-odds samples based on the computed mean and variance
    log_odds_samples = [
        np.random.normal(loc=mu, scale=np.sqrt(var), size=num_samples_uncertainty)
        for mu, var in zip(vib_miu_std, vib_var_std)
    ]
    
    # Convert log-odds to probabilities
    prob_samples = [1 / (1 + np.exp(-log_odds)) for log_odds in log_odds_samples]  # List of arrays
    prob_mean = np.array([1 / (1 + np.exp(-mu)) for mu in vib_miu_std])  # Shape: [num_embeddings,]
    prob_var = np.array([np.var(prob) for prob in prob_samples])  # Shape: [num_embeddings,]
    
    ## Step 4: Compute baseline metrics
    #try:
    #    max_auc = roc_auc_score(y, prob_mean)
    #except ValueError:
    #    max_auc = np.nan  # Handle cases where AUC cannot be computed
    #
    #ece_std_score = compute_ece(y, prob_mean)  # Expected Calibration Error
    #
    ## Compute ROC and PRC metrics
    #(fpr_1, tpr_1, auc_score_1,
    # fpr_2, tpr_2, auc_score_2,
    # precision_1, recall_1, auprc_1,
    # precision_2, recall_2, auprc_2) = compute_roc_prc(prob_var, prob_var, y, prob_mean, prob_mean)
    #
    ## Step 5: Compute additional metrics
    ## Brier Score
    #brier = (y - prob_mean) ** 2
    #brier_mean = np.mean(brier)
    #
    ## Entropy
    #entropy_pred = shannon_entropy(prob_mean)
    #entropy_mean = np.mean(entropy_pred)
    #
    ## Negative Log-Likelihood (NLL)
    #epsilon = 1e-10  # Small constant to prevent log(0)
    #nll = -y * np.log(prob_mean + epsilon) - (1 - y) * np.log(1 - prob_mean + epsilon)
    #nll_mean = np.mean(nll)
    #
    ## Predicted Probability Mean
    #pred_prob_mean = np.mean(prob_mean)
    #
    #ause = calculate_ause(
    #uncertainty=prob_var,
    #predictions=prob_mean,
    #targets=y,
    #min_fraction=0.0,
    #max_fraction=0.9)
    #
    ## Variance Mean
    #var_mean = np.mean(prob_var)
    #
    ## Step 6: Compile all metrics into a dictionary
    #baseline_values = {
    #    "max_auc": np.round(max_auc, 4),
    #    "normal_auc": np.round(max_auc, 4),  # Assuming normal_auc is the same as max_auc
    #    "uncertainty_auc": np.round(auc_score_1, 4),
    #    "auprc": np.round(auprc_1, 4),
    #    "ece": np.round(ece_std_score, 4),
    #    "ece": 0,
    #    "brier_score_mean": np.round(brier_mean, 4),
    #    "entropy_mean": np.round(entropy_mean, 4),
    #    "nll_mean": np.round(nll_mean, 4),
    #    "pred_prob_mean": np.round(pred_prob_mean, 4),
    #    "ause": np.round(ause, 4),
    #    "var_mean": np.round(var_mean, 4)
    #}
    #
    #return baseline_values
# ===== After you finish computing prob_mean and prob_var, and you have "y" and "n" ===== #
# Suppose 'prob_mean', 'prob_var', 'y', and 'n' are all 1D arrays (or lists) of the same length.
# Also assume 'truncations' is a sorted list of integer boundaries, e.g. [10, 20, 30].

# Step 4: For each group defined by truncations, compute baseline metrics
    baseline_values_list = []  # We'll store a dictionary of metrics for each group
    
    prev = 0
    for t in truncations:
        start, end = prev, t
        prev = t
        
        # Identify indices of samples whose sequence length falls within [start, end)
        idx_group = np.where((n >= start) & (n < end))[0]
        
        # If there's no sample in this group, return NaN metrics (or handle differently if desired)
        if len(idx_group) == 0:
            metrics_dict = {
                "max_auc": np.nan,
                "normal_auc": np.nan,
                "uncertainty_auc": np.nan,
                "auprc": np.nan,
                "ece": np.nan,
                "brier_score_mean": np.nan,
                "entropy_mean": np.nan,
                "nll_mean": np.nan,
                "pred_prob_mean": np.nan,
                "ause": np.nan,
                "var_mean": np.nan
            }
            baseline_values_list.append(metrics_dict)
            continue
        
        # Extract the subgroup data
        y_sub = y[idx_group]
        prob_mean_sub = prob_mean[idx_group]
        prob_var_sub = prob_var[idx_group]
        
        # Step 4: Compute baseline metrics (same as your original code, but on subgroup data)
        try:
            max_auc = roc_auc_score(y_sub, prob_mean_sub)
        except ValueError:
            max_auc = np.nan  # Handle cases where AUC cannot be computed
        
        ece_std_score = compute_ece(y_sub, prob_mean_sub)  # Expected Calibration Error
        
        (fpr_1, tpr_1, auc_score_1,
         fpr_2, tpr_2, auc_score_2,
         precision_1, recall_1, auprc_1,
         precision_2, recall_2, auprc_2) = compute_roc_prc(
             prob_var_sub, prob_var_sub, y_sub, prob_mean_sub, prob_mean_sub
        )
        
        # Step 5: Compute additional metrics
        brier = (y_sub - prob_mean_sub) ** 2
        brier_mean = np.mean(brier)
        
        entropy_pred = shannon_entropy(prob_mean_sub)
        entropy_mean = np.mean(entropy_pred)
        
        epsilon = 1e-10
        nll = -y_sub * np.log(prob_mean_sub + epsilon) - (1 - y_sub) * np.log(1 - prob_mean_sub + epsilon)
        nll_mean = np.mean(nll)
        
        pred_prob_mean = np.mean(prob_mean_sub)
        
        ause = calculate_ause(
            uncertainty=prob_var_sub,
            predictions=prob_mean_sub,
            targets=y_sub,
            min_fraction=0.0,
            max_fraction=0.9
        )
        
        var_mean = np.mean(prob_var_sub)
        
        # Step 6: Compile all metrics into a dictionary
        metrics_dict = {
            #"max_auc": np.round(max_auc, 4),
            
            "normal_auc": np.round(max_auc, 4),  # Assuming normal_auc is the same as max_auc
            "uncertainty_auc": np.round(auc_score_1, 4),
            "auprc": np.round(auprc_1, 4),
            "ece": np.round(ece_std_score, 4),
            #"ece": 0,  # If you need to override ECE again (as in your original code)
            "brier_score_mean": np.round(brier_mean, 4),
            "entropy_mean": np.round(entropy_mean, 4),
            "nll_mean": np.round(nll_mean, 4),
            "pred_prob_mean": np.round(pred_prob_mean, 4),
            "ause": np.round(ause, 4),
            "var_mean": np.round(var_mean, 4)
        }
        
        # Store this group's metrics in the result list
        baseline_values_list.append(metrics_dict)
    
    # Finally, return or otherwise use baseline_values_list.
    # Each element in this list corresponds to one truncation interval's metrics.
    return baseline_values_list



print("Functions outcome_show and baseline_ks have been defined.")
