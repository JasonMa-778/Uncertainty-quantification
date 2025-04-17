# visualization/plot_utils.py

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy.stats as stats

from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

################################################################################
# 这里开始放置您的可视化脚本中的所有函数，不做任何省略
################################################################################


def shannon_entropy(p, epsilon=1e-10):
    """
    Calculate Shannon Entropy

    Parameters:
    - p: Probability of being the positive class (array)
    - epsilon: Small constant to prevent log(0)

    Returns:
    - entropy: Shannon entropy values (array)
    """
    p = np.clip(p, epsilon, 1 - epsilon)  # Prevent log(0)
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return entropy

def calculate_ause(uncertainty, predictions, targets, metric='brier',
                   uncertainty_type='variance', bins=10, save_path=None, min_fraction=0.0,
                   max_fraction=0.9, ax=None, color='b', label='Sparsification Error'):
    """
    Calculate and plot AUSE (Area Under the Sparsification Error) curve.

    Parameters:
    - uncertainty (np.ndarray): Uncertainty measure (e.g., variance or confidence).
    - predictions (np.ndarray): Predicted probabilities.
    - targets (np.ndarray): Ground truth labels (0 or 1).
    - metric (str): Error metric ('brier', 'mae', 'ece', or 'auc').
    - uncertainty_type (str): Type of uncertainty ('variance' or 'confidence').
    - bins (int): Number of bins for ECE calculation, only used if metric='ece'.
    - save_path (str): If provided, save the figure to this path.
    - min_fraction (float): Minimum fraction of samples to remove.
    - max_fraction (float): Maximum fraction of samples to remove.
    - ax (matplotlib.axes.Axes): Axis on which to plot. If None, create a new figure.
    - color (str): Color of the plotted line.
    - label (str): Base label for the line in the plot legend.

    Returns:
    - ause_pred (float): Computed AUSE value.
    """

    if not (len(predictions) == len(targets) == len(uncertainty)):
        raise ValueError("predictions, targets, and uncertainty must have the same length.")

    if len(predictions) < 10:
        raise ValueError("Not enough predictions for sparsification.")

    # Define error functions
    def brier_score(preds, tgts):
        return np.mean((preds - tgts)**2)

    def mean_absolute_error(preds, tgts):
        return np.mean(np.abs(preds - tgts))

    def calculate_ece_metric(preds, tgts, n_bins=10):
        n = len(preds)
        if n == 0:
            return np.nan
        bin_edges = np.linspace(0., 1., n_bins + 1)
        bin_indices = np.digitize(preds, bin_edges, right=True) - 1
        ece = 0.0
        for i in range(n_bins):
            bin_mask = bin_indices == i
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                bin_confidence = np.mean(preds[bin_mask])
                bin_accuracy = np.mean(tgts[bin_mask])
                ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)
        return ece

    def calculate_auc_metric(preds, tgts):
        if len(np.unique(tgts)) < 2:
            return np.nan
        return roc_auc_score(tgts, preds)

    # Select error function
    if metric == 'brier':
        error_function = brier_score
    elif metric == 'mae':
        error_function = mean_absolute_error
    elif metric == 'ece':
        error_function = lambda p, t: calculate_ece_metric(p, t, n_bins=bins)
    elif metric == 'auc':
        error_function = calculate_auc_metric
    else:
        raise ValueError("Unknown metric. Choose from 'brier', 'mae', 'ece', or 'auc'.")

    sorted_inds_uncertainty = np.argsort(uncertainty)
    if metric in ['brier', 'mae']:
        if metric == 'brier':
            true_errors = (predictions - targets)**2
        else:  # mae
            true_errors = np.abs(predictions - targets)
        sorted_inds_error = np.argsort(true_errors)
    else:
        sorted_inds_error = None

    # Initialize lists
    uncertainty_scores = []
    oracle_scores = []

    fractions = np.linspace(min_fraction, max_fraction, int((max_fraction - min_fraction)/0.01) + 1)
    num_samples = len(predictions)

    for fraction in fractions:
        num_keep = max(1, int((1.0 - fraction) * num_samples))
        inds_keep_uncertainty = sorted_inds_uncertainty[:num_keep]
        remaining_preds = predictions[inds_keep_uncertainty]
        remaining_targets = targets[inds_keep_uncertainty]
        try:
            uncertainty_error_score = error_function(remaining_preds, remaining_targets)
        except ValueError as e:
            print(f"Error calculating {metric} at fraction {fraction:.2f}: {e}")
            uncertainty_error_score = np.nan
        uncertainty_scores.append(uncertainty_error_score)

        if sorted_inds_error is not None:
            inds_keep_oracle = sorted_inds_error[:num_keep]
            oracle_preds = predictions[inds_keep_oracle]
            oracle_targets = targets[inds_keep_oracle]
            try:
                oracle_error_score = error_function(oracle_preds, oracle_targets)
            except ValueError as e:
                print(f"Error calculating Oracle {metric} at fraction {fraction:.2f}: {e}")
                oracle_error_score = np.nan
            oracle_scores.append(oracle_error_score)
        else:
            oracle_scores.append(np.nan)

    uncertainty_scores = np.array(uncertainty_scores)
    oracle_scores = np.array(oracle_scores)

    # Normalize error scores
    if metric in ['brier', 'mae']:
        uncertainty_scores_normalized = uncertainty_scores / uncertainty_scores[0]
        if sorted_inds_error is not None:
            oracle_scores_normalized = oracle_scores / oracle_scores[0]
        else:
            oracle_scores_normalized = np.full_like(uncertainty_scores_normalized, np.nan)
    elif metric == 'ece':
        uncertainty_scores_normalized = uncertainty_scores / uncertainty_scores[0]
        oracle_scores_normalized = np.full_like(uncertainty_scores_normalized, np.nan)
    elif metric == 'auc':
        uncertainty_scores_normalized = (1 - uncertainty_scores) / (1 - uncertainty_scores[0])
        oracle_scores_normalized = np.full_like(uncertainty_scores_normalized, np.nan)
    else:
        raise ValueError("Unknown metric.")

    ause_pred = np.trapz(y=uncertainty_scores_normalized, x=fractions) / (max_fraction - min_fraction)

    if ax is None:
        #fig, ax = plt.subplots(figsize=(8, 6))
        return ause_pred
    else:
        
        if metric in ['brier', 'mae', 'ece', 'auc']:
            # Include the AUSE value in the label
            ax.plot(fractions, uncertainty_scores_normalized, label=f"{label} (AUSE={ause_pred:.4f})", color=color)
    
        ax.set_xlabel("Fraction of removed samples")
        ax.set_ylabel(f"{metric.upper()} Error (Normalized)")
        ax.set_ylim([0, 1.5])
        ax.set_title('Sparsification Curve')
        ax.legend()
        ax.grid(True)
    
        if save_path:
            plt.savefig(save_path)
        elif ax is None:
            plt.show()

    return ause_pred
    
def compute_ece(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).

    Parameters:
    - y_true: Array-like of shape (n_samples,). True labels (0 or 1).
    - y_prob: Array-like of shape (n_samples,). Predicted probabilities.
    - n_bins: int, Number of bins to divide the probability scores.

    Returns:
    - ece: float, Expected Calibration Error.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        bin_mask = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[bin_mask] == 1)
            bin_confidence = np.mean(y_prob[bin_mask])
            ece += np.abs(bin_accuracy - bin_confidence) * (bin_count / len(y_true))
    return ece

def calculate_auc(prob_df, y):
    """
    Calculate AUC for a given set of probability predictions.
    
    Args:
        prob_df: shape [n_samples, n_inferences]
        y: True labels, shape [n_samples].
        
    Returns:
        auc (float): Rounded AUC value.
    """
    prob_df = np.array(prob_df)
    pre = np.mean(prob_df, axis=1)
    auc_val = roc_auc_score(y, pre)
    return round(auc_val, 3)

def test_norm(Prob_df, vib_miu_std, vib_var_std):
    """
    Perform KS test to see if Prob_df's distribution ~ Normal(mean=vib_miu_std[i], std=sqrt(vib_var_std[i])).

    Args:
        Prob_df (np.ndarray): shape [n_samples, n_inferences]
        vib_miu_std (np.ndarray): shape [n_samples]
        vib_var_std (np.ndarray): shape [n_samples]

    Returns:
        List of (ks_statistic, pvalue) for each sample.
    """
    ks_statistic = [
        (
            stats.kstest(Prob_df[i, :], 'norm', args=(vib_miu_std[i], np.sqrt(vib_var_std[i]))).statistic,
            stats.kstest(Prob_df[i, :], 'norm', args=(vib_miu_std[i], np.sqrt(vib_var_std[i]))).pvalue
        )
        for i in range(Prob_df.shape[0])
    ]
    return ks_statistic

def logistic(prob_matrix):
    """
    Transform from pi to log odds
    """
    prob_matrix = np.clip(prob_matrix, 0.0001, 0.9999)
    logit_matrix = np.log(prob_matrix / (1 - prob_matrix))
    return logit_matrix

def uncertainty_auroc(y, pre, var):
    """
    correctness = (y == (pre > 0.5))
    Then compute an AUROC with 'var' as the predictor for correctness.
    """
    correctness = np.array(y == (pre > 0.5))
    fprs, tprs, thresholds = roc_curve(correctness, -1 * var)
    auc = roc_auc_score(correctness, -1 * var)
    return tprs, fprs, auc

def compute_roc_prc(variances_1, variances_2, labels, pre, std_pre):
    """
    ROC and PRC curves for variances
    """
    tpr_1, fpr_1, auc_score_1 = uncertainty_auroc(labels, std_pre, variances_1)
    tpr_2, fpr_2, auc_score_2 = uncertainty_auroc(labels, pre, variances_2)

    precision_1, recall_1, _ = precision_recall_curve(labels, std_pre)
    precision_2, recall_2, _ = precision_recall_curve(labels, pre)
    auprc_1 = average_precision_score(labels, std_pre)
    auprc_2 = average_precision_score(labels, pre)

    return fpr_1, tpr_1, auc_score_1, fpr_2, tpr_2, auc_score_2, precision_1, recall_1, auprc_1, precision_2, recall_2, auprc_2

def model_result_show(prob_df_set, y, n, vib_miu_std, vib_var_std, 
                      prob_miu_std, prob_var_std, df_names, 
                      plot_results=False,uncertainty_method="mcdropout",
                      truncations=None,truncation_colors=None,metric='brier',
                      models=None):
    """
    Show model results for ensemble or MCdropout.

    prob_df_set: List of probability DataFrame (each shape [n_samples, n_inferences])
    y: np.ndarray of shape [n_samples], true labels
    n: list -> number of embeddings for each sample
    vib_miu_std, vib_var_std: arrays from Bayesian baseline
    prob_miu_std, prob_var_std: arrays from samples
    df_names: List of model names (str)
    plot_results: bool, whether to plot subplots

    Return: metrics_df (pd.DataFrame)
    """
    num_dfs = len(prob_df_set)
    num_models=len(models)
    num_poolings=len(prob_df_set)//num_models
    n=np.array(n)
    subplot_width = 3 
    subplot_height = 3
    num_columns = 11  
    num_rows = num_poolings*num_models
    if plot_results:
        fig_width = subplot_width * num_columns + 2 
        fig_height = subplot_height * num_rows + 2  
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height), squeeze=False)
    else:
        fig, axes = None, None
    
    metrics_records = []
    
    for idx, Prob_df in enumerate(prob_df_set):
        
        model_idx=idx%num_models
        pool_idx=idx//num_models
        pooling_name = df_names[pool_idx]
        model_name = models[model_idx]
        #print(Prob_df)
        df_full = np.array(Prob_df)
        ax_sparsification=None
        if plot_results:
            # Assign the 10 subplots for the current pooling
            ax_ks_density        = axes[idx][0]
            ax_var_density       = axes[idx][1]
            ax_unc_roc           = axes[idx][2]
            ax_calibration       = axes[idx][3]
            ax_normal_roc        = axes[idx][4]
            ax_sparsification    = axes[idx][5]
            ax_brier_density     = axes[idx][6]
            ax_entropy_density   = axes[idx][7]
            ax_nll_density       = axes[idx][8]
            ax_pred_prob_density = axes[idx][9]
            ax_pr                = axes[idx][10]

        for t_idx, t_val in enumerate(truncations):

            if t_idx == 0:
                # First interval: n <= truncations[0]
                mask = (n <= t_val)

            else:
                # Subsequent intervals: truncations[t_idx-1] < n <= t_val
                mask = (n > truncations[t_idx - 1]) & (n <= t_val)

            # If there are no samples in this interval, skip
            if not np.any(mask):
                continue
            
            sub_df = df_full[mask]  # shape: (# of samples in interval, num_MC_predicts)
            sub_y  = y[mask]

            logit_df = logistic(sub_df)
            ks_result = test_norm(logit_df, vib_miu_std[mask], vib_var_std[mask])
            ks_stats = [i[0] for i in ks_result]
            ks_mean=round(np.mean(ks_stats), 4)
            
            # (2) Hexbin plot of variance
            # Calculate the fraction of data points in this interval
            fraction_data = sub_df.shape[0] / df_full.shape[0]

            # Compute mean and variance of predictions
            predictions = np.mean(sub_df, axis=1)
            variance = np.var(sub_df, axis=1)
            std_dev = np.sqrt(variance)

            # Standard ROC & AUC
            normal_fpr, normal_tpr, _ = roc_curve(sub_y, predictions)
            normal_auc = roc_auc_score(sub_y, predictions)

            # Uncertainty ROC & AUC
            unc_tpr, unc_fpr, unc_auc = uncertainty_auroc(sub_y, predictions, variance)

            # Calibration curve & ECE
            prob_true, prob_pred = calibration_curve(sub_y, predictions, n_bins=10)
            ece_score = compute_ece(sub_y, predictions)

            # Precision-Recall
            precision, recall, thresholds = precision_recall_curve(sub_y, predictions)
            auprc = average_precision_score(sub_y, predictions)

            # Brier Score
            brier = (sub_y - predictions) ** 2

            # NLL (binary)
            epsilon = 1e-10
            nll = -sub_y * np.log(predictions + epsilon) - (1 - sub_y) * np.log(1 - predictions + epsilon)

            # Entropy
            entropy_pred = shannon_entropy(predictions)

            
            # AUSE
            ause = calculate_ause(
                uncertainty=variance,
                predictions=predictions,
                targets=sub_y,
                metric=metric,
                min_fraction=0.0,
                max_fraction=0.9,
                ax=ax_sparsification,
                color=truncation_colors[t_idx],
                label=f"{t_val}"
            )

            # Averages for annotation
            variance_mean  = np.mean(variance)
            brier_mean     = np.mean(brier)
            entropy_mean   = np.mean(entropy_pred)
            nll_mean       = np.mean(nll)
            pred_prob_mean = np.mean(predictions)
            

            # Label for this interval
            if t_idx == 0:
                trunc_name = f"≤ {t_val}"
            else:
                trunc_name = f"{truncations[t_idx - 1]}-{t_val}"
    
            # Plotting
            c = truncation_colors[t_idx]
            if plot_results:    
                # new(1) KS statistic density plot
                sns.kdeplot(
                    ks_stats, 
                    ax=ax_ks_density,
                    label=f"{trunc_name} (avgVar={ks_mean:.4f}, ratio={fraction_data:.2f})",
                    fill=False,
                    color=c)
        
                # 1) Variance Density Plot
                sns.kdeplot(
                    variance,
                    ax=ax_var_density,
                    label=f"{trunc_name} (avgVar={variance_mean:.5f}, ratio={fraction_data:.2f})",
                    fill=False,
                    color=c
                )
        
                # 2) Uncertainty ROC
                ax_unc_roc.plot(
                    unc_fpr,
                    unc_tpr,
                    label=f"{trunc_name} (AUROC={unc_auc:.2f}, ratio={fraction_data:.2f})",
                    color=c
                )
        
                # 3) Calibration Curve
                ax_calibration.plot(
                    prob_pred,
                    prob_true,
                    marker="o",
                    label=f"{trunc_name} (ECE={ece_score:.2f}, ratio={fraction_data:.2f})",
                    color=c
                )
        
                # 4) Standard ROC
                ax_normal_roc.plot(
                    normal_fpr,
                    normal_tpr,
                    label=f"{trunc_name} (AUC={normal_auc:.2f}, ratio={fraction_data:.2f})",
                    color=c
                )
        
                # 5) Sparsification Curve
                # (Already handled by calculate_ause, just set axis properties below)
        
                # 6) Brier Score Density
                sns.kdeplot(
                    brier,
                    ax=ax_brier_density,
                    label=f"{trunc_name} (MeanBrier={brier_mean:.4f}, ratio={fraction_data:.2f})",
                    color=c
                )
        
                # 7) Prediction Distribution Entropy Density
                sns.kdeplot(
                    entropy_pred,
                    ax=ax_entropy_density,
                    label=f"{trunc_name} (MeanEntropy={entropy_mean:.4f}, ratio={fraction_data:.2f})",
                    color=c
                )
        
                # 8) NLL Density
                sns.kdeplot(
                    nll,
                    ax=ax_nll_density,
                    label=f"{trunc_name} (MeanNLL={nll_mean:.4f}, ratio={fraction_data:.2f})",
                    color=c
                )
        
                # 9) Predicted Probability Density
                sns.kdeplot(
                    predictions,
                    ax=ax_pred_prob_density,
                    label=f"{trunc_name} (MeanPred={pred_prob_mean:.4f}, ratio={fraction_data:.2f})",
                    color=c
                )
        
                # 10) Precision-Recall
                ax_pr.plot(
                    recall,
                    precision,
                    marker='.',
                    label=f"{trunc_name} (AUPRC={auprc:.2f}, ratio={fraction_data:.2f})",
                    color=c
                )

            # Store metrics in records
            metrics_records.append({
                "Model":model_name,
                "Pooling": pooling_name,
                "truncation": trunc_name,
                "ratio": fraction_data,
                "ks statistic": ks_mean,
                "normal_auc": normal_auc,
                "uncertainty_auc": unc_auc,
                "auprc": auprc,
                "ece": ece_score,
                "brier_score_mean": brier_mean,
                "entropy_mean": entropy_mean,
                "nll_mean": nll_mean,
                "pred_prob_mean": pred_prob_mean,
                "ause": ause,
                "var_mean": variance_mean
            })
            #print(metrics_records[-1])
        # ===== Format subplots for each pooling =====
        if plot_results:
            ax_ks_density.set_xlabel("KS Statistic")
            ax_ks_density.set_xlim([0, 1])
            ax_ks_density.set_ylim([0, 6])
            ax_ks_density.set_title(f"{pooling_name}_{model_name} - KS Plot")
            
            ax_var_density.set_title(f"{pooling_name}_{model_name}- Variance Density")
            ax_var_density.set_xlabel("Variance")
            ax_var_density.legend()
    
            ax_unc_roc.set_title(f"{pooling_name}_{model_name} - Uncertainty ROC")
            ax_unc_roc.set_xlabel("False Positive Rate")
            ax_unc_roc.set_ylabel("True Positive Rate")
            ax_unc_roc.plot([0, 1], [0, 1], '--', color='gray')
            ax_unc_roc.legend()
            ax_unc_roc.grid(True)
    
            ax_calibration.set_title(f"{pooling_name}_{model_name}- Calibration Curve")
            ax_calibration.set_xlabel("Mean Predicted Probability")
            ax_calibration.set_ylabel("Actual Positive Proportion")
            ax_calibration.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_calibration.legend()
    
            ax_normal_roc.set_title(f"{pooling_name}_{model_name} - ROC Curve")
            ax_normal_roc.set_xlabel("False Positive Rate")
            ax_normal_roc.set_ylabel("True Positive Rate")
            ax_normal_roc.plot([0, 1], [0, 1], '--', color="gray")
            ax_normal_roc.legend()
            ax_normal_roc.grid(True)
    
            ax_sparsification.set_title(f"{pooling_name}_{model_name} - Sparsification Curve (Metric: {metric.upper()})")
            ax_sparsification.set_ylim([-0.01, 1.05])
            ax_sparsification.legend()
            ax_sparsification.grid(True)
    
            ax_brier_density.set_title(f"{pooling_name}_{model_name} - Brier Score Density")
            ax_brier_density.set_xlabel("Brier Score")
            ax_brier_density.set_ylabel("Density")
            ax_brier_density.legend()
    
            ax_entropy_density.set_title(f"{pooling_name}_{model_name} - Prediction Dist. Entropy Density")
            ax_entropy_density.set_xlabel("Entropy")
            ax_entropy_density.set_ylabel("Density")
            ax_entropy_density.legend()
    
            ax_nll_density.set_title(f"{pooling_name}_{model_name}- Negative Log-Likelihood (NLL) Density")
            ax_nll_density.set_xlabel("NLL")
            ax_nll_density.set_ylabel("Density")
            ax_nll_density.legend()
    
            ax_pred_prob_density.set_title(f"{pooling_name}_{model_name} - Predicted Probability Density")
            ax_pred_prob_density.set_xlabel("Predicted Probability")
            ax_pred_prob_density.set_ylabel("Density")
            ax_pred_prob_density.legend()
    
            ax_pr.set_title(f"{pooling_name}_{model_name} - Precision-Recall Curve")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_xlim([0.0, 1.0])
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.legend()
            ax_pr.grid(True)

    if plot_results:
        plt.tight_layout()
        plt.show()

    metrics_df = pd.DataFrame(metrics_records)
    print(metrics_df)
    return metrics_df
  