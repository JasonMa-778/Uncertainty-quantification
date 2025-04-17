# train/edl_train.py

import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.edl import TransformerWithPoolingEDL
from utils.dataset import MyDataset
from utils.early_stopping import EarlyStopping
from utils.train_utils import train_predict, pad_sequences, generate_padding_mask,train_predict_edl
from data.generate_data import generate_data
from torch.utils.data import DataLoader

def logits2evidence(logits):
    """
    Convert logits to evidence using ReLU.
    
    Args:
        logits (torch.Tensor): Logits from the model.
    
    Returns:
        torch.Tensor: Evidence.
    """
    return F.relu(logits)

def exp_evidence(logits): 
    """
    Convert logits to evidence using exponentiation with clamping.
    
    Args:
        logits (torch.Tensor): Logits from the model.
    
    Returns:
        torch.Tensor: Evidence.
    """
    return torch.exp(torch.clamp(logits, min=-10, max=10))

def softplus_evidence(logits):
    """
    Convert logits to evidence using softplus.
    
    Args:
        logits (torch.Tensor): Logits from the model.
    
    Returns:
        torch.Tensor: Evidence.
    """
    return F.softplus(logits)

def KL(alpha, K=2, device='cpu'):
    """
    Calculate KL divergence.
    
    Args:
        alpha (torch.Tensor): Dirichlet parameters.
        K (int): Number of classes.
        device (str): Device.
    
    Returns:
        torch.Tensor: KL divergence.
    """
    beta = torch.ones((1, K), dtype=torch.float32).to(device)

    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, global_step=None, annealing_step=10, K=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Custom MSE loss function combining EDL principles.
    
    Args:
        p (torch.Tensor): True labels, shape [batch_size].
        alpha (torch.Tensor): Dirichlet parameters, shape [batch_size, num_classes].
        global_step (int, optional): Current global step for annealing.
        annealing_step (int): Annealing step for KL divergence weight.
        K (int): Number of classes.
        device (str): Device.
    
    Returns:
        torch.Tensor: Loss value.
    """
    print(alpha)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    p_one_hot = F.one_hot(p, num_classes=m.shape[-1]).float().to(device)
    A = torch.sum((p_one_hot - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    if global_step is not None:
        annealing_coef = min(1.0, global_step / annealing_step)
    else:
        annealing_coef = 1
    alp = E * (1 - p_one_hot) + 1  
    C = annealing_coef * KL(alp, K, device=alpha.device)
    return torch.mean(A + B + C)

def uncertainty_and_probabilities(alpha):
    """
    Calculate uncertainty and probabilities from Dirichlet parameters.
    
    Args:
        alpha (torch.Tensor): Dirichlet parameters, shape [batch_size, num_classes].
    
    Returns:
        tuple: (uncertainty, probabilities)
    """
    total_evidence = torch.sum(alpha, dim=1, keepdim=True)
    K = alpha.size(1)  # Number of classes
    uncertainty = K / total_evidence  # Uncertainty measure
    probabilities = alpha / total_evidence  # Class probabilities
    return uncertainty, probabilities


