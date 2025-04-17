# models/edl.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoderLayer, TransformerWithPooling
import math

class TransformerWithPoolingEDL(nn.Module):
    def __init__(self, embedding_size, nhead, dim_feedforward, num_layers, num_classes, pooling='cls', dropout=0.1):
        """
        Transformer model tailored for Evidential Deep Learning (EDL).
        
        Args:
            embedding_size (int): Embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            num_layers (int): Number of Transformer Encoder layers.
            num_classes (int): Number of classification classes.
            pooling (str): Pooling method ('cls', 'mean', 'mean_no_pad').
            dropout (float): Dropout rate.
        """
        super(TransformerWithPoolingEDL, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.pooling = pooling
        self.linear = nn.Linear(embedding_size, num_classes)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.norm4 = nn.LayerNorm(embedding_size)

        # Create multiple Transformer Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, nhead, dim_feedforward, dropout, pooling)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass.
        
        Args:
            src (torch.Tensor): Input sequence, shape [batch_size, seq_length, embedding_size].
            src_mask (torch.Tensor, optional): Mask.
            src_key_padding_mask (torch.Tensor, optional): Padding mask.
        
        Returns:
            torch.Tensor: Classification logits.
        """
        # Apply each Transformer Encoder layer
        for layer in self.encoder_layers:
            src = layer(src, src_mask, src_key_padding_mask)

        # Pooling
        if self.pooling == 'cls':
            x = src[:, 0, :]  # Use first token as CLS
        elif self.pooling == 'mean':
            x = src.mean(dim=1)  # Average all tokens, including padding
        elif self.pooling == 'mean_no_pad' or self.pooling == "mask_np_mp":
            if src_key_padding_mask is not None:
                mask = ~src_key_padding_mask
                mask = mask.unsqueeze(-1).float()  # [batch_size, seq_length, 1]
                x = (src * mask).sum(dim=1) / mask.sum(dim=1)  # Average excluding padding
            else:
                raise ValueError("Pooling method 'mean_no_pad' requires src_key_padding_mask.")
        else:
            raise ValueError("Unsupported pooling type. Choose 'cls', 'mean', or 'mean_no_pad'.")

        # Classification layer
        logits = self.linear(x)
        return logits  # Output logits without softmax

# Adding the custom loss functions here

def KL(alpha, K=2, device='cpu'):
    """
    Calculate KL divergence between Dirichlet distributions.
    
    Args:
        alpha (torch.Tensor): Dirichlet parameters, shape [batch_size, num_classes].
        K (int): Number of classes.
        device (str): Device ('cpu' or 'cuda').
    
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

def mse_loss(p, alpha, global_step=None, annealing_step=10, K=2, device='cpu'):
    """
    Custom MSE loss function combining EDL principles.
    
    Args:
        p (torch.Tensor): True labels, shape [batch_size].
        alpha (torch.Tensor): Dirichlet parameters, shape [batch_size, num_classes].
        global_step (int, optional): Current global step for annealing.
        annealing_step (int): Annealing step for KL divergence weight.
        K (int): Number of classes.
        device (str): Device ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: Loss value.
    """
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
