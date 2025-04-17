# models/mc_dropout.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoderLayer, TransformerWithPooling

class TransformerWithMCDropout(nn.Module):
    def __init__(self, embedding_size, nhead, dim_feedforward, num_layers, num_classes, pooling='cls', dropout=0.1):
        """
        Transformer model with MC Dropout.
        
        Args:
            embedding_size (int): Embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            num_layers (int): Number of Transformer Encoder layers.
            num_classes (int): Number of classification classes.
            pooling (str): Pooling method ('cls', 'mean', 'mean_no_pad').
            dropout (float): Dropout rate.
        """
        super(TransformerWithMCDropout, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.pooling = pooling
        self.linear = nn.Linear(embedding_size, num_classes)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.norm4 = nn.LayerNorm(embedding_size)

        # Create multiple Transformer Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_size, nhead, dim_feedforward, dropout)
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

    def enable_dropout(self):
        """
        Enable dropout layers during inference for MC Dropout.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
