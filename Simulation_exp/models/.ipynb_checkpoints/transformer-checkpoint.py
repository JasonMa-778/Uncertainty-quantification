# models/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, pooling="cls"):
        """
        Custom Transformer Encoder Layer.
        
        Args:
            d_model (int): Model dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
            pooling (str): Pooling method ('cls', 'mean', 'mean_no_pad').
        """
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pooling = pooling

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def self_attention(self, query, key, value, mask=None):
        """
        Implement self-attention mechanism.
        
        Args:
            query (torch.Tensor): Query vectors.
            key (torch.Tensor): Key vectors.
            value (torch.Tensor): Value vectors.
            mask (torch.Tensor, optional): Mask.
        
        Returns:
            Tuple: (attention_output, attention_weights)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output, attention_weights

    def self_attention2(self, query, key, value, mask=None):
        """
        Implement alternative self-attention mechanism.
        
        Args:
            query (torch.Tensor): Query vectors.
            key (torch.Tensor): Key vectors.
            value (torch.Tensor): Value vectors.
            mask (torch.Tensor, optional): Mask.
        
        Returns:
            Tuple: (attention_output, attention_weights)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = ~mask.transpose(-2, -1)   # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        
        if mask is not None and self.pooling == "mean_no_pad":
            mask = mask.squeeze(2)  # [batch_size, 1, seq_len]
            mask = mask.unsqueeze(-1).float()  # [batch_size, 1, seq_len, 1]
            attention_output = attention_output * mask
        return attention_output, attention_weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass.
        
        Args:
            src (torch.Tensor): Input sequence, shape [batch_size, seq_length, d_model].
            src_mask (torch.Tensor, optional): Mask.
            src_key_padding_mask (torch.Tensor, optional): Padding mask.
        
        Returns:
            torch.Tensor: Output sequence.
        """
        batch_size, seq_length, emb_dim = src.size()
        query = self.q_proj(src)
        key = self.k_proj(src)
        value = self.v_proj(src)

        # Split into multiple heads
        query = query.view(batch_size, seq_length, self.nhead, -1).transpose(1, 2)  # [batch_size, nhead, seq_length, head_dim]
        key = key.view(batch_size, seq_length, self.nhead, -1).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.nhead, -1).transpose(1, 2)

        # Self-attention
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, seq_length, 1]
        else:
            mask = None
        
        if self.pooling != "mask_np_mp":
            src2, weights = self.self_attention(query, key, value, mask)
        else:
            src2, weights = self.self_attention2(query, key, value, mask)
        
        # Combine multiple heads
        src2 = src2.transpose(1, 2).contiguous().view(batch_size, seq_length, emb_dim)
        src2 = self.out_proj(src2)

        # Residual connection and layer normalization
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerWithPooling(nn.Module):
    def __init__(self, embedding_size, nhead, dim_feedforward, num_layers, num_classes, pooling='cls', dropout=0.1):
        """
        Transformer model with pooling and classification layer.
        
        Args:
            embedding_size (int): Embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            num_layers (int): Number of Transformer Encoder layers.
            num_classes (int): Number of classification classes.
            pooling (str): Pooling method ('cls', 'mean', 'mean_no_pad').
            dropout (float): Dropout rate.
        """
        super(TransformerWithPooling, self).__init__()
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

    def pool(self, src, src_key_padding_mask=None):
        """
        Apply pooling to the encoder output.
        """
        if self.pooling == 'cls':
            # Use CLS token
            return src[:, 0, :]

        elif self.pooling == 'mean':
            # Mean over all tokens except CLS (but includes padding)
            return src[:, 1:, :].mean(dim=1)

        elif self.pooling in ['mean_no_pad', 'mask_np_mp']:
            if src_key_padding_mask is None:
                raise ValueError(f"Pooling method '{self.pooling}' requires src_key_padding_mask.")

            # Remove CLS from src and mask
            src_wo_cls = src[:, 1:, :]                     # [B, L-1, D]
            mask = ~src_key_padding_mask[:, 1:]            # [B, L-1]
            mask = mask.unsqueeze(-1).float()              # [B, L-1, 1]

            summed = (src_wo_cls * mask).sum(dim=1)        # [B, D]
            counts = mask.sum(dim=1).clamp(min=1e-6)       # Avoid div-by-zero
            return summed / counts

        else:
            raise ValueError("Unsupported pooling type. Choose from 'cls', 'mean', 'mean_no_pad', or 'mask_np_mp'.")


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass.

        Args:
            src (torch.Tensor): Input sequence, shape [batch_size, seq_length, embedding_size].
            src_mask (torch.Tensor, optional): Attention mask.
            src_key_padding_mask (torch.Tensor, optional): Padding mask (bool): True for PAD tokens.

        Returns:
            torch.Tensor: Classification logits.
        """
        # Pass through transformer layers
        for layer in self.encoder_layers:
            src = layer(src, src_mask, src_key_padding_mask)

        # Apply pooling
        x = self.pool(src, src_key_padding_mask)

        # Final classification layer
        logits = self.linear(x)
        return logits

