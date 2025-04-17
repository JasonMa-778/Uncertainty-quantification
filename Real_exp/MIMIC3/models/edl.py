import torch
import torch.nn as nn
import torch.nn.functional as F
from models.minute_embedding import PatientEmbedding
import torch.nn as nn

def apply_pooling(embeddings, strategy="cls", attention_mask=None):
    if strategy == "cls":
        return embeddings[:, 0, :]

    elif strategy == "mean":
        return embeddings.mean(dim=1)

    elif strategy == "mean_without_padding":
        if attention_mask is None:
            raise ValueError("attention_mask is required for mean_without_padding pooling")
        mask = attention_mask.unsqueeze(-1).expand_as(embeddings)
        sum_embeddings = (embeddings * mask).sum(dim=1)
        valid_token_count = mask.sum(dim=1)
        return sum_embeddings / valid_token_count

    elif strategy == "max":
        return embeddings.max(dim=1).values

    else:
        raise ValueError(f"Invalid pooling strategy: {strategy}")

class TransformerPredictorEDL(nn.Module):
    def __init__(self, d_embedding, d_model, dropout=0.1, n_layers=2, swe_pooling=None, tokenizer_codes=None, device='cpu', pooling="cls", num_classes=2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pooling = pooling  
        self.swe_pooling = swe_pooling
        self.embedding = PatientEmbedding(d_model=d_embedding, tokenizer_codes=tokenizer_codes, device=device)

        self.proj = nn.Linear(d_embedding, d_model).to(device)

        layer = nn.TransformerEncoderLayer(d_model, 2, dim_feedforward=2 * d_model, dropout=0.5, batch_first=True, device=device)
        self.transformer = nn.TransformerEncoder(layer, n_layers).to(device)

        self.cls = nn.Linear(d_model, num_classes, bias=True).to(device)
        self.prelu = nn.PReLU(num_parameters=1)


    def forward(self, codes, values, minutes, attention_mask=None):
        x = self.embedding(codes, values, minutes)
        x = self.prelu(x)

        x = self.proj(x)
        x = self.prelu(x)

        x = self.transformer(x)
        if self.pooling == "swe":
            x = self.swe_pooling(x, mask=attention_mask)
        else:
            x = apply_pooling(x, strategy=self.pooling, attention_mask=attention_mask)

        x = self.prelu(x)
        x = self.dropout(x)

        logits = self.cls(x)  
        logits = torch.clamp(logits, min=-10, max=10)  
        return logits
# models/edl_utils.py

import torch
import torch.nn.functional as F

def KL(alpha, K=2, device='cpu'):
    """
    计算两个 Dirichlet 分布之间的 KL 散度。

    Args:
        alpha (torch.Tensor): Dirichlet 参数，形状 [batch_size, num_classes]。
        K (int): 类别数量。
        device (str): 设备 ('cpu' 或 'cuda')。

    Returns:
        torch.Tensor: KL 散度。
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
    自定义的 MSE 损失函数，结合 EDL 原理。

    Args:
        p (torch.Tensor): 真实标签，形状 [batch_size]。
        alpha (torch.Tensor): Dirichlet 参数，形状 [batch_size, num_classes]。
        global_step (int, optional): 当前全局步骤，用于 annealing。
        annealing_step (int): Annealing 的步数。
        K (int): 类别数量。
        device (str): 设备 ('cpu' 或 'cuda')。

    Returns:
        torch.Tensor: 损失值。
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
    从 Dirichlet 参数计算不确定性和概率。

    Args:
        alpha (torch.Tensor): Dirichlet 参数，形状 [batch_size, num_classes]。

    Returns:
        tuple: (uncertainty, probabilities)
    """
    total_evidence = torch.sum(alpha, dim=1, keepdim=True)
    if torch.any(total_evidence <= 1e-6):  # 这里的阈值可以调整，例如 1e-6
        print("Warning: Total evidence contains zero or near-zero values!")
        print(f"Total evidence values: {total_evidence}")
    K = alpha.size(1)  # 类别数量
    uncertainty = K / total_evidence  # 不确定性度量
    probabilities = alpha / total_evidence  
    return uncertainty, probabilities

def logits2evidence(logits):
    """
    使用 ReLU 将 logits 转换为 evidence。

    Args:
        logits (torch.Tensor): 模型输出的 logits。

    Returns:
        torch.Tensor: evidence。
    """
    return F.relu(logits)
