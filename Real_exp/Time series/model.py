import torch
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
        valid_token_count = torch.clamp(valid_token_count, min=1)
        return sum_embeddings / valid_token_count
    elif strategy == "max":
        return embeddings.max(dim=1).values
    else:
        raise ValueError(f"Invalid pooling strategy: {strategy}")
class TransformerPredictor(nn.Module):
    def __init__(self, d_model=2, n_layers=2, dropout=0.1, pooling="cls", sequence_length=152, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pooling = pooling
        self.embedding = nn.Linear(1, d_model).to(device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(device)
        self.cls = nn.Linear(d_model, 1, bias=True).to(device)
    def forward(self, sequence, attention_mask=None):
        sequence = sequence.unsqueeze(-1)
        sequence = self.embedding(sequence)
        sequence = nn.ReLU()(sequence)
        sequence = self.transformer(sequence)
        sequence = apply_pooling(sequence, strategy=self.pooling, attention_mask=attention_mask)
        sequence = nn.ReLU()(sequence)
        sequence = self.dropout(sequence)
        return self.cls(sequence)