import torch
import torch.nn as nn
from models.minute_embedding import PatientEmbedding

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

class TransformerPredictor(nn.Module):

    def __init__(self, d_embedding, d_model, dropout=0.1, n_layers=2, swe_pooling=None, tokenizer_codes=None, device='cpu', pooling="cls"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pooling = pooling  
        self.swe_pooling = swe_pooling
        self.embedding = PatientEmbedding(d_model=d_embedding, tokenizer_codes=tokenizer_codes, device=device)

        self.proj = nn.Linear(d_embedding, d_model).to(device)

        layer = nn.TransformerEncoderLayer(d_model, 2, dim_feedforward=2 * d_model, dropout=0.5, batch_first=True, device=device)
        self.transformer = nn.TransformerEncoder(layer, n_layers).to(device)

        self.cls = nn.Linear(d_model, 1, bias=True).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, codes, values, minutes, attention_mask=None):
        x = self.embedding(codes, values, minutes)
        x = nn.ReLU()(x)
        x = self.proj(x)
        x = nn.ReLU()(x)

        x = self.transformer(x)
        if self.pooling == "swe":
            x = self.swe_pooling(x, mask=attention_mask)
        else:
            x = apply_pooling(x, strategy=self.pooling, attention_mask=attention_mask)

        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.cls(x)
        #x = self.sigmoid(x)
        return x
