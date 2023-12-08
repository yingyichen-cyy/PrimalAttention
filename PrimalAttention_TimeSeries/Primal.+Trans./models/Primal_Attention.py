import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat

from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

## Canonical softmax attention
class SoftmaxAttention(nn.Module):
    def __init__(self, d_keys, attention_dropout):
        super(SoftmaxAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.d_keys = d_keys

    def forward(self, queries, keys, values):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        dot = torch.matmul(queries, keys.transpose(-2, -1))
        dot = dot / math.sqrt(self.d_keys)

        attn = F.softmax(dot, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, values).transpose(1, 2).contiguous()

        return out

## Primal Attention with cosine similarity feature maps     
class PrimalCosAttention(nn.Module):
    def __init__(self, n_heads, d_keys, low_rank, attention_dropout):
        super(PrimalCosAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.Lambda = nn.Parameter(nn.init.uniform_(torch.Tensor(n_heads, low_rank)))

        self.concate_weight = nn.Linear(2*low_rank, d_keys)

    def feature_map(self, x):
        return F.normalize(x, p=2, dim=-1)

    def forward(self, queries, keys, we, wr):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        # feature_map
        queries = self.feature_map(queries)
        keys = self.feature_map(keys)
        # compute e-score and r-score
        escore = torch.einsum('...nd,...de->...ne', queries, we[0])
        rscore = torch.einsum('...nd,...de->...ne', keys, wr[0])
        score = torch.cat((escore, rscore), dim=-1)
        out = self.concate_weight(score).transpose(1,2).contiguous()

        return out, [escore, rscore, we[-1], wr[-1], queries, keys], self.Lambda

## Attention layer
class AttentionLayer(nn.Module):
    def __init__(self, max_seq_len, attn_type, d_model, n_heads, low_rank=20, rank_multi=10, nb_features=64,
                 attention_dropout=0.1, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        self.d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.attn_type = attn_type
        self.max_seq_len = max_seq_len

        if self.attn_type == "softmax":
            d_values = d_values or (d_model // n_heads)
            self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.out_projection = nn.Linear(self.d_keys * n_heads, d_model)
        self.n_heads = n_heads
        self.low_rank =  low_rank
        self.rank_multi = rank_multi
        self.len = min(self.max_seq_len, self.low_rank * self.rank_multi)

        if self.attn_type.startswith("primal"):
            self.We = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.n_heads, self.len, self.low_rank)))
            self.Wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.n_heads, self.len, self.low_rank)))

            if "cos" in self.attn_type:
                self.inner_attention = PrimalCosAttention(n_heads, self.d_keys, low_rank, attention_dropout)
            
        elif self.attn_type == "softmax":
            self.inner_attention = SoftmaxAttention(self.d_keys, attention_dropout)

    def forward(self, x):
        B, L, _ = x.shape
        _, S, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, S, H, -1)

        if self.attn_type == "softmax":
            values = self.value_projection(x).view(B, S, H, -1)
            out = self.inner_attention(queries, keys, values)
            out = out.reshape(B, L, -1)
            return self.out_projection(out)

        elif self.attn_type.startswith("primal"):
            # evenly sample
            indices = torch.linspace(0, L-1, self.len, dtype=int)
            x = x.transpose(-2,-1).reshape(B, H, self.d_keys, L)
            x = x[:, :, :, indices].transpose(1, 2)

            we = torch.einsum('bahd,hde->bahe', x, self.We.type_as(x)).transpose(1,2)
            wr = torch.einsum('bahd,hde->bahe', x, self.Wr.type_as(x)).transpose(1,2)

            out, scores, Lambda = self.inner_attention(queries, keys, [we.float(), self.We], [wr.float(), self.Wr])
            out = out.reshape(B, L, -1)
            return self.out_projection(out), scores, Lambda