import torch
import torch.nn as nn
import torch.nn.functional as F


class Primal_attention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, low_rank=30, rank_multi=10, drop_out=0.05):
        super(Primal_attention, self).__init__()
        self.d_keys = d_model // n_heads

        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.out_projection = nn.Linear(self.d_keys * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(drop_out)

        self.len = min(seq_len, low_rank * rank_multi)
        self.We = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.n_heads, self.d_keys, low_rank)))
        self.Wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.n_heads, self.d_keys, low_rank)))
        self.Lambda = nn.Parameter(nn.init.uniform_(torch.Tensor(self.n_heads, low_rank)))
        self.concate_weight = nn.Linear(2*low_rank, self.d_keys)

    def feature_map(self, x):
        return F.normalize(x, p=2, dim=-1)

    def forward(self, queries, keys, values, attention_mask=None):
        ## Note: queries, keys, values are not projected yet
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)

        ## generate weights for Primal attention
        normal = (((torch.arange(L)).float() + 1.0)).to(queries.device)

        # transpose the queries and keys
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        # we conduct cumsum before the non-linear map (Causal map)
        queries = queries.cumsum(dim=2) / normal[None, None, :, None]
        keys = keys.cumsum(dim=2) / normal[None, None, :, None]
        # feature_map
        queries = self.feature_map(queries)
        keys = self.feature_map(keys)
        # compute e-score and r-score
        escore = torch.einsum('...nd,...de->...ne', queries, self.We.unsqueeze(0).repeat(queries.size(0),1,1,1))
        rscore = torch.einsum('...nd,...de->...ne', keys, self.Wr.unsqueeze(0).repeat(keys.size(0),1,1,1))
        score = torch.cat((escore, rscore), dim=-1)
        out = self.concate_weight(score).transpose(1, 2).contiguous()
        # Final projection
        out = out.reshape(B, L, -1)
        out = self.out_projection(out)
        out = self.dropout(out)
        return out, [escore, rscore, self.We, self.Wr], self.Lambda
