import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrimalCosAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.low_rank = config["low_rank"]
        # here is not related to batch samples
        self.Lambda = nn.Parameter(nn.init.uniform_(torch.Tensor(self.num_head, self.low_rank)))

        self.concate_weight = nn.Linear(2 * self.low_rank, self.head_dim)
        self.trace_no_x = config["trace_no_x"]

    def feature_map(self, X):
        # normalization should be on dim=-1
        return F.normalize(X, p=2, dim=-1) 

    def forward(self, Q, K, we, wr, mask):
        # feature_map
        queries = self.feature_map(Q) 
        keys = self.feature_map(K) 
        # compute e-score and r-score
        escore = torch.einsum('...nd,...de->...ne', queries, we[0])
        rscore = torch.einsum('...nd,...de->...ne', keys, wr[0])
        # scores
        score = torch.cat((escore, rscore), dim=-1) 
        attn_out = self.concate_weight(score) * mask[:, None, :, None]

        if self.trace_no_x:
            return attn_out, [escore, rscore, we[-1], wr[-1], queries, keys], self.Lambda
        else:
            return attn_out, [escore, rscore, we[0], wr[0], queries, keys], self.Lambda
