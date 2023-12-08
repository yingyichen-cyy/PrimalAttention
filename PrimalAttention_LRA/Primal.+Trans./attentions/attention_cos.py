import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CosAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

    def feature_map(self, X):
        # normalization should be on dim=-1
        return F.normalize(X, p=2, dim=-1) 

    def forward(self, Q, K, V, mask):
        # feature_map
        Q = self.feature_map(Q) / math.sqrt(math.sqrt(Q.size(2)))
        K = self.feature_map(K) * mask[:, None, :, None] / math.sqrt(math.sqrt(K.size(2)))
        V = V * mask[:, None, :, None]  

        X = torch.matmul(Q, torch.matmul(torch.transpose(K, -2, -1), V))

        return X
