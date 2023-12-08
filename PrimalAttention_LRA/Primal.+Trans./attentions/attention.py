import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint
import sys
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        return V

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.dim = config["transformer_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        if self.attn_type.startswith("primal"):
            self.max_seq_len = config["max_seq_len"]
            self.low_rank = config["low_rank"]
            self.rank_multi = config["rank_multi"]

            self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
            # the weights should be based on batch sample
            self.We = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_head, self.low_rank * self.rank_multi, self.low_rank)))
            self.Wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_head, self.low_rank * self.rank_multi, self.low_rank)))

            if "cos" in self.attn_type:
                from attention_primal_cos import PrimalCosAttention
                self.attn = PrimalCosAttention(config)
        else:
            self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
            self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

            if self.attn_type == "softmax":
                self.attn = SoftmaxAttention(config)
            elif self.attn_type == "cos":
                from attention_cos import CosAttention
                self.attn = CosAttention(config)
            elif self.attn_type == "none":
                self.attn = NoneAttention(config)
            elif self.attn_type.startswith("linformer"):
                from attention_linformer import LinformerAttention
                self.attn = LinformerAttention(config)
            elif self.attn_type.startswith("reformer"):
                from attention_reformer import LSHAttention
                self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
            elif self.attn_type.startswith("nystrom"):
                from attention_nystrom import NystromAttention
                self.attn = NystromAttention(config)
            elif self.attn_type.startswith("performer"):
                from attention_performer import PerformerAttention
                self.attn = PerformerAttention(config)
            elif self.attn_type.startswith("linear"):
                from attention_linear import LinearAttention
                self.attn = LinearAttention(config)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):
        if self.attn_type.startswith("primal"):
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            # evenly sample
            indices = torch.linspace(0, X.shape[1]-1, self.low_rank * self.rank_multi, dtype=int)
            X = X.transpose(-2,-1).reshape(X.size(0), self.num_head, self.head_dim, X.size(1))
            X = X[:, :, :, indices].transpose(1, 2)

            if "cos" in self.attn_type:
                we = torch.einsum('bahd,hde->bahe', X, self.We.type_as(X)).transpose(1,2)
                wr = torch.einsum('bahd,hde->bahe', X, self.Wr.type_as(X)).transpose(1,2)

            with torch.cuda.amp.autocast(enabled = False):
                attn_out, scores, Lambda = self.attn(Q.float(), K.float(), [we.float(), self.We], [wr.float(), self.Wr], mask.float())
            attn_out = self.combine_heads(attn_out)
            out = self.ff(attn_out)

            return out, scores, Lambda

        else:
            if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
                with torch.cuda.amp.autocast(enabled = False):
                    attn_out = self.attn(X.float(), mask.float())
            else:
                Q = self.split_heads(self.W_q(X))
                K = self.split_heads(self.W_k(X))
                V = self.split_heads(self.W_v(X))
                with torch.cuda.amp.autocast(enabled = False):
                    if self.grad_checkpointing:
                        attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                    else:
                        attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
                attn_out = self.combine_heads(attn_out)

            out = self.ff(attn_out)

            return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X