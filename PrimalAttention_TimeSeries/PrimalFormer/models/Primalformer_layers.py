import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attn_type, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attn_type = attn_type 
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        if self.attn_type == "softmax":
            new_x = self.attention(x)
        elif self.attn_type.startswith("primal"):
            new_x, scores, Lambda = self.attention(x)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        if self.attn_type == "softmax":
            return self.norm2(x + y)
        elif self.attn_type.startswith("primal"):
            return self.norm2(x + y), scores, Lambda


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        score_list = []
        Lambda_list = []
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            if attn_layer.attention.inner_attention.__class__.__name__.startswith("Softmax"):
                x = attn_layer(x)
            elif attn_layer.attention.inner_attention.__class__.__name__.startswith("Primal"):
                x, scores, Lambda = attn_layer(x)
                score_list.append(scores)
                Lambda_list.append(Lambda)

        if self.norm is not None:
            x = self.norm(x)

        if attn_layer.attention.inner_attention.__class__.__name__.startswith("Softmax"):
            return x
        elif attn_layer.attention.inner_attention.__class__.__name__.startswith("Primal"):
            return x, score_list, Lambda_list