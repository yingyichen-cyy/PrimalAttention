import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class PrimalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, embed_len=197, low_rank=20, rank_multi=10, \
                qk_bias=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # for the we and wr in primal_former
        self.low_rank = low_rank
        self.rank_multi = rank_multi
        self.embed_len = embed_len
        self.we = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_heads, min(self.embed_len, self.low_rank * self.rank_multi), self.low_rank)))
        self.wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_heads, min(self.embed_len, self.low_rank * self.rank_multi), self.low_rank)))
        self.Lambda = nn.Parameter(nn.init.uniform_(torch.Tensor(self.num_heads, self.low_rank)))
        self.concate_weight = nn.Linear(2 * self.low_rank, self.head_dim)

    def gen_weights(self, x):
        # evenly sample
        if self.embed_len > self.low_rank * self.rank_multi:
            indices = torch.linspace(0, x.shape[1]-1, self.low_rank * self.rank_multi, dtype=int)
            x = x.transpose(-2,-1).reshape(x.size(0), self.num_heads, self.head_dim, x.size(1))
            x = x[:, :, :, indices].transpose(1, 2)
        else:
            x = x.transpose(-2,-1).reshape(x.size(0), self.num_heads, self.head_dim, x.size(1))
            x = x.transpose(1, 2)
        we = torch.einsum('bahd,hde->bahe', x, self.we.type_as(x)).transpose(1,2)
        wr = torch.einsum('bahd,hde->bahe', x, self.wr.type_as(x)).transpose(1,2)
        return we, wr

    def feature_map(self, x):
        # normalization should be on dim=-1
        return F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)

        we, wr = self.gen_weights(x)
        q = self.feature_map(q) 
        k = self.feature_map(k) 
        escore = torch.einsum('...nd,...de->...ne', q, we)
        rscore = torch.einsum('...nd,...de->...ne', k, wr)
        score = torch.cat((escore, rscore), dim=-1) 
        attn_out = self.concate_weight(score) 

        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out, [escore, rscore, self.we, self.wr], self.Lambda


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            embed_len,
            low_rank,
            rank_multi=10,
            mlp_ratio=4.,
            qk_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            init_values=1e-5
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrimalAttention(
            dim,
            num_heads=num_heads,
            embed_len=embed_len,
            low_rank=low_rank, 
            rank_multi=rank_multi,
            qk_bias=qk_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
       
    def forward(self, x):
        out, scores, Lambda = self.attn(self.norm1(x))
        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, scores, Lambda


class PrimalFormer(nn.Module):

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            low_rank: list = [20] * 12,
            rank_multi: int = 10,
            mlp_ratio: float = 4.,
            qk_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            low_rank: how many components to reserve for the low rank.
            rank_multi: how many components to sample from the sequence length.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qk_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention drop

            out rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layey.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                embed_len=embed_len,
                low_rank=low_rank[i],
                rank_multi=rank_multi,
                mlp_ratio=mlp_ratio,
                qk_bias=qk_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        score_list = []
        Lambda_list = []

        x = self.patch_embed(x)
        x = self._pos_embed(x)

        for block in self.blocks:
            x, scores, Lambda = block(x)
            score_list.append(scores)
            Lambda_list.append(Lambda)

        x = self.norm(x)
        return x, score_list, Lambda_list

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x, score_list, Lambda_list = self.forward_features(x)
        x = self.forward_head(x)
        return x, score_list, Lambda_list


@register_model
def primal_tiny_patch16_224(low_rank=[20]*12, rank_multi=10, **kwargs):
    model = PrimalFormer(
        low_rank=low_rank, rank_multi=rank_multi,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qk_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def primal_small_patch16_224(low_rank=[20]*12, rank_multi=10, **kwargs):
    model = PrimalFormer(
        low_rank=low_rank, rank_multi=rank_multi,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qk_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def primal_base_patch16_224(low_rank=[20]*12, rank_multi=10, **kwargs):
    model = PrimalFormer(
        low_rank=low_rank, rank_multi=rank_multi,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qk_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    net = primal_tiny_patch16_224(low_rank=[20]*12, rank_multi=10)
    net = net.cuda()
    img = torch.cuda.FloatTensor(6, 3, 224, 224)
    with torch.no_grad():
        outs, score_list, Lambda_list = net(img)