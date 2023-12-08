# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import models_primal


class ComboVisionTransformer(VisionTransformer):
    def __init__(self, low_rank, rank_multi, num_ksvd_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low_rank = low_rank
        self.rank_multi = rank_multi
        self.num_ksvd_layer = num_ksvd_layer
        self.depth = kwargs['depth']

        dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], self.depth)]  # stochastic depth decay rule
        for i in range(self.num_ksvd_layer):
            self.blocks[-(i+1)] = models_primal.Block(
                dim=self.embed_dim,
                num_heads=kwargs['num_heads'],
                embed_len=self.patch_embed.num_patches+1,
                low_rank=self.low_rank,
                rank_multi=self.rank_multi,
                mlp_ratio=kwargs['mlp_ratio'],
                qk_bias=kwargs['qkv_bias'],
                drop_path=dpr[-(i+1)],
                norm_layer=kwargs['norm_layer'],
            )

    def forward_features(self, x):
        score_list = []
        Lambda_list = []

        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for idx in range(self.depth):
            if idx < (self.depth - self.num_ksvd_layer):
                x = self.blocks[idx](x)
            else:
                x, scores, Lambda = self.blocks[idx](x)
                score_list.append(scores)
                Lambda_list.append(Lambda)

        x = self.norm(x)
        return x, score_list, Lambda_list

    def forward(self, x):
        x, score_list, Lambda_list = self.forward_features(x)
        x = self.head(x[:, 0])
        return x, score_list, Lambda_list


@register_model
def primal_tiny_patch16_224(low_rank=20, rank_multi=10, num_ksvd_layer=2, **kwargs):
    model = ComboVisionTransformer(
        low_rank=low_rank, rank_multi=rank_multi, num_ksvd_layer=num_ksvd_layer,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def primal_small_patch16_224(low_rank=20, rank_multi=10, num_ksvd_layer=2, **kwargs):
    model = ComboVisionTransformer(
        low_rank=low_rank, rank_multi=rank_multi, num_ksvd_layer=num_ksvd_layer,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def primal_base_patch16_224(low_rank=20, rank_multi=10, num_ksvd_layer=2, **kwargs):
    model = ComboVisionTransformer(
        low_rank=low_rank, rank_multi=rank_multi, num_ksvd_layer=num_ksvd_layer,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    net = primal_small_patch16_224(low_rank=20, rank_multi=10, num_ksvd_layer=2, drop_path_rate=0.1)
    net = net.cuda()
    img = torch.cuda.FloatTensor(6, 3, 224, 224)
    with torch.no_grad():
        outs, score_list, Lambda_list = net(img)