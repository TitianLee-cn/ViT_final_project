"""Hybrid CNN + Vision Transformer."""

import torch
from torch import nn

from .layers import TransformerEncoderBlock
from .resnet import build_resnet


class HybridVisionTransformer(nn.Module):
    def __init__(self, args, num_classes, image_size):
        super().__init__()
        self.backbone = build_resnet(args, num_classes)
        self.stage = args.hybrid_stage
        if args.hybrid_freeze_cnn:
            for p in self.backbone.parameters():
                p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat = self.backbone.forward_features(dummy, stage=self.stage)
        feat_channels = feat.shape[1]
        feat_h, feat_w = feat.shape[2], feat.shape[3]
        if feat_h % args.hybrid_patch_size != 0 or feat_w % args.hybrid_patch_size != 0:
            raise ValueError("hybrid feature map size must be divisible by hybrid_patch_size")

        self.token_embed = nn.Conv2d(
            feat_channels,
            args.vit_dim,
            kernel_size=args.hybrid_patch_size,
            stride=args.hybrid_patch_size,
        )
        num_patches = (feat_h // args.hybrid_patch_size) * (feat_w // args.hybrid_patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, args.vit_dim))
        self.pos_drop = nn.Dropout(args.dropout)
        drop_rates = torch.linspace(0, args.drop_path_rate, args.vit_depth).tolist()
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    args.vit_dim,
                    args.vit_heads,
                    args.vit_mlp_ratio,
                    args.qkv_bias,
                    args.dropout,
                    args.attention_dropout,
                    drop_rates[i],
                )
                for i in range(args.vit_depth)
            ]
        )
        self.norm = nn.LayerNorm(args.vit_dim)
        self.head = nn.Linear(args.vit_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        feat = self.backbone.forward_features(x, stage=self.stage)
        x = self.token_embed(feat).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])
