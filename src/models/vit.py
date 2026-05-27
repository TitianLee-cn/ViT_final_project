"""Vision Transformer implemented from PyTorch layers."""

import torch
from torch import nn

from .layers import TransformerEncoderBlock


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        use_cls_token=True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)
        token_count = self.patch_embed.num_patches + (1 if use_cls_token else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    dropout,
                    attention_dropout,
                    drop_rates[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0] if self.use_cls_token else x.mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))
