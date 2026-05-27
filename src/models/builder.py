"""Model factory."""

from .hybrid_vit import HybridVisionTransformer
from .resnet import build_resnet
from .vit import VisionTransformer


def build_model(args, num_classes, image_size):
    if args.model == "cnn":
        model = build_resnet(args, num_classes)
        return model, f"{args.dataset}_resnet{args.cnn_depth}"
    if args.model == "vit":
        model = VisionTransformer(
            image_size=image_size,
            patch_size=args.patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=args.vit_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=args.qkv_bias,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            drop_path_rate=args.drop_path_rate,
            use_cls_token=args.cls_token,
        )
        return model, f"{args.dataset}_vit_p{args.patch_size}_d{args.vit_depth}_h{args.vit_heads}_dim{args.vit_dim}"
    if args.model == "hybrid":
        model = HybridVisionTransformer(args, num_classes, image_size)
        return (
            model,
            f"{args.dataset}_hybrid_resnet{args.cnn_depth}_{args.hybrid_stage}"
            f"_vit_d{args.vit_depth}_h{args.vit_heads}_dim{args.vit_dim}",
        )
    raise ValueError(f"Unsupported model: {args.model}")
