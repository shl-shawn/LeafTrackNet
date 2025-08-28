import torch
import torch.nn as nn
from torchvision import models

_BACKBONES = {
    "resnet18":   ("resnet18",   "ResNet18_Weights"),
    "resnet34":   ("resnet34",   "ResNet34_Weights"),
    "resnet50":   ("resnet50",   "ResNet50_Weights"),
    "resnet101":  ("resnet101",  "ResNet101_Weights"),
    "mobilenet_v3": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "vit":        ("vit_b_16",   "ViT_B_16_Weights"),
}

def _build_backbone(backbone_type: str, use_pretrained: bool):
    """Create a torchvision backbone using the new 'weights' API if available,
    and fall back to 'pretrained=' for older torchvision versions."""
    key = backbone_type.lower()
    if key not in _BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone_type}")
    ctor_name, weights_name = _BACKBONES[key]

    ctor = getattr(models, ctor_name)  # e.g., tvm.resnet50
    weights = None
    if use_pretrained:
        try:
            weights_enum = getattr(models, weights_name)  # e.g., tvm.ResNet50_Weights
            weights = weights_enum.DEFAULT
        except AttributeError:
            weights = None
    try:
        return ctor(weights=weights)
    except TypeError:
        return ctor(pretrained=bool(use_pretrained))
    

class LeafReIDModel(nn.Module):
    """Backbone-agnostic embedding network for leaf re-identification."""
    def __init__(self, backbone_type: str = "mobilenet_v3", embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.backbone = _build_backbone(self.backbone_type, pretrained)
        # if self.backbone_type == "resnet50":
        #     backbone = models.resnet50(pretrained=pretrained)
        # elif self.backbone_type == "resnet18":
        #     backbone = models.resnet18(pretrained=pretrained)
        # elif self.backbone_type == "resnet34":
        #     backbone = models.resnet34(pretrained=pretrained)
        # elif self.backbone_type == "resnet101":
        #     backbone = models.resnet101(pretrained=pretrained)
        # elif self.backbone_type == "mobilenet_v3":
        #     backbone = models.mobilenet_v3_large(pretrained=pretrained)
        # elif self.backbone_type == "vit":
        #     backbone = models.vit_b_16(pretrained=pretrained)
        # else:
        #     raise ValueError(f"Unsupported backbone: {backbone_type}")
        # self.backbone = backbone

        # feature extractor
        if self.backbone_type.startswith("resnet") or self.backbone_type == "mobilenet_v3":
            self.feature_extractor = nn.Sequential(*(list(self.backbone.children())[:-1]))
        else:
            self.feature_extractor = None  # ViT handled specially

        # infer feature_dim
        if hasattr(self.backbone, "fc"):
            feature_dim = self.backbone.fc.in_features
        elif hasattr(self.backbone, "classifier"):
            # e.g., MobileNetV3 classifier is a Sequential
            feature_dim = None
            for layer in self.backbone.classifier:
                if isinstance(layer, nn.Linear):
                    feature_dim = layer.in_features
                    break
            if feature_dim is None:
                raise RuntimeError("Unable to infer feature_dim from classifier.")
        elif hasattr(self.backbone, "heads"):
            # ViT: look inside heads for Linear layer to get in_features
            feature_dim = None
            for m in self.backbone.heads.modules():
                if isinstance(m, nn.Linear):
                    feature_dim = m.in_features
                    break
            if feature_dim is None:
                raise RuntimeError("Unable to infer feature_dim from ViT heads.")
        else:
            raise RuntimeError("Unknown backbone type.")

        self.embedding = nn.Linear(feature_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        if self.backbone_type in {"resnet18","resnet34","resnet50","resnet101","mobilenet_v3"}:
            feats = self.feature_extractor(x)  # (B, C, 1, 1)
            feats = feats.view(feats.size(0), -1)
        else:
            # ViT path
            bck = self.backbone
            x_patches = bck._process_input(x)
            cls_tok = bck.class_token.expand(x_patches.size(0), -1, -1)
            feats = torch.cat((cls_tok, x_patches), dim=1)
            feats = bck.encoder(feats)
            feats = feats[:, 0, :]
        emb = self.embedding(feats)
        emb = self.bn(emb)
        return nn.functional.normalize(emb, p=2, dim=1)