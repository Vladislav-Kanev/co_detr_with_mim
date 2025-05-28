import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class ImprovedSimMIMHead(nn.Module):
    def __init__(
        self,
        in_channels=256,
        patch_size=16,
        loss=dict(type="L1Loss", loss_weight=0.1),
        training_only=True,
        use_backbone=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.loss_fn = build_loss(loss)
        self.training_only = training_only
        self.use_backbone = use_backbone

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3, kernel_size=1),
        )

    def forward(self, x, output_size=None):
        if not self.training and self.training_only:
            return None

        # x assumed shape: [B, C, H, W]
        out = self.decoder(x)  # output shape: [B, 3, H, W]

        if output_size is not None:
            out = F.interpolate(
                out, size=output_size, mode="bilinear", align_corners=False
            )

        return out

    def loss(self, pred, target):
        # Interpolate target if needed
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.loss_fn(pred, target)
