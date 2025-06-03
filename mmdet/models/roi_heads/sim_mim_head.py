from mmdet.models.builder import HEADS, build_loss
import torch.nn as nn
import torch.nn.functional as F


@HEADS.register_module()
class SimMIMStyleHead(nn.Module):
    def __init__(
        self,
        in_channels=768,
        patch_size=16,
        use_backbone=False,
        training_only=True,

        loss=dict(type="L1Loss", loss_weight=1.0),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(in_channels, patch_size * patch_size * 3)
        self.loss_fn = build_loss(loss)
        self.training_only = training_only

        self.use_backbone = use_backbone

    def forward(self, x, output_size=None):
        if not self.training and self.training_only:
            return None
        # print(f"SimMIMStyleHead: {[i.shape for i in x]}")
        B, C, H, W = x.shape  # [B, C, H, W]
        num_patches_h = H
        num_patches_w = W
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C] = [B, N, C]
        out = self.linear(x)  # [B, N, patch_area*3]

        # [B, N, 3, p, p]
        out = out.view(
            B, num_patches_h * num_patches_w, 3, self.patch_size, self.patch_size
        )

        # [B, H, W, 3, p, p]
        out = out.view(
            B, num_patches_h, num_patches_w, 3, self.patch_size, self.patch_size
        )

        # [B, 3, H*p, W*p]
        out = (
            out.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(
                B, 3, num_patches_h * self.patch_size, num_patches_w * self.patch_size
            )
        )

        if output_size is not None:
            out = F.interpolate(
                out, size=output_size, mode="bilinear", align_corners=False
            )

        return out

    def loss(self, pred, target):
        return self.loss_fn(pred, target)

    def use_backbone(self):
        return self.use_backbone
