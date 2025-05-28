_base_ = ["co_deformable_detr_mim_r50.py"]
pretrained = "models/swin_small_patch4_window7_224.pth"
# model settings

num_dec_layer = 2

model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformerV1",
        embed_dim=64,
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        out_indices=(1, 2, 3),
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained,
    ),
    neck=dict(in_channels=[128, 256, 512]),
    mim_head=dict(
        type="ImprovedSimMIMHead",
        in_channels=512,
        patch_size=8,
        loss=dict(type="MSELoss", loss_weight=0.5),
        training_only=True,
    ),
)

# optimizer
optimizer = dict(weight_decay=0.05)
