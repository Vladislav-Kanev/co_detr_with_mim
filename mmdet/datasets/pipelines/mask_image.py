# mmdet/datasets/pipelines/mask_image.py
import numpy as np
from typing import Tuple
from mmdet.datasets.builder import PIPELINES

MASK_VALUE = 0  # Или любой другой цвет (например, 127 для серого)


@PIPELINES.register_module()
class MaskImage:
    def __init__(
        self,
        patch_size=(16, 16),
        mask_percent=0.6,
        mask_value=0,
        return_mask=False,
        save_mask_as="mask",
    ):
        self.patch_size = patch_size
        self.mask_percent = mask_percent
        self.mask_value = mask_value
        self.return_mask = return_mask
        self.save_mask_as = save_mask_as

    def __call__(self, results):
        img = results["img"].copy()

        seg_mask = results.get("gt_masks", None)
        bboxes = results.get("gt_bboxes", None)

        masked_img, mask = self._mask_random_patches(img)

        if self.return_mask:
            results[self.save_mask_as] = mask

        if seg_mask is not None and bboxes is not None:
            keep_indices = self._filter_boxes_by_iou(seg_mask, mask)
            bboxes, seg_mask = bboxes[keep_indices], seg_mask[keep_indices]

            results["gt_bboxes"] = bboxes
            results["gt_masks"] = seg_mask

        results["img"] = masked_img

        return results

    def _mask_random_patches(
        self,
        image: np.ndarray,
    ):
        assert 0 <= self.mask_percent <= 1, "mask_percent must be between 0 and 1"

        masked_image = np.copy(image)
        H, W = image.shape[:2]
        ph, pw = self.patch_size
        patches = [(i, j) for i in range(0, H, ph) for j in range(0, W, pw)]
        np.random.shuffle(patches)
        masked_patches = patches[: int(round(self.mask_percent * len(patches)))]

        mask = np.ones((H, W), dtype=np.uint8)
        for i, j in masked_patches:
            masked_image[i : i + ph, j : j + pw] = self.mask_value
            mask[i : i + ph, j : j + pw] = 0

        return (masked_image, mask)

    def _filter_boxes_by_iou(self, masks, mask_map):
        keep = []
        for i in range(len(masks)):
            obj_mask = masks[i].numpy().astype(bool)
            intersection = np.logical_and(obj_mask, mask_map == 0).sum()
            iou = intersection / (obj_mask.sum() + 1e-6)
            if iou <= self.iou_threshold:
                keep.append(i)
        return keep
