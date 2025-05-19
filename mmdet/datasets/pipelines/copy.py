from ..builder import PIPELINES


@PIPELINES.register_module()
class CopyImage:
    def __init__(self, dst_key="clear_image"):
        self.dst_key = dst_key

    def __call__(self, results):
        # print(f"Copying image:", results["img"].shape)
        results[self.dst_key] = results["img"].copy()
        return results
