from .nodes import VideoMaskEditor

NODE_CLASS_MAPPINGS = {
    "VideoMaskEditor": VideoMaskEditor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMaskEditor": "Video Mask Editor (视频遮罩修复编辑器)"
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
