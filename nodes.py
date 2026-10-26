import os
import time
import json
import logging
import hashlib
import threading
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from .ratio_mask import RATIO_PRESETS, ALIGNMENTS, create_ratio_padding_mask

import comfy.model_management as mm
from server import PromptServer

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    ExecutionBlocker = None

from .server_routes import (
    sessions,
    get_disk_cache_dir,
    write_cache_metadata,
    detect_flicker_frames,
    register_routes
)

# Initialize API routes
register_routes()

logger = logging.getLogger("ComfyUI-VideoMaskEditor")

def compute_video_hash(images: torch.Tensor) -> str:
    """
    Compute a fast fingerprint for the input video sequence.
    Differentiates different videos by shape and sparse pixel sample.
    """
    B, H, W, C = images.shape
    step_b = max(1, B // 8)
    step_h = max(1, H // 16)
    step_w = max(1, W // 16)
    sample = images[::step_b, ::step_h, ::step_w, :].contiguous()
    h = hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16]
    return f"{B}f_{W}x{H}_{h}"

def apply_gaussian_feather(mask_tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Feather mask tensor [B, H, W] using 2D Gaussian convolution.
    """
    if radius <= 0:
        return mask_tensor
        
    kernel_size = radius * 2 + 1
    sigma = max(0.1, radius / 2.0)
    
    # 1D gaussian
    k = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss1d = torch.exp(-0.5 * (k / sigma) ** 2)
    gauss1d = gauss1d / gauss1d.sum()
    
    # 2D gaussian kernel
    gauss2d = gauss1d.unsqueeze(1) * gauss1d.unsqueeze(0)
    kernel = gauss2d.unsqueeze(0).unsqueeze(0).to(mask_tensor.device)
    
    # Pad and apply depthwise convolution
    masks_4d = mask_tensor.unsqueeze(1)
    padded = F.pad(masks_4d, (radius, radius, radius, radius), mode="replicate")
    feathered = F.conv2d(padded, kernel)
    return feathered.squeeze(1).clamp(0.0, 1.0)


class VideoMaskEditor:
    """
    Interactive Video Mask Editor for ComfyUI.
    Fixes flickering, missing, or corrupt mask frames from SAM3 or any tracking/segmentation node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "原视频帧序列 (IMAGE)"}),
                "mode": ([
                    "Interactive (Pause & Wait)",
                    "Use Edited or Passthrough",
                    "Block Downstream"
                ], {
                    "default": "Interactive (Pause & Wait)",
                    "tooltip": "模式说明：\n1. Interactive (Pause & Wait): 运行到本节点自动暂停并弹出编辑器，修改后点击【继续】下游无缝执行\n2. Use Edited or Passthrough: 直接使用已保存的编辑或原样输出（适合停止后再运行）\n3. Block Downstream: 阻断下游大模型执行，仅供预览与精修"
                }),
                "feather_edges": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "输出遮罩边缘羽化/平滑像素半径（0为原样保持）"
                }),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "输入遮罩模式下必接；比例补边模式下可不连接"}),
                "reset_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "勾选后将在本次运行时强制清空本视频的所有手动编辑缓存，恢复到初始输入遮罩"
                }),
                "mask_source": (["Input masks", "Aspect ratio padding"], {
                    "default": "Input masks",
                    "tooltip": "Input masks：使用输入遮罩；Aspect ratio padding：按原始画面比例生成补边区域遮罩，忽略输入遮罩和画面颜色"
                }),
                "original_ratio": (RATIO_PRESETS, {
                    "default": "16:9",
                    "tooltip": "补边前原始画面的宽高比。输入 images 必须是已经补边后的画面，目标比例由其尺寸自动确定"
                }),
                "ratio_width": ("FLOAT", {"default": 16.0, "min": 0.01, "max": 16384.0,
                    "step": 0.01, "tooltip": "original_ratio=Custom 时使用的原始比例宽度"}),
                "ratio_height": ("FLOAT", {"default": 9.0, "min": 0.01, "max": 16384.0,
                    "step": 0.01, "tooltip": "original_ratio=Custom 时使用的原始比例高度"}),
                "content_alignment": (ALIGNMENTS, {
                    "default": "Top",
                    "tooltip": "原画面在补边后画布中的位置。Top=顶部对齐，新增遮罩在下方；Center=居中，两侧均分；Bottom=底部对齐"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "images")
    FUNCTION = "process"
    CATEGORY = "mask/video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, images, masks=None, mode="Interactive (Pause & Wait)", feather_edges=0, reset_cache=False, unique_id=None, **kwargs):
        # Force re-execution if session version, mode, or feather parameters changed
        node_id = str(unique_id) if unique_id is not None else "default"
        version = sessions.get(node_id, {}).get("version", 0)
        return f"{node_id}_{version}_{mode}_{feather_edges}_{reset_cache}_{time.time()}"

    def process(self, images: torch.Tensor, masks: torch.Tensor = None,
                mode: str = "Interactive (Pause & Wait)", feather_edges: int = 0,
                reset_cache: bool = False, unique_id=None, mask_source="Input masks",
                original_ratio="16:9", ratio_width=16.0, ratio_height=9.0,
                content_alignment="Top", **kwargs):
        node_id = str(unique_id) if unique_id is not None else "default"

        # 1. Normalize dimensions
        # images: [B, H, W, 3]
        if images.ndim == 3:
            images = images.unsqueeze(0)
        B, H, W, C = images.shape

        ratio_cache_key = None
        if mask_source == "Aspect ratio padding":
            if original_ratio != "Custom":
                if original_ratio not in RATIO_PRESETS:
                    raise ValueError(f"Unknown original aspect ratio: {original_ratio}")
                ratio_width, ratio_height = map(float, original_ratio.split(":"))
            masks = create_ratio_padding_mask(
                B, H, W, ratio_width, ratio_height, content_alignment, device=images.device
            )
            # Different geometric masks must never reuse each other's manual edits.
            ratio_cache_key = hashlib.sha256(json.dumps(
                [float(ratio_width), float(ratio_height), content_alignment],
                separators=(",", ":")
            ).encode("utf-8")).hexdigest()[:16]
        elif mask_source == "Input masks":
            if masks is None:
                raise ValueError("Input masks 模式需要连接 masks；生成补边遮罩请选择 Aspect ratio padding。")
        else:
            raise ValueError(f"Unknown mask source: {mask_source}")

        orig_device = masks.device
        orig_dtype = masks.dtype

        # masks: [B, H, W]
        if masks.ndim == 4:
            if masks.shape[1] == 1:
                masks = masks.squeeze(1)
            elif masks.shape[-1] == 1:
                masks = masks.squeeze(-1)
            elif masks.shape[-1] == 3:
                masks = masks.mean(dim=-1)
        elif masks.ndim == 2:
            masks = masks.unsqueeze(0)

        # Match batch sizes if mismatched
        if masks.shape[0] == 1 and B > 1:
            masks = masks.repeat(B, 1, 1)
        elif masks.shape[0] > 1 and B == 1:
            images = images.repeat(masks.shape[0], 1, 1, 1)
            B = masks.shape[0]

        # Match spatial resolution if needed
        if masks.shape[1] != H or masks.shape[2] != W:
            logger.info(f"VideoMaskEditor [{node_id}]: Resizing mask {masks.shape[1:]} to image resolution {(H, W)}")
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

        # Move to CPU for editing session and detach from PyTorch InferenceMode
        cpu_images = torch.from_numpy(images.detach().cpu().float().numpy().copy())
        cpu_masks = torch.from_numpy(masks.detach().cpu().float().clamp(0.0, 1.0).numpy().copy())

        # Compute video hash to isolate edits per video sequence
        video_hash = compute_video_hash(cpu_images)
        if ratio_cache_key is not None:
            video_hash += f"_ratio_{ratio_cache_key}"
        logger.info(f"VideoMaskEditor [{node_id}]: Video fingerprint: {video_hash}")

        # 2. Detect flickering frames
        flicker_indices = detect_flicker_frames(cpu_masks)
        if flicker_indices:
            logger.warning(f"VideoMaskEditor [{node_id}]: ⚠️ Detected {len(flicker_indices)} potential flicker frame(s): {flicker_indices}")

        # 3. Load existing disk cache edits isolated by video_hash
        node_dir = get_disk_cache_dir(node_id, video_hash)
        
        # Clean any legacy loose png files in the parent directory
        parent_dir = os.path.dirname(node_dir)
        if os.path.exists(parent_dir):
            for f in os.listdir(parent_dir):
                if f.startswith("frame_") and f.endswith(".png"):
                    try:
                        os.remove(os.path.join(parent_dir, f))
                    except Exception:
                        pass

        # If reset_cache is True, wipe current video edits
        if reset_cache and os.path.exists(node_dir):
            logger.info(f"VideoMaskEditor [{node_id}]: 'reset_cache' is True. Wiping edits for video {video_hash}...")
            for f in os.listdir(node_dir):
                if f.startswith("frame_") and f.endswith(".png"):
                    try:
                        os.remove(os.path.join(node_dir, f))
                    except Exception:
                        pass

        edited_indices = set()
        active_masks = cpu_masks.clone()

        if not reset_cache and os.path.exists(node_dir):
            for f in os.listdir(node_dir):
                if f.startswith("frame_") and f.endswith(".png"):
                    try:
                        idx_str = f.replace("frame_", "").replace(".png", "")
                        f_idx = int(idx_str)
                        if 0 <= f_idx < B:
                            img_path = os.path.join(node_dir, f)
                            pil_m = Image.open(img_path).convert("L")
                            if pil_m.size != (W, H):
                                pil_m = pil_m.resize((W, H), Image.BILINEAR)
                            m_arr = np.array(pil_m, dtype=np.float32) / 255.0
                            active_masks[f_idx] = torch.from_numpy(m_arr)
                            edited_indices.add(f_idx)
                    except Exception as e:
                        logger.warning(f"Error loading cached frame {f}: {e}")

        # 4. Update in-memory session
        event = threading.Event()
        prev_sess = sessions.get(node_id, {})
        new_version = prev_sess.get("version", 0) + 1

        sessions[node_id] = {
            "images": cpu_images,
            "masks": active_masks,
            "original_masks": cpu_masks.clone(),
            "num_frames": B,
            "height": H,
            "width": W,
            "status": "waiting_for_user" if mode == "Interactive (Pause & Wait)" else "ready",
            "mode": mode,
            "edited_indices": edited_indices,
            "flicker_indices": flicker_indices,
            "event": event,
            "version": new_version,
            "video_hash": video_hash,
            "disk_cache_dir": node_dir
        }
        # Keep a manifest beside the PNG frames.  The editor can then list and
        # manage this cache even after ComfyUI has been restarted.
        try:
            write_cache_metadata(node_dir, node_id, sessions[node_id])
        except Exception as e:
            logger.warning(f"VideoMaskEditor [{node_id}]: Failed to write cache metadata: {e}")

        # 5. Notify Frontend via WebSocket
        register_routes()
        if hasattr(PromptServer, "instance") and PromptServer.instance is not None:
            try:
                PromptServer.instance.send_sync("video-mask-editor-update", {
                    "node_id": node_id,
                    "video_hash": video_hash,
                    "num_frames": B,
                    "width": W,
                    "height": H,
                    "mode": mode,
                    "flicker_indices": flicker_indices,
                    "edited_indices": sorted(list(edited_indices)),
                    "is_paused": mode == "Interactive (Pause & Wait)"
                })
            except Exception as e:
                logger.warning(f"Failed to send websocket notification: {e}")

        # 6. Mode handling
        if mode == "Block Downstream":
            logger.info(f"VideoMaskEditor [{node_id}]: Mode is 'Block Downstream'. Data cached, downstream blocked.")
            if ExecutionBlocker is not None:
                return (ExecutionBlocker(None), ExecutionBlocker(None))
            else:
                # Return empty tensors to prevent memory usage
                return (torch.zeros_like(masks[:1]), torch.zeros_like(images[:1]))

        elif mode == "Interactive (Pause & Wait)":
            logger.info(f"VideoMaskEditor [{node_id}]: ⏸️ Execution paused. Waiting for user interaction in web editor...")
            # Wait until user clicks "Save & Continue" or cancels
            while not event.is_set():
                if mm.processing_interrupted():
                    sessions[node_id]["status"] = "cancelled"
                    logger.info(f"VideoMaskEditor [{node_id}]: Cancelled by ComfyUI interrupt.")
                    raise mm.InterruptProcessingException()
                time.sleep(0.15)

            if sessions[node_id].get("status") == "cancelled":
                raise mm.InterruptProcessingException()

            logger.info(f"VideoMaskEditor [{node_id}]: ▶️ Resuming execution with edited masks!")

        else: # "Use Edited or Passthrough"
            logger.info(f"VideoMaskEditor [{node_id}]: Outputting masks directly ({len(edited_indices)} frames with manual edits).")

        # 7. Prepare final masks
        final_masks = sessions[node_id]["masks"].to(device=orig_device, dtype=orig_dtype)
        final_images = cpu_images.to(device=orig_device, dtype=images.dtype)

        # 8. Apply feathering if requested
        if feather_edges > 0:
            final_masks = apply_gaussian_feather(final_masks, feather_edges)

        return (final_masks, final_images)
