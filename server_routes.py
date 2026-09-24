import os
import io
import time
import json
import base64
import threading
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aiohttp import web
from server import PromptServer
import folder_paths

# Global sessions dictionary: { node_id: session_data }
sessions = {}

def get_disk_cache_dir(node_id: str, video_hash: str = None) -> str:
    """Return local disk cache directory for saving edited masks persistently, isolated per video."""
    cache_root = os.path.join(folder_paths.get_temp_directory(), "video_mask_editor")
    if video_hash:
        node_dir = os.path.join(cache_root, f"node_{node_id}", str(video_hash))
    else:
        node_dir = os.path.join(cache_root, f"node_{node_id}")
    os.makedirs(node_dir, exist_ok=True)
    return node_dir

def detect_flicker_frames(masks: torch.Tensor) -> list[int]:
    """
    Detect sudden drops in mask area indicating flickering/missing masks.
    masks: [B, H, W] float32 tensor
    """
    B = masks.shape[0]
    if B < 3:
        return []
    
    areas = [float(masks[i].sum().item()) for i in range(B)]
    flickers = []
    
    for i in range(B):
        curr = areas[i]
        if i == 0:
            neighbor_avg = areas[1]
        elif i == B - 1:
            neighbor_avg = areas[B - 2]
        else:
            neighbor_avg = (areas[i - 1] + areas[i + 1]) / 2.0
            
        if neighbor_avg > 50 and curr < 0.35 * neighbor_avg:
            flickers.append(i)
        elif neighbor_avg > 200 and curr < 20:
            flickers.append(i)
            
    return flickers

def tensor_to_jpeg_bytes(tensor_img: torch.Tensor, quality=85) -> bytes:
    """Convert [H, W, 3] float tensor (0..1) to JPEG bytes."""
    np_img = (tensor_img.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def tensor_mask_to_png_bytes(tensor_mask: torch.Tensor) -> bytes:
    """Convert [H, W] float tensor (0..1) to PNG bytes."""
    np_mask = (tensor_mask.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pil_mask = Image.fromarray(np_mask, mode="L")
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()

def base64_to_tensor_mask(b64_str: str, target_shape: tuple[int, int]) -> torch.Tensor:
    """Convert base64 PNG data url to [H, W] float tensor."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_data)).convert("L")
    H, W = target_shape
    if pil_img.size != (W, H):
        pil_img = pil_img.resize((W, H), Image.BILINEAR)
    np_arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(np_arr)

def base64_to_operation_mask(b64_str: str, target_shape: tuple[int, int]) -> torch.Tensor:
    """Decode a transparent canvas operation using its alpha channel."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_data)).convert("RGBA")
    H, W = target_shape
    if pil_img.size != (W, H):
        pil_img = pil_img.resize((W, H), Image.BILINEAR)
    alpha = np.asarray(pil_img, dtype=np.float32)[..., 3] / 255.0
    return torch.from_numpy(alpha.copy())

def save_cached_mask(node_id: str, frame_idx: int, mask: torch.Tensor):
    node_dir = get_disk_cache_dir(node_id)
    save_path = os.path.join(node_dir, f"frame_{frame_idx:05d}.png")
    np_mask = (mask.numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(np_mask, mode="L").save(save_path)

def translate_mask_tensor(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate [H, W] mask by (dx, dy) and zero-fill boundaries."""
    H, W = mask.shape
    new_mask = torch.zeros_like(mask)
    
    src_y_start = max(0, -dy)
    src_y_end = min(H, H - dy)
    dst_y_start = max(0, dy)
    dst_y_end = min(H, H + dy)
    
    src_x_start = max(0, -dx)
    src_x_end = min(W, W - dx)
    dst_x_start = max(0, dx)
    dst_x_end = min(W, W + dx)
    
    if src_y_end > src_y_start and src_x_end > src_x_start and dst_y_end > dst_y_start and dst_x_end > dst_x_start:
        new_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = mask[src_y_start:src_y_end, src_x_start:src_x_end]
        
    return new_mask

def resolve_session(node_id: str):
    """Find session by node_id or fallback to only active session."""
    node_id = str(node_id).strip()
    if node_id in sessions:
        return node_id, sessions[node_id]
    for k in sessions:
        if str(k) == node_id:
            return str(k), sessions[k]
    if len(sessions) == 1:
        single_k = list(sessions.keys())[0]
        return single_k, sessions[single_k]
    return None, None

# Register HTTP API Routes on PromptServer
_routes_registered = False

def register_routes():
    global _routes_registered
    if _routes_registered:
        return
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        return
        
    routes = PromptServer.instance.routes
    _routes_registered = True

    async def get_session(request):
        req_id = request.rel_url.query.get("node_id", "")
        node_id, sess = resolve_session(req_id)
        if not sess:
            return web.json_response({"has_session": False, "node_id": req_id})
        
        return web.json_response({
            "has_session": True,
            "node_id": node_id,
            "video_hash": sess.get("video_hash", ""),
            "num_frames": sess["num_frames"],
            "width": sess["width"],
            "height": sess["height"],
            "status": sess.get("status", "idle"),
            "mode": sess.get("mode", ""),
            "edited_indices": sorted(list(sess.get("edited_indices", set()))),
            "flicker_indices": sess.get("flicker_indices", []),
            "version": sess.get("version", 0)
        })

    async def get_frame(request):
        req_id = request.rel_url.query.get("node_id", "")
        frame_idx = int(request.rel_url.query.get("frame_idx", 0))
        frame_type = request.rel_url.query.get("type", "image")
        
        node_id, sess = resolve_session(req_id)
        if not sess:
            return web.Response(status=404, text="Session not found")
            
        if frame_idx < 0 or frame_idx >= sess["num_frames"]:
            return web.Response(status=400, text="Frame index out of range")
            
        if frame_type == "image":
            img_tensor = sess["images"][frame_idx]
            img_bytes = tensor_to_jpeg_bytes(img_tensor, quality=85)
            return web.Response(body=img_bytes, content_type="image/jpeg")
        else: # "mask"
            mask_tensor = sess["masks"][frame_idx]
            mask_bytes = tensor_mask_to_png_bytes(mask_tensor)
            return web.Response(body=mask_bytes, content_type="image/png")

    async def save_frame(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            frame_idx = int(data.get("frame_idx", 0))
            mask_data = data.get("mask_data", "")
            
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
                
            if frame_idx < 0 or frame_idx >= sess["num_frames"]:
                return web.json_response({"success": False, "error": "Invalid frame index"}, status=400)
                
            H, W = sess["height"], sess["width"]
            new_mask = base64_to_tensor_mask(mask_data, (H, W))
            try:
                sess["masks"][frame_idx] = new_mask
            except RuntimeError:
                sess["masks"] = torch.from_numpy(sess["masks"].cpu().numpy().copy())
                sess["masks"][frame_idx] = new_mask
                
            sess["edited_indices"].add(frame_idx)
            sess["version"] = sess.get("version", 0) + 1
            
            node_dir = sess.get("disk_cache_dir", get_disk_cache_dir(node_id, sess.get("video_hash")))
            save_path = os.path.join(node_dir, f"frame_{frame_idx:05d}.png")
            np_mask = (new_mask.numpy().clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(np_mask, mode="L").save(save_path)
            
            return web.json_response({
                "success": True,
                "frame_idx": frame_idx,
                "edited_indices": sorted(list(sess["edited_indices"])),
                "version": sess["version"]
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def apply_batch_operation(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            frame_indices = sorted(set(int(i) for i in data.get("frame_indices", [])))
            operation = data.get("operation", "add")
            operation_data = data.get("operation_data", "")

            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
            if not frame_indices or any(i < 0 or i >= sess["num_frames"] for i in frame_indices):
                return web.json_response({"success": False, "error": "Invalid frame selection"}, status=400)
            if operation not in ("add", "erase"):
                return web.json_response({"success": False, "error": "Invalid operation"}, status=400)

            H, W = sess["height"], sess["width"]
            op_mask = base64_to_operation_mask(operation_data, (H, W))
            edited_before = set(sess.get("edited_indices", set()))
            history_entry = {
                "frames": {i: tensor_mask_to_png_bytes(sess["masks"][i]) for i in frame_indices},
                "edited_before": edited_before,
            }

            for frame_idx in frame_indices:
                current = sess["masks"][frame_idx]
                # Match Canvas source-over / destination-out alpha compositing.
                updated = current + op_mask * (1.0 - current) if operation == "add" else current * (1.0 - op_mask)
                sess["masks"][frame_idx] = updated
                sess["edited_indices"].add(frame_idx)
                save_cached_mask(node_id, frame_idx, updated)

            history = sess.setdefault("batch_history", [])
            history.append(history_entry)
            if len(history) > 10:
                history.pop(0)
            sess["version"] = sess.get("version", 0) + 1
            return web.json_response({
                "success": True,
                "frame_indices": frame_indices,
                "edited_indices": sorted(sess["edited_indices"]),
                "version": sess["version"],
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def undo_batch_operation(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
            history = sess.setdefault("batch_history", [])
            if not history:
                return web.json_response({"success": False, "error": "No batch operation to undo"}, status=400)

            entry = history.pop()
            edited_before = entry["edited_before"]
            node_dir = get_disk_cache_dir(node_id)
            restored = []
            for frame_idx, png_bytes in entry["frames"].items():
                restored_mask = base64_to_tensor_mask(base64.b64encode(png_bytes).decode("ascii"), (sess["height"], sess["width"]))
                sess["masks"][frame_idx] = restored_mask
                restored.append(frame_idx)
                save_path = os.path.join(node_dir, f"frame_{frame_idx:05d}.png")
                if frame_idx in edited_before:
                    save_cached_mask(node_id, frame_idx, restored_mask)
                elif os.path.exists(save_path):
                    os.remove(save_path)
            sess["edited_indices"] = set(edited_before)
            sess["version"] = sess.get("version", 0) + 1
            return web.json_response({
                "success": True,
                "frame_indices": sorted(restored),
                "edited_indices": sorted(sess["edited_indices"]),
                "version": sess["version"],
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def copy_frame(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            target_frame = int(data.get("target_frame", 0))
            source_frame = int(data.get("source_frame", 0))
            
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
                
            num_frames = sess["num_frames"]
            if not (0 <= target_frame < num_frames and 0 <= source_frame < num_frames):
                return web.json_response({"success": False, "error": "Frame index out of bounds"}, status=400)
                
            try:
                sess["masks"][target_frame] = sess["masks"][source_frame].clone()
            except RuntimeError:
                sess["masks"] = torch.from_numpy(sess["masks"].cpu().numpy().copy())
                sess["masks"][target_frame] = sess["masks"][source_frame].clone()
                
            sess["edited_indices"].add(target_frame)
            sess["version"] = sess.get("version", 0) + 1
            
            node_dir = sess.get("disk_cache_dir", get_disk_cache_dir(node_id, sess.get("video_hash")))
            save_path = os.path.join(node_dir, f"frame_{target_frame:05d}.png")
            np_mask = (sess["masks"][target_frame].numpy().clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(np_mask, mode="L").save(save_path)
            
            return web.json_response({
                "success": True,
                "target_frame": target_frame,
                "source_frame": source_frame,
                "edited_indices": sorted(list(sess["edited_indices"])),
                "version": sess["version"]
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def translate_mask(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            frame_idx = int(data.get("frame_idx", 0))
            dx = int(data.get("dx", 0))
            dy = int(data.get("dy", 0))
            
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
                
            if not (0 <= frame_idx < sess["num_frames"]):
                return web.json_response({"success": False, "error": "Frame index out of bounds"}, status=400)
                
            curr_mask = sess["masks"][frame_idx]
            shifted_mask = translate_mask_tensor(curr_mask, dx, dy)
            try:
                sess["masks"][frame_idx] = shifted_mask
            except RuntimeError:
                sess["masks"] = torch.from_numpy(sess["masks"].cpu().numpy().copy())
                sess["masks"][frame_idx] = shifted_mask
                
            sess["edited_indices"].add(frame_idx)
            sess["version"] = sess.get("version", 0) + 1
            
            node_dir = sess.get("disk_cache_dir", get_disk_cache_dir(node_id, sess.get("video_hash")))
            save_path = os.path.join(node_dir, f"frame_{frame_idx:05d}.png")
            np_mask = (shifted_mask.numpy().clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(np_mask, mode="L").save(save_path)
            
            return web.json_response({
                "success": True,
                "frame_idx": frame_idx,
                "edited_indices": sorted(list(sess["edited_indices"])),
                "version": sess["version"]
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def reset_frame(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            frame_idx = int(data.get("frame_idx", 0))
            
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
                
            if not (0 <= frame_idx < sess["num_frames"]):
                return web.json_response({"success": False, "error": "Frame index out of bounds"}, status=400)
                
            try:
                sess["masks"][frame_idx] = sess["original_masks"][frame_idx].clone()
            except RuntimeError:
                sess["masks"] = torch.from_numpy(sess["masks"].cpu().numpy().copy())
                sess["masks"][frame_idx] = sess["original_masks"][frame_idx].clone()
                
            sess["edited_indices"].discard(frame_idx)
            sess["version"] = sess.get("version", 0) + 1
            
            node_dir = sess.get("disk_cache_dir", get_disk_cache_dir(node_id, sess.get("video_hash")))
            save_path = os.path.join(node_dir, f"frame_{frame_idx:05d}.png")
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
                    
            return web.json_response({
                "success": True,
                "frame_idx": frame_idx,
                "edited_indices": sorted(list(sess["edited_indices"])),
                "version": sess["version"]
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def reset_all(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
                
            sess["masks"] = sess["original_masks"].clone()
            sess["edited_indices"].clear()
            sess["version"] = sess.get("version", 0) + 1
            
            node_dir = sess.get("disk_cache_dir", get_disk_cache_dir(node_id, sess.get("video_hash")))
            if os.path.exists(node_dir):
                for f in os.listdir(node_dir):
                    if f.startswith("frame_") and f.endswith(".png"):
                        try:
                            os.remove(os.path.join(node_dir, f))
                        except Exception:
                            pass
                            
            return web.json_response({"success": True, "edited_indices": [], "version": sess["version"]})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def continue_execution(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
                
            sess["status"] = "resumed"
            if "event" in sess and isinstance(sess["event"], threading.Event):
                sess["event"].set()
                
            return web.json_response({"success": True, "status": "resumed"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def cancel_execution(request):
        try:
            data = await request.json()
            req_id = str(data.get("node_id", ""))
            
            node_id, sess = resolve_session(req_id)
            if not sess:
                return web.json_response({"success": False, "error": "Session not found"}, status=404)
                
            sess["status"] = "cancelled"
            if "event" in sess and isinstance(sess["event"], threading.Event):
                sess["event"].set()
                
            return web.json_response({"success": True, "status": "cancelled"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # Register both standard and /api prefixed routes
    for prefix in ["", "/api"]:
        routes.get(f"{prefix}/video_mask_editor/session")(get_session)
        routes.get(f"{prefix}/video_mask_editor/frame")(get_frame)
        routes.post(f"{prefix}/video_mask_editor/save_frame")(save_frame)
        routes.post(f"{prefix}/video_mask_editor/apply_batch_operation")(apply_batch_operation)
        routes.post(f"{prefix}/video_mask_editor/undo_batch_operation")(undo_batch_operation)
        routes.post(f"{prefix}/video_mask_editor/copy_frame")(copy_frame)
        routes.post(f"{prefix}/video_mask_editor/translate_mask")(translate_mask)
        routes.post(f"{prefix}/video_mask_editor/reset_frame")(reset_frame)
        routes.post(f"{prefix}/video_mask_editor/reset_all")(reset_all)
        routes.post(f"{prefix}/video_mask_editor/continue")(continue_execution)
        routes.post(f"{prefix}/video_mask_editor/cancel")(cancel_execution)
