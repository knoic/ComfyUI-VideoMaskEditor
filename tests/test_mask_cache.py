"""Cache regression tests using real tensors/PNGs and stubbed ComfyUI services.

Run with: python -m unittest discover -s tests -v
"""

import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import web
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


class JsonRequest:
    def __init__(self, data):
        self.data = data

    async def json(self):
        return self.data


class MaskCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = temporary.name

        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
            routes=web.RouteTableDef(), send_sync=lambda *args: None,
        ))
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: self.output
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")
        management.processing_interrupted = lambda: False
        comfy.model_management = management
        package = types.ModuleType("mask_cache_test_package")
        package.__path__ = [str(ROOT)]

        modules = patch.dict(sys.modules, {
            "server": server, "folder_paths": folder_paths, "comfy": comfy,
            "comfy.model_management": management,
            "mask_cache_test_package": package,
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.routes = self.load_module("server_routes")
        self.nodes = self.load_module("nodes")
        self.handlers = {
            route.path.rsplit("/", 1)[-1]: route.handler
            for route in server.PromptServer.instance.routes
        }
        self.images = torch.zeros((2, 4, 4, 3))

    def load_module(self, name):
        module_name = f"mask_cache_test_package.{name}"
        spec = importlib.util.spec_from_file_location(module_name, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def process(self, node_id, reset_cache=False, images=None):
        images = self.images if images is None else images
        masks = torch.zeros(images.shape[:3])
        return self.nodes.VideoMaskEditor().process(
            images, masks, "Use Edited or Passthrough",
            reset_cache=reset_cache, unique_id=node_id,
        )[0]

    async def call(self, route, expected_status=200, **data):
        response = await self.handlers[route](JsonRequest(data))
        body = json.loads(response.text)
        self.assertEqual(response.status, expected_status, body)
        return body

    async def save(self, node_id, value, frame_idx=0):
        mask = torch.full((4, 4), float(value))
        png = self.routes.tensor_mask_to_png_bytes(mask)
        return await self.call(
            "save_frame", node_id=node_id, frame_idx=frame_idx,
            mask_data=base64.b64encode(png).decode("ascii"),
        )

    async def snapshot(self, node_id):
        return (await self.call(
            "save_mask_snapshot", node_id=node_id, label="Regression snapshot",
        ))["cache_id"]

    async def test_same_video_nodes_remain_isolated_after_restart(self):
        self.process("101")
        await self.save("101", 1)
        self.assertEqual(self.process("202").sum().item(), 0)
        self.assertNotEqual(
            self.routes.sessions["101"]["disk_cache_dir"],
            self.routes.sessions["202"]["disk_cache_dir"],
        )
        self.routes.sessions.clear()  # Simulate loss of in-memory state.
        self.assertEqual(self.process("101")[0].sum().item(), 16)
        self.assertEqual(self.process("202").sum().item(), 0)

    async def test_reset_cache_preserves_other_node_and_snapshot(self):
        for node_id in ("101", "202"):
            self.process(node_id)
            await self.save(node_id, 1)
        snapshot = await self.snapshot("101")
        self.assertEqual(self.process("101", reset_cache=True).sum().item(), 0)
        self.assertEqual(self.process("202")[0].sum().item(), 16)
        snapshot_dir = Path(self.routes.get_cache_root(), snapshot)
        self.assertTrue((snapshot_dir / "frame_00000.png").is_file())

    async def test_reset_all_preserves_other_node_and_snapshot(self):
        for node_id in ("101", "202"):
            self.process(node_id)
            await self.save(node_id, 1)
        snapshot = await self.snapshot("101")
        await self.call("reset_all", node_id="101")
        self.assertEqual(self.process("101").sum().item(), 0)
        self.assertEqual(self.process("202")[0].sum().item(), 16)
        self.assertTrue(Path(self.routes.get_cache_root(), snapshot).is_dir())

    async def test_active_cache_delete_is_rejected_without_losing_edits(self):
        self.process("101")
        await self.save("101", 1)
        cache_dir = self.routes.sessions["101"]["disk_cache_dir"]
        result = await self.call(
            "delete_mask_cache", expected_status=409,
            cache_id=os.path.basename(cache_dir),
        )
        self.assertFalse(result["success"])
        self.assertTrue(Path(cache_dir, "frame_00000.png").is_file())
        self.assertEqual(self.routes.sessions["101"]["masks"][0].sum().item(), 16)
        await self.save("101", 1, frame_idx=1)

    async def test_inactive_cache_can_be_deleted(self):
        self.process("101")
        await self.save("101", 1)
        cache_dir = self.routes.sessions["101"]["disk_cache_dir"]
        self.routes.sessions.clear()
        result = await self.call("delete_mask_cache", cache_id=os.path.basename(cache_dir))
        self.assertTrue(result["success"])
        self.assertFalse(Path(cache_dir).exists())

    async def test_snapshot_is_independent_and_deletable(self):
        self.process("101")
        await self.save("101", 1)
        snapshot = await self.snapshot("101")
        await self.save("101", 0)
        await self.call("apply_mask_cache", node_id="101", cache_id=snapshot)
        self.assertEqual(self.routes.sessions["101"]["masks"][0].sum().item(), 16)
        await self.call("delete_mask_cache", cache_id=snapshot)
        self.assertTrue(Path(self.routes.sessions["101"]["disk_cache_dir"]).is_dir())
        self.assertFalse(Path(self.routes.get_cache_root(), snapshot).exists())

    async def test_batch_write_recovers_missing_cache_directory(self):
        self.process("101")
        cache_dir = self.routes.sessions["101"]["disk_cache_dir"]
        shutil.rmtree(cache_dir)
        buffer = io.BytesIO()
        Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(buffer, format="PNG")
        await self.call(
            "apply_batch_operation", node_id="101", frame_indices=[0, 1],
            operation="add", operation_data=base64.b64encode(buffer.getvalue()).decode("ascii"),
        )
        self.assertEqual(self.routes.sessions["101"]["masks"].sum().item(), 32)
        self.assertTrue(Path(cache_dir, "metadata.json").is_file())
        self.assertEqual(self.routes.cache_frame_indices(cache_dir), [0, 1])
        await self.call("undo_batch_operation", node_id="101")
        self.assertEqual(self.routes.sessions["101"]["masks"].sum().item(), 0)

    async def test_apply_snapshot_recovers_missing_directory_and_resizes(self):
        self.process("101")
        await self.save("101", 1)
        snapshot = await self.snapshot("101")
        self.process("202", images=torch.zeros((1, 8, 8, 3)))
        cache_dir = self.routes.sessions["202"]["disk_cache_dir"]
        shutil.rmtree(cache_dir)
        result = await self.call("apply_mask_cache", node_id="202", cache_id=snapshot)
        self.assertEqual(result["applied_frames"], [0])
        self.assertEqual(self.routes.sessions["202"]["masks"].sum().item(), 64)
        self.assertTrue(Path(cache_dir, "frame_00000.png").is_file())
        self.assertTrue(Path(cache_dir, "metadata.json").is_file())

    async def test_nested_node_ids_stay_inside_cache_root(self):
        cache_dir = self.routes.get_disk_cache_dir("../nested/node", "video_hash")
        self.assertEqual(Path(cache_dir).parent, Path(self.routes.get_cache_root()))
        self.assertEqual(cache_dir, self.routes.get_disk_cache_dir("../nested/node", "video_hash"))

    async def test_ratio_configuration_keeps_separate_persistent_edits(self):
        def generate(alignment):
            return self.nodes.VideoMaskEditor().process(
                self.images, mode="Use Edited or Passthrough", unique_id="101",
                mask_source="Aspect ratio padding", content_alignment=alignment,
            )[0]

        generate("Top")
        top_dir = self.routes.sessions["101"]["disk_cache_dir"]
        await self.save("101", 1)
        bottom_masks = generate("Bottom")
        self.assertNotEqual(top_dir, self.routes.sessions["101"]["disk_cache_dir"])
        self.assertEqual(bottom_masks[0, -1].sum().item(), 0)
        self.assertEqual(generate("Top")[0].sum().item(), 16)


if __name__ == "__main__":
    unittest.main()
