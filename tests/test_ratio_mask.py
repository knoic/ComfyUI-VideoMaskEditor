"""Run: python -m unittest discover -s tests -v (ComfyUI not required)."""

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "vme_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.ratio_mask", ROOT / "ratio_mask.py")
ratio_mask = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ratio_mask
spec.loader.exec_module(ratio_mask)


class RatioMaskTests(unittest.TestCase):
    def test_4k_bottom_padding(self):
        masks = ratio_mask.create_ratio_padding_mask(1, 2560, 3840, 16, 9)
        self.assertEqual(tuple(masks.shape), (1, 2560, 3840))
        self.assertEqual(masks[:, :2160].count_nonzero().item(), 0)
        self.assertTrue(torch.all(masks[:, 2160:] == 1).item())

    def test_vertical_alignment(self):
        for alignment, expected_y in [("Top", 0), ("Center", 200), ("Bottom", 400)]:
            with self.subTest(alignment=alignment):
                self.assertEqual(ratio_mask.content_bounds(3840, 2560, 16, 9, alignment),
                                 (0, expected_y, 3840, 2160))

    def test_side_padding_and_corners(self):
        for alignment, expected_x in [("Left", 0), ("Center", 20), ("Right", 40),
                                      ("Top Left", 0), ("Bottom Right", 40)]:
            with self.subTest(alignment=alignment):
                self.assertEqual(ratio_mask.content_bounds(160, 90, 4, 3, alignment),
                                 (expected_x, 0, 120, 90))

    def test_same_ratio_is_empty(self):
        self.assertEqual(ratio_mask.create_ratio_padding_mask(2, 90, 160, 16, 9).sum().item(), 0)

    def test_rounding_and_odd_gap(self):
        self.assertEqual(ratio_mask.content_bounds(100, 80, 16, 9), (0, 0, 100, 56))
        self.assertEqual(ratio_mask.content_bounds(100, 80, 8, 5), (0, 0, 100, 63))
        self.assertEqual(ratio_mask.content_bounds(100, 80, 8, 5, "Center"), (0, 8, 100, 63))

    def test_batch_is_editable_independently(self):
        masks = ratio_mask.create_ratio_padding_mask(3, 64, 96, 16, 9)
        self.assertTrue(torch.equal(masks[0], masks[2]))
        masks[0, 0, 0] = 1
        self.assertEqual(masks[1, 0, 0].item(), 0)

    def test_invalid_parameters(self):
        for w, h in [(0, 9), (-1, 9), (16, float("nan")), (float("inf"), 9)]:
            with self.subTest(ratio=(w, h)), self.assertRaises(ValueError):
                ratio_mask.content_bounds(96, 64, w, h)
        with self.assertRaises(ValueError):
            ratio_mask.content_bounds(96, 64, 16, 9, "unknown")


class NodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        routes = types.ModuleType(f"{PACKAGE}.server_routes")
        routes.sessions = {}
        def cache_dir(node_id, video_hash):
            directory = pathlib.Path(self.temp.name) / node_id / video_hash
            directory.mkdir(parents=True, exist_ok=True)
            return str(directory)
        routes.get_disk_cache_dir = cache_dir
        routes.detect_flicker_frames = lambda masks: []
        routes.register_routes = lambda: None
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=None)
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")
        management.processing_interrupted = lambda: False
        self.modules = patch.dict(sys.modules, {
            routes.__name__: routes, "server": server,
            "comfy": comfy, "comfy.model_management": management,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(f"{PACKAGE}.nodes", ROOT / "nodes.py")
        self.nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.nodes)
        self.node = self.nodes.VideoMaskEditor()
        self.images = torch.rand(2, 64, 96, 3)

    def generate(self, **kwargs):
        return self.node.process(self.images, mode="Use Edited or Passthrough",
                                 mask_source="Aspect ratio padding", unique_id="1", **kwargs)

    def test_ratio_mode_without_masks_and_preserves_images(self):
        masks, images = self.generate()
        self.assertEqual(tuple(masks.shape), (2, 64, 96))
        self.assertEqual(masks[:, :54].count_nonzero().item(), 0)
        self.assertTrue(torch.all(masks[:, 54:] == 1).item())
        self.assertTrue(torch.equal(images, self.images))

    def test_ignores_connected_masks_in_ratio_mode(self):
        masks, _ = self.generate(masks=torch.ones(7, 3, 3))
        self.assertEqual(tuple(masks.shape), (2, 64, 96))
        self.assertEqual(masks[0, 0, 0].item(), 0)

    def test_custom_ratio(self):
        masks, _ = self.generate(original_ratio="Custom", ratio_width=3, ratio_height=2)
        self.assertEqual(masks.count_nonzero().item(), 0)

    def test_input_mask_passthrough_and_missing_error(self):
        original = torch.rand(1, 64, 96)
        masks, _ = self.node.process(self.images, original, "Use Edited or Passthrough")
        self.assertTrue(torch.equal(masks, original.repeat(2, 1, 1)))
        with self.assertRaisesRegex(ValueError, "Input masks"):
            self.node.process(self.images, mode="Use Edited or Passthrough")

    def test_cache_isolation_and_reset(self):
        self.generate()
        first = self.nodes.sessions["1"]
        Image.fromarray(np.full((64, 96), 255, dtype=np.uint8)).save(
            pathlib.Path(first["disk_cache_dir"]) / "frame_00000.png")
        masks, _ = self.generate()
        self.assertTrue(torch.all(masks[0] == 1).item())
        masks, _ = self.generate(content_alignment="Bottom")
        self.assertNotEqual(first["video_hash"], self.nodes.sessions["1"]["video_hash"])
        self.assertTrue(torch.all(masks[:, :10] == 1).item())
        self.assertEqual(masks[:, 10:].count_nonzero().item(), 0)
        masks, _ = self.generate(original_ratio="1:1")
        self.assertEqual(masks[0, 0, 48].item(), 0)
        masks, _ = self.generate(reset_cache=True)
        self.assertEqual(masks[0, 0, 0].item(), 0)
        self.node.process(self.images, torch.zeros(2, 64, 96), "Use Edited or Passthrough", unique_id="1")
        self.assertNotIn("_ratio_", self.nodes.sessions["1"]["video_hash"])

    def test_schema_optional_mask_and_feather(self):
        schema = self.node.INPUT_TYPES()
        self.assertIn("masks", schema["optional"])
        self.assertNotIn("masks", schema["required"])
        masks, _ = self.generate(feather_edges=2)
        self.assertTrue(((masks > 0) & (masks < 1)).any().item())
        self.assertTrue(self.node.IS_CHANGED(self.images, mode="Use Edited or Passthrough"))


if __name__ == "__main__":
    unittest.main()
