#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path
from PIL import Image


class ImageConversionTest(unittest.TestCase):
    def test_rgb_resize_and_generated_bytes(self):
        root = Path(__file__).resolve().parents[1]
        tool = root / "tools" / "image_to_header.py"
        spec = importlib.util.spec_from_file_location("image_to_header", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.png"
            header = directory / "input_image.h"
            # Known RGB corners; a non-square source also verifies that the
            # repository policy is resize/stretch rather than center crop.
            image = Image.new("RGB", (2, 1))
            image.putdata([(255, 0, 0), (0, 0, 255)])
            image.save(source)
            module.convert(source, header)

            raw = header.with_suffix(".dat").read_bytes()
            expected = Image.open(source).convert("RGB").resize(
                (224, 224), Image.Resampling.BILINEAR).tobytes()
            self.assertEqual(raw, expected)
            self.assertEqual(len(raw), 224 * 224 * 3)
            text = header.read_text()
            self.assertIn("input_image_len = 150528", text)
            self.assertIn("interleaved RGB", text)


if __name__ == "__main__":
    unittest.main()
