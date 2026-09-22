#!/usr/bin/env python3
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_image_model import fused_image


class FusedImageModelTest(unittest.TestCase):
    def test_same_padding_uses_zero_point_on_the_fly(self):
        data = [1, 2, 3, 4]
        out, oh, ow = fused_image(
            data, 2, 2, 1,
            [[1]], [0],
            [[1] for _ in range(9)], [0],
            [[1]], [0],
            stride=1, padding=1, input_zero_point=0)
        self.assertEqual((oh, ow), (2, 2))
        self.assertEqual(out, [10, 10, 10, 10])

    def test_stride_two_and_non_aligned_dimensions(self):
        data = list(range(-7, 8))  # 3x5x1
        out, oh, ow = fused_image(
            data, 3, 5, 1,
            [[1]], [0],
            [[1] for _ in range(9)], [0],
            [[1]], [0],
            stride=2, padding=1, input_zero_point=0)
        self.assertEqual((oh, ow), (2, 3))
        self.assertEqual(len(out), 6)

    def test_signed_multichannel_and_saturation(self):
        rng = random.Random(19)
        height, width, cin, cexp, cout = 4, 3, 3, 5, 4
        data = [rng.randrange(-128, 128) for _ in range(height * width * cin)]
        ew = [[rng.randrange(-8, 9) for _ in range(cin)] for _ in range(cexp)]
        eb = [rng.randrange(-1000, 1001) for _ in range(cexp)]
        dw = [[rng.randrange(-8, 9) for _ in range(cexp)] for _ in range(9)]
        db = [rng.randrange(-1000, 1001) for _ in range(cexp)]
        pw = [[rng.randrange(-8, 9) for _ in range(cexp)] for _ in range(cout)]
        pb = [rng.randrange(-1000, 1001) for _ in range(cout)]
        out, oh, ow = fused_image(data, height, width, cin, ew, eb, dw, db,
                                  pw, pb, shifts=(3, 4, 3), stride=1,
                                  padding=1, input_zero_point=-3)
        self.assertEqual(len(out), oh * ow * cout)
        self.assertTrue(all(-128 <= value <= 127 for value in out))

    def test_valid_padding(self):
        data = [1] * (4 * 4)
        out, oh, ow = fused_image(data, 4, 4, 1, [[1]], [0],
                                  [[1] for _ in range(9)], [0], [[1]], [0],
                                  stride=1, padding=0)
        self.assertEqual((oh, ow), (2, 2))
        self.assertEqual(out, [9, 9, 9, 9])


if __name__ == "__main__":
    unittest.main()
