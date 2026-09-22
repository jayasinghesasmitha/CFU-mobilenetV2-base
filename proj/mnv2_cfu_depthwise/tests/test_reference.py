#!/usr/bin/env python3
import random
import unittest


def clamp(value):
    return max(-128, min(127, value))


def fused(patch, ew, eb, dw, db, pw, pb, shifts):
    es, ds, ps = shifts
    cin = len(patch[0])
    x = [[clamp((eb[e] + sum(patch[t][i] * ew[e][i]
                             for i in range(cin))) >> es)
          for e in range(len(eb))] for t in range(9)]
    d = [clamp((db[e] + sum(x[t][e] * dw[t][e]
                            for t in range(9))) >> ds)
         for e in range(len(eb))]
    return [clamp((pb[o] + sum(d[e] * pw[o][e]
                                for e in range(len(eb)))) >> ps)
            for o in range(len(pb))]


class ReferenceKernelTest(unittest.TestCase):
    def test_deterministic_and_random_shapes(self):
        rng = random.Random(7)
        for cin, cexp, cout in [(1, 1, 1), (2, 3, 2), (4, 2, 3)]:
            patch = [[rng.randrange(-8, 9) for _ in range(cin)] for _ in range(9)]
            ew = [[rng.randrange(-4, 5) for _ in range(cin)] for _ in range(cexp)]
            eb = [rng.randrange(-10, 11) for _ in range(cexp)]
            dw = [[rng.randrange(-4, 5) for _ in range(cexp)] for _ in range(9)]
            db = [rng.randrange(-10, 11) for _ in range(cexp)]
            pw = [[rng.randrange(-4, 5) for _ in range(cexp)] for _ in range(cout)]
            pb = [rng.randrange(-10, 11) for _ in range(cout)]
            result = fused(patch, ew, eb, dw, db, pw, pb, (1, 2, 1))
            self.assertEqual(len(result), cout)
            self.assertTrue(all(-128 <= value <= 127 for value in result))


if __name__ == "__main__":
    unittest.main()
