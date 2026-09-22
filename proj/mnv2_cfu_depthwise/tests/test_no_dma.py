#!/usr/bin/env python3
import re
import unittest
from pathlib import Path


class NoDmaInterfaceTest(unittest.TestCase):
    def test_gateware_has_only_cfu_ports_and_no_bus_master_vocabulary(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gateware" / "fused_depthwise.py").read_text()
        forbidden = [r"\bDMA\w*\s*\(", r"\baxi\b", r"\bwishbone\b",
                     r"bus_master", r"system_memory", r"memory_address"]
        for pattern in forbidden:
            self.assertIsNone(re.search(pattern, source, re.IGNORECASE), pattern)
        self.assertIn("simple_cfu", source)

    def test_software_api_streams_values_not_addresses(self):
        root = Path(__file__).resolve().parents[1]
        header = (root / "src" / "mnv2_cfu.h").read_text()
        self.assertIn("MNV2_FUSED_PUSH_INPUT", header)
        self.assertIn("MNV2_FUSED_PUSH_EXP_WEIGHT", header)
        self.assertIn("MNV2_FUSED_POP_OUTPUT", header)
        self.assertNotIn("DMA", header.upper())


if __name__ == "__main__":
    unittest.main()
