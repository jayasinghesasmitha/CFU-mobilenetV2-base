#!/usr/bin/env python3

import sys
import hashlib
from pathlib import Path
from PIL import Image

WIDTH = 224
HEIGHT = 224


def fnv1a(data):
    hash_value = 2166136261

    for value in data:
        hash_value ^= value
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF

    return hash_value


def image_to_header(input_file, output_file):

    print("========================================")
    print(" MobileNetV2 Image Preprocessing")
    print("========================================")

    print("Input image :", input_file)
    print("Output file :", output_file)
    print("Resize      :", f"{WIDTH}x{HEIGHT}")
    print("Format      : RGB")
    print("Interpolation: BILINEAR")

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    image = Image.open(input_file)

    print("Original size:", image.size)
    print("Original mode:", image.mode)

    # --------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------
    # Resize to MobileNetV2 input size
    # --------------------------------------------------

    image = image.resize(
        (WIDTH, HEIGHT),
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------
    # Extract RGB bytes
    # --------------------------------------------------

    pixels = list(image.getdata())

    rgb_data = []

    for pixel in pixels:

        r, g, b = pixel

        rgb_data.extend([
            r,
            g,
            b
        ])

    expected_size = WIDTH * HEIGHT * 3

    if len(rgb_data) != expected_size:

        raise RuntimeError(
            f"Wrong data size: {len(rgb_data)}, "
            f"expected {expected_size}"
        )

    # --------------------------------------------------
    # Convert to bytes
    # --------------------------------------------------

    rgb_bytes = bytes(rgb_data)

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    rgb_min = min(rgb_data)
    rgb_max = max(rgb_data)

    rgb_sum = sum(rgb_data)

    rgb_mean_scaled = (
        rgb_sum * 1000000
    ) // len(rgb_data)

    rgb_sha256 = hashlib.sha256(
        rgb_bytes
    ).hexdigest()

    rgb_fnv1a = fnv1a(rgb_data)

    # --------------------------------------------------
    # Print image information
    # --------------------------------------------------

    print()
    print("========== RGB IMAGE ==========")

    print(
        "Shape       :",
        f"{WIDTH} x {HEIGHT} x 3"
    )

    print(
        "Total bytes :",
        len(rgb_data)
    )

    print(
        "Min         :",
        rgb_min
    )

    print(
        "Max         :",
        rgb_max
    )

    print(
        "Mean        :",
        f"{rgb_mean_scaled // 1000000}."
        f"{rgb_mean_scaled % 1000000:06d}"
    )

    print(
        "SHA256      :",
        rgb_sha256
    )

    print(
        "FNV1a       :",
        f"0x{rgb_fnv1a:08x}"
    )

    # --------------------------------------------------
    # First bytes
    # --------------------------------------------------

    print()
    print("First 30 RGB bytes:")

    print(rgb_data[:30])

    # --------------------------------------------------
    # First pixels
    # --------------------------------------------------

    print()
    print("First 10 RGB pixels:")

    for i in range(10):

        p = i * 3

        print(
            f"Pixel {i:2d}: "
            f"R={rgb_data[p]} "
            f"G={rgb_data[p + 1]} "
            f"B={rgb_data[p + 2]}"
        )

    # --------------------------------------------------
    # Generate C header
    #
    # IMPORTANT:
    #
    # The model expects:
    #
    # UINT8 [0,255]
    #
    # Therefore we DO NOT modify the RGB values.
    # --------------------------------------------------

    output_path = Path(output_file)

    with open(output_path, "w") as f:

        f.write(
            "#ifndef CAT_IMAGE_H\n"
        )

        f.write(
            "#define CAT_IMAGE_H\n\n"
        )

        f.write(
            "#include <stdint.h>\n\n"
        )

        f.write(
            "/*\n"
            " * MobileNetV2 a0.35 INT8 input image.\n"
            " *\n"
            " * External model input:\n"
            " *   UINT8 [1,224,224,3]\n"
            " *\n"
            " * Raw RGB bytes are stored unchanged.\n"
            " * The TFLite model performs the internal\n"
            " * quantization itself.\n"
            " */\n"
        )

        f.write(
            "static const unsigned char cat_image[] = {\n"
        )

        for i in range(
            0,
            len(rgb_data),
            12
        ):

            chunk = rgb_data[
                i:i + 12
            ]

            f.write("    ")

            f.write(
                ", ".join(
                    f"0x{value:02x}"
                    for value in chunk
                )
            )

            f.write(",\n")

        f.write(
            "};\n\n"
        )

        f.write(
            "static const unsigned int "
            "cat_image_len = "
            f"{len(rgb_data)};\n\n"
        )

        f.write(
            "#endif\n"
        )

    # --------------------------------------------------
    # Generate DAT file
    #
    # This contains the EXACT SAME raw RGB bytes.
    # --------------------------------------------------

    dat_file = output_path.with_suffix(".dat")

    with open(
        dat_file,
        "wb"
    ) as f:

        f.write(rgb_bytes)

    # --------------------------------------------------
    # Final information
    # --------------------------------------------------

    print()
    print("========== OUTPUT FILES ==========")

    print(
        "Header:",
        output_file
    )

    print(
        "DAT   :",
        dat_file
    )

    print()
    print("Generated successfully.")

    print(
        "Bytes   :",
        len(rgb_data)
    )

    print(
        "Expected:",
        expected_size
    )

    print()
    print("Verification:")

    print(
        "  RGB values      : 0..255"
    )

    print(
        "  Resolution      : 224x224"
    )

    print(
        "  Channels        : RGB"
    )

    print(
        "  Total RGB bytes :",
        len(rgb_data)
    )

    print(
        "  SHA256          :",
        rgb_sha256
    )

    print(
        "  FNV1a            :",
        f"0x{rgb_fnv1a:08x}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "  python3 image_to_header.py "
            "<input.jpg> <output.h>"
        )

        sys.exit(1)

    image_to_header(
        sys.argv[1],
        sys.argv[2]
    )