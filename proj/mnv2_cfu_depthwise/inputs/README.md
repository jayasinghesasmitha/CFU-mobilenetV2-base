# Custom image location

Put your source JPEG or PNG in this directory, for example:

```text
proj/mnv2_cfu_depthwise/inputs/my_image.jpg
```

Then run from the repository root:

```bash
scripts/pyrun proj/mnv2_cfu_depthwise/tools/image_to_header.py \
  proj/mnv2_cfu_depthwise/inputs/my_image.jpg \
  proj/mnv2_cfu_depthwise/src/input_image.h
```

This embeds the image into firmware. The board/ Renode runtime does not read the
JPEG directly and there is no filesystem or DMA image loader.
