#ifndef MNV2_CFU_DEPTHWISE_INPUT_IMAGE_H_
#define MNV2_CFU_DEPTHWISE_INPUT_IMAGE_H_

// Placeholder input so the project builds before a custom image is supplied.
// Replace this file with:
//   scripts/pyrun proj/mnv2_cfu_depthwise/tools/image_to_header.py \
//     /path/to/image.jpg proj/mnv2_cfu_depthwise/src/input_image.h
// Unspecified elements are initialized to zero (black RGB image).
static const unsigned char input_image[224 * 224 * 3] = {0};
static const unsigned int input_image_len = 224 * 224 * 3;
static const unsigned int input_image_width = 224;
static const unsigned int input_image_height = 224;
static const unsigned int input_image_channels = 3;

#endif  // MNV2_CFU_DEPTHWISE_INPUT_IMAGE_H_
