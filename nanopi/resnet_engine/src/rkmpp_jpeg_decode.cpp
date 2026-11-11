// Decode one JPEG with Rockchip MPP, convert NV12 -> RGB with RGA,
// apply the ResNet50V2 FP32 normalization and FP16 rounding, then save PNG.

#define MODULE_TAG "rkmpp_jpeg_decode"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include "rk_type.h"
#include "rk_mpi.h"
#include "rk_mpi_cmd.h"
#include "mpp_buffer.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "mpp_meta.h"

#include <im2d.h>
#include <im2d_buffer.h>

namespace {

constexpr int kTargetWidth = 320;
constexpr int kTargetHeight = 180;
constexpr size_t kMaxJpegBytes = 128U * 1024U * 1024U;

struct MppResources {
    MppCtx context = nullptr;
    MppApi* mpi = nullptr;
    MppBufferGroup buffer_group = nullptr;
    MppBuffer output_buffer = nullptr;
    MppBufferGroup input_buffer_group = nullptr;
    MppBuffer input_buffer = nullptr;
    MppFrame output_frame = nullptr;
    MppPacket packet = nullptr;
    MppFrame decoded_frame = nullptr;
    MppDecCfg decoder_config = nullptr;

    ~MppResources() {
        if (packet) {
            mpp_packet_deinit(&packet);
        }
        if (input_buffer) {
            mpp_buffer_put(input_buffer);
        }
        if (input_buffer_group) {
            mpp_buffer_group_put(input_buffer_group);
        }

        // The normal MJPEG path returns the caller-provided output_frame.
        // If MPP ever returns a different frame object, release both objects.
        if (decoded_frame && decoded_frame != output_frame) {
            mpp_frame_deinit(&decoded_frame);
        }
        if (output_frame) {
            mpp_frame_deinit(&output_frame);
        }
        if (context) {
            mpp_destroy(context);
        }
        if (decoder_config) {
            mpp_dec_cfg_deinit(decoder_config);
        }
        if (output_buffer) {
            mpp_buffer_put(output_buffer);
        }
        if (buffer_group) {
            mpp_buffer_group_put(buffer_group);
        }
    }

    MppResources() = default;
    MppResources(const MppResources&) = delete;
    MppResources& operator=(const MppResources&) = delete;
};

struct RgaResources {
    rga_buffer_handle_t source_handle = 0;
    rga_buffer_handle_t destination_handle = 0;

    ~RgaResources() {
        if (destination_handle > 0) {
            releasebuffer_handle(destination_handle);
        }
        if (source_handle > 0) {
            releasebuffer_handle(source_handle);
        }
    }

    RgaResources() = default;
    RgaResources(const RgaResources&) = delete;
    RgaResources& operator=(const RgaResources&) = delete;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail("Could not open input JPEG: " + path.string());
    }

    const std::streamoff length = file.tellg();
    if (length <= 0) {
        fail("Input JPEG is empty: " + path.string());
    }
    if (static_cast<uint64_t>(length) > kMaxJpegBytes) {
        fail("Input JPEG exceeds the 128 MiB safety limit.");
    }

    std::vector<uint8_t> data(static_cast<size_t>(length));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), length);
    if (!file || file.gcount() != length) {
        fail("Could not read the complete input JPEG: " + path.string());
    }
    return data;
}

uint16_t read_be16(const std::vector<uint8_t>& data, size_t offset) {
    if (offset + 1 >= data.size()) {
        fail("Malformed JPEG: truncated 16-bit field.");
    }
    return static_cast<uint16_t>((static_cast<uint16_t>(data[offset]) << 8) |
                                 static_cast<uint16_t>(data[offset + 1]));
}

bool is_sof_marker(uint8_t marker) {
    switch (marker) {
        case 0xC0: case 0xC1: case 0xC2: case 0xC3:
        case 0xC5: case 0xC6: case 0xC7:
        case 0xC9: case 0xCA: case 0xCB:
        case 0xCD: case 0xCE: case 0xCF:
            return true;
        default:
            return false;
    }
}

std::pair<int, int> jpeg_dimensions(const std::vector<uint8_t>& data) {
    if (data.size() < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        fail("Input file is not a valid JPEG (missing SOI marker).");
    }

    size_t position = 2;
    while (position + 3 < data.size()) {
        if (data[position] != 0xFF) {
            ++position;
            continue;
        }
        while (position < data.size() && data[position] == 0xFF) {
            ++position;
        }
        if (position >= data.size()) {
            break;
        }

        const uint8_t marker = data[position++];
        if (marker == 0xD9 || marker == 0xDA) {
            break; // EOI or start of scan without a preceding supported SOF.
        }
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            continue; // Standalone markers have no length field.
        }
        if (position + 1 >= data.size()) {
            break;
        }

        const uint16_t segment_length = read_be16(data, position);
        if (segment_length < 2 || position + segment_length > data.size()) {
            fail("Malformed JPEG: invalid segment length.");
        }

        if (is_sof_marker(marker)) {
            if (segment_length < 8) {
                fail("Malformed JPEG: SOF segment is too short.");
            }
            const int height = read_be16(data, position + 3);
            const int width = read_be16(data, position + 5);
            if (width <= 0 || height <= 0) {
                fail("JPEG contains invalid image dimensions.");
            }
            return {width, height};
        }
        position += segment_length;
    }

    fail("Could not find a supported JPEG SOF marker.");
}

uint32_t align_up_16(uint32_t value) {
    return (value + 15U) & ~15U;
}

std::string mpp_error(const char* operation, MPP_RET code) {
    return std::string(operation) + " failed with MPP error " + std::to_string(code) + ".";
}

// IEEE-754 binary32 -> binary16, round-to-nearest-even. The normalization range
// [-1, 1] used here does not contain NaN/Inf, but the helper handles them too.
uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16U) & 0x8000U;
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa == 0U) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        uint32_t payload = mantissa >> 13U;
        if (payload == 0U) payload = 1U;
        return static_cast<uint16_t>(sign | 0x7C00U | payload);
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000U;
        const unsigned shift = static_cast<unsigned>(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder_mask = (1U << shift) - 1U;
        const uint32_t remainder = mantissa & remainder_mask;
        const uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t rounded_mantissa = mantissa >> 13U;
    const uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U ||
        (remainder == 0x1000U && (rounded_mantissa & 1U))) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400U) {
            rounded_mantissa = 0U;
            ++half_exponent;
            if (half_exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<uint16_t>(sign |
                                 (static_cast<uint32_t>(half_exponent) << 10U) |
                                 rounded_mantissa);
}

float half_to_float(uint16_t half) {
    const uint32_t sign = (static_cast<uint32_t>(half & 0x8000U)) << 16U;
    uint32_t exponent = (half >> 10U) & 0x1FU;
    uint32_t mantissa = half & 0x03FFU;
    uint32_t bits = 0;

    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x03FFU;
            const uint32_t float_exponent = static_cast<uint32_t>(127 - 15 - shift + 1);
            bits = sign | (float_exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        exponent = exponent + (127U - 15U);
        bits = sign | (exponent << 23U) | (mantissa << 13U);
    }

    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void write_be32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>((value >> 24U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 16U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
}

void append_png_chunk(std::vector<uint8_t>& png, const char type[4],
                      const uint8_t* payload, size_t payload_size) {
    if (payload_size > std::numeric_limits<uint32_t>::max()) {
        fail("PNG chunk exceeds the maximum supported size.");
    }

    write_be32(png, static_cast<uint32_t>(payload_size));
    const size_t crc_start = png.size();
    png.insert(png.end(), type, type + 4);
    if (payload_size > 0) {
        png.insert(png.end(), payload, payload + payload_size);
    }
    const uLong crc = crc32(0L, png.data() + crc_start,
                            static_cast<uInt>(4U + payload_size));
    write_be32(png, static_cast<uint32_t>(crc));
}

void write_png_rgb(const std::filesystem::path& output_path,
                   const std::vector<uint8_t>& rgb,
                   int width, int height) {
    const size_t row_bytes = static_cast<size_t>(width) * 3U;
    if (rgb.size() != row_bytes * static_cast<size_t>(height)) {
        fail("RGB buffer size does not match PNG dimensions.");
    }

    std::vector<uint8_t> filtered((row_bytes + 1U) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        const size_t destination_offset = static_cast<size_t>(y) * (row_bytes + 1U);
        const size_t source_offset = static_cast<size_t>(y) * row_bytes;
        filtered[destination_offset] = 0; // PNG filter type 0: None.
        std::memcpy(filtered.data() + destination_offset + 1U,
                    rgb.data() + source_offset, row_bytes);
    }

    uLongf compressed_size = compressBound(static_cast<uLong>(filtered.size()));
    std::vector<uint8_t> compressed(static_cast<size_t>(compressed_size));
    const int zret = compress2(compressed.data(), &compressed_size,
                               filtered.data(), static_cast<uLong>(filtered.size()),
                               Z_BEST_SPEED);
    if (zret != Z_OK) {
        fail("zlib compress2 failed with code " + std::to_string(zret) + ".");
    }
    compressed.resize(static_cast<size_t>(compressed_size));

    std::vector<uint8_t> png;
    png.reserve(compressed.size() + 128U);
    const std::array<uint8_t, 8> signature =
        {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    png.insert(png.end(), signature.begin(), signature.end());

    std::array<uint8_t, 13> ihdr{};
    ihdr[0] = static_cast<uint8_t>((static_cast<uint32_t>(width) >> 24U) & 0xFFU);
    ihdr[1] = static_cast<uint8_t>((static_cast<uint32_t>(width) >> 16U) & 0xFFU);
    ihdr[2] = static_cast<uint8_t>((static_cast<uint32_t>(width) >> 8U) & 0xFFU);
    ihdr[3] = static_cast<uint8_t>(static_cast<uint32_t>(width) & 0xFFU);
    ihdr[4] = static_cast<uint8_t>((static_cast<uint32_t>(height) >> 24U) & 0xFFU);
    ihdr[5] = static_cast<uint8_t>((static_cast<uint32_t>(height) >> 16U) & 0xFFU);
    ihdr[6] = static_cast<uint8_t>((static_cast<uint32_t>(height) >> 8U) & 0xFFU);
    ihdr[7] = static_cast<uint8_t>(static_cast<uint32_t>(height) & 0xFFU);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type: truecolor RGB
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;

    append_png_chunk(png, "IHDR", ihdr.data(), ihdr.size());
    append_png_chunk(png, "IDAT", compressed.data(), compressed.size());
    append_png_chunk(png, "IEND", nullptr, 0);

    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file) {
        fail("Could not create output PNG: " + output_path.string());
    }
    file.write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));
    if (!file) {
        fail("Failed while writing output PNG: " + output_path.string());
    }
}

std::vector<uint8_t> apply_resnet_preprocess_fp16_and_restore_png(
    const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> restored(rgb.size());

    for (size_t index = 0; index < rgb.size(); ++index) {
        const float pixel = static_cast<float>(rgb[index]);
        const float normalized = pixel / 127.5F - 1.0F;
        const uint16_t normalized_fp16 = float_to_half(normalized);
        const float normalized_after_fp16 = half_to_float(normalized_fp16);
        const float reconstructed = (normalized_after_fp16 + 1.0F) * 127.5F;
        const long rounded = std::lround(reconstructed);
        restored[index] = static_cast<uint8_t>(std::clamp(rounded, 0L, 255L));
    }

    return restored;
}

void convert_nv12_to_rgb_with_rga(MppFrame frame,
                                  std::vector<uint8_t>& rgb,
                                  int width, int height) {
    MppBuffer mpp_buffer = mpp_frame_get_buffer(frame);
    if (!mpp_buffer) {
        fail("MPP frame has no buffer for RGA conversion.");
    }

    const int source_fd = mpp_buffer_get_fd(mpp_buffer);
    const size_t source_size = mpp_buffer_get_size(mpp_buffer);
    const int horizontal_stride = static_cast<int>(mpp_frame_get_hor_stride(frame));
    const int vertical_stride = static_cast<int>(mpp_frame_get_ver_stride(frame));
    if (source_fd < 0 || horizontal_stride < width || vertical_stride < height ||
        (horizontal_stride & 1) != 0 || (vertical_stride & 1) != 0) {
        fail("MPP returned an invalid DMA-BUF descriptor or NV12 stride.");
    }
    const size_t minimum_nv12_size = static_cast<size_t>(horizontal_stride) *
                                     static_cast<size_t>(vertical_stride) * 3U / 2U;
    if (source_size < minimum_nv12_size) {
        fail("MPP DMA-BUF is smaller than the required linear NV12 image size.");
    }

    // Use the four-argument import overload. Its width/height parameters are
    // the pixel strides of the allocated buffer, avoiding version-specific
    // im_handle_param_t member names across librga releases.
    RgaResources rga;
    rga.source_handle = importbuffer_fd(
        source_fd, horizontal_stride, vertical_stride, RK_FORMAT_YCbCr_420_SP);
    if (rga.source_handle <= 0) {
        fail("RGA importbuffer_fd failed for the MPP NV12 DMA-BUF.");
    }

    rga.destination_handle = importbuffer_virtualaddr(
        rgb.data(), width, height, RK_FORMAT_RGB_888);
    if (rga.destination_handle <= 0) {
        fail("RGA importbuffer_virtualaddr failed for the RGB output buffer.");
    }

    const rga_buffer_t source_image = wrapbuffer_handle(
        rga.source_handle, width, height, RK_FORMAT_YCbCr_420_SP,
        horizontal_stride, vertical_stride);
    const rga_buffer_t destination_image = wrapbuffer_handle(
        rga.destination_handle, width, height, RK_FORMAT_RGB_888,
        width, height);

    // The reference JPEG was reported by FFmpeg as yuvj420p(pc, bt470bg):
    // full-range BT.601/BT.470BG. Use the corresponding explicit RGA mode.
    const IM_STATUS status = imcvtcolor(
        source_image, destination_image,
        RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888,
        IM_YUV_TO_RGB_BT601_FULL, 1);

    if (status != IM_STATUS_SUCCESS) {
        fail("RGA imcvtcolor(NV12 -> RGB888, BT.601 full-range) failed with status " +
             std::to_string(static_cast<int>(status)) + ".");
    }

    std::cout << "[RGA] NV12 -> RGB888, BT.601 full-range, sync operation completed."
              << " Source fd=" << source_fd
              << ", buffer=" << source_size << " bytes.\n";
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <input.jpg>\n"
              << "\n"
              << "Decodes a 320x180 JPEG with MPP, converts NV12 to RGB888 with RGA,\n"
              << "applies ResNet50V2 normalization and FP16 rounding, then saves PNG\n"
              << "in the current directory as <input name without extension>.png.\n"
              << "Color conversion is fixed to BT.601 full range to match the tested\n"
              << "yuvj420p(pc, bt470bg) input.\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc != 2) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const std::filesystem::path input_path(argv[1]);

        // Output goes in the current working directory: input file name with
        // its extension replaced by .png (any directory part of the input is dropped).
        const std::string output_stem = input_path.stem().string();
        if (output_stem.empty()) {
            fail("Could not derive an output file name from input path: " + input_path.string());
        }
        const std::filesystem::path output_path = std::filesystem::path(output_stem + ".png");

        std::cout << "====================================================================\n"
                  << "MPP + RGA JPEG PREPROCESSING TEST (C++)\n"
                  << "====================================================================\n"
                  << "[Input]  " << input_path << '\n'
                  << "[Output] " << output_path << '\n'
                  << "[Target] " << kTargetWidth << 'x' << kTargetHeight
                  << " RGB888 -> ResNet50V2 normalized FP16 -> PNG\n"
                  << "[Color]  BT.601 full range (JPEG / PC range)\n";

        const std::vector<uint8_t> jpeg = read_file(input_path);
        const auto [jpeg_width, jpeg_height] = jpeg_dimensions(jpeg);
        std::cout << "[JPEG]   " << jpeg_width << 'x' << jpeg_height
                  << ", " << jpeg.size() << " bytes\n";

        if (jpeg_width != kTargetWidth || jpeg_height != kTargetHeight) {
            fail("This first validation utility intentionally accepts only 320x180 JPEGs, "
                 "matching the current ResNet feature-extractor input. Received " +
                 std::to_string(jpeg_width) + "x" + std::to_string(jpeg_height) + ".");
        }

        MppResources mpp;
        MPP_RET ret = mpp_create(&mpp.context, &mpp.mpi);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_create", ret));
        }
        ret = mpp_init(mpp.context, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_init(MJPEG decoder)", ret));
        }

        // Match Rockchip's own advanced MJPEG decode sample: enable the
        // internal parser for a complete JPEG packet before decoding.
        ret = mpp_dec_cfg_init(&mpp.decoder_config);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_dec_cfg_init", ret));
        }
        ret = mpp.mpi->control(mpp.context, MPP_DEC_GET_CFG, mpp.decoder_config);
        if (ret != MPP_OK) {
            fail(mpp_error("MPP_DEC_GET_CFG", ret));
        }
        ret = mpp_dec_cfg_set_u32(mpp.decoder_config, "base:split_parse", 1U);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_dec_cfg_set_u32(base:split_parse)", ret));
        }
        ret = mpp.mpi->control(mpp.context, MPP_DEC_SET_CFG, mpp.decoder_config);
        if (ret != MPP_OK) {
            fail(mpp_error("MPP_DEC_SET_CFG", ret));
        }

        MppFrameFormat output_format = MPP_FMT_YUV420SP;
        ret = mpp.mpi->control(mpp.context, MPP_DEC_SET_OUTPUT_FORMAT, &output_format);
        if (ret != MPP_OK) {
            fail(mpp_error("MPP_DEC_SET_OUTPUT_FORMAT(MPP_FMT_YUV420SP)", ret));
        }
        RK_S64 output_timeout_ms = -1;
        ret = mpp.mpi->control(mpp.context, MPP_SET_OUTPUT_TIMEOUT, &output_timeout_ms);
        if (ret != MPP_OK) {
            fail(mpp_error("MPP_SET_OUTPUT_TIMEOUT", ret));
        }

        const uint32_t horizontal_stride = align_up_16(static_cast<uint32_t>(jpeg_width));
        const uint32_t vertical_stride = align_up_16(static_cast<uint32_t>(jpeg_height));
        const size_t output_buffer_size = static_cast<size_t>(horizontal_stride) *
                                          static_cast<size_t>(vertical_stride) * 4U;

        ret = mpp_buffer_group_get(&mpp.buffer_group, MPP_BUFFER_TYPE_DRM,
                                   MPP_BUFFER_INTERNAL, MODULE_TAG, __FUNCTION__);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_buffer_group_get_internal(DRM)", ret));
        }
        ret = mpp_buffer_get(mpp.buffer_group, &mpp.output_buffer, output_buffer_size);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_buffer_get(MJPEG output)", ret));
        }
        ret = mpp_frame_init(&mpp.output_frame);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_frame_init", ret));
        }
        mpp_frame_set_buffer(mpp.output_frame, mpp.output_buffer);

        // Rockchip's JPEG decode test path uses an MppBuffer-backed input
        // packet. Do the same here instead of wrapping the std::vector memory
        // directly: the JPEG decoder can then access the compressed bitstream
        // through the DMA-BUF associated with the packet.
        ret = mpp_buffer_group_get(&mpp.input_buffer_group, MPP_BUFFER_TYPE_DRM,
                                   MPP_BUFFER_INTERNAL, MODULE_TAG, "jpeg_input");
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_buffer_group_get_internal(DRM, JPEG input)", ret));
        }

        const size_t input_buffer_size = (jpeg.size() + 15U) & ~static_cast<size_t>(15U);
        ret = mpp_buffer_get(mpp.input_buffer_group, &mpp.input_buffer, input_buffer_size);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_buffer_get(JPEG input)", ret));
        }

        void* input_ptr = mpp_buffer_get_ptr(mpp.input_buffer);
        if (!input_ptr) {
            fail("MPP returned a null CPU mapping for the JPEG input DMA-BUF.");
        }
        std::memcpy(input_ptr, jpeg.data(), jpeg.size());

        ret = mpp_packet_init_with_buffer(&mpp.packet, mpp.input_buffer);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_packet_init_with_buffer(JPEG input)", ret));
        }
        mpp_packet_set_pos(mpp.packet, input_ptr);
        mpp_packet_set_length(mpp.packet, jpeg.size());

        std::cout << "[MPP] Input DMA-BUF: fd=" << mpp_buffer_get_fd(mpp.input_buffer)
                  << ", allocated=" << mpp_buffer_get_size(mpp.input_buffer)
                  << " bytes, JPEG payload=" << jpeg.size() << " bytes.\n";

        // This packet contains the entire MJPEG stream (one JPEG image),
        // so mark it as the final packet before submitting it to the decoder.
        mpp_packet_set_eos(mpp.packet);
        std::cout << "[MPP] Input packet marked as EOS.\n";

        MppMeta packet_meta = mpp_packet_get_meta(mpp.packet);
        if (!packet_meta) {
            fail("mpp_packet_get_meta returned null.");
        }
        ret = mpp_meta_set_frame(packet_meta, KEY_OUTPUT_FRAME, mpp.output_frame);
        if (ret != MPP_OK) {
            fail(mpp_error("mpp_meta_set_frame(KEY_OUTPUT_FRAME)", ret));
        }

        std::cout << "[MPP] Submitting JPEG to the hardware decoder...\n";
        ret = mpp.mpi->decode_put_packet(mpp.context, mpp.packet);
        if (ret != MPP_OK) {
            fail(mpp_error("decode_put_packet", ret));
        }
        std::cout << "[MPP] Packet length after decode_put_packet: "
                  << mpp_packet_get_length(mpp.packet) << " bytes.\n";

        ret = mpp.mpi->decode_get_frame(mpp.context, &mpp.decoded_frame);
        if (ret != MPP_OK) {
            fail(mpp_error("decode_get_frame", ret));
        }
        if (!mpp.decoded_frame) {
            fail("MPP returned success but no decoded frame.");
        }
        const RK_U32 decoded_errinfo = mpp_frame_get_errinfo(mpp.decoded_frame);
        const RK_U32 decoded_discard = mpp_frame_get_discard(mpp.decoded_frame);
        const RK_U32 decoded_eos = mpp_frame_get_eos(mpp.decoded_frame);
        const RK_U32 decoded_info_change = mpp_frame_get_info_change(mpp.decoded_frame);
        std::cout << "[MPP] Frame status: errinfo=0x" << std::hex << decoded_errinfo
                  << ", discard=0x" << decoded_discard
                  << ", eos=" << std::dec << decoded_eos
                  << ", info_change=" << decoded_info_change << '\n';

        if (mpp_frame_get_info_change(mpp.decoded_frame)) {
            fail("MPP returned an info-change event instead of an image frame.");
        }
        if (mpp_frame_get_errinfo(mpp.decoded_frame) != 0U ||
            mpp_frame_get_discard(mpp.decoded_frame) != 0U) {
            fail("MPP marked the decoded frame as erroneous or discarded (errinfo=0x" +
                 [&]() { std::ostringstream out; out << std::hex << decoded_errinfo; return out.str(); }() +
                 ", discard=0x" +
                 [&]() { std::ostringstream out; out << std::hex << decoded_discard; return out.str(); }() +
                 ").");
        }

        const int decoded_width = static_cast<int>(mpp_frame_get_width(mpp.decoded_frame));
        const int decoded_height = static_cast<int>(mpp_frame_get_height(mpp.decoded_frame));
        const MppFrameFormat decoded_format = mpp_frame_get_fmt(mpp.decoded_frame);
        const uint32_t decoded_hor_stride = mpp_frame_get_hor_stride(mpp.decoded_frame);
        const uint32_t decoded_ver_stride = mpp_frame_get_ver_stride(mpp.decoded_frame);

        std::cout << "[MPP] Decoded frame: " << decoded_width << 'x' << decoded_height
                  << ", stride=" << decoded_hor_stride << 'x' << decoded_ver_stride
                  << ", format=" << static_cast<int>(decoded_format)
                  << ", color_range=" << static_cast<int>(mpp_frame_get_color_range(mpp.decoded_frame))
                  << ", colorspace=" << static_cast<int>(mpp_frame_get_colorspace(mpp.decoded_frame))
                  << ", chroma_location=" << static_cast<int>(mpp_frame_get_chroma_location(mpp.decoded_frame))
                  << '\n';

        if (mpp.decoded_frame != mpp.output_frame) {
            std::cerr << "[Warning] MPP returned a frame object different from the supplied output frame.\n";
        }
        if (decoded_width != kTargetWidth || decoded_height != kTargetHeight) {
            fail("MPP output dimensions are not 320x180.");
        }
        if (decoded_format != MPP_FMT_YUV420SP) {
            fail("MPP output is not NV12 (MPP_FMT_YUV420SP).");
        }

        std::vector<uint8_t> rgb(static_cast<size_t>(kTargetWidth) *
                                 static_cast<size_t>(kTargetHeight) * 3U);
        convert_nv12_to_rgb_with_rga(mpp.decoded_frame, rgb,
                                     kTargetWidth, kTargetHeight);

        std::cout << "[Preprocess] RGB888 -> [-1, 1] FP32 normalization -> FP16 rounding...\n";
        std::vector<uint8_t> final_rgb = apply_resnet_preprocess_fp16_and_restore_png(rgb);

        write_png_rgb(output_path, final_rgb, kTargetWidth, kTargetHeight);

        std::cout << "[PASS] C++ MPP + RGA preprocessing completed.\n"
                  << "[PNG]  " << output_path << " ("
                  << kTargetWidth << 'x' << kTargetHeight << ", RGB, lossless)\n"
                  << "[Info] PNG contains the image reconstructed after the exact\n"
                  << "       ResNet50V2 normalization and FP16 input rounding.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[Error] " << error.what() << '\n';
        return 1;
    }
}
