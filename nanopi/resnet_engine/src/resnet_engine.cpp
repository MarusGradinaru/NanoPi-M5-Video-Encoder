#include <resnet_engine.h>

#include <rknn_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "rk_type.h"
#include "rk_mpi.h"
#include "rk_mpi_cmd.h"
#include "mpp_buffer.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "mpp_meta.h"

#include <im2d.h>
#include <im2d_buffer.h>

namespace fs = std::filesystem;

namespace resnet_npu {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct PreprocessTiming {
    double jpeg_preprocess_ms = 0.0;
    double jpeg_file_read_ms = 0.0;
    double jpeg_dma_copy_ms = 0.0;
    double mpp_decode_ms = 0.0;
    double rga_convert_ms = 0.0;
    double fp16_preprocess_ms = 0.0;
};

constexpr uint32_t kRk3576MaximumNpuCores = 2;
constexpr size_t kInitialJpegBufferBytes = 64U * 1024U;
constexpr size_t kMaximumJpegBytes = 128U * 1024U * 1024U;

class ImageProcessingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check_rknn(int result, const char* operation) {
    if (result != RKNN_SUCC) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with RKNN error " + std::to_string(result) + ".");
    }
}

uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
        return static_cast<uint16_t>(sign | 0x7e00u);
    }

    int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder_mask = (1u << shift) - 1u;
        const uint32_t remainder = mantissa & remainder_mask;
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) {
            ++half_mantissa;
        }
        return static_cast<uint16_t>(sign | half_mantissa);
    }

    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<uint16_t>(sign |
                                 (static_cast<uint32_t>(half_exponent) << 10) |
                                 half_mantissa);
}

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<uint32_t>(unbiased_exponent + 127) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent += 127u - 15u;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<uint8_t> load_model_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open RKNN model: " + path);
    }
    const std::streamoff length = file.tellg();
    if (length <= 0 || static_cast<uint64_t>(length) > UINT32_MAX) {
        throw std::runtime_error("Invalid RKNN model size: " + path);
    }
    std::vector<uint8_t> model(static_cast<size_t>(length));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(model.data()), length);
    if (!file || file.gcount() != length) {
        throw std::runtime_error("Could not read complete RKNN model: " + path);
    }
    return model;
}

std::vector<uint8_t> read_jpeg_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw ImageProcessingError("Could not open JPEG file.");
    }
    const std::streamoff length = file.tellg();
    if (length <= 0) {
        throw ImageProcessingError("JPEG file is empty.");
    }
    if (static_cast<uint64_t>(length) > kMaximumJpegBytes) {
        throw ImageProcessingError("JPEG exceeds the 128 MiB safety limit.");
    }

    std::vector<uint8_t> data(static_cast<size_t>(length));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), length);
    if (!file || file.gcount() != length) {
        throw ImageProcessingError("Could not read the complete JPEG file.");
    }
    return data;
}

uint16_t read_jpeg_be16(const std::vector<uint8_t>& data, size_t offset) {
    if (offset + 1 >= data.size()) {
        throw ImageProcessingError("Malformed JPEG: truncated 16-bit field.");
    }
    return static_cast<uint16_t>((static_cast<uint16_t>(data[offset]) << 8) |
                                 static_cast<uint16_t>(data[offset + 1]));
}

bool is_jpeg_sof_marker(uint8_t marker) {
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

std::pair<uint32_t, uint32_t> jpeg_dimensions(const std::vector<uint8_t>& data) {
    if (data.size() < 4 || data[0] != 0xff || data[1] != 0xd8) {
        throw ImageProcessingError("Not a JPEG file (missing SOI marker).");
    }

    size_t position = 2;
    while (position + 3 < data.size()) {
        if (data[position] != 0xff) {
            ++position;
            continue;
        }
        while (position < data.size() && data[position] == 0xff) {
            ++position;
        }
        if (position >= data.size()) {
            break;
        }

        const uint8_t marker = data[position++];
        if (marker == 0xd9 || marker == 0xda) {
            break;
        }
        if (marker == 0xd8 || marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
            continue;
        }
        if (position + 1 >= data.size()) {
            break;
        }

        const uint16_t segment_length = read_jpeg_be16(data, position);
        if (segment_length < 2 || position + segment_length > data.size()) {
            throw ImageProcessingError("Malformed JPEG: invalid segment length.");
        }
        if (is_jpeg_sof_marker(marker)) {
            if (segment_length < 8) {
                throw ImageProcessingError("Malformed JPEG: SOF segment is too short.");
            }
            const uint32_t height = read_jpeg_be16(data, position + 3);
            const uint32_t width = read_jpeg_be16(data, position + 5);
            if (width == 0 || height == 0) {
                throw ImageProcessingError("JPEG contains invalid dimensions.");
            }
            return {width, height};
        }
        position += segment_length;
    }
    throw ImageProcessingError("Could not find a supported JPEG SOF marker.");
}

uint32_t align_up_16(uint32_t value) {
    return (value + 15U) & ~15U;
}

std::string mpp_error_text(const char* operation, MPP_RET code) {
    return std::string(operation) + " failed with MPP error " + std::to_string(code) + ".";
}

struct MppPacketGuard {
    MppPacket packet = nullptr;
    ~MppPacketGuard() {
        if (packet != nullptr) {
            mpp_packet_deinit(&packet);
        }
    }
};

struct MppFrameGuard {
    MppFrame frame = nullptr;
    MppFrame reusable_frame = nullptr;
    ~MppFrameGuard() {
        if (frame != nullptr && frame != reusable_frame) {
            mpp_frame_deinit(&frame);
        }
    }
};

class MppRgaPreprocessor {
public:
    MppRgaPreprocessor(uint32_t width, uint32_t height)
        : width_(width), height_(height), rgb_(static_cast<size_t>(width) * height * 3U) {
        if (width_ == 0 || height_ == 0 || (width_ & 1U) != 0U || (height_ & 1U) != 0U) {
            throw std::runtime_error("MPP/RGA requires non-zero even model input dimensions.");
        }
        for (size_t value = 0; value < normalization_lut_.size(); ++value) {
            const float normalized = static_cast<float>(value) * (1.0F / 127.5F) - 1.0F;
            normalization_lut_[value] = float_to_half(normalized);
        }
        try {
            initialize();
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~MppRgaPreprocessor() {
        cleanup();
    }

    MppRgaPreprocessor(const MppRgaPreprocessor&) = delete;
    MppRgaPreprocessor& operator=(const MppRgaPreprocessor&) = delete;

    void process_to(const fs::path& path,
                    uint16_t* destination,
                    PreprocessTiming* timing = nullptr) {
        if (destination == nullptr) {
            throw std::runtime_error("Null destination passed to MPP/RGA preprocessing.");
        }

        const auto preprocess_start = Clock::now();
        auto stage_start = preprocess_start;
        const std::vector<uint8_t> jpeg = read_jpeg_file(path);
        auto stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->jpeg_file_read_ms += elapsed_ms(stage_start, stage_stop);
        }

        const auto dimensions = jpeg_dimensions(jpeg);
        if (dimensions.first != width_ || dimensions.second != height_) {
            throw ImageProcessingError(
                "Expected " + std::to_string(width_) + "x" + std::to_string(height_) +
                " JPEG, got " + std::to_string(dimensions.first) + "x" +
                std::to_string(dimensions.second) + ".");
        }

        stage_start = Clock::now();
        ensure_input_buffer_capacity(jpeg.size());
        void* input_ptr = mpp_buffer_get_ptr(input_buffer_);
        if (input_ptr == nullptr) {
            throw std::runtime_error("MPP input DMA-BUF has no CPU mapping.");
        }
        std::memcpy(input_ptr, jpeg.data(), jpeg.size());
        stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->jpeg_dma_copy_ms += elapsed_ms(stage_start, stage_stop);
        }

        MppPacketGuard packet;
        MPP_RET ret = mpp_packet_init_with_buffer(&packet.packet, input_buffer_);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_packet_init_with_buffer", ret));
        }
        mpp_packet_set_pos(packet.packet, input_ptr);
        mpp_packet_set_length(packet.packet, jpeg.size());
        // Do not signal EOS here: the decoder is a persistent engine and must
        // remain usable across chunks and multiple process_folder/process_files calls.

        MppMeta packet_meta = mpp_packet_get_meta(packet.packet);
        if (packet_meta == nullptr) {
            throw std::runtime_error("mpp_packet_get_meta returned null.");
        }
        ret = mpp_meta_set_frame(packet_meta, KEY_OUTPUT_FRAME, output_frame_);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_meta_set_frame(KEY_OUTPUT_FRAME)", ret));
        }

        stage_start = Clock::now();
        ret = mpi_->decode_put_packet(context_, packet.packet);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("decode_put_packet", ret));
        }

        MppFrameGuard decoded;
        decoded.reusable_frame = output_frame_;
        ret = mpi_->decode_get_frame(context_, &decoded.frame);
        stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->mpp_decode_ms += elapsed_ms(stage_start, stage_stop);
        }
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("decode_get_frame", ret));
        }
        if (decoded.frame == nullptr) {
            throw ImageProcessingError("MPP returned no decoded JPEG frame.");
        }

        const RK_U32 errinfo = mpp_frame_get_errinfo(decoded.frame);
        const RK_U32 discard = mpp_frame_get_discard(decoded.frame);
        if (mpp_frame_get_info_change(decoded.frame) != 0U) {
            throw ImageProcessingError("MPP returned an info-change event instead of a JPEG frame.");
        }
        if (errinfo != 0U || discard != 0U) {
            std::ostringstream message;
            message << "MPP marked the JPEG frame as erroneous or discarded (errinfo=0x"
                    << std::hex << errinfo << ", discard=0x" << discard << ").";
            throw ImageProcessingError(message.str());
        }

        const uint32_t decoded_width = mpp_frame_get_width(decoded.frame);
        const uint32_t decoded_height = mpp_frame_get_height(decoded.frame);
        const MppFrameFormat decoded_format = mpp_frame_get_fmt(decoded.frame);
        const uint32_t horizontal_stride = mpp_frame_get_hor_stride(decoded.frame);
        const uint32_t vertical_stride = mpp_frame_get_ver_stride(decoded.frame);
        MppBuffer decoded_buffer = mpp_frame_get_buffer(decoded.frame);
        if (decoded_width != width_ || decoded_height != height_ ||
            decoded_format != MPP_FMT_YUV420SP ||
            horizontal_stride != horizontal_stride_ || vertical_stride != vertical_stride_ ||
            decoded_buffer == nullptr ||
            mpp_buffer_get_fd(decoded_buffer) != mpp_buffer_get_fd(output_buffer_)) {
            throw std::runtime_error(
                "MPP returned an unexpected frame format, dimensions, stride or DMA-BUF.");
        }

        stage_start = Clock::now();
        const IM_STATUS rga_status = imcvtcolor(
            source_image_, destination_image_,
            RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888,
            IM_YUV_TO_RGB_BT601_FULL, 1);
        stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->rga_convert_ms += elapsed_ms(stage_start, stage_stop);
        }
        if (rga_status != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA NV12 -> RGB888 failed with status " +
                                     std::to_string(static_cast<int>(rga_status)) + ".");
        }

        stage_start = Clock::now();
        const size_t pixel_values = rgb_.size();
        for (size_t index = 0; index < pixel_values; ++index) {
            destination[index] = normalization_lut_[rgb_[index]];
        }
        stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->fp16_preprocess_ms += elapsed_ms(stage_start, stage_stop);
            timing->jpeg_preprocess_ms += elapsed_ms(preprocess_start, stage_stop);
        }
    }

private:
    void ensure_input_buffer_capacity(size_t required_bytes) {
        if (required_bytes == 0 || required_bytes > kMaximumJpegBytes) {
            throw ImageProcessingError("Invalid JPEG size for MPP input buffer.");
        }
        if (input_buffer_ != nullptr && required_bytes <= input_buffer_capacity_) {
            return;
        }

        const size_t new_capacity = (required_bytes + 15U) & ~static_cast<size_t>(15U);
        MppBuffer replacement = nullptr;
        const MPP_RET ret = mpp_buffer_get(input_group_, &replacement, new_capacity);
        if (ret != MPP_OK || replacement == nullptr) {
            if (replacement != nullptr) {
                mpp_buffer_put(replacement);
            }
            throw std::runtime_error(mpp_error_text("mpp_buffer_get(input resize)", ret));
        }
        if (mpp_buffer_get_ptr(replacement) == nullptr || mpp_buffer_get_fd(replacement) < 0) {
            mpp_buffer_put(replacement);
            throw std::runtime_error("Resized MPP input buffer has no CPU mapping or DMA-BUF.");
        }

        MppBuffer previous = input_buffer_;
        input_buffer_ = replacement;
        input_buffer_capacity_ = new_capacity;
        if (previous != nullptr) {
            mpp_buffer_put(previous);
        }
    }

    void initialize() {
        MPP_RET ret = mpp_create(&context_, &mpi_);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_create", ret));
        }
        ret = mpp_init(context_, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_init(MJPEG)", ret));
        }

        ret = mpp_dec_cfg_init(&decoder_config_);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_dec_cfg_init", ret));
        }
        ret = mpi_->control(context_, MPP_DEC_GET_CFG, decoder_config_);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("MPP_DEC_GET_CFG", ret));
        }
        ret = mpp_dec_cfg_set_u32(decoder_config_, "base:split_parse", 1U);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_dec_cfg_set_u32(base:split_parse)", ret));
        }
        ret = mpi_->control(context_, MPP_DEC_SET_CFG, decoder_config_);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("MPP_DEC_SET_CFG", ret));
        }

        MppFrameFormat output_format = MPP_FMT_YUV420SP;
        ret = mpi_->control(context_, MPP_DEC_SET_OUTPUT_FORMAT, &output_format);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("MPP_DEC_SET_OUTPUT_FORMAT", ret));
        }
        RK_S64 output_timeout_ms = -1;
        ret = mpi_->control(context_, MPP_SET_OUTPUT_TIMEOUT, &output_timeout_ms);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("MPP_SET_OUTPUT_TIMEOUT", ret));
        }

        horizontal_stride_ = align_up_16(width_);
        vertical_stride_ = align_up_16(height_);
        const size_t output_buffer_size = static_cast<size_t>(horizontal_stride_) *
                                          vertical_stride_ * 4U;
        ret = mpp_buffer_group_get(&output_group_, MPP_BUFFER_TYPE_DRM,
                                   MPP_BUFFER_INTERNAL, "resnet_npu", "mpp_output");
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_buffer_group_get(output)", ret));
        }
        ret = mpp_buffer_get(output_group_, &output_buffer_, output_buffer_size);
        if (ret != MPP_OK || output_buffer_ == nullptr) {
            throw std::runtime_error(mpp_error_text("mpp_buffer_get(output)", ret));
        }
        ret = mpp_frame_init(&output_frame_);
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_frame_init(output)", ret));
        }
        mpp_frame_set_buffer(output_frame_, output_buffer_);

        ret = mpp_buffer_group_get(&input_group_, MPP_BUFFER_TYPE_DRM,
                                   MPP_BUFFER_INTERNAL, "resnet_npu", "mpp_input");
        if (ret != MPP_OK) {
            throw std::runtime_error(mpp_error_text("mpp_buffer_group_get(input)", ret));
        }
        ensure_input_buffer_capacity(kInitialJpegBufferBytes);

        const int output_fd = mpp_buffer_get_fd(output_buffer_);
        if (output_fd < 0) {
            throw std::runtime_error("MPP output buffer has no DMA-BUF descriptor for RGA.");
        }
        rga_.source = importbuffer_fd(output_fd, static_cast<int>(horizontal_stride_),
                                      static_cast<int>(vertical_stride_), RK_FORMAT_YCbCr_420_SP);
        if (rga_.source <= 0) {
            throw std::runtime_error("RGA importbuffer_fd failed for the MPP output buffer.");
        }
        rga_.destination = importbuffer_virtualaddr(
            rgb_.data(), static_cast<int>(width_), static_cast<int>(height_), RK_FORMAT_RGB_888);
        if (rga_.destination <= 0) {
            throw std::runtime_error("RGA importbuffer_virtualaddr failed for the RGB buffer.");
        }
        source_image_ = wrapbuffer_handle(
            rga_.source, static_cast<int>(width_), static_cast<int>(height_),
            RK_FORMAT_YCbCr_420_SP, static_cast<int>(horizontal_stride_),
            static_cast<int>(vertical_stride_));
        destination_image_ = wrapbuffer_handle(
            rga_.destination, static_cast<int>(width_), static_cast<int>(height_),
            RK_FORMAT_RGB_888, static_cast<int>(width_), static_cast<int>(height_));
    }

    void cleanup() noexcept {
        if (rga_.destination > 0) {
            releasebuffer_handle(rga_.destination);
            rga_.destination = 0;
        }
        if (rga_.source > 0) {
            releasebuffer_handle(rga_.source);
            rga_.source = 0;
        }
        if (output_frame_ != nullptr) {
            mpp_frame_deinit(&output_frame_);
        }
        if (context_ != nullptr) {
            mpp_destroy(context_);
            context_ = nullptr;
            mpi_ = nullptr;
        }
        if (decoder_config_ != nullptr) {
            mpp_dec_cfg_deinit(decoder_config_);
            decoder_config_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            mpp_buffer_put(input_buffer_);
            input_buffer_ = nullptr;
        }
        if (input_group_ != nullptr) {
            mpp_buffer_group_put(input_group_);
            input_group_ = nullptr;
        }
        if (output_buffer_ != nullptr) {
            mpp_buffer_put(output_buffer_);
            output_buffer_ = nullptr;
        }
        if (output_group_ != nullptr) {
            mpp_buffer_group_put(output_group_);
            output_group_ = nullptr;
        }
    }

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t horizontal_stride_ = 0;
    uint32_t vertical_stride_ = 0;
    size_t input_buffer_capacity_ = 0;
    std::vector<uint8_t> rgb_;
    std::array<uint16_t, 256> normalization_lut_{};

    MppCtx context_ = nullptr;
    MppApi* mpi_ = nullptr;
    MppDecCfg decoder_config_ = nullptr;
    MppBufferGroup output_group_ = nullptr;
    MppBuffer output_buffer_ = nullptr;
    MppFrame output_frame_ = nullptr;
    MppBufferGroup input_group_ = nullptr;
    MppBuffer input_buffer_ = nullptr;
    struct RgaCleanup {
        rga_buffer_handle_t source = 0;
        rga_buffer_handle_t destination = 0;
    } rga_;
    rga_buffer_t source_image_{};
    rga_buffer_t destination_image_{};
};

std::vector<fs::path> discover_images(const fs::path& root, bool recursive) {
    std::error_code error;
    if (!fs::exists(root, error) || error || !fs::is_directory(root, error) || error) {
        throw std::runtime_error("Image folder does not exist or is not a directory: " + root.string());
    }

    std::vector<fs::path> files;
    const auto add_if_jpeg = [&files](const fs::directory_entry& entry) {
        std::error_code entry_error;
        if (!entry.is_regular_file(entry_error) || entry_error) {
            return;
        }
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (extension == ".jpg" || extension == ".jpeg") {
            files.push_back(entry.path());
        }
    };

    if (recursive) {
        fs::recursive_directory_iterator iterator(
            root, fs::directory_options::skip_permission_denied, error);
        const fs::recursive_directory_iterator end;
        if (error) {
            throw std::runtime_error("Could not scan image folder: " + error.message());
        }
        while (iterator != end) {
            add_if_jpeg(*iterator);
            iterator.increment(error);
            if (error) {
                error.clear();
            }
        }
    } else {
        fs::directory_iterator iterator(
            root, fs::directory_options::skip_permission_denied, error);
        const fs::directory_iterator end;
        if (error) {
            throw std::runtime_error("Could not scan image folder: " + error.message());
        }
        while (iterator != end) {
            add_if_jpeg(*iterator);
            iterator.increment(error);
            if (error) {
                error.clear();
            }
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path& left, const fs::path& right) {
        return left.generic_string() < right.generic_string();
    });
    return files;
}

}  // namespace

struct ResNetEngine::Impl {
    explicit Impl(const std::string& path, uint32_t requested_batch, uint32_t requested_cores)
        : model_path(path), batch_size(requested_batch), npu_core_count(requested_cores),
          model_data(load_model_file(path)) {
        if (batch_size == 0 || (batch_size != 1 && (batch_size % 2U) != 0U)) {
            throw std::invalid_argument("batch_size must be 1 or a positive even integer.");
        }
        if (npu_core_count == 0 || npu_core_count > kRk3576MaximumNpuCores) {
            throw std::invalid_argument("npu_core_count must be between 1 and 2 on RK3576.");
        }
        try {
            initialize_rknn();
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() {
        cleanup();
    }

    void initialize_rknn() {
        check_rknn(rknn_init(&context, model_data.data(),
                             static_cast<uint32_t>(model_data.size()), 0, nullptr),
                   "rknn_init");
        context_initialized = true;

        rknn_sdk_version runtime_version{};
        check_rknn(rknn_query(context, RKNN_QUERY_SDK_VERSION,
                              &runtime_version, sizeof(runtime_version)),
                   "RKNN_QUERY_SDK_VERSION");
        runtime_api_version = runtime_version.api_version;
        driver_version = runtime_version.drv_version;

        if (batch_size > 1) {
            check_rknn(rknn_set_batch_core_num(context, npu_core_count),
                       "rknn_set_batch_core_num");
        }

        rknn_input_output_num io_num{};
        check_rknn(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
                   "RKNN_QUERY_IN_OUT_NUM");
        if (io_num.n_input != 1 || io_num.n_output != 1) {
            throw std::runtime_error("ResNet feature extractor must have exactly one input and one output.");
        }

        input_attr = {};
        input_attr.index = 0;
        check_rknn(rknn_query(context, RKNN_QUERY_NATIVE_INPUT_ATTR,
                              &input_attr, sizeof(input_attr)),
                   "RKNN_QUERY_NATIVE_INPUT_ATTR");
        output_attr = {};
        output_attr.index = 0;
        check_rknn(rknn_query(context, RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                              &output_attr, sizeof(output_attr)),
                   "RKNN_QUERY_NATIVE_OUTPUT_ATTR");

        if (input_attr.n_dims != 4 || input_attr.dims[0] != batch_size ||
            input_attr.fmt != RKNN_TENSOR_NHWC || input_attr.type != RKNN_TENSOR_FLOAT16 ||
            input_attr.dims[3] != 3) {
            throw std::runtime_error(
                "Model native input must be FLOAT16 NHWC [batch,height,width,3] and match batch_size.");
        }
        input_height = input_attr.dims[1];
        input_width = input_attr.dims[2];
        input_channels = input_attr.dims[3];
        if (input_width == 0 || input_height == 0 || (input_width & 1U) != 0U ||
            (input_height & 1U) != 0U) {
            throw std::runtime_error("Model input dimensions must be non-zero and even.");
        }

        if (output_attr.n_dims != 2 || output_attr.dims[0] != batch_size ||
            output_attr.dims[1] == 0 || output_attr.type != RKNN_TENSOR_FLOAT16) {
            throw std::runtime_error(
                "Model native output must be FLOAT16 [batch,feature_count] and match batch_size.");
        }
        feature_count = output_attr.dims[1];

        const uint64_t expected_input_bytes = static_cast<uint64_t>(batch_size) * input_height *
            input_width * input_channels * sizeof(uint16_t);
        const uint64_t expected_output_bytes = static_cast<uint64_t>(batch_size) *
            feature_count * sizeof(uint16_t);
        if (input_attr.size_with_stride != expected_input_bytes ||
            output_attr.size_with_stride != expected_output_bytes) {
            throw std::runtime_error("Model native input/output tensors contain unsupported padding or stride.");
        }

        input_attr.pass_through = 1;
        input_mem = rknn_create_mem(context, input_attr.size_with_stride);
        if (input_mem == nullptr || input_mem->virt_addr == nullptr) {
            throw std::runtime_error("Could not allocate RKNN Zero-Copy input memory.");
        }
        output_mem = rknn_create_mem(context, output_attr.size_with_stride);
        if (output_mem == nullptr || output_mem->virt_addr == nullptr) {
            throw std::runtime_error("Could not allocate RKNN Zero-Copy output memory.");
        }
        check_rknn(rknn_set_io_mem(context, input_mem, &input_attr), "rknn_set_io_mem(input)");
        check_rknn(rknn_set_io_mem(context, output_mem, &output_attr), "rknn_set_io_mem(output)");

        sample_elements = static_cast<size_t>(input_height) * input_width * input_channels;
        input_sample_bytes = sample_elements * sizeof(uint16_t);
        if (input_sample_bytes * batch_size != input_attr.size_with_stride ||
            static_cast<size_t>(batch_size) * feature_count * sizeof(uint16_t) != output_attr.size_with_stride) {
            throw std::runtime_error("RKNN tensor allocation is not contiguous as expected.");
        }
    }

    void ensure_preprocessor() {
        if (!preprocessor) {
            preprocessor = std::make_unique<MppRgaPreprocessor>(input_width, input_height);
        }
    }

    std::vector<float> run_batch(size_t valid_samples, BatchTiming* timing = nullptr) {
        if (valid_samples == 0 || valid_samples > batch_size) {
            throw std::logic_error("Invalid number of valid samples passed to run_batch.");
        }
        auto stage_start = Clock::now();
        check_rknn(rknn_mem_sync(context, input_mem, RKNN_MEMORY_SYNC_TO_DEVICE),
                   "rknn_mem_sync(input, TO_DEVICE)");
        auto stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->input_cache_sync_ms += elapsed_ms(stage_start, stage_stop);
        }

        stage_start = Clock::now();
        check_rknn(rknn_run(context, nullptr), "rknn_run");
        stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->inference_ms += elapsed_ms(stage_start, stage_stop);
        }

        stage_start = Clock::now();
        check_rknn(rknn_mem_sync(context, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE),
                   "rknn_mem_sync(output, FROM_DEVICE)");
        stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->output_cache_sync_ms += elapsed_ms(stage_start, stage_stop);
        }

        stage_start = Clock::now();
        const auto* native_output = static_cast<const uint16_t*>(output_mem->virt_addr);
        std::vector<float> result(valid_samples * feature_count);
        for (size_t sample = 0; sample < valid_samples; ++sample) {
            for (size_t feature = 0; feature < feature_count; ++feature) {
                const float value = half_to_float(native_output[sample * feature_count + feature]);
                if (!std::isfinite(value)) {
                    throw std::runtime_error("RKNN output contains NaN or infinity.");
                }
                result[sample * feature_count + feature] = value;
            }
        }
        stage_stop = Clock::now();
        if (timing != nullptr) {
            timing->output_read_ms += elapsed_ms(stage_start, stage_stop);
        }
        return result;
    }

    void cleanup() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed) {
            return;
        }
        preprocessor.reset();
        if (context != 0) {
            if (input_mem != nullptr) {
                // Destructors must not throw; cleanup is best effort.
                (void)rknn_destroy_mem(context, input_mem);
                input_mem = nullptr;
            }
            if (output_mem != nullptr) {
                (void)rknn_destroy_mem(context, output_mem);
                output_mem = nullptr;
            }
            if (context_initialized) {
                rknn_destroy(context);
                context_initialized = false;
            }
            context = 0;
        }
        std::vector<uint8_t>().swap(model_data);
        closed = true;
    }

    std::string model_path;
    uint32_t batch_size = 0;
    uint32_t npu_core_count = 0;
    std::vector<uint8_t> model_data;

    mutable std::mutex mutex;
    bool closed = false;
    bool context_initialized = false;
    rknn_context context = 0;
    rknn_tensor_attr input_attr{};
    rknn_tensor_attr output_attr{};
    rknn_tensor_mem* input_mem = nullptr;
    rknn_tensor_mem* output_mem = nullptr;
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t input_channels = 0;
    uint32_t feature_count = 0;
    std::string runtime_api_version;
    std::string driver_version;
    size_t sample_elements = 0;
    size_t input_sample_bytes = 0;
    std::unique_ptr<MppRgaPreprocessor> preprocessor;
};

ResNetEngine::ResNetEngine(const std::string& model_path,
                           uint32_t batch_size,
                           uint32_t npu_core_count)
    : impl_(std::make_unique<Impl>(model_path, batch_size, npu_core_count)) {}

ResNetEngine::~ResNetEngine() = default;

EngineInfo ResNetEngine::info() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    EngineInfo result;
    result.batch_size = impl_->batch_size;
    result.input_width = impl_->input_width;
    result.input_height = impl_->input_height;
    result.input_channels = impl_->input_channels;
    result.feature_count = impl_->feature_count;
    result.npu_core_count = impl_->batch_size > 1 ? impl_->npu_core_count : 1;
    result.runtime_api_version = impl_->runtime_api_version;
    result.driver_version = impl_->driver_version;
    return result;
}

std::string ResNetEngine::model_path() const {
    return impl_->model_path;
}

bool ResNetEngine::is_closed() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->closed;
}

void ResNetEngine::close() {
    impl_->cleanup();
}

std::shared_ptr<FeatureStream> ResNetEngine::process_folder(
    const std::string& folder_path,
    size_t chunk_size,
    bool recursive) {
    if (chunk_size == 0) {
        throw std::invalid_argument("chunk_size must be greater than zero.");
    }
    std::vector<fs::path> paths = discover_images(fs::path(folder_path), recursive);
    return std::make_shared<FeatureStream>(shared_from_this(), std::move(paths), chunk_size);
}

std::shared_ptr<FeatureStream> ResNetEngine::process_files(
    const std::vector<std::string>& image_paths,
    size_t chunk_size) {
    if (chunk_size == 0) {
        throw std::invalid_argument("chunk_size must be greater than zero.");
    }
    std::vector<fs::path> paths;
    paths.reserve(image_paths.size());
    for (const std::string& path : image_paths) {
        if (path.empty()) {
            throw std::invalid_argument("process_files received an empty path.");
        }
        paths.emplace_back(path);
    }
    return std::make_shared<FeatureStream>(shared_from_this(), std::move(paths), chunk_size);
}

std::vector<float> ResNetEngine::infer_preprocessed(
    const float* tensor,
    size_t sample_count,
    TensorLayout layout) {
    if (sample_count == 0) {
        return {};
    }
    if (tensor == nullptr) {
        throw std::invalid_argument("infer_preprocessed received a null tensor.");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) {
        throw std::runtime_error("ResNetEngine is closed.");
    }

    const size_t input_samples_count = sample_count;
    const size_t input_sample_elements = impl_->sample_elements;
    const size_t input_image_plane = static_cast<size_t>(impl_->input_width) * impl_->input_height;
    auto* native_input = static_cast<uint16_t*>(impl_->input_mem->virt_addr);
    std::vector<float> result;
    result.reserve(sample_count * impl_->feature_count);

    size_t current = 0;
    while (current < input_samples_count) {
        const size_t valid = std::min<size_t>(impl_->batch_size, input_samples_count - current);
        for (size_t slot = 0; slot < valid; ++slot) {
            const size_t source_sample = current + slot;
            uint16_t* destination = native_input + slot * input_sample_elements;
            for (uint32_t y = 0; y < impl_->input_height; ++y) {
                for (uint32_t x = 0; x < impl_->input_width; ++x) {
                    for (uint32_t channel = 0; channel < impl_->input_channels; ++channel) {
                        size_t source_index = 0;
                        if (layout == TensorLayout::NCHW) {
                            source_index = source_sample * input_sample_elements +
                                static_cast<size_t>(channel) * input_image_plane +
                                static_cast<size_t>(y) * impl_->input_width + x;
                        } else {
                            source_index = source_sample * input_sample_elements +
                                (static_cast<size_t>(y) * impl_->input_width + x) *
                                impl_->input_channels + channel;
                        }
                        const float value = tensor[source_index];
                        if (!std::isfinite(value)) {
                            throw std::invalid_argument("Preprocessed tensor contains NaN or infinity.");
                        }
                        destination[(static_cast<size_t>(y) * impl_->input_width + x) *
                                   impl_->input_channels + channel] = float_to_half(value);
                    }
                }
            }
        }

        if (valid < impl_->batch_size) {
            const uint16_t* last_sample = native_input + (valid - 1U) * input_sample_elements;
            for (size_t slot = valid; slot < impl_->batch_size; ++slot) {
                std::memcpy(native_input + slot * input_sample_elements,
                            last_sample, impl_->input_sample_bytes);
            }
        }

        std::vector<float> batch_result = impl_->run_batch(valid);
        result.insert(result.end(), batch_result.begin(), batch_result.end());
        current += valid;
    }
    return result;
}

FeatureChunk ResNetEngine::process_stream_chunk(
    const std::vector<fs::path>& paths,
    size_t& next_index,
    size_t chunk_size,
    StreamStats& stats) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->closed) {
        throw std::runtime_error("ResNetEngine is closed.");
    }

    FeatureChunk chunk;
    chunk.image_paths.reserve(chunk_size + impl_->batch_size - 1U);
    chunk.features.reserve((chunk_size + impl_->batch_size - 1U) * impl_->feature_count);

    const size_t sample_elements = impl_->sample_elements;
    auto* native_input = static_cast<uint16_t*>(impl_->input_mem->virt_addr);

    while (chunk.image_paths.size() < chunk_size && next_index < paths.size()) {
        const auto batch_start = Clock::now();
        size_t valid = 0;
        std::vector<std::string> batch_paths;
        batch_paths.reserve(impl_->batch_size);

        while (valid < impl_->batch_size && next_index < paths.size()) {
            const fs::path& path = paths[next_index++];
            PreprocessTiming preprocess_timing{};
            try {
                impl_->ensure_preprocessor();
                uint16_t* destination = native_input + valid * sample_elements;
                impl_->preprocessor->process_to(path, destination, &preprocess_timing);
                batch_paths.push_back(path.string());
                ++valid;
            } catch (const ImageProcessingError& error) {
                ++stats.images_failed;
                stats.failures.push_back({path.string(), error.what()});
            }
            stats.timings.jpeg_preprocess_ms += preprocess_timing.jpeg_preprocess_ms;
            stats.timings.jpeg_file_read_ms += preprocess_timing.jpeg_file_read_ms;
            stats.timings.jpeg_dma_copy_ms += preprocess_timing.jpeg_dma_copy_ms;
            stats.timings.mpp_decode_ms += preprocess_timing.mpp_decode_ms;
            stats.timings.rga_convert_ms += preprocess_timing.rga_convert_ms;
            stats.timings.fp16_preprocess_ms += preprocess_timing.fp16_preprocess_ms;
        }

        if (valid == 0) {
            continue;
        }

        if (valid < impl_->batch_size) {
            const uint16_t* last_sample = native_input + (valid - 1U) * sample_elements;
            for (size_t slot = valid; slot < impl_->batch_size; ++slot) {
                std::memcpy(native_input + slot * sample_elements,
                            last_sample, impl_->input_sample_bytes);
            }
        }

        BatchTiming batch_timing{};
        std::vector<float> batch_features = impl_->run_batch(valid, &batch_timing);
        batch_timing.wall_ms = elapsed_ms(batch_start, Clock::now());
        stats.timings.input_cache_sync_ms += batch_timing.input_cache_sync_ms;
        stats.timings.inference_ms += batch_timing.inference_ms;
        stats.timings.output_cache_sync_ms += batch_timing.output_cache_sync_ms;
        stats.timings.output_read_ms += batch_timing.output_read_ms;
        stats.batch_timings.push_back(batch_timing);
        chunk.image_paths.insert(chunk.image_paths.end(), batch_paths.begin(), batch_paths.end());
        chunk.features.insert(chunk.features.end(), batch_features.begin(), batch_features.end());
        ++stats.batches_executed;
        stats.images_processed += valid;
        stats.features_returned += valid;
    }

    if (next_index >= paths.size()) {
        stats.completed = true;
    }
    return chunk;
}

FeatureStream::FeatureStream(std::shared_ptr<ResNetEngine> engine,
                             std::vector<fs::path> paths,
                             size_t chunk_size)
    : engine_(std::move(engine)), paths_(std::move(paths)), chunk_size_(chunk_size) {
    stats_.files_found = paths_.size();
}

bool FeatureStream::next(FeatureChunk& chunk) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    if (closed_) {
        return false;
    }
    try {
        chunk = engine_->process_stream_chunk(paths_, next_index_, chunk_size_, stats_);
    } catch (...) {
        closed_ = true;
        throw;
    }
    if (chunk.image_paths.empty()) {
        closed_ = true;
        stats_.completed = (next_index_ >= paths_.size());
        return false;
    }
    return true;
}

void FeatureStream::close() {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    if (!closed_) {
        stats_.closed_early = next_index_ < paths_.size();
        closed_ = true;
    }
}

StreamStats FeatureStream::stats() const {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    return stats_;
}

}  // namespace resnet_npu
