#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define RESNET_ENGINE_API __attribute__((visibility("default")))
#else
#define RESNET_ENGINE_API
#endif

namespace resnet_npu {

enum class TensorLayout {
    NCHW,
    NHWC,
};

struct RESNET_ENGINE_API EngineInfo {
    uint32_t batch_size = 0;
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t input_channels = 0;
    uint32_t feature_count = 0;
    uint32_t npu_core_count = 0;
    std::string runtime_api_version;
    std::string driver_version;
};

struct RESNET_ENGINE_API ImageFailure {
    std::string path;
    std::string reason;
};

// Profiling counters collected by the native JPEG-to-feature pipeline.
// Times are cumulative milliseconds across preprocessing attempts and executed batches.
struct RESNET_ENGINE_API StreamTimings {
    double jpeg_preprocess_ms = 0.0;
    double jpeg_file_read_ms = 0.0;
    double jpeg_dma_copy_ms = 0.0;
    double mpp_decode_ms = 0.0;
    double rga_convert_ms = 0.0;
    double fp16_preprocess_ms = 0.0;
    double input_cache_sync_ms = 0.0;
    double inference_ms = 0.0;
    double output_cache_sync_ms = 0.0;
    double output_read_ms = 0.0;
};

// Per-inference-call timings. One record is appended for every RKNN batch.
struct RESNET_ENGINE_API BatchTiming {
    double input_cache_sync_ms = 0.0;
    double inference_ms = 0.0;
    double output_cache_sync_ms = 0.0;
    double output_read_ms = 0.0;
    double wall_ms = 0.0;
};

struct RESNET_ENGINE_API StreamStats {
    size_t files_found = 0;
    size_t images_processed = 0;
    size_t images_failed = 0;
    size_t batches_executed = 0;
    size_t features_returned = 0;
    bool completed = false;
    bool closed_early = false;
    StreamTimings timings;
    std::vector<BatchTiming> batch_timings;
    std::vector<ImageFailure> failures;
};

struct RESNET_ENGINE_API FeatureChunk {
    std::vector<std::string> image_paths;
    std::vector<float> features;  // Row-major [image_paths.size(), feature_count].
};

class FeatureStream;

class RESNET_ENGINE_API ResNetEngine : public std::enable_shared_from_this<ResNetEngine> {
public:
    ResNetEngine(const std::string& model_path,
                 uint32_t batch_size = 4,
                 uint32_t npu_core_count = 2);
    ~ResNetEngine();

    ResNetEngine(const ResNetEngine&) = delete;
    ResNetEngine& operator=(const ResNetEngine&) = delete;
    ResNetEngine(ResNetEngine&&) = delete;
    ResNetEngine& operator=(ResNetEngine&&) = delete;

    EngineInfo info() const;
    std::string model_path() const;
    bool is_closed() const;
    void close();

    std::shared_ptr<FeatureStream> process_folder(
        const std::string& folder_path,
        size_t chunk_size = 64,
        bool recursive = false);

    std::shared_ptr<FeatureStream> process_files(
        const std::vector<std::string>& image_paths,
        size_t chunk_size = 64);

    // Accepts already-normalized float32 values in [-1, 1], in NCHW or NHWC
    // layout. The returned row-major vector contains float32 feature vectors.
    std::vector<float> infer_preprocessed(
        const float* tensor,
        size_t sample_count,
        TensorLayout layout);

private:
    friend class FeatureStream;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    FeatureChunk process_stream_chunk(
        const std::vector<std::filesystem::path>& paths,
        size_t& next_index,
        size_t chunk_size,
        StreamStats& stats);
};

class RESNET_ENGINE_API FeatureStream {
public:
    FeatureStream(std::shared_ptr<ResNetEngine> engine,
                  std::vector<std::filesystem::path> paths,
                  size_t chunk_size);
    ~FeatureStream() = default;

    FeatureStream(const FeatureStream&) = delete;
    FeatureStream& operator=(const FeatureStream&) = delete;

    // Returns false when the stream is exhausted or explicitly closed.
    bool next(FeatureChunk& chunk);
    void close();
    StreamStats stats() const;

private:
    std::shared_ptr<ResNetEngine> engine_;
    std::vector<std::filesystem::path> paths_;
    size_t chunk_size_ = 64;
    size_t next_index_ = 0;
    bool closed_ = false;
    mutable std::mutex stream_mutex_;
    StreamStats stats_;
};

}  // namespace resnet_npu
