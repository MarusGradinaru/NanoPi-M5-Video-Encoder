#include "resnet_engine.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace resnet_npu;

namespace {

constexpr uint32_t kDefaultBatchSize = 4;
constexpr uint32_t kDefaultImageCount = 256;
constexpr uint32_t kDefaultWarmupTargetImages = 16;
constexpr uint32_t kRk3576NpuCoreCount = 2;
constexpr size_t kReservedWarmupImages = 2;
constexpr char kDefaultModelPath[] =
    "../../resnet/models/resnet50v2_feature_extractor_rk3576_fp16_320x180_batch4.rknn";
constexpr char kDefaultImagesDir[] = "../../resnet/dataset/andreea-esca";

// Directory that physically contains the running executable (Linux).
// /proc/self/exe is resolved by the kernel to the real file, so symlinks to the
// executable are followed. Falls back to the current directory if unavailable.
fs::path executable_dir() {
    std::error_code error;
    const fs::path exe = fs::read_symlink("/proc/self/exe", error);
    if (!error && exe.has_parent_path()) {
        return exe.parent_path();
    }
    error.clear();
    fs::path cwd = fs::current_path(error);
    return error ? fs::path(".") : cwd;
}

// Resolve a default path relative to the executable's directory, not the CWD.
std::string from_executable_dir(const char* relative_path) {
    return (executable_dir() / relative_path).lexically_normal().string();
}

struct Options {
    std::string model_path = from_executable_dir(kDefaultModelPath);
    std::string images_dir = from_executable_dir(kDefaultImagesDir);
    uint32_t batch_size = kDefaultBatchSize;
    uint32_t used_images = kDefaultImageCount;
    uint32_t warm_target = kDefaultWarmupTargetImages;
};

double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

uint32_t parse_positive_u32(const std::string& text, const char* option_name) {
    try {
        size_t consumed = 0;
        const unsigned long value = std::stoul(text, &consumed, 10);
        if (consumed != text.size() || value == 0 ||
            value > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("range");
        }
        return static_cast<uint32_t>(value);
    } catch (...) {
        throw std::runtime_error(std::string(option_name) +
                                 " must be a positive 32-bit integer.");
    }
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "Benchmark the shared ResNetEngine C++ library using real JPEG images.\n"
        << "Default paths are resolved relative to the directory of the executable itself,\n"
        << "so it can be started from any working directory. Paths passed on the command\n"
        << "line are interpreted relative to the current working directory.\n\n"
        << "Options:\n";

    const auto print_option = [](const char* syntax, const std::string& description) {
        std::cout << "  " << std::left << std::setw(24) << syntax
                  << description << '\n';
    };
    print_option("--model PATH", "RKNN model file (default: " +
                 from_executable_dir(kDefaultModelPath) + ")");
    print_option("--batch-size N", "Model batch size: 1 or a positive even number (default: " +
                 std::to_string(kDefaultBatchSize) + ")");
    print_option("--images-dir PATH", "Directory containing .jpg/.jpeg files (default: " +
                 from_executable_dir(kDefaultImagesDir) + ")");
    print_option("--used-images N", "Number of measured JPEG images to process (default: " +
                 std::to_string(kDefaultImageCount) + ")");
    print_option("--warm-target N", "Target warm-up image samples; rounded up to a full batch (default: " +
                 std::to_string(kDefaultWarmupTargetImages) + ")");
    print_option("-h, --help", "Show this help");

    std::cout
        << "\nMPP + RGA preprocessing currently accepts native 320x180 JPEG images.\n"
        << "\n"
        << "The first two sorted JPEG files are always reserved for warm-up and excluded\n"
        << "from measurement. Warm-up batches repeat these two images in A/B order; for\n"
        << "batch size 1, only image A is used.\n"
        << "\n"
        << "--warm-target is a target sample count. The number of complete inference calls\n"
        << "is rounded up, so actual warm-up samples may exceed the target.\n"
        << "\n"
        << "If --used-images is not divisible by the batch size, the incomplete tail batch\n"
        << "is skipped. Only complete batches are included in the benchmark.\n"
        << "\n"
        << "The benchmark calls libresnet_engine.so\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name + ".");
            }
            return argv[++i];
        };

        if (arg == "--model") {
            options.model_path = require_value("--model");
        } else if (arg == "--batch-size") {
            options.batch_size = parse_positive_u32(require_value("--batch-size"), "--batch-size");
        } else if (arg == "--images-dir") {
            options.images_dir = require_value("--images-dir");
        } else if (arg == "--used-images") {
            options.used_images = parse_positive_u32(require_value("--used-images"), "--used-images");
        } else if (arg == "--warm-target") {
            options.warm_target = parse_positive_u32(require_value("--warm-target"), "--warm-target");
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.batch_size != 1 && (options.batch_size % 2U) != 0U) {
        throw std::runtime_error("--batch-size must be 1 or an even positive integer.");
    }
    return options;
}

std::string lower_extension(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

std::vector<fs::path> discover_jpeg_images(const std::string& directory) {
    const fs::path root(directory);
    std::error_code error;
    if (!fs::exists(root, error) || error || !fs::is_directory(root, error) || error) {
        throw std::runtime_error("Image directory does not exist or is not a directory: " + directory);
    }

    std::vector<fs::path> files;
    fs::directory_iterator iterator(root, fs::directory_options::skip_permission_denied, error);
    if (error) {
        throw std::runtime_error("Could not list image directory: " + directory + ": " + error.message());
    }
    for (const fs::directory_entry& entry : iterator) {
        std::error_code entry_error;
        if (!entry.is_regular_file(entry_error) || entry_error) {
            continue;
        }
        const std::string extension = lower_extension(entry.path());
        if (extension == ".jpg" || extension == ".jpeg") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path& left, const fs::path& right) {
        return left.filename().string() < right.filename().string();
    });
    return files;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(fraction * values.size())) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

void print_stage(const char* label, double total_ms, size_t batch_count, size_t image_count) {
    std::cout << "  " << std::left << std::setw(34) << label << std::right
              << std::fixed << std::setprecision(3) << total_ms << " ms"
              << " | " << (batch_count ? total_ms / batch_count : 0.0) << " ms/batch"
              << " | " << (image_count ? total_ms / image_count : 0.0) << " ms/image\n";
}

std::vector<std::string> to_string_paths(const std::vector<fs::path>& paths) {
    std::vector<std::string> result;
    result.reserve(paths.size());
    for (const fs::path& path : paths) {
        result.push_back(path.string());
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    std::shared_ptr<ResNetEngine> engine;
    int exit_code = 1;

    try {
        const Options options = parse_options(argc, argv);
        std::cout << "====================================================================\n"
                  << "RK3576 MULTI-BATCH ZERO-COPY PERFORMANCE BENCHMARK\n"
                  << "====================================================================\n"
                  << "[Model]       " << options.model_path << '\n'
                  << "[Images dir]  " << options.images_dir << '\n'
                  << "[Batch size]  " << options.batch_size << '\n'
                  << "[Used images] " << options.used_images << " requested\n"
                  << "[Warm target] " << options.warm_target << " image samples\n\n";

        const std::vector<fs::path> discovered = discover_jpeg_images(options.images_dir);
        if (discovered.size() < kReservedWarmupImages) {
            throw std::runtime_error(
                "At least two JPEG files are required: the first two are reserved for warm-up; found " +
                std::to_string(discovered.size()) + ".");
        }

        const uint64_t warmup_runs_u64 =
            (static_cast<uint64_t>(options.warm_target) + options.batch_size - 1U) /
            options.batch_size;
        const uint64_t warmup_samples_u64 = warmup_runs_u64 * options.batch_size;
        if (warmup_runs_u64 > std::numeric_limits<uint32_t>::max() ||
            warmup_samples_u64 > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Calculated warm-up size exceeds the platform limits.");
        }
        const uint32_t warmup_runs = static_cast<uint32_t>(warmup_runs_u64);
        const size_t warmup_samples = static_cast<size_t>(warmup_samples_u64);

        std::vector<fs::path> warmup_images(
            discovered.begin(), discovered.begin() + kReservedWarmupImages);
        std::vector<std::string> warmup_paths;
        warmup_paths.reserve(warmup_samples);
        for (uint32_t run = 0; run < warmup_runs; ++run) {
            for (uint32_t slot = 0; slot < options.batch_size; ++slot) {
                warmup_paths.push_back(
                    warmup_images[slot % kReservedWarmupImages].string());
            }
        }

        const size_t available_measured = discovered.size() - kReservedWarmupImages;
        const size_t selected_count = std::min<size_t>(options.used_images, available_measured);
        const size_t usable_count = (selected_count / options.batch_size) * options.batch_size;
        const size_t skipped_remainder = selected_count - usable_count;
        if (usable_count == 0) {
            throw std::runtime_error(
                "Not enough JPEG files after reserving the first two warm-up images for one complete batch.");
        }

        std::vector<fs::path> selected(
            discovered.begin() + static_cast<std::ptrdiff_t>(kReservedWarmupImages),
            discovered.begin() + static_cast<std::ptrdiff_t>(kReservedWarmupImages + usable_count));

        std::cout << "[Images] Found JPEG files:         " << discovered.size() << '\n'
                  << "[Images] Reserved for warm-up:     " << kReservedWarmupImages << '\n'
                  << "[Images] Warm-up target:           " << options.warm_target << " samples\n"
                  << "[Images] Warm-up actual:           " << warmup_samples << " samples\n"
                  << "[Images] Warm-up inference runs:   " << warmup_runs << '\n'
                  << "[Images] Warm-up source A: " << warmup_images[0].filename().string() << '\n'
                  << "[Images] Warm-up source B: " << warmup_images[1].filename().string() << '\n'
                  << "[Images] Selected for measurement: " << selected_count << '\n'
                  << "[Images] Measured/processed:       " << usable_count << '\n'
                  << "[Images] Skipped incomplete tail:  " << skipped_remainder << '\n'
                  << "[Images] Measured first: " << selected.front().filename().string() << '\n'
                  << "[Images] Measured last:  " << selected.back().filename().string() << '\n';

        std::cout << "\n[Runtime] Loading shared ResNetEngine library...\n";
        engine = std::make_shared<ResNetEngine>(
            options.model_path, options.batch_size, kRk3576NpuCoreCount);
        const EngineInfo info = engine->info();
        if (info.batch_size != options.batch_size || info.feature_count == 0 ||
            info.input_width != 320 || info.input_height != 180 || info.input_channels != 3) {
            throw std::runtime_error("Loaded ResNetEngine has unexpected model input/output attributes.");
        }
        std::cout << "[Engine]      " << info.input_width << 'x' << info.input_height << 'x'
                  << info.input_channels << " input, " << info.feature_count
                  << " features, batch " << info.batch_size << '\n'
                  << "[Runtime] API: " << info.runtime_api_version
                  << " | Driver: " << info.driver_version << '\n'
                  << "[Runtime] Multi-Batch NPU core count set to " << info.npu_core_count << ".\n";

        // The shared engine creates its MPP/RGA preprocessor lazily on the first
        // JPEG. Use the reserved A/B pair for warm-up so initialization is excluded
        // from the measured stream while the persistent MPP/RGA resources are reused.
        auto warmup_stream = engine->process_files(warmup_paths, warmup_paths.size());
        FeatureChunk warmup_chunk;
        size_t warmup_processed = 0;
        while (warmup_stream->next(warmup_chunk)) {
            warmup_processed += warmup_chunk.image_paths.size();
        }
        const StreamStats warmup_stats = warmup_stream->stats();
        if (!warmup_stats.completed || warmup_stats.images_failed != 0 ||
            warmup_processed != warmup_samples) {
            throw std::runtime_error("Warm-up did not process all reserved JPEG samples successfully.");
        }
        std::cout << "[Warm-up] Completed " << warmup_stats.batches_executed
                  << " inference runs using " << warmup_processed
                  << " reserved A/B JPEG samples.\n"
                  << "[Preprocess] MPP MJPEG decoder + RGA initialized once by the shared engine; "
                  << "reused for all measured images.\n";

        const std::vector<std::string> measured_paths = to_string_paths(selected);
        auto measured_stream = engine->process_files(measured_paths, options.batch_size);

        double output_checksum = 0.0;
        size_t progress_batches = 0;
        size_t progress_images = 0;
        const auto benchmark_start = Clock::now();
        FeatureChunk chunk;
        while (measured_stream->next(chunk)) {
            if (chunk.features.size() != chunk.image_paths.size() * info.feature_count) {
                throw std::runtime_error("ResNetEngine returned an inconsistent feature chunk size.");
            }
            for (float value : chunk.features) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Engine output contains NaN or infinity.");
                }
                output_checksum += value;
            }
            progress_images += chunk.image_paths.size();
            ++progress_batches;
            if ((progress_batches % 16U) == 0U || progress_images == usable_count) {
                std::cout << "[Progress] Completed " << progress_batches << "/"
                          << (usable_count / options.batch_size) << " batches ("
                          << progress_images << "/" << usable_count << " images).\n";
            }
        }
        const double benchmark_wall_ms = elapsed_ms(benchmark_start, Clock::now());
        const StreamStats stats = measured_stream->stats();
        if (!stats.completed) {
            throw std::runtime_error("Measured feature stream did not complete.");
        }
        if (stats.images_failed != 0) {
            throw std::runtime_error(
                "Benchmark encountered " + std::to_string(stats.images_failed) +
                " invalid JPEG image(s); first failure: " + stats.failures.front().path +
                " (" + stats.failures.front().reason + ").");
        }
        if (stats.images_processed != usable_count || stats.batches_executed == 0 ||
            stats.batch_timings.size() != stats.batches_executed) {
            throw std::runtime_error("Measured image/batch counts do not match the requested benchmark workload.");
        }

        const size_t batch_count = stats.batches_executed;
        const size_t image_count = stats.images_processed;
        const double calls = static_cast<double>(batch_count);
        const double images = static_cast<double>(image_count);
        const StreamTimings& timings = stats.timings;
        const double preprocess_ms = timings.jpeg_preprocess_ms;

        std::vector<double> inference_times;
        std::vector<double> batch_wall_times;
        inference_times.reserve(stats.batch_timings.size());
        batch_wall_times.reserve(stats.batch_timings.size());
        double inference_min = std::numeric_limits<double>::infinity();
        double inference_max = 0.0;
        double batch_wall_min = std::numeric_limits<double>::infinity();
        for (const BatchTiming& timing : stats.batch_timings) {
            inference_times.push_back(timing.inference_ms);
            batch_wall_times.push_back(timing.wall_ms);
            inference_min = std::min(inference_min, timing.inference_ms);
            inference_max = std::max(inference_max, timing.inference_ms);
            batch_wall_min = std::min(batch_wall_min, timing.wall_ms);
        }

        std::cout << "\n[Benchmark results]\n"
                  << "  Processed images:             " << image_count << '\n'
                  << "  Inference calls:              " << batch_count << '\n'
                  << "  Batch size:                   " << options.batch_size << '\n'
                  << "  NPU cores requested:          " << info.npu_core_count << '\n';

        std::cout << "\n[Stage timing] (warm-up and model initialization excluded)\n";
        print_stage("JPEG load + MPP/RGA preprocess", preprocess_ms, batch_count, image_count);
        std::cout << "  [Preprocess breakdown]\n";
        print_stage("  JPEG file read", timings.jpeg_file_read_ms, batch_count, image_count);
        print_stage("  Copy JPEG into MPP DMA-BUF", timings.jpeg_dma_copy_ms, batch_count, image_count);
        print_stage("  MPP hardware JPEG decode", timings.mpp_decode_ms, batch_count, image_count);
        print_stage("  RGA NV12 -> RGB888", timings.rga_convert_ms, batch_count, image_count);
        print_stage("  Normalize + FP16 conversion", timings.fp16_preprocess_ms, batch_count, image_count);
        std::cout << "  Input copy into Zero-Copy buffer: eliminated; preprocessing writes directly into RKNN memory.\n";
        print_stage("Input cache sync (CPU -> NPU)", timings.input_cache_sync_ms, batch_count, image_count);
        print_stage("rknn_run()", timings.inference_ms, batch_count, image_count);
        print_stage("Output cache sync (NPU -> CPU)", timings.output_cache_sync_ms, batch_count, image_count);
        print_stage("Read/check output tensor", timings.output_read_ms, batch_count, image_count);

        std::cout << "\n[Inference-only statistics]\n"
                  << "  Total rknn_run time:           " << std::fixed << std::setprecision(3)
                  << timings.inference_ms << " ms\n"
                  << "  Mean time per batch:           " << timings.inference_ms / calls << " ms\n"
                  << "  Amortized time per image:      " << timings.inference_ms / images << " ms\n"
                  << "  Min / median / p95 / max:      " << inference_min << " / "
                  << percentile(inference_times, 0.50) << " / "
                  << percentile(inference_times, 0.95) << " / " << inference_max << " ms\n";

        std::cout << "\n[End-to-end benchmark] (includes loading every measured image from disk)\n"
                  << "  Total wall time:               " << benchmark_wall_ms << " ms\n"
                  << "  Mean time per batch:           " << benchmark_wall_ms / calls << " ms\n"
                  << "  Amortized time per image:      " << benchmark_wall_ms / images << " ms\n"
                  << "  Throughput:                    " << (images * 1000.0 / benchmark_wall_ms)
                  << " images/s\n"
                  << "  Batch wall min / median / p95: " << batch_wall_min << " / "
                  << percentile(batch_wall_times, 0.50) << " / "
                  << percentile(batch_wall_times, 0.95) << " ms\n"
                  << "  Output checksum:               " << output_checksum << '\n';

        std::cout << "\n[Notes]\n"
                  << "  - The benchmark uses the same libresnet_engine.so as the Python extension.\n"
                  << "  - Model loading, RKNN initialization, MPP/RGA initialization, directory scan\n"
                  << "    and warm-up are excluded.\n"
                  << "  - MPP/RGA preprocessing writes normalized FP16 data directly into the\n"
                  << "    RKNN Zero-Copy input buffer; there is no separate input memcpy stage.\n"
                  << "  - JPEG file read, MPP decode, RGA conversion, FP16 preprocessing, input\n"
                  << "    cache synchronization, inference, output synchronization and output\n"
                  << "    conversion/checking are included in end-to-end wall time.\n"
                  << "  - Non-JPEG files are ignored. Only top-level .jpg/.jpeg files are scanned.\n"
                  << "  - Non-divisible remainder samples are skipped, never padded.\n"
                  << "  - MPP/RGA preprocessing currently requires native 320x180 JPEGs; no resize is performed.\n"
                  << "  - The first two sorted JPEGs are reserved for warm-up; the same A/B pair\n"
                  << "    is repeated to fill warm-up batches and is never measured.\n";

        exit_code = 0;
    } catch (const std::exception& error) {
        std::cerr << "\n[Error] " << error.what() << '\n';
        exit_code = 1;
    }

    if (engine) {
        try {
            engine->close();
            std::cout << "[Cleanup] ResNetEngine closed.\n";
        } catch (const std::exception& error) {
            std::cerr << "[Cleanup] ResNetEngine close failed: " << error.what() << '\n';
            if (exit_code == 0) {
                exit_code = 1;
            }
        }
    }
    return exit_code;
}
