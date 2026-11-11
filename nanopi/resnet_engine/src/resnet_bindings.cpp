#include <resnet_engine.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace resnet_npu;

namespace {

py::array_t<float> make_feature_array(const std::vector<float>& values,
                                      size_t rows,
                                      size_t columns) {
    py::array_t<float> result({static_cast<py::ssize_t>(rows),
                               static_cast<py::ssize_t>(columns)});
    if (!values.empty()) {
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return result;
}

py::dict make_stats_dict(const StreamStats& stats) {
    py::dict result;
    result["files_found"] = stats.files_found;
    result["images_processed"] = stats.images_processed;
    result["images_failed"] = stats.images_failed;
    result["batches_executed"] = stats.batches_executed;
    result["features_returned"] = stats.features_returned;
    result["completed"] = stats.completed;
    result["closed_early"] = stats.closed_early;

    py::list failures;
    for (const ImageFailure& failure : stats.failures) {
        py::dict item;
        item["path"] = failure.path;
        item["reason"] = failure.reason;
        failures.append(std::move(item));
    }
    result["failures"] = std::move(failures);
    return result;
}

std::string path_like_to_string(py::handle value) {
    py::object path_value = py::module_::import("os").attr("fspath")(value);
    if (!py::isinstance<py::str>(path_value)) {
        throw py::type_error("Each image path must resolve to a Python str (bytes paths are not supported).");
    }
    return path_value.cast<std::string>();
}

}  // namespace

PYBIND11_MODULE(resnet_engine, module) {
    module.doc() = "Native Rockchip MPP + RGA + RKNN ResNet feature extraction engine.";

    py::class_<FeatureStream, std::shared_ptr<FeatureStream>>(module, "FeatureStream")
        .def("__iter__",
             [](FeatureStream& stream) -> FeatureStream& { return stream; },
             py::return_value_policy::reference_internal)
        .def("__next__", [](FeatureStream& stream) -> py::object {
            FeatureChunk chunk;
            bool available = false;
            {
                py::gil_scoped_release release;
                available = stream.next(chunk);
            }
            if (!available) {
                throw py::stop_iteration();
            }

            py::list paths;
            for (const std::string& path : chunk.image_paths) {
                paths.append(path);
            }
            const size_t rows = chunk.image_paths.size();
            const size_t columns = rows == 0 ? 0 : chunk.features.size() / rows;
            py::array_t<float> features = make_feature_array(chunk.features, rows, columns);
            return py::make_tuple(std::move(paths), std::move(features));
        })
        .def("close", &FeatureStream::close,
             "Stop the stream before all files have been processed.")
        .def_property_readonly("stats", [](const FeatureStream& stream) {
            return make_stats_dict(stream.stats());
        }, "Processing counters and per-file failures accumulated so far.");

    py::class_<EngineInfo>(module, "EngineInfo")
        .def_readonly("batch_size", &EngineInfo::batch_size)
        .def_readonly("input_width", &EngineInfo::input_width)
        .def_readonly("input_height", &EngineInfo::input_height)
        .def_readonly("input_channels", &EngineInfo::input_channels)
        .def_readonly("feature_count", &EngineInfo::feature_count)
        .def_readonly("npu_core_count", &EngineInfo::npu_core_count)
        .def_readonly("runtime_api_version", &EngineInfo::runtime_api_version)
        .def_readonly("driver_version", &EngineInfo::driver_version)
        .def("__repr__", [](const EngineInfo& info) {
            return "EngineInfo(batch_size=" + std::to_string(info.batch_size) +
                ", input=" + std::to_string(info.input_width) + "x" +
                std::to_string(info.input_height) + "x" +
                std::to_string(info.input_channels) + ", feature_count=" +
                std::to_string(info.feature_count) + ", npu_core_count=" +
                std::to_string(info.npu_core_count) + ", runtime_api_version=\"" +
                info.runtime_api_version + "\", driver_version=\"" +
                info.driver_version + "\")";
        });

    py::class_<ResNetEngine, std::shared_ptr<ResNetEngine>>(module, "ResNetEngine")
        .def(py::init<const std::string&, uint32_t, uint32_t>(),
             py::arg("model_path"), py::arg("batch_size") = 4,
             py::arg("npu_core_count") = 2,
             "Load an RKNN ResNet model and initialize its Zero-Copy runtime.")
        .def_property_readonly("info", &ResNetEngine::info)
        .def_property_readonly("model_path", &ResNetEngine::model_path)
        .def_property_readonly("closed", &ResNetEngine::is_closed)
        .def("close", [](ResNetEngine& engine) {
            py::gil_scoped_release release;
            engine.close();
        }, "Release MPP, RGA and RKNN resources. Safe to call more than once.")
        .def("process_folder",
             [](const std::shared_ptr<ResNetEngine>& engine,
                const std::string& folder_path,
                size_t chunk_size,
                bool recursive) {
                 std::shared_ptr<FeatureStream> stream;
                 {
                     py::gil_scoped_release release;
                     stream = engine->process_folder(folder_path, chunk_size, recursive);
                 }
                 return stream;
             },
             py::arg("folder_path"), py::arg("chunk_size") = 64,
             py::arg("recursive") = false,
             "Lazily process JPEGs in a folder; yields (image_paths, features) chunks.")
        .def("process_files",
             [](const std::shared_ptr<ResNetEngine>& engine,
                const py::iterable& image_paths,
                size_t chunk_size) {
                 if (py::isinstance<py::str>(image_paths) || py::isinstance<py::bytes>(image_paths)) {
                     throw py::type_error("image_paths must be an iterable of paths, not a single string.");
                 }
                 std::vector<std::string> paths;
                 for (py::handle value : image_paths) {
                     paths.push_back(path_like_to_string(value));
                 }
                 std::shared_ptr<FeatureStream> stream;
                 {
                     py::gil_scoped_release release;
                     stream = engine->process_files(paths, chunk_size);
                 }
                 return stream;
             },
             py::arg("image_paths"), py::arg("chunk_size") = 64,
             "Process an explicit ordered list of JPEG paths; yields result chunks.")
        .def("infer_preprocessed",
             [](ResNetEngine& engine, py::array input_tensor) {
                 using FloatArray = py::array_t<float,
                     py::array::c_style | py::array::forcecast>;
                 FloatArray tensor = FloatArray::ensure(input_tensor);
                 if (!tensor) {
                     throw py::type_error("input_tensor must be convertible to a float32 NumPy array.");
                 }
                 const py::buffer_info buffer = tensor.request();
                 if (buffer.ndim != 4) {
                     throw py::value_error("input_tensor must be a 4D NCHW or NHWC array.");
                 }

                 const EngineInfo info = engine.info();
                 const auto* shape = buffer.shape.data();
                 TensorLayout layout;
                 if (shape[1] == static_cast<py::ssize_t>(info.input_channels) &&
                     shape[2] == static_cast<py::ssize_t>(info.input_height) &&
                     shape[3] == static_cast<py::ssize_t>(info.input_width)) {
                     layout = TensorLayout::NCHW;
                 } else if (shape[1] == static_cast<py::ssize_t>(info.input_height) &&
                            shape[2] == static_cast<py::ssize_t>(info.input_width) &&
                            shape[3] == static_cast<py::ssize_t>(info.input_channels)) {
                     layout = TensorLayout::NHWC;
                 } else {
                     throw py::value_error(
                         "input_tensor shape does not match model input dimensions; expected "
                         "(N,3,H,W) NCHW or (N,H,W,3) NHWC.");
                 }
                 if (shape[0] < 0) {
                     throw py::value_error("input_tensor has an invalid sample dimension.");
                 }

                 const size_t samples = static_cast<size_t>(shape[0]);
                 const float* data = static_cast<const float*>(buffer.ptr);
                 std::vector<float> input_values;
                 if (buffer.size > 0) {
                     input_values.assign(data, data + buffer.size);
                 }
                 std::vector<float> features;
                 {
                     py::gil_scoped_release release;
                     features = engine.infer_preprocessed(input_values.data(), samples, layout);
                 }
                 return make_feature_array(features, samples, info.feature_count);
             },
             py::arg("input_tensor"),
             "Run already-normalized float32 NCHW or NHWC tensors through RKNN. "
             "Incomplete final batches are padded and padding outputs are discarded.")
        .def("__enter__", [](const std::shared_ptr<ResNetEngine>& engine) { return engine; })
        .def("__exit__", [](ResNetEngine& engine, py::object, py::object, py::object) {
            py::gil_scoped_release release;
            engine.close();
        });
}
