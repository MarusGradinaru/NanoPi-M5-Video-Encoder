# NanoPi M5 Video Encoder

A hardware-accelerated video processing and encoding pipeline for detecting specific TV news presenters in video recordings, extracting their appearances, and combining the selected segments into a single HEVC video.

The project runs on a FriendlyElec NanoPi M5 equipped with the Rockchip RK3576 SoC. It uses FFmpeg with Rockchip MPP and RGA for video decoding and image processing, a native C++ ResNet50V2 feature-extraction engine connected to the RK3576 NPU, and a multiclass Logistic Regression classifier.

The current classifier is configured for **Andreea Esca**, a Romanian TV news presenter. The training pipeline supports additional classes, allowing other presenters to be included as the dataset expands.

## Why this project exists

The motivation is deliberately specific: I keep a personal collection of Romania's most sexy and beautiful TV news presenters. Building that collection the traditional way was a lot of work: I had to find the exact timecodes for each appearance, cut the scenes manually, and encode the clips in software on a PC. That took a great deal of time, and keeping entire news recordings just to save a few short appearances wasted a lot of storage space.

This project automates that process, selecting the appearances I want and combining them into a much smaller video using the RK3576's hardware encoder.

In other words, this is a video-encoding project with a very particular fan-driven use case. 🙂

## Processing workflow

The main processing pipeline no longer relies on FFmpeg scene-change detection (`scdet`). Instead, it continuously samples the source video and classifies the sampled frames.

The workflow consists of the following stages:

1. **Frame sampling.** FFmpeg decodes the source video using Rockchip MPP and produces RGB24 frames at a configurable sampling interval. The default interval is 160 ms.
2. **Feature extraction.** Each sampled frame is resized to 320×180 using Rockchip RGA and passed directly to the native `ResNetEngine`.
3. **NPU inference.** The C++ engine performs ResNet50V2 feature extraction through RKNN Runtime, using fixed-size batches and the RK3576 NPU.
4. **Classification.** A separately trained multiclass Logistic Regression classifier evaluates the 2048-dimensional feature vectors and calculates the probability of each class.
5. **Segment validation.** A rolling validation window checks classification consistency before accepting a candidate. Minimum run length and minimum duration requirements prevent isolated positive predictions from becoming video segments.
6. **Review handling.** Segments are separated into automatically validated results and lower-confidence candidates that can be reviewed manually.
7. **Video encoding.** The validated time ranges are encoded using the Rockchip HEVC hardware encoder.
8. **Final muxing.** The resulting segments are combined into `output_video.mkv`.

The frame-processing stage uses a streaming pipeline. Decoded RGB24 frames are transferred from FFmpeg directly to the native ResNet engine without creating an intermediate collection of JPEG files.

The classification and encoding stages can also be executed independently. This allows the detection results to be inspected before encoding starts.

### Segment validation

The classifier uses three mutually exclusive states:

- **VALID** — target-class probability is at or above the main threshold.
- **REVIEW** — probability is below the main threshold but at or above the optional review threshold.
- **NONE** — probability is below both thresholds.

A rolling validation window confirms transitions between these states using a configurable number of matching votes and a required consecutive run at the end of the window.

A candidate must also satisfy the minimum number of consecutive samples and minimum continuous positive coverage before it becomes an active segment.

This approach is designed to avoid accepting isolated detections or short classification fluctuations as genuine appearances.

## Requirements

### NanoPi M5

- FriendlyElec **NanoPi M5 / Rockchip RK3576**, running 64-bit ARM Ubuntu.
- A working RK3576 NPU driver and RKNN Runtime.
- Rockchip MPP and RGA libraries, headers, and `pkg-config` files.
- A Rockchip-enabled FFmpeg build with MPP and RGA support.
- `pyenv` with Python **3.12.x** installed.
- C++ build tools (`g++`, `pkg-config`) for the native ResNet engine.
- Access to the project's Hugging Face bucket to download model files and, when required, the training dataset.

The standard NanoPi inference path uses the native C++ `ResNetEngine` linked directly to `librknnrt.so`. It does not require RKNN Toolkit Lite 2.

The Python environment provides the application dependencies and the thin pybind11 extension.

### Optional PC / WSL environment for model conversion

- Windows with WSL2 and Ubuntu on x86-64.
- `pyenv` with Python **3.11.x** installed.
- CPU-only PyTorch 2.4.0.
- SSH/SCP connectivity to the NanoPi if you want to transfer generated model artifacts automatically.

The PC/WSL workflow is only needed when exporting or updating the RKNN feature-extractor model. It is not required for ordinary video processing.

## Repository layout

| Path | Purpose |
|---|---|
| `nanopi/` | Main NanoPi workflow: frame sampling, classification, segment encoding, classifier training, model validation, and helper scripts. |
| `nanopi/resnet_engine/` | Native C++ ResNet engine, Python bindings, benchmark, JPEG diagnostic utility, and RKNN Runtime SDK download helper. See its [dedicated README](nanopi/resnet_engine/README.md). |
| `pc-wsl/` | Optional PC/WSL workflow that exports the Keras ResNet50V2 feature extractor to ONNX and converts it to RKNN. |
| `resnet/` | Local model files, training dataset, and build/validation artifacts. These files are downloaded or generated locally and are not stored in Git. |
| `ffmpeg/` | `setup-ffmpeg.sh`, which builds the Rockchip-enabled MPP, RGA, and FFmpeg stack. |
| `windows/` | Windows helpers for SSH configuration, launching the remote processing workflow, and Explorer context-menu shortcuts. |

The `.gitignore` excludes generated build products, virtual environments, downloaded model files, datasets, and local artifacts. The repository contains the scripts and source required to reproduce the workflow, not the large model and dataset files.

## Getting started on the NanoPi

The following commands assume that the repository is cloned into `~/NanoPi-M5-Video-Encoder`. Adjust paths if your checkout is elsewhere.

### 1. Clone the repository

```bash
git clone https://github.com/MarusGradinaru/NanoPi-M5-Video-Encoder.git
cd NanoPi-M5-Video-Encoder
```

### 2. Build and install the Rockchip-enabled FFmpeg stack

```bash
cd ffmpeg
./setup-ffmpeg.sh
```

The script builds pinned revisions of Rockchip MPP, RGA, and FFmpeg. The default installation prefix is `/opt/rkmpp-stack`, which is the path expected by the NanoPi processing and encoding scripts.

To limit parallel build jobs (not recommended):

```bash
./setup-ffmpeg.sh --jobs 4
```

Use `./setup-ffmpeg.sh --help` for the available options. A custom installation prefix can be selected with `--dir`, but the scripts and build configuration must then be adjusted to use that prefix.

### 3. Create the NanoPi Python environment

Run the setup script from the `nanopi/` directory:

```bash
cd ../nanopi
./setup-venv-resnet.sh
source .venv-resnet/bin/activate
```

The script selects an installed Python 3.12.x release through `pyenv`, creates or reuses `.venv-resnet`, and installs the pinned application dependencies, including scikit-learn, Pillow, Hugging Face Hub, and pybind11.

### 4. Build the native ResNet engine

The standard NPU workflow requires both the shared C++ library and its Python extension.

Run these commands from `nanopi/`:

```bash
./resnet_engine/get-rknn-lib.sh
./resnet_engine/src/build-resnet-engine-library.sh
./resnet_engine/src/build-resnet-engine-python-ext.sh
```

The first command downloads the RKNN Runtime headers and AArch64 runtime library from the `v2.3.2` tag of the upstream Rockchip RKNN Toolkit2 repository.

The second builds the native engine against the installed MPP/RGA stack.

The third builds the Python extension used by the main processing script.

The resulting shared library and Python extension are local build products and are not committed to Git.

The engine architecture, build process, and programming interfaces are documented in [nanopi/resnet_engine/README.md](nanopi/resnet_engine/README.md).

### 5. Download the model files

The model files are stored in the project's [Hugging Face bucket](https://huggingface.co/buckets/MarusGradinaru/romanian-tv-resnet-classifier). With the NanoPi virtual environment active, run the downloader from the repository's `resnet/` directory:

```bash
cd ../resnet
./get-resnet.sh models
```

To download the training dataset as well:

```bash
./get-resnet.sh dataset
```

Or synchronize both folders at once:

```bash
./get-resnet.sh models dataset
```

> **Important:** `get-resnet.sh` uses `hf buckets sync --delete`. Extra local files inside each selected folder may be deleted if they are not present in the remote bucket. Back up custom files before synchronizing those folders.

The `models/` directory must contain:

- `resnet50v2_logistic_classifier.npz` — trained multiclass classifier parameters.
- The RKNN ResNet50V2 feature-extractor model corresponding to the selected inference batch size.

The default batch size is 4, so the normal default extractor is the `_batch4.rknn` model.

## Running Stage 1 — Processing the video

Go to `/nanopi` folder, activate the virtual environment (if it's not active), and run `process_video.py`:

```bash
cd ~/NanoPi-M5-Video-Encoder/nanopi
source .venv-resnet/bin/activate

python process_video.py \
  --target-class andreea-esca \
  --sampling 160 \
  --threshold 0.75 \
  "/path/to/my_video.ts"
```

The `--target-class` option is required and must match a class name stored in the classifier model.

By default, the script creates a working directory named after the input video, next to the source file.

For example:

```text
/recordings/my_video.ts
```

produces:

```text
/recordings/my_video/
```

The output directory must not already contain files. Use `--out` to select another location or move the previous results first.

### Processing output

The processing directory contains:

| Path | Purpose |
|---|---|
| `seg_validated/` | Automatically validated target-class segments. These are the segments used by the encoder. |
| `seg_review/` | Candidate segments that meet the review threshold but not the main validation threshold. |
| `source_link.<ext>` | Relative symbolic link to the original source video, used by the encoder. |
| `delete_source.sh` | Helper script for deleting the source video after processing. |
| `delete_source.bat` | Windows launcher for the source-deletion helper. |
| `log-class-work.txt` | Optional processing console log. |
| `log-class-frames.txt` | Optional per-frame classification probabilities. |

The timestamp information is stored in the filenames of the extracted segment thumbnails:

```text
HH-MM-SS-mmm - HH-MM-SS-mmm.jpg
```

The first timestamp represents the segment start; the second represents its end.

After processing, inspect the images in `seg_review/`. Move any candidate you want to include into `seg_validated/`.

### Useful processing options

| Option | Default / behavior |
|---|---|
| `--target-class NAME` | Required. Class name to detect. |
| `--threshold N` | `0.75`; minimum target-class probability for automatic validation. |
| `--review-thr N` | `0`; lower probability threshold for review candidates. `0` disables review handling. |
| `--sampling MS` | `160`; minimum interval between sampled frames, in milliseconds. Accepted range: 5–1000 ms. |
| `--batch-size N` | `4`; number of images per inference batch. Selects the corresponding RKNN feature-extractor model. |
| `--classifier PATH` | Explicit multiclass classifier file. |
| `--extractor PATH` | Explicit RKNN feature-extractor model. When supplied, its filename determines the effective batch size. |
| `--window N` | `5`; samples in the rolling validation window. |
| `--votes N` | `4`; matching-category samples required inside the validation window. |
| `--streak N` | `2`; consecutive matching samples required at the end of the window. |
| `--min-time MS` | `1000`; minimum continuous positive coverage required to validate a candidate. |
| `--min-run N` | `3`; minimum consecutive matching-category samples required to validate a candidate. |
| `--deint 0/1` | `1`; enable or disable MPP hardware deinterlacing. |
| `--out PATH` | Custom working directory. |
| `--offset SECONDS` | Manual timestamp correction. Completely overrides the offset automatically calculated by default using FFprobe metadata. |
| `--verbose` | Show additional detection events and diagnostics. |
| `--log-work` | Save displayed console output to `log-class-work.txt`, excluding transient progress updates. |
| `--log-frames` | Write a classification-probability record for each sampled frame to `log-class-frames.txt`. |

The accepted RKNN batch sizes are `1` or positive even integers (`2`, `4`, `6`, `8`, and so on). Each batch size requires a matching fixed-batch RKNN model.

When `--extractor` is supplied explicitly, the model filename is authoritative: `_batch8` means batch 8, while a filename without `_batchN` means batch 1.

For the complete option list:

```bash
python process_video.py --help
```

### Timestamp handling

The processing script obtains the container start time and video stream start time through FFprobe. The difference between these timestamps is used to calculate the default offset automatically. This is important for transport-stream recordings where container and video timestamps do not necessarily begin at the same point. The calculated offset can be overridden with `--offset`.

### Progress and performance information

During processing, the script reports:

- Number of sampled frames.
- Average sampling interval.
- Number of validated and review segments.
- Current position in the source recording.
- Realtime processing factor.
- Current and maximum monitored SoC temperature.

The processing summary also includes the time spent reading frames, packing RGB24 batches, running ResNet inference, and executing the Logistic Regression classifier.

## Running Stage 2 — Encoding the selected segments

Pass the processing directory to `encode_segments.py`:

```bash
python encode_segments.py \
  --qp 20 --tier high --level 5.1 \
  "/path/to/my_video"
```

The input must be the processing directory generated by `process_video.py`.

The encoder reads the intervals from `seg_validated/`, accesses the original recording through `source_link.<ext>`, encodes the selected ranges into individual MP4 files, and muxes them into `output_video.mkv`. Individual encoded clips are stored in `encoded_seg/`.

Existing encoded clips are skipped unless `--force-encode` is specified.

### Encoding options

| Option | Default / behavior |
|---|---|
| `--qp N` | `20`; HEVC encoder quantization parameter. |
| `--tier NAME` | `high`; HEVC tier: `main` or `high`. |
| `--level NAME` | `5.1`; HEVC level. |
| `--edge-trim MS` | `40`; trim this amount from both ends of each resulting segment, after adjacent ranges have been merged. |
| `--decode-mode NAME` | `auto`; input decoding mode: `auto`, `hardware`, or `software`. Automatic mode retries with software decoding if hardware decoding fails. |
| `--scan-type NAME` | `auto`; source scan type: `auto`, `detect`, `progressive`, or `interlaced`. |
| `--seek-back SECONDS` | `5.0`; seek before a segment start to improve accurate trimming. |
| `--sb TIME` | `00:00:00.000`; beginning of the timing-correction range. |
| `--se TIME` | Recording end; exclusive end of the timing-correction range. |
| `--offset MS` | `0`; signed video and audio time correction in milliseconds. |
| `--audio-delay MS` | `0`; signed audio-only synchronization correction in milliseconds. |
| `--skip-mux` | Encode missing clips without creating or updating `output_video.mkv`. |
| `--just-mux` | Skip encoding and mux existing timestamp-named MP4 clips in `encoded_seg/`. Overrides `--skip-mux`. |
| `--force-encode` | Re-encode and overwrite all selected clips. |
| `--log-work` | Save displayed console output to `log-enc-work.txt`. |

The supported HEVC levels are:

```text
1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2
```

### Video scan type and decoding

The encoder supports both progressive and interlaced sources.

In automatic scan-type mode, the script uses explicit FFprobe `field_order` metadata. If the metadata is insufficient, processing stops rather than guessing. Use `--scan-type detect` to run the slower FFmpeg `idet` analysis, or explicitly specify `progressive` or `interlaced` when the source type is already known.

For interlaced sources, the output frame rate is doubled after deinterlacing.

In automatic decoding mode, the script tries Rockchip MPP hardware decoding and falls back to software decoding if necessary. Software decoding can also be selected explicitly.

HEVC output encoding always uses the Rockchip hardware encoder.

### Timing correction

The correction range allows timing adjustments to be applied selectively to segments based on their original source start times.

For example:

```bash
python encode_segments.py \
  --qp 20 --tier high --level 5.1 \
  --sb 00:12:00.000 --se 00:25:00.000 \
  --offset -120 \
  "/path/to/my_video"
```

The `--offset` parameter shifts both video and audio boundaries. Positive values move them forward; negative values move them backward.

The `--audio-delay` parameter applies an additional audio-only synchronization correction. Positive values delay audio, while negative values advance it.

Existing encoded clips are not automatically replaced when corrections change. Use `--force-encode` to apply new settings to existing clips.

For the complete option list:

```bash
python encode_segments.py --help
```

## Understanding the offset adjustment in these two stages

The project uses two independent timestamp offset mechanisms. They operate at different stages of the processing pipeline and serve different purposes.

### 1. `process_video.py` — Automatic timestamp correction

The processing script automatically calculates a timestamp offset using FFprobe:

```text
offset = video_stream.start_time - format.start_time
```

This correction compensates for the difference between the video stream's initial timestamp and the container's start time. It is primarily intended for the MPEG-TS recordings used by this project, where FFmpeg's default input timestamp handling can otherwise produce video timestamps that do not reflect the stream's actual position relative to the recording timeline.

The calculated offset is applied using FFmpeg's `setpts` filter, before timestamps are passed to the Python processing pipeline. Consequently, the detected segment boundaries and the timestamps stored in the generated image filenames are aligned with the corrected video timeline.

Frame selection is not affected by this correction. The `select` filter compares the time difference between consecutive selected frames, so adding the same constant offset to all timestamps does not change which frames are selected.

**Manual override:**

The optional `--offset` argument specifies a manual timestamp offset in seconds.

- When omitted, the automatically calculated offset is used.
- When specified, the manual value completely replaces the automatically calculated offset. It is not added to it.
- Specifying `--offset 0` disables the automatic correction.

Examples:

```bash
--offset 0.250    # Apply a manual +250 ms offset
--offset -0.120   # Apply a manual -120 ms offset
--offset 0        # Disable timestamp correction
```

### 2. `encode_segments.py` — Manual segment boundary correction

The encoding script provides a separate `--offset` argument, expressed in milliseconds.

Unlike the processing offset, this correction does not modify the timestamps generated by the processing stage. Instead, it shifts the start and end boundaries used when extracting segments from the original recording.

The offset is added equally to both boundaries.
A positive offset moves the extraction window later in the recording, while a negative offset moves it earlier. The segment duration remains unchanged.

The encoder subsequently adds the container's original `start_time` because FFmpeg is invoked with `-copyts`, which preserves the input timestamp domain. The configured edge trimming is then applied to both boundaries.

**Selective correction range:**

The encoder allows this manual correction to be restricted to a specific portion of the recording:

- `--sb` defines the beginning of the correction range (inclusive).
- `--se` defines the end of the correction range (exclusive).

A segment is selected for correction when its **start timestamp** falls inside this range. By default, the range covers the entire recording.

Example:

```bash
--sb 00:12:00.000 --se 00:18:00.000 --offset 180
```

This shifts the extraction boundaries of segments starting between 12 and 18 minutes by +180 ms.

The correction is applied to both video and audio extraction boundaries, preserving their relative synchronization. For audio-only synchronization adjustments, use the separate `--audio-delay` argument.

**Intended use:**

This mechanism is useful when a recording contains localized timestamp irregularities or discontinuities that cause the detected segment boundaries to become slightly misaligned with the actual content as presented by a video player.

A constant offset can compensate for a corresponding constant timing error in the affected portion of the recording. It does not repair the original TS timestamps or correct arbitrary timestamp drift within a segment.

Note that already encoded segments are skipped by default. Use `--force-encode` to re-encode existing segments after changing the correction parameters.

### Summary

| Script | Offset type | Unit | Purpose |
|---|---|---|---|
| `process_video.py` | Automatic, with manual override | Seconds | Correct the global video timestamp position before segment detection. |
| `encode_segments.py` | Manual, optionally range-limited | Milliseconds | Shift selected segment extraction boundaries to compensate for local timing errors. |

The two corrections are independent: the encoder's manual offset is applied on top of the segment timestamps produced by the processing stage. It does not replace or recalculate the processing-stage correction.

## Standalone scene-change thumbnail extraction

The repository also includes: `nanopi/scdet-extract.sh`

This is a separate utility for extracting thumbnails at FFmpeg scene-change boundaries. It is independent of the ResNet classification pipeline and is not used by `process_video.py`.

Run it from the NanoPi:

```bash
./scdet-extract.sh "/path/to/my_video.ts"
```

By default, the script creates:

```text
/path/to/my_video_scenes/
```

The initial frame is also extracted, and the timestamps are converted into consecutive thumbnail intervals.

### Options

| Option | Default / behavior |
|---|---|
| `--width PIXELS` | `320`; thumbnail width. Must be an even number. Height preserves the source aspect ratio. |
| `--scene-threshold N` | `10`; FFmpeg scene-change sensitivity, from 0 to 100. |
| `--offset SECONDS` | Manually override the automatically calculated timestamp offset. |
| `--out DIRECTORY` | Custom output directory. |

## Running the complete workflow with process-and-encode.sh

`process-and-encode.sh` provides a guided two-stage workflow that runs classification followed by encoding and final muxing.

Before running it, a configuration file must be available in the `nanopi/` directory:

```text
process-and-encode.ini
```

The script looks for the configuration in this order:

1. `/dev/shm/process-and-encode.ini` — temporary configuration uploaded by the Windows launcher.
2. `nanopi/process-and-encode.ini` — local configuration file.

The temporary RAM configuration takes priority and is automatically removed when the wrapper exits.

The configuration is parsed and validated before processing starts. Unknown parameters generate warnings and are ignored.

### Configuration example

Create `nanopi/process-and-encode.ini` and configure the parameters according to your preferences.

The Windows directory includes a ready-to-use template:

```text
windows/process-and-encode-template.ini
```

Copy it and rename the copy to:

```text
process-and-encode.ini
```

Do not edit or rename the original template. It is intended to remain unchanged as a reference for users.

The configuration contains two main sections.

**Classifier parameters**

| Parameter | Default | Description |
|---|---|---|
| `target-class` | Required | Class name to detect. |
| `threshold` | `0.75` | Probability threshold for validated segments. |
| `review-thr` | `0.0` | Lower threshold for review candidates; `0` disables review. |
| `batch-size` | `4` | Number of images per inference batch. |
| `deint` | `1` | Hardware deinterlacing: `0` off, `1` on. |
| `verbose` | `0` | Show additional diagnostic information. |
| `log-class-work` | `0` | Save classifier console output. |
| `log-class-frames` | `0` | Save per-frame classification probabilities. |

**Encoder parameters**

| Parameter | Default | Description |
|---|---|---|
| `qp` | `20` | HEVC quantization parameter. |
| `tier` | `high` | HEVC encoding tier. |
| `level` | `5.1` | HEVC encoding level. |
| `edge-trim` | `40` | Milliseconds trimmed from both segment edges. |
| `scan-type` | `auto` | Video scan type. |
| `sel-begin` | `start` | Beginning of the timing-correction range. |
| `sel-end` | `end` | End of the timing-correction range. |
| `offset` | `0` | Signed A/V correction in milliseconds. |
| `audio-delay` | `0` | Signed audio-only correction in milliseconds. |
| `log-enc-work` | `0` | Save encoder console output. |

All boolean parameters use `0` or `1`.

The selection range uses either `start`/`end` or an explicit `HH:MM:SS.mmm` timestamp.

### Running the wrapper

Run it from `nanopi/`:

```bash
./process-and-encode.sh "/absolute/path/to/my_video.ts"
```

The wrapper expects exactly one absolute Linux path to the input video.

It creates the processing directory next to the source recording, using the video filename without its extension.

The workflow is divided into two stages:

1. **Classification:** the video is sampled and classified. When new results have been generated, the script waits for user confirmation so that review candidates can be inspected.
2. **Encoding:** the validated segments are encoded and muxed into the final video.

If completed processing results already exist, the wrapper skips classification and proceeds directly to encoding.

If an incomplete processing directory exists, its contents are removed before a fresh run. Keep any files you need before reusing an incomplete directory.

## Training or updating the presenter classifier

The training dataset uses one subdirectory per class. The directory name becomes the class label.

Example:

```text
resnet/dataset/
├── andreea-esca/
│   ├── image_001.jpg
│   └── ...
└── other-presenter/
    ├── image_001.jpg
    └── ...
```

At least two non-empty classes are required, with at least two JPEG images in each class. Larger and more varied datasets are strongly recommended.

Run training from `nanopi/` with the NanoPi virtual environment active:

```bash
python train_classifier.py
```

The default feature-extraction backend uses the RK3576 NPU and the batch-4 RKNN model.

Other installed batch variants can be selected explicitly:

```bash
python train_classifier.py --batch-size 8
```

Or specify a particular feature-extractor model:

```bash
python train_classifier.py \
  --extractor ../resnet/models/resnet50v2_feature_extractor_rk3576_fp16_320x180_batch2.rknn
```

An explicit extractor path takes precedence and its filename determines the effective batch size.

The ResNet50V2 feature extractor remains frozen. The training pipeline extracts 2048-dimensional feature vectors and trains a separate multiclass Logistic Regression classifier.

The trained classifier parameters are saved in:

```text
resnet/models/resnet50v2_logistic_classifier.npz
```

The classifier can then be used immediately by `process_video.py`.

## Optional PC / WSL model conversion workflow

The `pc-wsl/` directory contains the tools required to export the ResNet50V2 feature extractor and convert it into an RKNN model for the RK3576 NPU. This workflow is not required for normal video processing when the required RKNN model is already available on the NanoPi. Run it from WSL Ubuntu on an x86-64 PC.

1. Clone the repository:

   ```bash
   git clone https://github.com/MarusGradinaru/NanoPi-M5-Video-Encoder.git
   cd NanoPi-M5-Video-Encoder/pc-wsl
   ```

2. Prepare the Python 3.11 conversion environment:

   ```bash
   ./setup-venv-resnet.sh
   ```

   There is no need to activate the environment (`source .venv-resnet/bin/activate`), unless you plan to execute the Python scripts separately.

3. Install the SSH alias (just and example; use your own data):

   ```bash
   ./install-ssh-alias.sh marus 192.168.0.50 marus_key marus 192.168.0.30 marus_key
   ```

   ...and don't forget to copy your acces keys in `~/.ssh` folder.

4. On first run, get the dataset (model conversion needs some validation images):

   ```bash
   source .venv-resnet/bin/activate
   cd ../resnet
   ./get-resnet.sh dataset
   cd ../pc-wsl
   ```

5. Run the complete build. The default batch size is 4. 

   ```bash
   ./build.sh
   ```

   Batch 1 or any positive even batch size can be requested like this:

   ```bash
   ./build.sh --batch-size 4
   ```

   Add `--forced` when you also want to force the ONNX export step:

   ```bash
   ./build.sh --batch-size 4 --forced
   ```

   The build checks the pinned Python environment, exports the frozen Keras ResNet50V2 model to ONNX, converts the ONNX input layout from NHWC to NCHW, and converts the result to a fixed-batch FP16 RKNN model without INT8 quantization. It also creates a validation archive containing normalized input samples and ONNX reference features.

6. The script offers to transfer the RKNN model and matching validation archive to the NanoPi. 

   You can enter a different SSH target and remote destination when prompted, or
   use the default SSH target `rkvenc-nanopi` and default remote directory `~/NanoPi-M5-Video-Encoder/resnet/artifacts/`.

   The local build artifacts remain in `resnet/artifacts/` if transfer is skipped.

7. On the NanoPi, validate and promote the candidate model:

   ```bash
   cd ~/NanoPi-M5-Video-Encoder/nanopi
   source .venv-resnet/bin/activate
   python validate_rknn_export.py
   ```

   The device validation runs the candidate through the native ResNetEngine and compares its feature vectors with the ONNX reference data. Every sample must reach cosine similarity of at least `0.95`. Only a passing candidate is promoted from `resnet/artifacts/` to `resnet/models/`; the matching validation archive is then removed.

Model filenames use a fixed-batch naming convention:

| Batch | RKNN model filename |
|---:|---|
| 1 | `resnet50v2_feature_extractor_rk3576_fp16_320x180.rknn` |
| 2 | `resnet50v2_feature_extractor_rk3576_fp16_320x180_batch2.rknn` |
| 4 | `resnet50v2_feature_extractor_rk3576_fp16_320x180_batch4.rknn` |
| 8 | `resnet50v2_feature_extractor_rk3576_fp16_320x180_batch8.rknn` |

The conversion environment uses RKNN Toolkit2 2.3.2 on the PC/WSL side. The NanoPi's inference runtime is separate: the native C++ engine links to the RKNN Runtime library downloaded by `nanopi/resnet_engine/get-rknn-lib.sh`.

## Windows integration

The `windows/` directory contains optional helpers for launching the NanoPi workflow directly from Windows.

The integration uses:

- Windows OpenSSH (`ssh.exe` and `scp.exe`).
- SSH configuration aliases named `rkvenc-nanopi` and `rkvenc-nas`.
- A mapped Windows network drive pointing to the same NAS storage mounted on the NanoPi.
- The Windows batch launcher and its INI configuration file.

### SSH configuration

The Windows scripts use the SSH aliases to acces the NanoPi and NAS. Configure this aliases in your Windows SSH configuration `%USERPROFILE%\.ssh\config`. The repository includes an SSH alias installation helper:

```text
windows/install-ssh-alias.bat
```

Run it and follow its instructions. Copy your acces keys in `%USERPROFILE%\.ssh` folder, if you use one.

### Configure the Windows launcher

Copy the template `process-and-encode-template.ini` and rename it to `process-and-encode.ini`. Edit the copied file according to your preferences. The original template must remain unchanged. It documents the supported configuration keys and their expected formats.

The Windows launcher reads this local INI file and uploads it to `/dev/shm/process-and-encode.ini` on the NanoPi using SCP. The remote workflow then uses this temporary configuration instead of the local NanoPi configuration, and removes the uploaded file automatically when processing finishes.

### Explorer context-menu integration

Use the provided scripts in `windows/` folder to install and remove an Explorer context-menu entry for a selected video file extension.

To install the context-menu entry:

```bat
add-menu-shortcut.bat ts /NAS
```

The extension and NAS mount point can also be entered interactively.

After installation, right-click a supported video file in Windows Explorer and select
`Send to NanoPi Encoder...`. The script checks that the video is stored on a mapped network drive, translates the Windows path into its corresponding Linux NAS path, uploads the configuration to the NanoPi, and starts the remote processing workflow.

To remove the entry:

```bat
remove-menu-shortcut.bat ts
```

The Windows integration is optional. All processing and encoding operations can also be started directly from the NanoPi terminal.


## Utility scripts

| Script | Purpose |
|---|---|
| `nanopi/adj-audio.sh` | Adjust audio/video synchronization without re-encoding. A positive offset in milliseconds delays audio; a negative offset delays video. |
| `nanopi/mux-seg.sh` | Mux timestamp-named MP4 segments from one folder, or from multiple folders in the supplied folder order. Files are sorted chronologically within each folder, and overlapping ranges within one folder are rejected. |
| `nanopi/validate_rknn_export.py` | Validate a newly converted RKNN feature extractor on the NanoPi before promoting it to `resnet/models/`. |
| `resnet/get-resnet.sh` | Synchronize selected model, dataset, or artifact folders from the Hugging Face bucket. |
| `nanopi/resnet_engine/` | Build and test the native C++ ResNet inference engine. See [its README](nanopi/resnet_engine/README.md). |

For the complete list of command-line options, run the script with `--help` where supported.

## Notes and limitations

- The included classifier is currently configured for `andreea-esca`; it will not recognize a presenter that has not been represented by the trained classes.
- Classification quality depends on the training examples and selected probability thresholds. Review borderline scenes, especially after changing the dataset or classifier.
- The native MPP/RGA preprocessing path expects JPEG thumbnails with the dimensions required by the RKNN model. The feature extractor distributed for this project uses 320x180 input images.
- Hardware acceleration depends on a compatible RK3576 kernel/driver/runtime stack and the custom FFmpeg build. The scripts cannot make an incompatible system image compatible by themselves.

## Marus Non-Commercial License v1.0

This project is free for personal and hobby use only.  
Commercial use requires a paid license from the author.  

Third-party libraries keep their own licenses:
- FFmpeg and the Rockchip MPP/RGA stack for hardware video processing.
- RKNN Runtime for NPU inference.
- ResNet50V2 for image feature extraction.
- Logistic Regression for multiclass classification.
- Python and native C++ for the processing pipeline.
- HEVC hardware encoding through Rockchip MPP.
