# NanoPi M5 Video Encoder

This project detects scene changes in TV recordings, classifies the resulting thumbnails with a ResNet50V2-based presenter classifier, and encodes the selected scenes into a single HEVC video using the Rockchip RK3576 hardware encoder.

The main workflow runs on a FriendlyElec NanoPi M5. An optional Windows/WSL workflow can regenerate the RKNN feature-extractor model, while the NanoPi performs normal inference through a native C++ engine linked directly to the RKNN Runtime.

## Why this project exists

The motivation is deliberately specific: I keep a personal collection of Romania's most sexy and beautiful TV news presenters. Building that collection the traditional way was a lot of work: I had to find the exact timecodes for each appearance, cut the scenes manually, and encode the clips in software on a PC. That took a great deal of time, and keeping entire news recordings just to save a few short appearances wasted a lot of storage space.

This project automates that process, selecting the scenes I want and combining them into a much smaller video using the RK3576's hardware encoder. The current classifier is configured for **Andreea Esca**, a Romanian TV news presenter. The training workflow supports additional classes, so other presenters can be included as the dataset is expanded. In other words, this is a video-encoding project with a very particular fan-driven use case. 🙂

## Processing workflow

1. **Detect scene changes.** The Rockchip-enabled FFmpeg build decodes and analyzes the source video, then extracts a thumbnail for each detected scene boundary.
2. **Extract image features.** The native `ResNetEngine` uses MPP for JPEG decoding, RGA for color conversion, and RKNN Runtime for multi-batch inference on the RK3576 NPU.
3. **Classify scenes.** A separately trained multiclass Logistic Regression classifier evaluates the 2048-dimensional ResNet50V2 feature vectors.
4. **Review uncertain predictions.** The processing script creates an Excel report and can move borderline candidates into a review folder.
5. **Encode and mux.** The selected time ranges are encoded with the Rockchip HEVC encoder and combined into `output_video.mkv`.

Scene processing and video encoding can be run separately. This allows the predictions to be inspected and corrected before encoding starts.

## Requirements

### NanoPi M5

- FriendlyElec **NanoPi M5 / Rockchip RK3576**, running 64-bit ARM Ubuntu.
- A working RK3576 NPU driver and RKNN Runtime. The helper script in `nanopi/resnet_engine/` downloads the matching RKNN Runtime SDK files used by the native engine.
- Rockchip MPP and RGA libraries and headers. The project builds the pinned MPP, RGA, and FFmpeg sources with `ffmpeg/setup-ffmpeg.sh`.
- `pyenv` with Python **3.12.x** installed.
- C++ build tools (`g++`, `pkg-config`) for the native engine.
- Access to the project's Hugging Face bucket to download model files and, when required, the training dataset.

The standard NanoPi inference path does **not** use RKNN Toolkit Lite 2. It uses the native C++ `ResNetEngine` linked to `librknnrt.so`. The Python environment provides the application dependencies and the thin pybind11 extension.

### Optional PC / WSL environment for model conversion

- Windows with WSL2 and Ubuntu on x86-64.
- `pyenv` with Python **3.11.x** installed.
- SSH/SCP connectivity to the NanoPi if you want the build script to transfer the generated model and validation archive automatically.

The PC/WSL workflow is only needed when exporting or updating the RKNN feature-extractor model. It is not required for ordinary video processing when the required model files are already installed on the NanoPi.

## Repository layout

| Path | Purpose |
|---|---|
| `nanopi/` | Main NanoPi workflow: scene processing, segment encoding, classifier training, model validation, and helper scripts. |
| `nanopi/resnet_engine/` | Native C++ ResNet engine, Python bindings, benchmark, JPEG diagnostic utility, and RKNN Runtime SDK download helper. See its [dedicated README](nanopi/resnet_engine/README.md). |
| `pc-wsl/` | Optional PC/WSL workflow that exports the frozen Keras ResNet50V2 feature extractor to ONNX and converts it to RKNN. |
| `resnet/` | Local model files, dataset, and build/validation artifacts. These files are downloaded or generated locally and are not stored in Git. |
| `ffmpeg/` | `setup-ffmpeg.sh`, which builds the Rockchip-enabled MPP, RGA, and FFmpeg stack. Downloaded source trees and build files are ignored by Git. |
| `windows/` | Optional Windows helpers for SSH configuration, launching the remote workflow, and Explorer context-menu shortcuts. |

The `.gitignore` excludes generated build products, virtual environments, downloaded model files, datasets, and local artifacts. The repository contains the scripts and source needed to reproduce the workflow, not the large model and dataset files.

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

The script selects an installed Python 3.12.x release through `pyenv`, creates or reuses `.venv-resnet`, and installs the pinned application dependencies, including scikit-learn, Pillow, openpyxl, Hugging Face Hub, and pybind11.

### 4. Build the native ResNet engine

The standard NPU workflow needs both the shared C++ library and its Python extension. The first command downloads the RKNN Runtime headers and AArch64 runtime library from the `v2.3.2` tag of the upstream Rockchip RKNN Toolkit2 repository. The next command builds the engine against the MPP/RGA stack installed in `/opt/rkmpp-stack`. And the last one builds the resnet engine python extension.

Run these commands from `nanopi/`:

```bash
./resnet_engine/get-rknn-lib.sh
./resnet_engine/src/build-resnet-engine-library.sh
./resnet_engine/src/build-resnet-engine-python-ext.sh
```

The resulting shared library and Python extension are local build products and are not committed to Git. The engine build, its architecture, and its API are documented in [nanopi/resnet_engine/README.md](nanopi/resnet_engine/README.md).

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

The `models/` folder must contain the multiclass classifier and the RKNN feature-extractor variant you intend to use. The default processing and training batch size is 4, so the normal default extractor is the `_batch4.rknn` model.

### 6. Process a video

With virtual environment activated, run `process_video.py`. The `--target-class` option is required and must match a class name in the classifier model.

```bash
cd ../nanopi
python process_video.py \
  --target-class andreea-esca \
  --candidate-threshold 0.4 \
  "/path/to/my_video.ts"
```

The default target-class probability threshold is `0.75`. With `--candidate-threshold 0.4`, non-validated predictions whose target-class probability falls between `0.4` and `0.75` are placed in `scenes_to_review/`. A candidate threshold of `0` disables review-candidate handling. The candidate threshold must not exceed the main threshold.

By default, the script creates a working directory named after the input video, next to the source file. For example, `/recordings/my_video.ts` produces `/recordings/my_video/`. The working directory must not already contain files; use `--out` to choose another location or move the previous results first.

The processing directory contains:

- `scenes_all_other/` — the extracted thumbnails that have not been moved into a classified folder.
- `scenes_validated/` — scenes accepted by the classifier. These are the scenes used by the encoder.
- `scenes_to_review/` — borderline candidates that can be inspected and manually moved into `scenes_validated/` if they should be included.
- `scene_predictions.xlsx` — the formatted prediction report, always generated.
- `scene_predictions.csv` — an optional machine-readable report, generated with `--csv-predict`.
- `source_link.<extension>` — a relative symbolic link to the original video, used by the encoding script.
- `delete_source.sh` and `delete_source.bat` — generated helper launchers for deleting the source video after encoding.

After processing, inspect `scene_predictions.xlsx` and the images in `scenes_to_review/`. Move any review candidate that you want to encode into `scenes_validated/`.

#### Useful processing options

| Option | Default / behavior |
|---|---|
| `--threshold` | `0.75`; target-class probability required for automatic validation. |
| `--candidate-threshold` | `0`; lower probability bound for review candidates. `0` disables the review range. |
| `--out PATH` | Select the working directory instead of using `<video_stem>` beside the input video. |
| `--scene-threshold` | `10`; FFmpeg scene-change detection sensitivity. |
| `--width` | `320`; thumbnail width. The native ResNet engine expects 320x180 JPEGs. |
| `--offset SECONDS` | Manually override the timestamp offset used during scene extraction. |
| `--skip-scdet` | Reuse thumbnails already present in an existing working directory and rerun classification. |
| `--csv-predict` | Also write `scene_predictions.csv`. |
| `--resnet-cpu` | Use TensorFlow ResNet50V2 on the CPU instead of the RK3576 NPU. TensorFlow must be installed in the active environment. |
| `--batch-size N` | Default `4`. Selects the matching RKNN model when `--extractor` is omitted. With `--resnet-cpu`, it controls the CPU inference batch size. |
| `--extractor PATH` | Explicit RKNN model path. When supplied, the batch size is inferred from its filename and overrides `--batch-size`. A filename without `_batchN` means batch 1. Ignored with `--resnet-cpu`. |
| `--update-size N` | Default `8` inference calls between progress updates. With batch 4, this reports progress every 32 images. |

The accepted RKNN batch sizes are `1` or positive even integers (`2`, `4`, `6`, and so on). Without `--extractor`, `--batch-size` chooses the corresponding filename in `resnet/models`. If an explicit extractor path is supplied, its filename is authoritative: `_batch8` means batch 8, while a filename with no `_batchN` suffix means batch 1.

When using `--skip-scdet`, an existing working directory and its `scenes_all_other/` thumbnail directory are required. If no input video is supplied, `--out` is required. The script restores images from `scenes_validated/` and `scenes_to_review/` so that they can all be classified again. A valid `source_link.<extension>` is still required later if the results are to be encoded.

### 7. Encode the selected scenes

Pass the processing directory to `encode_segments.py`:

```bash
python encode_segments.py \
  --qp 20 --tier high --level 5.1 \
  "/path/to/my_video"
```

`--qp`, `--tier`, and `--level` are required for encoding. The example values are the quality/file-size settings used in this project; adjust them for your own material. The script reads the selected intervals from `scenes_validated/`, uses `source_link.<extension>` to access the source recording, creates individual clips in `encoded_seg/`, and muxes them into `output_video.mkv`.

Existing encoded clips are skipped unless `--force-encode` is specified. Useful additional options include:

| Option | Behavior |
|---|---|
| `--skip-mux` | Encode missing clips but do not create/update `output_video.mkv`. |
| `--just-mux` | Skip encoding and mux all existing timestamp-named MP4 files in `encoded_seg/`. Does not require `scenes_validated/` or `source_link.*`. |
| `--force-encode` | Re-encode and overwrite every selected clip. |
| `--decode-mode auto\|hardware\|software` | Default `auto`: try Rockchip MPP decoding and fall back to software decoding if hardware decoding fails. `hardware` disables fallback; `software` forces software decoding. HEVC output encoding remains hardware-accelerated. |
| `--scan-type auto\|detect\|progressive\|interlaced` | Default `auto` trusts explicit `ffprobe` scan metadata and errors if it is unknown. Use `detect` for the slower FFmpeg `idet` scan, or specify the known source type. |
| `--edge-trim SECONDS` | Default `0.060`; trim this amount from both ends of the resulting segment. |
| `--seek-back SECONDS` | Default `5.0`; seek before a segment start to improve accurate trimming. |
| `--sb` / `--se` | Define a source-time correction range. |
| `--offset SECONDS` | Apply a signed source-time shift to video and audio boundaries inside the correction range. |
| `--audio-delay MS` | Apply an audio-only synchronization correction inside the correction range. Positive values delay audio; negative values advance it. |

For the complete option list, run `python encode_segments.py --help`.

### Alternative: run the two-step wrapper

`process-and-encode.sh` runs scene processing and encoding as a guided workflow. Before using it, edit the configuration block near the top of the script, especially:

- `TARGET_CLASS`
- `PROJECT_DIR`
- `CANDIDATE_THRESHOLD`
- `RESNET_BATCH_SIZE` and `RESNET_UPDATE_SIZE`
- `ENCODER_QP`, `ENCODER_TIER`, and `ENCODER_LEVEL`

Then run it from `nanopi/`:

```bash
./process-and-encode.sh "my_video.ts"
```

The wrapper resolves a filename relative to `PROJECT_DIR`, or accepts an absolute input path. It pauses after a new classification run so the images can be reviewed before encoding. If a completed processing directory already exists (with `scenes_validated/` and `scene_predictions.xlsx`), it skips scene processing and proceeds to encoding. **If the processing directory exists but does not meet that completion check, the wrapper removes its contents before starting a new run.** Keep any files you need before reusing an incomplete directory.

The wrapper builds the extractor filename from `RESNET_BATCH_SIZE`, so the matching RKNN model must be present in `resnet/models/`.

## Training or updating the presenter classifier

The dataset uses one subdirectory per class. The directory name is the class label, and the training script reads `.jpg` and `.jpeg` images:

```text
resnet/dataset/
├── andreea-esca/
│   ├── image_001.jpg
│   └── ...
└── other-presenter/
    ├── image_001.jpg
    └── ...
```

At least two non-empty classes are required, with at least two JPEG images in each class. More numerous and varied examples are strongly recommended.

Run training from `nanopi/` with the NanoPi virtual environment active:

```bash
cd ~/NanoPi-M5-Video-Encoder/nanopi
source .venv-resnet/bin/activate
python train_classifier.py
```

The default backend extracts features with the RK3576 NPU using the batch-4 RKNN model. You can select another installed batch variant with `--batch-size` or provide an explicit model with `--extractor`:

```bash
python train_classifier.py --batch-size 8
python train_classifier.py --extractor ../resnet/models/resnet50v2_feature_extractor_rk3576_fp16_320x180_batch2.rknn
```

As with `process_video.py`, an explicit `--extractor` takes precedence and its filename determines the batch size. A filename without `_batchN` means batch 1. `--use-cpu` selects TensorFlow/CPU feature extraction and ignores the RKNN extractor and batch selection; TensorFlow must be available in the environment.

The training pipeline keeps ResNet50V2 frozen and trains a separate multiclass Logistic Regression classifier on its 2048-dimensional feature vectors. `class_weight="balanced"` compensates for differences in the number of images per class. The 80/20 stratified split is used for preliminary evaluation only; the final classifier is fitted again using the complete dataset.

The script writes:

- Feature cache: `resnet/artifacts/resnet50v2_classifier_cache.npz`
- Evaluation report: `resnet/artifacts/resnet50v2_classifier_training_report.txt`
- Final classifier: `resnet/models/resnet50v2_logistic_classifier.npz`
- Previous classifier backup: `resnet/models/resnet50v2_logistic_classifier.npz.bak`

The feature cache is keyed by backend and fixed RKNN batch variant. It also checks each image's relative path, file size, and modification time. Changing the RKNN batch size therefore causes the script to rebuild the feature cache for that variant rather than mixing features extracted by different variants.

The evaluation split is random at image level. Related frames from the same recording may therefore appear in both the training and evaluation subsets, making the reported score optimistic. For a reliable estimate of generalization, evaluate on recordings that were not used to build the training dataset.

## Updating the RKNN feature-extractor model from PC/WSL

This optional workflow regenerates the frozen ResNet50V2 feature extractor and converts it for the RK3576. Run it from WSL Ubuntu on an x86-64 PC.

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

The optional scripts in `windows/` can launch the NanoPi workflow from Windows when the source recordings are available through a mapped NAS drive.

### Configure SSH aliases

Run `windows/install-ssh-alias.bat` with six arguments:

```cmd
install-ssh-alias.bat <nanopi_user> <nanopi_host_ip> <nanopi_key> <nas_user> <nas_host_ip> <nas_key>
```

Private-key filenames must be located in `%USERPROFILE%\.ssh\`. Use `-` for a key argument when no dedicated private key is required. The script installs the aliases `rkvenc-nanopi` and `rkvenc-nas`; an alias that already exists is left unchanged.

### Launch processing from Windows

`windows/process-and-encode.bat` accepts one video path on the mapped `N:` drive:

```cmd
process-and-encode.bat "N:\WDRed2\TV Rec\News\recording.ts"
```

The launcher maps `N:\...` to `/NAS/...` and runs `process-and-encode.sh` remotely through the `rkvenc-nanopi` SSH alias. The NAS path must also be mounted under `/NAS` on the NanoPi. The current BAT file contains the NanoPi checkout path `/home/marus/NanoPi-M5-Video-Encoder/nanopi`; edit that path if your repository is installed elsewhere.

### Add or remove an Explorer context-menu entry

Run the relevant script with a file extension, with or without the leading dot:

```cmd
add-menu-shortcut.bat ts
remove-menu-shortcut.bat ts
```

The context-menu command invokes `process-and-encode.bat` for the selected file.

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
- Each normal processing run creates a new working directory. If it already exists and is non-empty, `process_video.py` refuses to overwrite it. The wrapper has its own cleanup behavior for incomplete directories; see the wrapper notes above.

## Marus Non-Commercial License v1.0

This project is free for personal and hobby use only.  
Commercial use requires a paid license from the author.  
Third-party libraries keep their own licenses.
