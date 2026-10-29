# NanoPi M5 Video Encoder

This project can detect scene changes in a video, classifies the resulting scene thumbnails with a trained ResNet-based classifier, and encodes the selected scenes into one compact video using the Rockchip RK3576 hardware HEVC encoder.

## Why this project exists

The motivation is deliberately specific: I keep a personal collection of Romania's most sexy and beautiful TV news presenters. Building that collection the traditional way was a lot of work: I had to find the exact timecodes for each appearance, cut the scenes manually, and encode the clips in software on a PC. That took a great deal of time, and keeping entire news recordings just to save a few short appearances wasted a lot of storage space.

This project automates that process, selecting the scenes I want and combining them into a much smaller video using the RK3576's hardware encoder, which does a better job than any other hardware encoder I’ve tried on PC from Intel or AMD. With "HEVC 8bit QP:20 high 5.1", the image quality is incredibly good, almost identical to the source, and the file size is decent.

The current classifier is configured to select **Andreea Esca**, a Romanian TV news presenter. The training workflow supports adding more classes, so other presenters can be included as the dataset is expanded. In other words, this is a video-encoding project with a very particular fan-driven use case. 🙂

## How it works

1. **Detect scenes.** FFmpeg finds scene changes and extracts a thumbnail for each candidate scene.
2. **Classify thumbnails.** A ResNet50V2 feature extractor runs on the RK3576 NPU. A separately trained classifier determines which scenes match the requested presenter.
3. **Review uncertain results.** The script creates a review folder and an Excel report so borderline detections can be checked before encoding.
4. **Encode selected scenes.** FFmpeg uses the Rockchip MPP/RGA-enabled build to process the selected time ranges and encode the output using hardware HEVC.
5. **Create one video.** The encoded segments are muxed into a single `output_video.mkv` file.

The processing and encoding steps can be run separately, giving the user a chance to review the results before spending time on encoding.

## Requirements

### NanoPi

- FriendlyElec **NanoPi M5 / Rockchip RK3576** running Ubuntu on ARM64.
- A working RKNN Lite 2 runtime and the Rockchip system drivers required by the board's NPU and video hardware.
- The custom FFmpeg stack built by `ffmpeg/setup-ffmpeg.sh`.
- `pyenv` with Python **3.12.x** installed for the NanoPi Python environment.
- Network access to download the model files and, when needed, the dataset.

### PC for model conversion (optional)

- Windows with **WSL2 and Ubuntu** on x86-64.
- `pyenv` with Python **3.11.x** installed.
- SSH/SCP connectivity to the NanoPi if you want `build.sh` to transfer the generated model and validation data automatically.

You only need the PC/WSL workflow when exporting or updating the RKNN feature-extractor model. It is not required for normal video processing if the required models are already available.

## Repository layout

| Directory | Purpose |
|---|---|
| `nanopi/` | Main working area on the NanoPi: video processing, segment encoding, classifier training, model validation, and helper tools. |
| `pc-wsl/` | Optional PC/WSL workflow that exports the Keras ResNet50V2 model to ONNX and converts it to RKNN for the RK3576. |
| `resnet/` | Shared location for the model files, dataset, and temporary build/validation artifacts. These files are downloaded or generated locally and are not stored in Git. |
| `ffmpeg/` | `setup-ffmpeg.sh`, which builds and installs the Rockchip-enabled MPP, RGA, and FFmpeg stack. Downloaded sources and build files are ignored by Git. |
| `windows/` | Optional Windows helper for configuring the `nanopi-encoder` SSH alias used by the source-file cleanup workflow. |

The repository's `.gitignore` intentionally excludes generated build files, virtual environments, datasets, model binaries, and artifacts. Only the scripts/configuration needed to reproduce the workflow belong in Git.

## Getting started on the NanoPi

Run these steps from the cloned repository on the NanoPi. The commands below use the default paths expected by the processing scripts.

### 1. Clone the repository

```bash
git clone https://github.com/MarusGradinaru/NanoPi-M5-Video-Encoder.git
cd NanoPi-M5-Video-Encoder
```

### 2. Build and install the custom FFmpeg stack

```bash
cd ffmpeg
./setup-ffmpeg.sh
```

The script installs the required build tools using `sudo`, downloads the pinned source revisions, builds Rockchip MPP and RGA, then builds and verifies FFmpeg. The default installation prefix is `/opt/rkmpp-stack`, which is the path expected by the encoding scripts.

The number of parallel build jobs can be changed if necessary:

```bash
./setup-ffmpeg.sh --jobs 4
```

Use the default installation prefix unless you are also prepared to configure the processing scripts to use a custom FFmpeg/FFprobe path. For other options, run `./setup-ffmpeg.sh --help`.

### 3. Create the NanoPi Python environment

```bash
cd ../nanopi
./setup_venv_resnet.sh
source .venv-resnet/bin/activate
```

The setup script checks the environment and installs the Python packages used by the NanoPi tools. It creates `.venv-resnet` with Python 3.12 and can be rerun after correcting an installation error.

### 4. Download the ResNet resources

The repository does not contain the model binaries or dataset. They are stored in the project's [Hugging Face bucket](https://huggingface.co/buckets/MarusGradinaru/romanian-tv-resnet-classifier).

With the NanoPi virtual environment activated, run the downloader from the `resnet/` directory:

```bash
cd ../resnet
./get-resnet.sh models
```

This downloads the model files needed for normal processing. If you also need the training dataset, run:

```bash
./get-resnet.sh dataset
```

Or request both folders at once:

```bash
./get-resnet.sh models dataset
```

**Important:** `get-resnet.sh` synchronizes the selected folders with the remote bucket using `--delete`. Extra local files inside a folder being synchronized may be deleted if they are not present in the bucket. Keep custom files elsewhere or back them up before syncing.

### 5. Process a video

Return to `nanopi/` and run the scene-processing script. Replace the example path with the location of your recording:

```bash
cd ../nanopi
python process_video.py \
  --target-class andreea-esca \
  --candidate-threshold 0.4 \
  "/path/to/my_video.ts"
```

By default, the script creates a working directory named after the input video, next to the input file. For example, processing `/recordings/my_video.ts` creates `/recordings/my_video/`. The destination must not already contain files; use a different output path with `--out` or move the previous result before starting again.

The working directory contains the extracted scene images, a `scenes_validated/` folder for scenes accepted by the classifier, a `scenes_to_review/` folder for borderline candidates, an Excel report named `scene_predictions.xlsx`, and batch/bash helper scripts to delete the associated video source file and its symlink.

The default target probability threshold is `0.75`. With `--candidate-threshold 0.4`, scenes with lower confidence that fall into the review range are kept in `scenes_to_review/` rather than automatically selected. Inspect `scene_predictions.xlsx` and the review images. If a candidate should be included, move its image into `scenes_validated/` before encoding. Only scenes in `scenes_validated/` are used by the encoder.

### 6. Encode the selected scenes

Pass the working directory created in the previous step to `encode_segments.py`:

```bash
python encode_segments.py "/path/to/my_video" \
  --qp 20 \
  --tier high \
  --level 5.1
```

The QP, tier, and level above are the values ​​for the best quality/file size ratio that I found; choose values appropriate for your use case. The script creates individual encoded clips in `encoded_seg/` and combines them into `output_video.mkv` inside the processing directory.

### Alternative: use the two-step wrapper

`process-and-encode.sh` runs processing and encoding as one guided workflow, pausing after scene classification so you can review the results before continuing. Before using it, edit the configuration constants near the top of the script, especially `PROJECT_DIR`, `TARGET_CLASS`, `CANDIDATE_THRESHOLD`, `ENCODER_QP`, `ENCODER_TIER`, and `ENCODER_LEVEL`.

```bash
./process-and-encode.sh "my_video.ts"
```

This wrapper is convenient when processing many recordings with the same settings. For a one-off run or more flexible options, use `process_video.py` and `encode_segments.py` directly.

## Training or updating the presenter classifier

If you add images, remove images, or introduce another presenter class, update the dataset under `resnet/dataset/`. Use one subdirectory per class, with the directory name serving as the class label. The training script requires at least two non-empty classes, with at least two usable images in each class; a larger and more varied dataset is strongly recommended.

Run the training script from the repository's `nanopi/` directory with the NanoPi virtual environment active:

```bash
cd /path/to/NanoPi-M5-Video-Encoder/nanopi
source .venv-resnet/bin/activate
python train_classifier.py
```

By default, feature extraction uses the RK3576 NPU and the installed RKNN feature-extractor model. The resulting classifier parameters are saved under `resnet/models/`. This is separate from converting the ResNet feature-extractor model itself.

## Updating the RKNN feature-extractor model from PC/WSL

This workflow is optional. Use it when you need to export or regenerate the ResNet50V2 feature-extractor model for the RK3576.

1. On WSL Ubuntu, prepare the conversion environment:

   ```bash
   cd /path/to/NanoPi-M5-Video-Encoder/pc-wsl
   ./setup_venv_resnet.sh
   source .venv-resnet/bin/activate
   ```

2. For validation against real images, download the dataset in the WSL checkout too. The dataset is optional for export, but is used automatically when it is available:

   ```bash
   cd ../resnet
   ./get-resnet.sh dataset
   cd ../pc-wsl
   ```

3. Run the automated build:

   ```bash
   ./build.sh
   ```

   `build.sh` checks the pinned environment and runs the conversion stages: Keras ResNet50V2 to ONNX, NHWC-to-NCHW layout conversion, then ONNX to RKNN. It also produces validation inputs and ONNX reference features.

4. When prompted, you can transfer the RKNN model and validation data to the NanoPi using SCP. The default remote destination is `~/NanoPi-M5-Video-Encoder/resnet/artifacts/`. Ensure that directory exists on the NanoPi before the first transfer:

   ```bash
   mkdir -p ~/NanoPi-M5-Video-Encoder/resnet/artifacts
   ```

   If you use an SSH alias for this transfer, configure it in WSL's SSH configuration (`~/.ssh/config`). The Windows helper only configures the Windows-side alias used by generated `.bat` launchers.

5. On the NanoPi, validate the candidate model from the repository's `nanopi/` directory:

   ```bash
   cd /path/to/NanoPi-M5-Video-Encoder/nanopi
   source .venv-resnet/bin/activate
   python validate_rknn_export.py
   ```

   The script compares RK3576 NPU outputs with the ONNX reference features. It promotes the candidate model from `resnet/artifacts/` to `resnet/models/` only when validation passes.

## Utility scripts

- **`nanopi/adj-audio.sh`** — adjust audio/video synchronization without re-encoding. It accepts a signed offset in milliseconds; a positive value delays audio, while a negative value delays video instead.
- **`nanopi/mux-seg.sh`** — combine timestamp-named MP4 segments from one folder or from several folders. This is useful when a long recording was split into multiple source files: process each part separately with `encode_segments.py --skip-mux`, then mux the resulting `encoded_seg/` folders in the required order.
- **`nanopi/validate_rknn_export.py`** — validate a newly exported RKNN feature extractor on the NanoPi before promoting it to the active model directory.
- **`nanopi/train_classifier.py`** — retrain the lightweight presenter classifier from the local dataset.
- **`windows/` helper** — configure the Windows SSH alias `nanopi-encoder` used by generated cleanup helpers when recordings are accessed from Windows and stored on a NAS.

For the complete list of command-line options for any tool, use its `--help` option where available. 

## Notes and limitations

- The included classifier is currently configured for `andreea-esca`; it will not automatically recognize a presenter that has not been represented by the trained classes.
- Classification quality depends on the training examples and the selected confidence thresholds. Review borderline scenes before encoding, especially after changing the dataset or classifier.
- Hardware acceleration depends on a compatible NanoPi M5/RK3576 software stack and the custom FFmpeg build. The scripts do not make an incompatible kernel/driver stack compatible by themselves.
- Processing creates a new working directory for each input video. If it already exists and is not empty, choose a different output directory or move the old results first.

## Marus Non-Commercial License v1.0

This project is free for personal and hobby use only.  
Commercial use requires a paid license from the author.  
Third-party libraries keep their own licenses.