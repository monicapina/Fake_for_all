# 🎭 Fake_for_all

![Sample Frame Examples](1751303029659659.jpg)

## 📁 Dataset Overview

This dataset was constructed using a large-scale face swapping pipeline based on [Roop](https://github.com/s0md3v/roop), which systematically combines face images with target videos. All synthetic videos were generated following the methodology illustrated in the [pipeline diagram](pipeline_dataset.png), ensuring consistency, reproducibility, and demographic balance.

### 📊 Dataset Composition

The final dataset consists of:

- **188,487 manipulated (synthetic) videos**
- **22,067 real (original) videos**

A key objective during the dataset design was to achieve **broad demographic coverage** across five major ethnic groups:

- *Afrodescendant*
- *Asian*
- *Caucasian*
- *Latino-Hispanic*
- *Middle Eastern–North African*

Moreover, the dataset maintains a **near-equal gender distribution**, enabling fair evaluation and training of models across gender and ethnicity. This balance supports research in areas such as deepfake detection, fairness in computer vision, and synthetic data generation.

---

## ⚙️ Installation

### 📋 Prerequisites

Ensure that you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed before proceeding.

### 📦 Environment Setup

No need to create a new environment. Simply install the required dependencies in your existing environment:

```bash
pip install -r requirements.txt
```

If you're using the Real-ESRGAN module, follow the extra instructions below.

### ⚡ Optional Enhancements

- **GPU Acceleration**: Install appropriate CUDA and cuDNN libraries. See [CUDA Installation Guide](https://docs.nvidia.com/cuda/).
- **YouTube Integration**: For video downloading, insert your [YouTube Data API Key](https://developers.google.com/youtube/v3/getting-started) in `video_downloader.py`.

---

## 🚀 Core Usage

![Pipeline](pipeline_dataset.png)

## ▶️ Download Videos from YouTube

```bash
python video_downloader.py
```

## ✂️ Extract Face-Based Segments from Videos

```bash
python video_segment_extractor.py --input /path/to/videos --output /path/to/output
```


## 🔼 Upscaling Faces with Real-ESRGAN

This step enhances the visual quality of extracted face crops or final results using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

### 📥 Repository Setup

  1. Clone the official Real-ESRGAN repository:

  ```bash
  git clone https://github.com/xinntao/Real-ESRGAN.git
  cd Real-ESRGAN
  ```

  2. Install the required dependencies (in your current environment):

  ```bash
  pip install -r requirements.txt
  python setup.py develop
  ```

  3. Copy the `upscale_frames.py` script into the Real-ESRGAN folder:

  ```bash
  cp /path/to/upscale_frames.py ./upscale_frames.py
  ```

### 🚀 Run the Upscaler

Upscale a folder of images:

  ```bash
  python upscale_frames.py --input /path/to/images --output /path/to/output --gpus 0
  ```

> ℹ️ The script will automatically create the output folder if it does not exist.

## 🚻 Gender Classification in Video Segments

```bash
python video_gender_classifier.py --input /path/to/videos --output /path/to/output --result_csv gender_results.csv
```

---

## 🧑‍🔬 Face Swapping with Roop

Use the `roop_generator.py` script to generate synthetic videos by replacing faces in existing footage.

### 🗂️ Directory Requirements

Prepare the following:

- `source_path`: folder with face images
- `videos_path`: folder with target videos
- `output_path`: folder to store generated videos

Also configure a `config.json`:

```json
{
  "execution_provider": ["cuda"],
  "temp_folder": "./temp",
  "log_level": "info",
  "execution_threads": 4
}
```

### ⚙️ Run Face Swapping

```bash
python roop_generator.py   --gpu 0   --source_path /path/to/images   --videos_path /path/to/videos   --output_path /path/to/output
```

Each image-video pair will produce a new video with the swapped face. Processing is checkpointed for recovery.

### 💡 Tips

- Output filenames combine the source image and target video names.
- Videos with <5 valid frames are skipped.
- Audio is restored unless `skip_audio` is set.

### 📌 Post-processing

After face swapping, run the metadata extraction step again:

```bash
python media_metadata_extractor.py --root /path/to/output --output /path/to/metadata.json.gz
```

---


