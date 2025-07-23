# Fake_for_all


## Overview
This project provides tools for processing videos, detecting faces, extracting metadata from media files, and more. It supports GPU acceleration and can be used to analyze videos frame-by-frame for tasks like gender classification, face segment extraction, and more.

## Installation

### Prerequisites

Make sure you have Conda installed. You can download Conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

### Setting up the Environment

Follow these steps to set up a Conda environment and install the dependencies for this project:

1. **Create a Conda Environment**:
   Open a terminal and create a new Conda environment with Python 3.8 (or your preferred version):

   ```bash
   conda create -n your_env_name python=3.8

2. **Activate the Environment**:
Activate the environment you just created:
    ```bash
    conda activate your_env_name
    
3. **Install Dependencies**:
Download the requirements.txt file (it should be included in the project directory), and use pip to install all required libraries. This will install all necessary dependencies, including packages for video processing, machine learning, and face detection:

```bash
pip install -r requirements.txt
```




### **Additional Setup (Optional)**

- **CUDA & GPU Support**: For GPU acceleration, ensure that you have the necessary CUDA and cuDNN libraries installed for your system. You can follow the official [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/) for more information.

- **YouTube API**: If you're working with YouTube video downloads, make sure to add your own [YouTube Data API Key](https://developers.google.com/youtube/v3/getting-started) in the relevant script file (`video_downloader.py`).

## **Usage**

Once you have the environment set up, you can run the scripts directly.

To download videos from YouTube:
```bash
python video_downloader.py
```

To process videos and extract face-based segments:
```bash
python video_segment_extractor.py --input /path/to/your/videos --output /path/to/output/folder
```
To classify gender in videos:
```bash
python video_gender_classifier.py --input /path/to/videos --output /path/to/output/folder --result_csv gender_results.csv
```

To extract metadata from videos and images:
```bash
python media_metadata_extractor.py --root /path/to/media/folder --output /path/to/output/metadata.json.gz
```















