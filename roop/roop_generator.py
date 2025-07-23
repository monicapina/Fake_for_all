import os
import sys
import json
import argparse
import shutil
import time
import subprocess
import torch
import onnxruntime as ort
import cv2

# Roop-specific imports (assumes roop is cloned in the same directory)
from roop.core import decode_execution_providers
import roop.globals
from roop.utilities import (
    has_image_extension, is_video, detect_fps, create_video, extract_frames,
    get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, get_temp_directory_path
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.face_analyser import get_one_face

# Suppress ONNX Runtime logs
ort.set_default_logger_severity(3)

CONFIG_FILE = "config.json"


def configure_environment(cuda_device):
    """Selects the GPU to be used by setting CUDA_VISIBLE_DEVICES."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)


def load_config(gpu_id):
    """Loads configuration parameters from a JSON file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"WARNING: {CONFIG_FILE} not found, using default values.")
        return

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    print(f"Loading config: {config}")

    for key, value in config.items():
        setattr(roop.globals, key, value)

    roop.globals.execution_providers = decode_execution_providers(config.get("execution_provider", ["cuda"]))
    roop.globals.log_level = config.get("log_level", "error")
    roop.globals.execution_threads = config.get("execution_threads", 4)
    roop.globals.temp_frame_format = config.get("temp_frame_format", "png")
    roop.globals.temp_frame_quality = config.get("temp_frame_quality", 80)

    base_temp_path = config.get("temp_folder", "./temp")
    roop.globals.temp_path = os.path.join(base_temp_path, f"gpu{gpu_id}")
    os.makedirs(roop.globals.temp_path, exist_ok=True)
    print(f"✅ Temp path for GPU {gpu_id}: {roop.globals.temp_path}")


def configure_globals(source_path, videos_path, output_path):
    """Sets global paths for source images, videos and output directory."""
    roop.globals.source_path = source_path
    roop.globals.videos_paths = [videos_path]
    roop.globals.output_path = output_path


def get_relative_path(base_path, full_path):
    return os.path.relpath(full_path, base_path)


def get_all_videos():
    """Recursively retrieves all video files from the specified paths."""
    all_videos = []
    for video_path in roop.globals.videos_paths:
        if os.path.exists(video_path):
            for root, _, files in os.walk(video_path):
                for vid in files:
                    full_path = os.path.join(root, vid)
                    if is_video(full_path):
                        all_videos.append(full_path)
    return all_videos


def save_checkpoint(image_index, video_index, checkpoint_file):
    with open(checkpoint_file, "w") as f:
        json.dump({"image_index": image_index, "video_index": video_index}, f)


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"image_index": 0, "video_index": 0}


def delete_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def safe_clean_temp(temp_directory_path):
    """Attempts to remove a temporary directory safely."""
    try:
        shutil.rmtree(temp_directory_path)
    except OSError as e:
        print(f"⚠️ Could not delete {temp_directory_path}: {e}")


def process_image_with_video(image_path, video_path, output_base_path, image_index, video_index, checkpoint_file):
    """Processes a single image and applies it to a video."""
    start_time = time.time()
    relative_image_path = get_relative_path(roop.globals.source_path, os.path.dirname(image_path))
    output_folder = os.path.join(output_base_path, relative_image_path)
    os.makedirs(output_folder, exist_ok=True)
    output_video_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{os.path.basename(video_path)}")

    if os.path.exists(output_video_path):
        print(f"✅ Already processed: {output_video_path}")
        return

    print(f"🔄 Processing {image_path} with {video_path}")
    create_temp(video_path)
    fps = detect_fps(video_path) if roop.globals.keep_fps else 30
    extract_frames(video_path, fps)
    temp_frame_paths = get_temp_frame_paths(video_path)

    valid_frame_paths = [p for p in temp_frame_paths if cv2.imread(p) is not None]

    if len(valid_frame_paths) < 5:
        print(f"❌ Too few valid frames ({len(valid_frame_paths)}) for {video_path}, skipping.")
        safe_clean_temp(get_temp_directory_path(video_path))
        return

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        frame_processor.process_video(image_path, valid_frame_paths)
        frame_processor.post_process()

    create_video(video_path, fps)
    if roop.globals.skip_audio:
        move_temp(video_path, output_video_path)
    else:
        restore_audio(video_path, output_video_path)

    safe_clean_temp(get_temp_directory_path(video_path))

    if os.path.exists(output_video_path):
        print(f"✅ Successfully processed: {output_video_path}")
    else:
        print("❌ ERROR: Output video not generated.")

    end_time = time.time()
    print(f"⏳ Time: {output_video_path} - {end_time - start_time:.2f}s")
    save_checkpoint(image_index, video_index, checkpoint_file)


def start_batch_processing(checkpoint_file):
    """Runs batch processing for all image-video combinations."""
    source_images = [
        os.path.join(root, img)
        for root, _, files in os.walk(roop.globals.source_path)
        for img in files if has_image_extension(img)
    ]

    if not source_images:
        print("❌ No source images found.")
        return

    videos = get_all_videos()
    checkpoint = load_checkpoint(checkpoint_file)
    image_index, video_index = checkpoint["image_index"], checkpoint["video_index"]

    for i, image_path in enumerate(list(reversed(source_images))[image_index:], start=image_index):
        print(f"\n📸 Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None or get_one_face(image) is None:
            print(f"❌ No detectable face in {image_path}, skipping.")
            continue

        for j, video_path in enumerate(videos[video_index:], start=video_index):
            try:
                process_image_with_video(image_path, video_path, roop.globals.output_path, i, j, checkpoint_file)
            except KeyboardInterrupt:
                print("\n🛑 Interrupted. Saving checkpoint...")
                save_checkpoint(i, j, checkpoint_file)
                sys.exit()

        video_index = 0

    delete_checkpoint(checkpoint_file)


def main():
    parser = argparse.ArgumentParser(description="Batch face-swapping processor using Roop")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--source_path", required=True, help="Folder with source face images")
    parser.add_argument("--videos_path", required=True, help="Folder with target videos")
    parser.add_argument("--output_path", required=True, help="Output folder for results")
    args = parser.parse_args()

    checkpoint_file = f"checkpoint_gpu{args.gpu}.json"
    configure_environment(args.gpu)
    load_config(args.gpu)
    configure_globals(args.source_path, args.videos_path, args.output_path)

    print(f"🔍 GPU {args.gpu} ready | Checkpoint: {checkpoint_file}")
    print(f"📁 Source: {args.source_path}")
    print(f"🎥 Videos: {args.videos_path}")
    print(f"📤 Output: {args.output_path}")
    print(f"📝 Checkpoint: {checkpoint_file}")

    start_batch_processing(checkpoint_file)


if __name__ == "__main__":
    main()
