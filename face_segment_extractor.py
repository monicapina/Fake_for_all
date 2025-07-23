"""
General-purpose script to process videos by detecting consistent face appearances.
It extracts video segments where the same face appears for a configurable number of frames.
Supports interruption handling and GPU acceleration via InsightFace, TensorFlow, and PyTorch.
"""

import os
import cv2
import json
import torch
import signal
import logging
import warnings
import argparse
import subprocess
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# Suppress future warnings from InsightFace
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface.utils.transform')

# Global variables used to track current progress
current_video = None
current_frame = 0
progress_file = ""

# --- Signal Handling (Safe interruption) ---
def handle_interrupt(signum, frame):
    logging.warning("⚠ Interruption detected! Saving progress and exiting...")
    if current_video:
        save_progress(current_video, current_frame)
    exit(1)

signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

# --- Save and Load Progress ---
def save_progress(video_path, frame_number):
    """Saves the last processed video and frame to a progress file."""
    progress = {"last_video": video_path, "last_frame": frame_number}
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def load_progress():
    """Loads the last saved video and frame position."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return progress.get("last_video", None), progress.get("last_frame", 0)
    return None, 0

# --- Face and Embedding Processing ---
def compare_embeddings(prev_embedding, current_embedding, l2_threshold=1.2):
    """Compares two embeddings using cosine similarity and L2 distance."""
    cosine_similarity = 1 - cosine(prev_embedding, current_embedding)
    l2_distance = np.linalg.norm(prev_embedding - current_embedding)

    print(f"🔹 Cosine similarity: {cosine_similarity:.4f}")
    print(f"🔹 L2 distance: {l2_distance:.4f}")

    return l2_distance < l2_threshold

def detect_faces_and_embeddings(frame):
    """Detects the largest face in the frame and extracts its embedding."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_frame)

    if not faces:
        return None, None, None

    # Select the largest face based on bounding box area
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    x, y, w, h = map(int, face.bbox)

    # Add margin to the bounding box
    margin = int(0.2 * max(w, h))
    x, y = max(0, x - margin), max(0, y - margin)
    w, h = min(frame.shape[1] - x, w + 2 * margin), min(frame.shape[0] - y, h + 2 * margin)

    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        return None, None, None

    return face_crop, face.normed_embedding, (x, y, w, h)

# --- Video Segment Extraction ---
def extract_video_segment(input_path, output_path, start_time, end_time):
    """Uses ffmpeg to extract a video segment between start_time and end_time."""
    duration = end_time - start_time
    if duration < 2:
        logging.warning(f"⚠ Segment discarded (too short): {duration:.2f} seconds")
        return

    command = [
        'ffmpeg', '-y', '-loglevel', 'info', '-ss', str(start_time), '-i', input_path,
        '-t', str(duration), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-strict', 'experimental', output_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0 and os.path.exists(output_path):
        logging.info(f"✅ Segment saved: {output_path}")
    else:
        logging.error(f"❌ Failed to generate file: {output_path}")

# --- Main Video Processing ---
def process_video(video_path, output_folder, identity_threshold=0.5, min_video_duration=2.0, frames_threshold=2):
    """Processes a video, detects faces, and extracts segments based on identity consistency."""
    global current_video, current_frame
    logging.info(f"📹 Processing video: {video_path}")

    current_video = video_path
    current_frame = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"❌ Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    _, start_frame = load_progress()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_embedding = None
    same_face_counter = 0
    start_time = None
    fragment_count = 0

    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, os.path.dirname(os.path.relpath(video_path, input_folder)))
    os.makedirs(video_output_folder, exist_ok=True)

    min_consecutive_frames = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        face_crop, embedding, bbox = detect_faces_and_embeddings(frame)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if embedding is not None:
            if prev_embedding is not None:
                l2_distance = np.linalg.norm(np.asarray(prev_embedding) - np.asarray(embedding))
                same_face_counter = same_face_counter + 1 if l2_distance < identity_threshold else 0
            else:
                same_face_counter = 1

            prev_embedding = embedding

            if same_face_counter >= min_consecutive_frames:
                if start_time is None:
                    start_time = current_time
                    logging.info(f"🟢 Segment started at {start_time:.2f}s")
            else:
                if start_time is not None:
                    segment_duration = current_time - start_time
                    if segment_duration >= min_video_duration:
                        output_path = os.path.join(video_output_folder, f"{video_base_name}_fragment_{fragment_count}.mp4")
                        extract_video_segment(video_path, output_path, start_time, current_time)
                        fragment_count += 1
                    start_time = None
        else:
            if start_time is not None:
                segment_duration = current_time - start_time
                if segment_duration >= min_video_duration:
                    output_path = os.path.join(video_output_folder, f"{video_base_name}_fragment_{fragment_count}.mp4")
                    extract_video_segment(video_path, output_path, start_time, current_time)
                    fragment_count += 1
                start_time = None
            prev_embedding = None
            same_face_counter = 0

        save_progress(video_path, current_frame)

    cap.release()
    logging.info(f"✅ Finished processing: {video_path}")

# --- Process All Videos in Folder ---
def process_videos_in_folder(input_folder, output_folder):
    last_video, last_frame = load_progress()
    skip = last_video is not None

    for root, _, files in os.walk(input_folder):
        for file in sorted(files):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                if skip:
                    if video_path == last_video:
                        skip = False
                    logging.info(f"⏭ Skipping video: {video_path}")
                    continue
                process_video(video_path, output_folder)
                save_progress(video_path, 0)

"""
General-purpose script to process videos by detecting consistent face appearances.
It extracts video segments where the same face appears for a configurable number of frames.
Supports interruption handling and GPU acceleration via InsightFace, TensorFlow, and PyTorch.
"""

import os
import cv2
import json
import torch
import signal
import logging
import warnings
import argparse
import subprocess
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# Suppress future warnings from InsightFace
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface.utils.transform')

# Global variables used to track current progress
current_video = None
current_frame = 0
progress_file = ""

# --- Signal Handling (Safe interruption) ---
def handle_interrupt(signum, frame):
    logging.warning("⚠ Interruption detected! Saving progress and exiting...")
    if current_video:
        save_progress(current_video, current_frame)
    exit(1)

signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

# --- Save and Load Progress ---
def save_progress(video_path, frame_number):
    """Saves the last processed video and frame to a progress file."""
    progress = {"last_video": video_path, "last_frame": frame_number}
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def load_progress():
    """Loads the last saved video and frame position."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return progress.get("last_video", None), progress.get("last_frame", 0)
    return None, 0

# --- Face and Embedding Processing ---
def compare_embeddings(prev_embedding, current_embedding, l2_threshold=1.2):
    """Compares two embeddings using cosine similarity and L2 distance."""
    cosine_similarity = 1 - cosine(prev_embedding, current_embedding)
    l2_distance = np.linalg.norm(prev_embedding - current_embedding)

    print(f"🔹 Cosine similarity: {cosine_similarity:.4f}")
    print(f"🔹 L2 distance: {l2_distance:.4f}")

    return l2_distance < l2_threshold

def detect_faces_and_embeddings(frame):
    """Detects the largest face in the frame and extracts its embedding."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb_frame)

    if not faces:
        return None, None, None

    # Select the largest face based on bounding box area
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    x, y, w, h = map(int, face.bbox)

    # Add margin to the bounding box
    margin = int(0.2 * max(w, h))
    x, y = max(0, x - margin), max(0, y - margin)
    w, h = min(frame.shape[1] - x, w + 2 * margin), min(frame.shape[0] - y, h + 2 * margin)

    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        return None, None, None

    return face_crop, face.normed_embedding, (x, y, w, h)

# --- Video Segment Extraction ---
def extract_video_segment(input_path, output_path, start_time, end_time):
    """Uses ffmpeg to extract a video segment between start_time and end_time."""
    duration = end_time - start_time
    if duration < 2:
        logging.warning(f"⚠ Segment discarded (too short): {duration:.2f} seconds")
        return

    command = [
        'ffmpeg', '-y', '-loglevel', 'info', '-ss', str(start_time), '-i', input_path,
        '-t', str(duration), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-strict', 'experimental', output_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0 and os.path.exists(output_path):
        logging.info(f"✅ Segment saved: {output_path}")
    else:
        logging.error(f"❌ Failed to generate file: {output_path}")

# --- Main Video Processing ---
def process_video(video_path, output_folder, identity_threshold=0.5, min_video_duration=2.0, frames_threshold=2):
    """Processes a video, detects faces, and extracts segments based on identity consistency."""
    global current_video, current_frame
    logging.info(f"📹 Processing video: {video_path}")

    current_video = video_path
    current_frame = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"❌ Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    _, start_frame = load_progress()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_embedding = None
    same_face_counter = 0
    start_time = None
    fragment_count = 0

    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, os.path.dirname(os.path.relpath(video_path, input_folder)))
    os.makedirs(video_output_folder, exist_ok=True)

    min_consecutive_frames = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        face_crop, embedding, bbox = detect_faces_and_embeddings(frame)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if embedding is not None:
            if prev_embedding is not None:
                l2_distance = np.linalg.norm(np.asarray(prev_embedding) - np.asarray(embedding))
                same_face_counter = same_face_counter + 1 if l2_distance < identity_threshold else 0
            else:
                same_face_counter = 1

            prev_embedding = embedding

            if same_face_counter >= min_consecutive_frames:
                if start_time is None:
                    start_time = current_time
                    logging.info(f"🟢 Segment started at {start_time:.2f}s")
            else:
                if start_time is not None:
                    segment_duration = current_time - start_time
                    if segment_duration >= min_video_duration:
                        output_path = os.path.join(video_output_folder, f"{video_base_name}_fragment_{fragment_count}.mp4")
                        extract_video_segment(video_path, output_path, start_time, current_time)
                        fragment_count += 1
                    start_time = None
        else:
            if start_time is not None:
                segment_duration = current_time - start_time
                if segment_duration >= min_video_duration:
                    output_path = os.path.join(video_output_folder, f"{video_base_name}_fragment_{fragment_count}.mp4")
                    extract_video_segment(video_path, output_path, start_time, current_time)
                    fragment_count += 1
                start_time = None
            prev_embedding = None
            same_face_counter = 0

        save_progress(video_path, current_frame)

    cap.release()
    logging.info(f"✅ Finished processing: {video_path}")

# --- Process All Videos in Folder ---
def process_videos_in_folder(input_folder, output_folder):
    last_video, last_frame = load_progress()
    skip = last_video is not None

    for root, _, files in os.walk(input_folder):
        for file in sorted(files):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                if skip:
                    if video_path == last_video:
                        skip = False
                    logging.info(f"⏭ Skipping video: {video_path}")
                    continue
                process_video(video_path, output_folder)
                save_progress(video_path, 0)

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Process videos to extract face-based segments.")
    parser.add_argument("--input", type=str, required=True, help="Path to input folder with videos")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video segments")

    args = parser.parse_args()

    global input_folder, output_folder, progress_file
    input_folder = args.input
    output_folder = args.output

    # Create dynamic progress file based on input folder name
    input_tag = os.path.basename(os.path.normpath(input_folder)).lower().replace(" ", "_")
    progress_file = f"processing_progress_{input_tag}.json"

    # Initialize InsightFace (uses GPU if available, falls back to CPU)
    global app
    app = FaceAnalysis(name="buffalo_l")  # No ctx_id → automatic provider selection
    app.prepare(det_size=(320, 320))      # Face detection resolution

    process_videos_in_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()


