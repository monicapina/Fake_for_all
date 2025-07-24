import os
import cv2
import numpy as np
import subprocess
import argparse
from pathlib import Path
from PIL import Image
import imquality.brisque as brisque
import shutil
from insightface.app import FaceAnalysis

EXTRACT_SCRIPT = "extract.py"  # Path to the face extraction script
LAST_PROCESSED_FILE = "last_processed.txt"

# Parameters to weight face selection criteria
PARAMS = {
    "eye_openness_weight": 300,
    "frontality_weight": 50,
    "occlusion_weight": 100,
    "brisque_weight": -0.5,
    "min_face_size": 50,
    "resize_to": (224, 224)
}

face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare()

def load_last_processed():
    if os.path.exists(LAST_PROCESSED_FILE):
        with open(LAST_PROCESSED_FILE, "r") as f:
            return f.read().strip()
    return None

def save_last_processed(video_identifier):
    with open(LAST_PROCESSED_FILE, "w") as f:
        f.write(video_identifier)

def run_extract_script(video_path, temp_dir):
    command = ["python", EXTRACT_SCRIPT, "-i", video_path, "-o", temp_dir]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed to execute extract.py: {result.stderr}")
        return False
    return True

def calculate_face_score(face_path, face_analyzer):
    image = cv2.imread(face_path)
    if image is None:
        return -np.inf

    faces = face_analyzer.get(image)
    if not faces or faces[0].landmark is None:
        return -np.inf

    landmarks = faces[0].landmark

    try:
        left_eye = np.linalg.norm(landmarks[37] - landmarks[41])
        right_eye = np.linalg.norm(landmarks[43] - landmarks[47])
        eye_openness = (left_eye + right_eye) / 2
    except IndexError:
        return -np.inf

    occlusion_score = len(landmarks) / 68

    try:
        quality = -brisque.score(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    except:
        quality = -100

    total_score = (
        quality * PARAMS["brisque_weight"] +
        eye_openness * PARAMS["eye_openness_weight"] +
        occlusion_score * PARAMS["occlusion_weight"]
    )
    return total_score

def select_best_face(temp_dir, target_dir, video_filename, face_analyzer):
    faces_info = []
    for f in os.listdir(temp_dir):
        if f.endswith(".jpg"):
            face_path = os.path.join(temp_dir, f)
            score = calculate_face_score(face_path, face_analyzer)
            faces_info.append((face_path, score))

    if not faces_info:
        print(f"[WARNING] No faces found in {video_filename}.")
        return

    best_face = max(faces_info, key=lambda x: x[1])
    best_face_path, best_score = best_face

    target_path = os.path.join(target_dir, os.path.basename(best_face_path))
    shutil.copy(best_face_path, target_path)
    print(f"[INFO] Best face saved to {target_path} with score {best_score:.2f}")

    shutil.rmtree(temp_dir)

def process_videos(source_dir, temp_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    last_processed = load_last_processed()
    skip = True if last_processed else False

    for root, _, files in os.walk(source_dir):
        video_files = sorted([f for f in files if f.endswith(('.mp4', '.avi', '.mov'))])
        if not video_files:
            continue

        relative_path = os.path.relpath(root, source_dir)
        output_subfolder = os.path.join(target_dir, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)
        print(f"[INFO] Processing {len(video_files)} videos in {root}")

        for video_file in video_files:
            video_identifier = os.path.join(relative_path, video_file)
            if skip:
                if video_identifier == last_processed:
                    skip = False
                continue

            video_path = os.path.join(root, video_file)
            output_face_path = os.path.join(output_subfolder, f"{Path(video_file).stem}.jpg")

            if os.path.exists(output_face_path):
                print(f"[INFO] {video_file} already processed. Skipping...")
                continue

            print(f"[INFO] Processing {video_file}")
            os.makedirs(temp_dir, exist_ok=True)

            success = run_extract_script(video_path, temp_dir)
            if not success or not os.listdir(temp_dir):
                print(f"[WARNING] No faces extracted from {video_file}. Skipping...")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

            select_best_face(temp_dir, output_subfolder, video_file, face_analyzer)
            save_last_processed(video_identifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the best face per video")
    parser.add_argument("--source", required=True, help="Path to folder containing input videos")
    parser.add_argument("--temp", default="./temp", help="Temporary folder for extracted faces")
    parser.add_argument("--target", required=True, help="Folder where best faces will be saved")
    args = parser.parse_args()

    process_videos(args.source, args.temp, args.target)
