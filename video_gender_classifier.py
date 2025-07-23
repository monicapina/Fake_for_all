"""
General-purpose script to classify the dominant gender in videos using face analysis.
It organizes the videos into subfolders by predicted gender and logs the results to a CSV.
Works with multiple folders or a single folder containing videos.
"""

import os
from pathlib import Path
import shutil
import cv2
import csv
from insightface.app import FaceAnalysis


def get_main_face(faces):
    """
    Selects the largest face from a list of detected faces.
    
    Args:
        faces (list): List of detected face objects.
    
    Returns:
        Face object with the largest bounding box.
    """
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def classify_gender_from_video(app, video_path, num_samples=20):
    """
    Analyzes sampled frames from a video and predicts the dominant gender.
    
    Args:
        app (FaceAnalysis): Initialized face analysis application.
        video_path (Path): Path to the video file.
        num_samples (int): Number of frames to sample throughout the video.
    
    Returns:
        (str, float): Predicted gender ("man" or "woman") and confidence score.
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    genders = []

    for i in range(num_samples):
        frame_idx = int(frame_count * (i + 1) / (num_samples + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)
        if not faces:
            continue

        main_face = get_main_face(faces)
        if hasattr(main_face, "gender"):
            genders.append(main_face.gender)

    cap.release()

    if not genders:
        return None, 0.0

    # Use the most frequent gender detected
    final_gender = max(set(genders), key=genders.count)
    label = "woman" if final_gender == 0 else "man"
    confidence = genders.count(final_gender) / len(genders)

    return label, confidence


def process_all_folders(input_root, output_root, result_csv_path):
    """
    Processes all videos inside input folders, predicts gender, copies videos by gender, and logs results.

    Args:
        input_root (str or Path): Root input folder containing video folders or videos directly.
        output_root (str or Path): Output folder to store videos grouped by predicted gender.
        result_csv_path (str or Path): Path to CSV file for storing gender predictions.
    """
    app = FaceAnalysis(allowed_modules=['detection', 'genderage'])
    app.prepare(det_size=(640, 640))  # Use GPU if available

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(result_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "Video", "PredictedGender", "ConfidenceScore"])

        # Case 1: multiple subfolders (e.g., groups, categories, ethnicities)
        subdirs = [d for d in input_root.iterdir() if d.is_dir()]
        if subdirs:
            for group_dir in subdirs:
                print(f"📂 Processing folder: {group_dir.name}")
                for video_path in group_dir.rglob("*.mp4"):
                    gender, score = classify_gender_from_video(app, video_path)
                    if gender:
                        gender_folder = output_root / group_dir.name / gender
                        gender_folder.mkdir(parents=True, exist_ok=True)
                        shutil.copy(video_path, gender_folder / video_path.name)
                        writer.writerow([group_dir.name, video_path.name, gender, round(score, 3)])
        else:
            # Case 2: input_root directly contains videos
            group_name = input_root.name
            print(f"📂 Processing single video folder: {group_name}")
            for video_path in input_root.rglob("*.mp4"):
                gender, score = classify_gender_from_video(app, video_path)
                if gender:
                    gender_folder = output_root / group_name / gender
                    gender_folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy(video_path, gender_folder / video_path.name)
                    writer.writerow([group_name, video_path.name, gender, round(score, 3)])


# Entry point
if __name__ == "__main__":
    input_root = "/path/to/input_videos"
    output_root = "/path/to/output_by_gender"
    result_csv = "gender_results.csv"

    process_all_folders(input_root, output_root, result_csv)
