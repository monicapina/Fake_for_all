"""
General-purpose script to extract metadata from video and image files organized in folders.
It supports structured datasets and logs properties such as resolution, FPS, duration,
as well as inferred attributes from filenames.
The final metadata is saved as a compressed JSON file (.json.gz).
"""

import os
import json
import gzip
import cv2
from pathlib import Path
from tqdm import tqdm

def infer_source_type(source_id):
    """
    Infers the origin/type of a video/image based on its filename.

    Args:
        source_id (str): Identifier or filename prefix.

    Returns:
        str: One of ["dfdc", "trailer", "celebv"] based on simple rules.
    """
    if "__" in source_id:
        return "dfdc"
    elif "fragment" in source_id.lower():
        return "trailer"
    else:
        return "celebv"

def extract_video_properties(video_path):
    """
    Extracts resolution, FPS, duration, and frame count from a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict or None: Metadata dictionary or None if extraction failed.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else None
        
        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_sec": round(duration, 2) if duration else None
        }
    except Exception:
        return None

def extract_image_properties(image_path):
    """
    Extracts resolution from an image file.

    Args:
        image_path (Path): Path to the image.

    Returns:
        dict or None: Metadata dictionary or None if extraction failed.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        height, width = img.shape[:2]
        return {
            "width": width,
            "height": height,
            "fps": None,
            "frame_count": 1,
            "duration_sec": None
        }
    except Exception:
        return None

def extract_metadata_from_videos(root_dir):
    """
    Traverses a directory to extract metadata from all video/image files.

    Assumes files are organized in subfolders like: root/label1/label2/filename.ext

    Args:
        root_dir (str): Root directory of the dataset.

    Returns:
        list: List of dictionaries containing metadata for each media file.
    """
    video_metadata = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in tqdm(filenames, desc="Processing files"):
            full_path = Path(dirpath) / filename

            # Assumes folder structure like: root/group/category/filename
            parts = Path(dirpath).parts
            ethnicity = parts[-2] if len(parts) >= 2 else "unknown"
            gender = parts[-1] if len(parts) >= 1 else "unknown"

            if filename.endswith(".mp4"):
                if "_out_" in filename:
                    # Synthetic pair (source_out_target.mp4)
                    parts = filename.split("_out_")
                    if len(parts) != 2:
                        continue

                    source_id = parts[0]
                    target_id = parts[1].replace(".mp4", "")
                    source_type = infer_source_type(source_id)
                    target_type = infer_source_type(target_id)

                    video_props = extract_video_properties(full_path)
                    if video_props is None:
                        continue

                    video_metadata.append({
                        "filename": filename,
                        "source_id": source_id,
                        "source_type": source_type,
                        "target_id": target_id,
                        "target_type": target_type,
                        "ethnicity": ethnicity,
                        "gender": gender,
                        "is_real": False,
                        **video_props
                    })
                else:
                    # Real video (no _out_ in filename)
                    source_id = filename.replace(".mp4", "")
                    source_type = infer_source_type(source_id)
                    video_props = extract_video_properties(full_path)
                    if video_props is None:
                        continue

                    video_metadata.append({
                        "filename": filename,
                        "source_id": source_id,
                        "source_type": source_type,
                        "target_id": None,
                        "target_type": None,
                        "ethnicity": ethnicity,
                        "gender": gender,
                        "is_real": True,
                        **video_props
                    })

            elif filename.endswith(".jpg"):
                # Real image
                source_id = filename.replace(".jpg", "")
                source_type = infer_source_type(source_id)
                image_props = extract_image_properties(full_path)
                if image_props is None:
                    continue

                video_metadata.append({
                    "filename": filename,
                    "source_id": source_id,
                    "source_type": source_type,
                    "target_id": None,
                    "target_type": None,
                    "ethnicity": ethnicity,
                    "gender": gender,
                    "is_real": True,
                    **image_props
                })

    return video_metadata


# === Run script ===
if __name__ == "__main__":
    # Define input and output paths
    input_root = "/path/to/your/media/folder"
    output_json_path = "/path/to/save/metadata.json.gz"

    # Extract metadata
    metadata = extract_metadata_from_videos(input_root)

    # Save as compressed JSON
    with gzip.open(output_json_path, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, separators=(",", ":"))

    print(f"✅ Metadata extracted and saved to: {output_json_path}")
