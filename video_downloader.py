import os
import re
import json
import time
import subprocess
import googleapiclient.discovery
import googleapiclient.errors

# YouTube Data API Key (replace with your own)
API_KEY = 'YOUR_API_KEY_HERE'

# File to keep track of downloaded video IDs
download_log_path = "downloaded_videos.json"

# Load download history if it exists
if os.path.exists(download_log_path):
    with open(download_log_path, "r") as f:
        downloaded_videos = set(json.load(f))
else:
    downloaded_videos = set()

# Folder to store downloaded videos
output_folder = "downloads"
os.makedirs(output_folder, exist_ok=True)

# Keywords to search per region (Example: Korea, Japan, China)
countries_keywords = {
    'Caucasian': {
        'regionCode': 'US',
        'keywords': [
            'Hollywood actress interview full video',
            'European actress podcast with subtitles',
            'British celebrity talk show',
            'French actress interview with English subtitles',
            'German actress documentary full video'
        ]
    },

    'Afroamerican': {
        'regionCode': 'US',
        'keywords': [
            'African American actress interview',
            'Black female celebrity documentary',
            'African American women in film podcast',
            'interview Black Hollywood actress',
            'Black actress talk show HD'
        ]
    },

    'Asian': {
        'regionCode': 'KR',  # Change between KR, JP, CN for diversity
        'keywords': [
            'Korean actress interview full video',
            'KDrama female podcast',
            'Japanese actress talk show HD',
            'Chinese actress documentary with subtitles',
            'Asian female celebrity face to face interview'
        ]
    },

    'LatinoHispanic': {
        'regionCode': 'MX',  # You can rotate with AR, CO, etc.
        'keywords': [
            'entrevista actriz mexicana HD',
            'actriz latina en programa de entrevistas',
            'entrevista a celebridad hispana',
            'actriz argentina entrevista completa',
            'telenovela escena actriz latinoamericana'
        ]
    },

    'MENA': {
        'regionCode': 'EG',  # You can rotate with SA, MA, IR, TR
        'keywords': [
            'مقابلة مع ممثلة مصرية',  # Egyptian actress interview
            'فيديو وثائقي عن ممثلة سعودية',  # Saudi actress documentary
            'actrice marocaine interview complète',
            'actriz libanesa entrevista en árabe',
            'actriz iraní entrevista subtitulada'
        ]
    }
}


def get_youtube_videos(query, region_code, max_results=50):
    """
    Search YouTube videos by keyword and region, prioritizing long videos (20+ min).
    Returns a list of valid video metadata.
    """
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

    request = youtube.search().list(
        part="snippet",
        maxResults=max_results,
        q=query,
        type="video",
        regionCode=region_code,
        videoDuration="long"
    )
    
    response = request.execute()
    videos = []

    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        title = item["snippet"]["title"]
        channel_id = item["snippet"]["channelId"]

        if video_id not in downloaded_videos:
            try:
                # Attempt to verify the region of the channel
                channel_info = youtube.channels().list(part="snippet", id=channel_id).execute()
                country = channel_info["items"][0]["snippet"].get("country", "Unknown")

                if country == region_code or country == "Unknown":
                    videos.append({"video_id": video_id, "url": video_url, "title": title})
                    downloaded_videos.add(video_id)
            except Exception:
                print(f"⚠️ Warning: Unable to verify region for channel {channel_id}, skipping.")

    return videos

def sanitize_filename(filename):
    """
    Remove illegal characters from filename and truncate if too long.
    """
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename[:100]

def download_video(url, output_path):
    """
    Download a YouTube video using yt-dlp and browser cookies for age-restricted content.
    """
    try:
        command = [
            "yt-dlp",
            "--cookies", "cookies.txt",  # Optional: export your browser cookies here
            "-f", "bestvideo[height<=1080]+bestaudio/best",
            "-o", output_path,
            "--merge-output-format", "mp4",
            "--retries", "5",
            "--sleep-interval", "2",
            "--max-sleep-interval", "4",
            "--limit-rate", "1M",
            "--no-playlist",
            "--socket-timeout", "60",
            url
        ]

        print(f"Executing: {' '.join(command)}")
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {url}: {str(e)}")
        return False

# === Main loop: Iterate over countries and search/download videos ===
for country, data in countries_keywords.items():
    print(f"🔍 Searching videos for country: {country}")

    # Create subfolder per country
    country_folder = os.path.join(output_folder, country)
    os.makedirs(country_folder, exist_ok=True)

    video_count = 1
    total_downloaded = 0

    for keyword in data["keywords"]:
        print(f"  🔑 Keyword: '{keyword}'")
        videos = get_youtube_videos(keyword, region_code=data["regionCode"], max_results=15)

        for video in videos:
            video_url = video["url"]
            video_title = sanitize_filename(video["title"])
            video_filename = f"{country}_{video_count:03d}.mp4"
            download_path = os.path.join(country_folder, video_filename)

            if os.path.exists(download_path):
                print(f"  ⚠️ Already exists: {video_filename}, skipping.")
                continue

            print(f"  📥 Downloading [{video_count}]: {video_url}")
            if download_video(video_url, download_path):
                video_count += 1
                total_downloaded += 1
                with open(download_log_path, "w") as f:
                    json.dump(list(downloaded_videos), f)

    print(f"✅ Finished {country}. Total videos downloaded: {total_downloaded}\n")
