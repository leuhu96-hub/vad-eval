"""Tải audio từ YouTube (video hoặc playlist) bằng yt-dlp.

Cài đặt:
    pip install yt-dlp
    # Cần ffmpeg trong PATH để chuyển đổi sang wav/mp3 (https://ffmpeg.org/download.html)

Ví dụ:
    python download_data.py https://www.youtube.com/watch?v=XXXX
    python download_data.py URL1 URL2 -o data/audio -f wav --sr 16000 --mono
    python download_data.py -i urls.txt -f mp3
"""
import argparse
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("Thiếu yt-dlp. Cài bằng: pip install yt-dlp")


def download_audio(urls, out_dir, fmt="wav", sample_rate=16000, mono=True, clip=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    postprocessor_args = []
    if sample_rate:
        postprocessor_args += ["-ar", str(sample_rate)]
    if mono:
        postprocessor_args += ["-ac", "1"]

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title).150B [%(id)s].%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": fmt,
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": {"ffmpeg": postprocessor_args},
        "download_archive": str(out_dir / "downloaded.txt"),  # bỏ qua file đã tải
        "ignoreerrors": True,  # lỗi 1 video không dừng cả playlist
        "noplaylist": False,
        "restrictfilenames": False,
        "windowsfilenames": True,
    }

    if clip:  # chỉ tải `clip` giây đầu của mỗi video
        opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, clip)])
        opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download(urls)


def main():
    p = argparse.ArgumentParser(description="Download audio from YouTube")
    p.add_argument("urls", nargs="*", help="URL video/playlist YouTube")
    p.add_argument("-i", "--input-file", help="File txt chứa mỗi dòng một URL")
    p.add_argument("-o", "--output", default="audio", help="Thư mục lưu (mặc định: audio)")
    p.add_argument("-f", "--format", default="wav",
                   choices=["wav", "mp3", "m4a", "flac", "opus"], help="Định dạng audio")
    p.add_argument("--sr", type=int, default=16000,
                   help="Sample rate (0 = giữ nguyên). Mặc định 16000 cho VAD")
    p.add_argument("--mono", action="store_true", help="Chuyển về mono")
    p.add_argument("--clip", type=int, default=0,
                   help="Chỉ tải N giây đầu mỗi video (0 = tải hết)")
    args = p.parse_args()

    urls = []
    files = list(args.input_file and [args.input_file] or [])
    for item in args.urls:
        # Tham số là đường dẫn file .txt tồn tại -> đọc URL từ file
        if item.lower().endswith(".txt") and Path(item).is_file():
            files.append(item)
        else:
            urls.append(item)
    for f in files:
        lines = Path(f).read_text(encoding="utf-8-sig").splitlines()
        urls += [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    if not urls:
        p.error("Cần ít nhất một URL hoặc --input-file")

    download_audio(urls, args.output, args.format, args.sr, args.mono, args.clip)


if __name__ == "__main__":
    main()
