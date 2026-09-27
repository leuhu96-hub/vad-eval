"""Tìm URL YouTube theo chủ đề của Groundtruth, ghi ra data/urls/<Topic>.txt.

    python search_urls.py            # ~30 URL mỗi chủ đề
    python search_urls.py -n 30
Sau đó tải: python download_data.py urls/Clean.txt -o audio/Clean --mono
"""
import argparse
from pathlib import Path

import yt_dlp

TOPICS = {
    "Clean": [
        "clean speech recording no background noise",
        "podcast interview studio clean voice",
        "audiobook reading english",
        "TED talk speech",
        "news anchor speaking",
    ],
    "Babble": [
        "crowd chatter background babble noise",
        "restaurant ambience people talking",
        "cafe ambience conversation background",
        "busy crowd talking ambience",
        "party crowd talking noise",
    ],
    "Music": [
        "song with vocals official audio",
        "instrumental music no lyrics",
        "acoustic pop song lyrics",
        "piano instrumental music",
        "rap song audio",
    ],
    "Noise": [
        "street traffic noise ambience",
        "rain and wind noise sound",
        "white noise sound",
        "factory machine noise",
        "train station noise ambience",
    ],
    "Asm": [
        "ASMR whispering",
        "ASMR tapping sounds",
        "ASMR mouth sounds",
        "ASMR triggers no talking",
        "ASMR role play soft spoken",
    ],
}

MIN_DUR, MAX_DUR = 20, 4 * 3600  # giây: bỏ video quá ngắn / quá dài (dùng --clip khi tải)


def search(query, n):
    opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return info.get("entries", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30, help="số URL mỗi chủ đề")
    ap.add_argument("-o", default="urls", help="thư mục ghi file txt")
    args = ap.parse_args()

    out = Path(__file__).parent / args.o
    out.mkdir(exist_ok=True)
    per_query = max(args.n // 2, 10)

    for topic, queries in TOPICS.items():
        seen, urls = set(), []
        for q in queries:
            for e in search(q, per_query):
                if not e or e["id"] in seen:
                    continue
                d = e.get("duration")
                if d is None or not (MIN_DUR <= d <= MAX_DUR):
                    continue
                seen.add(e["id"])
                urls.append(f"https://www.youtube.com/watch?v={e['id']}")
            if len(urls) >= args.n:
                break
        urls = urls[: args.n]
        (out / f"{topic}.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
        print(f"{topic}: {len(urls)} URL")


if __name__ == "__main__":
    main()
