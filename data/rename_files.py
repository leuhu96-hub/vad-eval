"""Đổi tên file audio theo category (tên thư mục con) + YouTube ID.

    audio/Babble/[Barista Vlog] Working in a Cafe ... [sUOTOHCVGGk].wav
 -> audio/Babble/Babble_sUOTOHCVGGk.wav

Chỉ đổi file có dạng "... [<id>].<ext>"; bỏ qua downloaded.txt, file .part đang tải dở.
Tên mới không có tiêu đề nên download_archive (downloaded.txt, lưu theo ID) vẫn dùng được.

Ví dụ:
    python rename_files.py              # đổi tên mọi category trong audio/
    python rename_files.py -n           # chỉ in ra, không đổi (dry-run)
    python rename_files.py -d audio -c Babble Music
"""
import argparse
import re
from pathlib import Path

ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]$")  # YouTube ID ở cuối tên (trước đuôi)
SKIP_EXT = {".part", ".ytdl", ".txt"}


def rename_category(cat_dir, dry_run=False):
    cat = cat_dir.name
    n = 0
    for f in sorted(cat_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() in SKIP_EXT:
            continue
        m = ID_RE.search(f.stem)
        if not m:
            continue
        dst = f.with_name(f"{cat}_{m.group(1)}{f.suffix}")
        if dst == f:
            continue
        if dst.exists():
            print(f"[BỎ QUA] đã tồn tại: {dst.name}  <-  {f.name}")
            continue
        print(f"{f.name}  ->  {dst.name}")
        if not dry_run:
            f.rename(dst)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description="Rename audio files to <category>_<youtube_id>")
    p.add_argument("-d", "--dir", default=Path(__file__).parent / "audio", type=Path,
                   help="Thư mục chứa các thư mục category (mặc định: data/audio)")
    p.add_argument("-c", "--categories", nargs="*",
                   help="Chỉ đổi các category này (mặc định: tất cả)")
    p.add_argument("-n", "--dry-run", action="store_true", help="Chỉ in ra, không đổi tên")
    args = p.parse_args()

    cats = [d for d in sorted(args.dir.iterdir()) if d.is_dir()]
    if args.categories:
        cats = [d for d in cats if d.name in args.categories]

    total = 0
    for d in cats:
        k = rename_category(d, args.dry_run)
        print(f"== {d.name}: {k} file{' (dry-run)' if args.dry_run else ''}\n")
        total += k
    print(f"Tổng: {total} file")


if __name__ == "__main__":
    main()
