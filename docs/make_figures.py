"""Vẽ sơ đồ và hình minh hoạ cho HUONG_DAN_SU_DUNG.md -> docs/img/.

Các hình minh hoạ hậu xử lý dùng tín hiệu tự tạo (không cần dữ liệu) và đúng các hàm của vadlib, nên luôn khớp code.
    python docs/make_figures.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vadlib import (rescore_hop, hysteresis, frames_to_segs, postprocess_segs, segs_to_bins_fast, boundary_mask,
                    HOP, BIN, WIN, HALF_WIN)

OUT = ROOT / "docs" / "img"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3b3a37", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
INK, INK2, MUTED, GRID, TINT = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f4f3ef"


def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig); print("saved", name)


# ---------------- khối vẽ sơ đồ ----------------
def box(ax, x, y, w, h, title, lines=(), color=BLUE, kind="script", fs=10.5):
    """kind: script = nền màu nhạt viền đậm; data = viền nét đứt; out = nền trắng viền xám."""
    fc = {"script": color, "data": "white", "out": "white"}[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, alpha=0.12 if kind == "script" else 1,
                       ec="none", zorder=1)
    ax.add_patch(p)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc="none",
                                ec=color if kind != "out" else MUTED, lw=1.6 if kind == "script" else 1.2,
                                ls="--" if kind == "data" else "-", zorder=2))
    n = len(lines); top = y + h / 2 + (0.13 * n if n else 0)
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK, zorder=3)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, top - 0.28 - 0.25 * i, t, ha="center", va="center", fontsize=fs - 2.2, color=INK2, zorder=3)


def arrow(ax, a, b, color=MUTED, text=None, rad=0.0, tpos=0.5, toff=(0, 0.12), fs=8.2):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=2, shrinkB=2,
                                                    connectionstyle=f"arc3,rad={rad}"), zorder=0)
    if text:
        mx, my = a[0] + (b[0] - a[0]) * tpos + toff[0], a[1] + (b[1] - a[1]) * tpos + toff[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=INK2,
                bbox=dict(fc="white", ec="none", pad=1.2), zorder=4)


def canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_aspect("equal")
    return fig, ax


# ======================= 1. Luồng công việc =======================
def fig_workflow():
    fig, ax = canvas(16, 7.2, (13, 5.9))
    ax.text(0.1, 6.95, "Các phần của vad-eval và dữ liệu đi qua chúng", fontsize=13, fontweight="bold", color=INK, va="top")
    box(ax, 0.2, 4.6, 3.0, 1.1, "audio/", ["<Cat>/<Cat>_<id>.wav"], MUTED, "data")
    box(ax, 0.2, 2.7, 3.0, 1.1, "Model/", ["điểm mỗi cửa sổ 0.96 s", "(output lib.so)"], MUTED, "data")
    box(ax, 4.2, 4.6, 3.2, 1.1, "Phần 1: auto_label.py", ["FireRedVAD + Silero", "(tuỳ chọn)"], PURPLE)
    box(ax, 8.4, 4.6, 3.0, 1.1, "Ground_truth/", ["nhãn bin 0.5 s", "(sửa tay theo review.csv)"], MUTED, "data")
    box(ax, 4.2, 0.6, 3.2, 1.1, "cache Silero", ["người gán 2", "(--silero-cache)"], MUTED, "data")
    box(ax, 8.4, 2.45, 3.0, 1.6, "Phần 2 + 3:", ["evaluate_real.py", "kiểm tra · AUC · ngưỡng ·", "MR/FAR/F1/DCF · so sánh", "hậu xử lý · --tune"], BLUE, fs=10.5)
    box(ax, 12.4, 2.7, 3.2, 1.1, "results_xxx/", ["csv · summary.json", "fig1 … fig7"], MUTED, "out")
    box(ax, 12.4, 0.6, 3.2, 1.1, "Phần 4:", ["make_excel_real.py → Tong_hop_…xlsx"], GREEN, fs=10.5)
    box(ax, 12.4, 4.6, 3.2, 1.1, "Phần 6: vadlib.py", ["hàm dùng chung", "(import trong code riêng)"], ORANGE, fs=10)
    arrow(ax, (3.2, 5.15), (4.2, 5.15))
    arrow(ax, (7.4, 5.15), (8.4, 5.15), text="nhãn")
    arrow(ax, (5.8, 4.6), (5.8, 1.7), text="cache", tpos=0.22)
    arrow(ax, (3.2, 3.25), (8.4, 3.25), text="--root / --raw")
    arrow(ax, (9.9, 4.6), (9.9, 4.05), text="--gt")
    arrow(ax, (7.4, 1.15), (8.9, 2.45), text="nhãn 2", rad=0.15)
    arrow(ax, (11.4, 3.25), (12.4, 3.25))
    arrow(ax, (14.0, 2.7), (14.0, 1.7), text="--res")
    arrow(ax, (12.4, 5.15), (11.0, 4.05), color=ORANGE, rad=-0.1)
    ax.text(0.2, 0.15, "Khung nét đứt = dữ liệu bạn chuẩn bị · khung màu = script · khung xám = kết quả", fontsize=8.5, color=MUTED)
    save(fig, "01_luong_cong_viec.png")


# ======================= 2. Cấu trúc thư mục + ghép tên =======================
def fig_folders():
    fig, ax = canvas(16, 7, (13, 5.7))
    ax.text(0.1, 6.8, "Đặt dữ liệu thế nào và file được ghép với nhau ra sao", fontsize=13, fontweight="bold", color=INK, va="top")
    tree = ["<root>/", "├── Model/", "│   ├── Asm_1.txt", "│   ├── Asm_2.txt", "│   └── Music_1.txt", "├── audio/            (tuỳ chọn)",
            "│   ├── Asm/Asm_1.wav", "│   └── Music/Music_1.wav", "└── Ground_truth/", "    ├── Asm/groundtruth/Gt_asm_1.txt",
            "    └── Music/groundtruth/Gt_music_1.txt"]
    ax.add_patch(FancyBboxPatch((0.2, 0.5), 6.4, 5.7, boxstyle="round,pad=0,rounding_size=0.1", fc=TINT, ec=GRID))
    tcol = [INK] + [BLUE] * 4 + [PURPLE] * 3 + [GREEN] * 3
    for i, t in enumerate(tree):
        ax.text(0.45, 5.85 - 0.47 * i, t, family="DejaVu Sans Mono", fontsize=9.6, color=tcol[i], va="center")
    # ghép
    ax.text(7.4, 5.9, "Ghép theo khoá tên file", fontsize=11, fontweight="bold", color=INK)
    rows = [("Model/Asm_1.txt", BLUE, "Asm_1"), ("Ground_truth/Asm/groundtruth/Gt_asm_1.txt", GREEN, "bỏ tiền tố Gt_"),
            ("audio/Asm/Asm_1.wav", PURPLE, "Asm_1")]
    for i, (t, c, how) in enumerate(rows):
        y = 5.0 - 1.05 * i
        ax.add_patch(FancyBboxPatch((7.4, y - 0.32), 5.0, 0.64, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=c, lw=1.4))
        ax.text(7.55, y, t, family="DejaVu Sans Mono", fontsize=8.6, color=c, va="center")
        arrow(ax, (12.4, y), (13.5, 3.95), color=c)
        ax.text(12.35, y + 0.42, how, fontsize=7.8, color=INK2, ha="right")
    ax.add_patch(FancyBboxPatch((13.5, 3.55), 2.2, 0.8, boxstyle="round,pad=0,rounding_size=0.1", fc=YELLOW, alpha=0.18, ec="none"))
    ax.add_patch(FancyBboxPatch((13.5, 3.55), 2.2, 0.8, boxstyle="round,pad=0,rounding_size=0.1", fc="none", ec=YELLOW, lw=1.6))
    ax.text(14.6, 3.95, "khoá: asm_1", ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    notes = ["• Không phân biệt hoa thường; bỏ Gt_, label_, _gt …",
             "• Category = thư mục con (Asm, Music) hoặc phần tên trước '_'",
             "• File thiếu một nguồn bị bỏ → summary.json → problems",
             "• Thư mục ở chỗ khác: dùng --raw / --gt / --audio"]
    for i, t in enumerate(notes): ax.text(7.4, 1.6 - 0.36 * i, t, fontsize=9, color=INK2, va="center")
    save(fig, "02_cau_truc_thu_muc.png")


# ======================= 3. Cửa sổ model và bin nhãn =======================
def fig_windows():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    T = 3.0; gt = np.array([0, 1, 1, 1, 0, 0])      # 6 bin 0.5 s
    yb = 9.6
    for k, v in enumerate(gt):
        ax.add_patch(Rectangle((k * BIN, yb), BIN, 0.8, fc=BLUE if v else "white", alpha=0.35 if v else 1, ec=INK2, lw=0.8))
        ax.text(k * BIN + BIN / 2, yb + 0.4, "speech" if v else "non-speech", ha="center", va="center", fontsize=8, color=INK)
    ax.text(-0.05, yb + 0.4, "Nhãn: bin 0.5 s", ha="right", va="center", fontsize=9.5, fontweight="bold")
    n_show = 14
    for i in range(n_show):
        st = i * HOP; y = 8.6 - 0.55 * i; hi = i == 6
        ax.add_patch(Rectangle((st, y), WIN, 0.36, fc=ORANGE if hi else MUTED, alpha=0.55 if hi else 0.22, ec="none"))
        ax.plot(st + HALF_WIN, y + 0.18, "o", ms=4.5, color=ORANGE if hi else INK2, zorder=3)
    ax.text(-0.05, 8.6 - 0.55 * 6.5, "Model: cửa sổ 0.96 s\ntrượt mỗi 0.08 s", ha="right", va="center", fontsize=9.5, fontweight="bold")
    st6 = 6 * HOP; y6 = 8.6 - 0.55 * 6
    ax.annotate("", xy=(st6, y6 + 0.62), xytext=(st6 + WIN, y6 + 0.62), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2))
    ax.text(st6 + WIN + 0.05, y6 + 0.62, "1 cửa sổ = 0.96 s (1 dòng output: start, end, score)", fontsize=8.5, color=ORANGE, va="center")
    ax.annotate("điểm đặt tại tâm cửa sổ = start + 0.48 s", xy=(st6 + HALF_WIN, y6 + 0.18), xytext=(1.75, y6 - 1.25), fontsize=8.5,
                color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    yh = 0.4
    for i in range(int(T / HOP) + 1):
        ax.add_patch(Rectangle((i * HOP, yh), HOP, 0.5, fc=GREEN if i % 2 else "white", alpha=0.25, ec=GREEN, lw=0.5))
    ax.text(-0.05, yh + 0.25, "Hop 0.08 s (frame\nsau Rebin)", ha="right", va="center", fontsize=9.5, fontweight="bold")
    t0 = 1.32
    ax.axvline(t0, ymin=0.05, ymax=0.83, color=GREEN, lw=1, ls=":")
    ax.text(t0 + 0.04, 1.15, "mỗi hop nằm trong tối đa 12 cửa sổ → Rebin gộp 12 điểm thành 1", fontsize=8.5, color=GREEN)
    ax.set_xlim(-0.02, T); ax.set_ylim(0, 10.7); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("thời gian (s)"); ax.set_xticks(np.arange(0, T + 0.01, 0.5))
    ax.set_title("Output model (cửa sổ 0.96 s, hop 0.08 s) và nhãn (bin 0.5 s) nằm trên hai lưới thời gian khác nhau",
                 fontsize=11.5, fontweight="bold", loc="left", x=-0.2)
    save(fig, "03_cua_so_va_bin.png")


# ======================= tín hiệu mẫu cho hình hậu xử lý =======================
def make_example(seed=4):
    """GT 10 s: câu có chỗ ngắt hơi ngắn, hai câu ngắn, một tiếng ho, một câu có đoạn nói nhỏ (điểm tụt)."""
    rng = np.random.default_rng(seed); FINE = 0.02; t = np.arange(0, 10, FINE)
    sp = [(0.9, 2.3), (2.8, 4.0), (5.0, 5.8), (7.7, 9.4)]
    g = np.zeros(len(t))
    for a, b in sp: g[(t >= a) & (t < b)] = 1
    st = np.round(np.arange(0, 10 - WIN + 1e-9, HOP), 4); c = st + HALF_WIN
    frac = np.array([g[(t >= s) & (t < s + WIN)].mean() for s in st])            # tỉ lệ speech trong cửa sổ
    score = 0.06 + 0.9 / (1 + np.exp(-12 * (frac - 0.5)))
    for mu, sg, amp in ((2.55, 0.12, -0.15), (6.75, 0.16, 0.95), (8.6, 0.2, -0.62)):   # ngắt hơi, tiếng ho, nói nhỏ
        score += amp * np.exp(-0.5 * ((c - mu) / sg) ** 2)
    score = np.clip(score + rng.normal(0, 0.03, len(st)), 0.01, 0.99)
    m = pd.DataFrame(dict(start=st, end=st + WIN, score=score)); m["center"] = c
    gt_bins = (np.add.reduceat(g, np.arange(0, len(g), int(BIN / FINE))) / (BIN / FINE) >= 0.5).astype(int)
    return m, sp, gt_bins


def seg_row(ax, y, segs, color, h=0.62, label=None, alpha=0.85):
    for a, b in segs: ax.add_patch(Rectangle((a, y - h / 2), b - a, h, fc=color, alpha=alpha, ec="none"))
    if label: ax.text(-0.15, y, label, ha="right", va="center", fontsize=9.3, color=INK)


def note(ax, x, y, t, color=INK2, dx=0, dy=0.55, ha="center"):
    ax.annotate(t, xy=(x, y), xytext=(x + dx, y + dy), ha=ha, va="bottom", fontsize=8.2, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9), bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5), zorder=5)


# ======================= 4. Hậu xử lý từng bước =======================
def fig_steps():
    m, sp, gt_bins = make_example(); dur = 10.0; on, off = 0.60, 0.45
    fr = rescore_hop(m, dur, weight="tri", smooth=3, smooth_kind="median")
    tc = (np.arange(len(fr)) + 0.5) * HOP
    single = frames_to_segs(fr >= on, HOP); hyst = frames_to_segs(hysteresis(fr, on, off), HOP)
    merged = postprocess_segs(hyst, 0, 0, 0.5, 0, dur, "mdp")
    dropped = postprocess_segs(hyst, 0, 0, 0.5, 0.4, dur, "mdp")
    padded = postprocess_segs(hyst, 0.08, 0.08, 0.5, 0.4, dur, "mdp")
    final = postprocess_segs(hyst, 0, 0, 0.5, 0.4, dur, "mdp")
    pb = segs_to_bins_fast(final, len(gt_bins))

    fig = plt.figure(figsize=(13, 9.2)); gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.05, 2.6], hspace=0.28)
    a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))
    for a in (a0, a1):
        for s, e in sp: a.axvspan(s, e, color=BLUE, alpha=0.09, lw=0)
        a.set_xlim(-0.05, dur); a.set_ylim(-0.03, 1.08); a.grid(axis="y", color=GRID, lw=0.6); a.set_yticks([0, 0.5, 1])
    a0.plot(m.center, m.score, "o", ms=2.6, color=MUTED, label="điểm từng cửa sổ (đặt tại tâm)")
    a0.set_title("Đầu vào: điểm model mỗi cửa sổ 0.96 s, chấm đặt tại tâm cửa sổ (nền xanh = speech theo nhãn)", loc="left", fontsize=10.5, fontweight="bold")
    note(a0, 6.75, 0.85, "tiếng ho → gai điểm ngắn", ORANGE, dx=-0.3, dy=0.12, ha="right")
    note(a0, 8.6, 0.42, "nói nhỏ giữa câu → điểm tụt", RED, dx=1.35, dy=-0.38, ha="right")
    a1.plot(tc, fr, color=BLUE, lw=1.8, label="+ median 3 hop (điểm mỗi hop 0.08 s)")
    a1.axhline(on, color=INK, lw=1, ls="--"); a1.axhline(off, color=INK2, lw=1, ls=":")
    a1.text(dur - 0.02, on + 0.02, "ngưỡng bật 0.60", ha="right", va="bottom", fontsize=8, color=INK)
    a1.text(dur - 0.02, off - 0.02, "ngưỡng tắt 0.45", ha="right", va="top", fontsize=8, color=INK2)
    a1.set_title("① Rebin: điểm mỗi hop 0.08 s = trọng số tam giác của 12 cửa sổ phủ hop, rồi làm mượt median 3 hop", loc="left", fontsize=10.5, fontweight="bold")

    rows = [("Nhãn (GT)", [(a, b) for a, b in sp], BLUE),
            ("② Ngưỡng đơn ≥ 0.60", single, MUTED), ("② Ngưỡng kép 0.60 / 0.45", hyst, PURPLE),
            ("③ Merge khoảng lặng < 0.5 s", merged, GREEN), ("④ Drop đoạn < 0.4 s", dropped, ORANGE),
            ("⑤ Pad (ví dụ +80 ms mỗi bên)", padded, YELLOW)]
    ys = np.arange(len(rows) + 1)[::-1] * 1.0 + 1
    for (lab, sg, c), y in zip(rows, ys[:-1]):
        a2.axhline(y, color=GRID, lw=0.6, zorder=0); seg_row(a2, y, sg, c, label=lab)
    # bin 0.5 s so với nhãn
    yb = ys[-1]
    for k, (g_, p_) in enumerate(zip(gt_bins, pb)):
        c = BLUE if g_ and p_ else ORANGE if p_ and not g_ else RED if g_ and not p_ else "white"
        a2.add_patch(Rectangle((k * BIN + 0.02, yb - 0.31), BIN - 0.04, 0.62, fc=c, alpha=0.8 if c != "white" else 1,
                               ec=GRID if c == "white" else "none"))
    a2.text(-0.15, yb, "Bin 0.5 s vs nhãn", ha="right", va="center", fontsize=9.3)
    # chú thích thay đổi
    gaps_single = [(b, c) for (a, b), (c, d) in zip(single[:-1], single[1:]) if 7.5 < b < 9.4]
    if gaps_single: note(a2, np.mean(gaps_single[0]), ys[1] + 0.05, "ngưỡng đơn cắt câu ở chỗ điểm tụt", RED, dx=-0.9, dy=0.32, ha="right")
    gap_m = [(b, c) for (a, b), (c, d) in zip(hyst[:-1], hyst[1:]) if c - b < 0.5]
    for b, c in gap_m[:1]: note(a2, (b + c) / 2, ys[3] + 0.32, "khoảng lặng ngắn được lấp", GREEN, dx=1.6, dy=0.12, ha="left")
    blip = [s for s in merged if s[1] - s[0] < 0.4]
    for a, b in blip[:1]:
        a2.plot([(a + b) / 2], [ys[4]], marker="x", ms=11, mew=2.2, color=RED)
        note(a2, (a + b) / 2, ys[4] + 0.3, "tiếng ho bị bỏ", RED, dy=0.35)
    a2.set_xlim(-0.05, dur); a2.set_ylim(0.4, ys[0] + 0.9); a2.set_yticks([]); a2.spines["left"].set_visible(False)
    a2.set_xlabel("thời gian (s)")
    a2.set_title("②–⑤ Từ điểm sang đoạn speech (thứ tự merge → drop → pad); hàng cuối: bin 0.5 s so với nhãn (pad 0)",
                 loc="left", fontsize=10.5, fontweight="bold")
    for c, lab in ((BLUE, "đúng speech"), (ORANGE, "báo nhầm (FP)"), (RED, "bỏ sót (FN)")):
        a2.add_patch(Rectangle((0, 0), 0, 0, fc=c, label=lab))
    a2.legend(loc="lower right", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(1, -0.2))
    fig.suptitle("Hậu xử lý từng bước trên một file mẫu 10 s", fontsize=13, fontweight="bold", x=0.08, ha="left", y=0.955)
    save(fig, "04_hau_xu_ly_tung_buoc.png")


# ======================= 5. Ngưỡng kép =======================
def fig_hysteresis():
    rng = np.random.default_rng(7); t = np.arange(0, 4, HOP) + HOP / 2
    s = 0.12 + 0.55 / (1 + np.exp(-(t - 0.9) * 9)) - 0.55 / (1 + np.exp(-(t - 3.1) * 9))
    s = s + 0.09 * np.sin(t * 13) + rng.normal(0, 0.035, len(t)); s = np.clip(s, 0, 1)
    on, off = 0.60, 0.45
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.8), gridspec_kw=dict(height_ratios=[2.2, 1.2], hspace=0.12), sharex=True)
    ax[0].plot(t, s, "-o", ms=3, color=BLUE, lw=1.5)
    ax[0].axhline(on, color=INK, ls="--", lw=1); ax[0].axhline(off, color=INK2, ls=":", lw=1)
    ax[0].fill_between(t, off, on, color=YELLOW, alpha=0.12, lw=0)
    ax[0].text(3.98, on + 0.015, "ngưỡng bật 0.60: bắt đầu đoạn khi điểm ≥ 0.60", ha="right", va="bottom", fontsize=8.5)
    ax[0].text(3.98, off - 0.015, "ngưỡng tắt 0.45 (= 0.60 − Δ 0.15): chỉ kết thúc khi điểm < 0.45", ha="right", va="top", fontsize=8.5, color=INK2)
    ax[0].text(2.0, 0.53, "vùng giữa hai ngưỡng: giữ nguyên trạng thái", ha="center", fontsize=8.3, color="#8a6a00")
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("điểm"); ax[0].grid(axis="y", color=GRID, lw=0.6)
    ax[0].set_title("Ngưỡng kép (--hyst) tránh việc đoạn speech bật tắt liên tục khi điểm dao động quanh ngưỡng",
                    loc="left", fontsize=11, fontweight="bold")
    a1 = ax[1]
    sg1 = frames_to_segs(s >= on, HOP); sg2 = frames_to_segs(hysteresis(s, on, off), HOP)
    seg_row(a1, 2, sg1, MUTED, h=0.55); seg_row(a1, 1, sg2, PURPLE, h=0.55)
    a1.text(-0.04, 2, f"ngưỡng đơn: {len(sg1)} đoạn vụn", ha="right", va="center", fontsize=9)
    a1.text(-0.04, 1, f"ngưỡng kép: {len(sg2)} đoạn liền", ha="right", va="center", fontsize=9)
    a1.set_ylim(0.4, 2.6); a1.set_yticks([]); a1.spines["left"].set_visible(False); a1.set_xlabel("thời gian (s)")
    a1.set_xlim(0, 4)
    save(fig, "05_nguong_kep.png")


# ======================= 6. Thứ tự pdm vs mdp =======================
def fig_order():
    dur = 4.6
    segs = [[1.0, 1.16], [1.4, 1.56], [1.8, 1.96], [2.2, 2.36], [3.6, 3.76]]   # câu hát ngắt quãng + 1 tiếng ho
    pre, post = 0.10, 0.12
    def pad(s): return [[max(0, a - pre), min(dur, b + post)] for a, b in s]
    def keep(s, d): return [x for x in s if x[1] - x[0] >= d - 1e-9]
    p1 = pad(segs)
    m1 = postprocess_segs(segs, 0, 0, 0.5, 0, dur, "mdp"); m2 = postprocess_segs(segs, 0, 0, 0.5, 0.32, dur, "mdp")
    panels = [("pdm, drop 0.32 (hiện tại)", [("Đầu vào sau ngưỡng", segs), ("pad 100/120 ms", p1), ("drop < 0.32 s (đo sau pad)", keep(p1, 0.32)),
                                           ("merge < 0.5 s", postprocess_segs(segs, pre, post, 0.5, 0.32, dur, "pdm"))], ORANGE,
               "pad làm mỗi mẩu 0.16 s thành 0.38 s → tiếng ho không bị drop", RED),
              ("pdm, drop 0.5 (để bỏ tiếng ho)", [("Đầu vào sau ngưỡng", segs), ("pad 100/120 ms", p1), ("drop < 0.5 s (đo sau pad)", keep(p1, 0.5)),
                                                ("merge < 0.5 s", postprocess_segs(segs, pre, post, 0.5, 0.5, dur, "pdm"))], ORANGE,
               "mỗi mẩu bị drop trước khi kịp merge → mất cả câu", RED),
              ("mdp (đề xuất): merge → drop → pad", [("Đầu vào sau ngưỡng", segs), ("merge < 0.5 s", m1), ("drop < 0.32 s (đo trước pad)", m2),
                                                   ("pad 0 / 0", m2)], GREEN, "câu giữ nguyên, tiếng ho bị drop", GREEN)]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), sharey=True)
    for a, (title, rows, col, msg, mcol) in zip(ax, panels):
        for i, (lab, sg) in enumerate(rows):
            y = 4 - i; a.axhline(y, color=GRID, lw=0.6, zorder=0)
            seg_row(a, y, sg, MUTED if i == 0 else col, h=0.5)
            a.text(0.05, y + 0.36, lab, fontsize=8.6, color=INK)
        a.set_title(title, loc="left", fontsize=10.8, fontweight="bold", color=INK)
        a.set_xlim(0, dur); a.set_ylim(0.3, 4.75); a.set_yticks([]); a.spines["left"].set_visible(False)
        a.set_xlabel("thời gian (s)")
        a.axvspan(0.95, 2.4, color=BLUE, alpha=0.06, lw=0); a.text(1.67, 4.6, "câu hát ngắt quãng", ha="center", fontsize=8, color=BLUE)
        a.axvspan(3.55, 3.8, color=RED, alpha=0.06, lw=0); a.text(3.67, 4.6, "tiếng ho", ha="center", fontsize=8, color=RED)
        a.text(dur / 2, 0.45, msg, ha="center", fontsize=8.4, color=mcol)
    fig.suptitle("Thứ tự pad / drop / merge (--order): với pdm, drop đo sau pad nên không thể vừa bỏ tiếng ho vừa giữ câu bị vỡ",
                 fontsize=12.2, fontweight="bold", x=0.02, ha="left", y=1.02)
    save(fig, "06_thu_tu_pdm_mdp.png")


# ======================= 7. Loại lỗi =======================
def fig_errors():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    p = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])
    b = boundary_mask(y); n = len(y)
    fig, ax = plt.subplots(figsize=(12, 3.9))
    cmap = {"TP": BLUE, "TN": "white", "FEC": RED, "MSC": RED, "OVER": ORANGE, "NDS": ORANGE}
    for k in range(n):
        for row, v in ((2, y[k]), (1, p[k])):
            ax.add_patch(Rectangle((k, row - 0.35), 1, 0.7, fc=BLUE if v else "white", alpha=0.45 if v else 1, ec=INK2, lw=0.6))
        if b[k]: ax.add_patch(Rectangle((k, 2.45), 1, 0.18, fc=YELLOW, ec="none", alpha=0.7))
        if y[k] != p[k]:
            kind = ("FEC" if b[k] else "MSC") if y[k] else ("OVER" if b[k] else "NDS")
            c = cmap[kind]
            ax.add_patch(Rectangle((k + 0.08, -0.15), 0.84, 0.5, fc=c, alpha=0.9 if kind in ("MSC", "NDS") else 0.45, ec="none"))
            ax.text(k + 0.5, 0.1, kind, ha="center", va="center", fontsize=7.6, color="white" if kind in ("MSC", "NDS") else INK,
                    fontweight="bold")
    ax.text(-0.2, 2, "Nhãn", ha="right", va="center", fontsize=9.5); ax.text(-0.2, 1, "Dự đoán", ha="right", va="center", fontsize=9.5)
    ax.text(-0.2, 0.1, "Lỗi", ha="right", va="center", fontsize=9.5); ax.text(-0.2, 2.54, "bin biên", ha="right", va="center", fontsize=8, color="#8a6a00")
    ax.set_xlim(-0.1, n + 0.1); ax.set_ylim(-0.4, 3.4); ax.axis("off")
    ax.text(0, 3.2, "Phân loại lỗi theo bin 0.5 s: lỗi biên (nhạt) là lệch ±1 bin quanh chuyển trạng thái, lỗi thật (đậm) mới là chỗ cần sửa",
            fontsize=11, fontweight="bold")
    leg = [("FEC", RED, 0.45, "bỏ sót ở đầu/cuối đoạn speech"), ("MSC", RED, 0.9, "bỏ sót giữa đoạn"),
           ("OVER", ORANGE, 0.45, "báo nhầm sát đoạn speech"), ("NDS", ORANGE, 0.9, "báo nhầm giữa vùng non-speech")]
    for i, (k, c, al, t) in enumerate(leg):
        x, yl = (i % 2) * 10.0, -0.85 - 0.5 * (i // 2)
        ax.add_patch(Rectangle((x, yl - 0.17), 0.6, 0.34, fc=c, alpha=al, ec="none", clip_on=False))
        ax.text(x + 0.8, yl, f"{k}: {t}", fontsize=8.8, va="center")
    save(fig, "07_loai_loi.png")


# ======================= 8. Các mục của evaluate_real.py =======================
def fig_eval_steps():
    fig, ax = canvas(18.2, 6.4, (14.5, 5.1))
    ax.text(0.1, 6.25, "evaluate_real.py chạy những gì (theo thứ tự) và ghi ra file nào", fontsize=13, fontweight="bold", va="top")
    st = [("1. Ghép &\nkiểm tra dữ liệu", ["structure.csv", "lag_per_file.csv", "shift_sweep.csv", "problems"], MUTED),
          ("2. Chia dev / test\ntheo file", ["per_file.csv (split)", "dataset_profile.csv", "fig5, fig6"], MUTED),
          ("3. Tầng A:\nAUC điểm thô", ["tierA.csv", "fig2 (ROC, PR, DET)"], BLUE),
          ("4. Chọn ngưỡng\ntrên dev", ["thr_sweep.csv", "threshold_transfer.csv"], PURPLE),
          ("5. Tầng B\ntrên test", ["tierB.csv", "category.csv", "event_by_threshold.csv", "fig3, fig4"], GREEN),
          ("6. Ablation &\nso sánh hậu xử lý", ["ablation.csv", "grid_dev.csv", "pp_compare.csv", "pp_tune_trials.csv", "fig7"], ORANGE)]
    w, gap = 2.72, 0.28
    for i, (t, outs, c) in enumerate(st):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 3.6), w, 1.5, boxstyle="round,pad=0,rounding_size=0.1", fc=c, alpha=0.13, ec="none"))
        ax.add_patch(FancyBboxPatch((x, 3.6), w, 1.5, boxstyle="round,pad=0,rounding_size=0.1", fc="none", ec=c, lw=1.5))
        ax.text(x + w / 2, 4.35, t, ha="center", va="center", fontsize=10, fontweight="bold")
        for j, o in enumerate(outs): ax.text(x + 0.08, 3.15 - 0.36 * j, "▸ " + o, fontsize=7.9, color=INK2, family="DejaVu Sans Mono")
        if i < len(st) - 1: arrow(ax, (x + w, 4.35), (x + w + gap, 4.35))
    # dải dev / test
    ax.add_patch(Rectangle((0.2 + 3 * (w + gap), 5.3), w, 0.35, fc=PURPLE, alpha=0.25, ec="none"))
    ax.text(0.2 + 3 * (w + gap) + w / 2, 5.47, "chỉ dùng dev", ha="center", va="center", fontsize=8.5)
    ax.add_patch(Rectangle((0.2 + 4 * (w + gap), 5.3), 2 * w + gap, 0.35, fc=GREEN, alpha=0.25, ec="none"))
    ax.text(0.2 + 4 * (w + gap) + w + gap / 2, 5.47, "đánh giá trên test (ngưỡng từ dev)", ha="center", va="center", fontsize=8.5)
    ax.add_patch(Rectangle((0.2, 5.3), 3 * w + 2 * gap, 0.35, fc=MUTED, alpha=0.18, ec="none"))
    ax.text(0.2 + 1.5 * w + gap, 5.47, "toàn bộ file", ha="center", va="center", fontsize=8.5)
    ax.text(0.2, 1.2, "Mọi lần chạy đều ghi summary.json (tóm tắt + tham số đã dùng) → make_excel_real.py đọc thư mục này.",
            fontsize=9, color=INK2)
    save(fig, "08_evaluate_real_cac_buoc.png")


# ======================= 9. Quy trình chọn cấu hình hậu xử lý =======================
def fig_tune_flow():
    fig, ax = canvas(17, 5.2, (14, 4.4))
    ax.text(0.1, 5.05, "Quy trình chọn cấu hình hậu xử lý (Phần 3)", fontsize=13, fontweight="bold", va="top")
    steps = [("Bước 1", "So sánh từng thay đổi", ["evaluate_real.py", "→ fig7, pp_compare.csv"], BLUE),
             ("Bước 2", "Tìm cấu hình trên dev", ["--tune 300", "→ dòng T"], PURPLE),
             ("Bước 3", "Kiểm tra độ ổn định", ["--seed 1, 2, 3 …", "--tune-seed khác"], ORANGE),
             ("Bước 4", "Chạy báo cáo cuối", ["--preset de-xuat", "hoặc tham số của T"], GREEN),
             ("Bước 5", "Áp vào lib.so", ["đổi tham số; cân nhắc", "hop 0.16 / 0.24 s"], MUTED)]
    w, gap = 3.0, 0.42
    for i, (k, t, ls, c) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.9), w, 1.9, boxstyle="round,pad=0,rounding_size=0.12", fc=c, alpha=0.13, ec="none"))
        ax.add_patch(FancyBboxPatch((x, 1.9), w, 1.9, boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec=c, lw=1.5))
        ax.text(x + w / 2, 3.45, k, ha="center", fontsize=8.5, color=INK2)
        ax.text(x + w / 2, 3.0, t, ha="center", fontsize=9.8, fontweight="bold")
        for j, l in enumerate(ls): ax.text(x + w / 2, 2.5 - 0.32 * j, l, ha="center", fontsize=8.6, color=INK2)
        if i < len(steps) - 1: arrow(ax, (x + w, 2.85), (x + w + gap, 2.85))
    x3 = 0.2 + 2 * (w + gap)
    arrow(ax, (x3 + w / 2, 1.9), (0.2 + w / 2, 1.9), color=RED, rad=-0.35)
    ax.text(x3 / 2 + w / 2, 0.55, "không lặp lại được / làm hỏng một category → quay lại, giữ thay đổi ổn định hơn",
            ha="center", fontsize=8.6, color=RED)
    save(fig, "09_quy_trinh_hau_xu_ly.png")


# ======================= 10. Cách đọc CI (fig7) =======================
def fig_read_ci():
    fig, ax = plt.subplots(figsize=(10, 3.3))
    rows = [("C2  tốt hơn B0", 0.012, 0.004, 0.020, GREEN), ("C6  chưa phân biệt được", 0.002, -0.004, 0.008, MUTED),
            ("C1  kém hơn B0", -0.013, -0.021, -0.005, RED)]
    for i, (lab, v, lo, hi, c) in enumerate(rows):
        yv = 2 - i; ax.hlines(yv, lo, hi, color=c, lw=3); ax.plot(v, yv, "o", ms=8, color=c)
        ax.text(-0.03, yv, lab, ha="right", va="center", fontsize=9.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.annotate("điểm = ΔF1 trên test", xy=(0.012, 2), xytext=(0.017, 2.45), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.annotate("vạch = CI 95% (bootstrap ghép cặp theo file)", xy=(0.006, 1), xytext=(0.011, 1.4), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.annotate("vạch 0 = bằng B0", xy=(0, -0.3), xytext=(0.004, -0.55), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(0.024, 2, "CI nằm hẳn bên phải 0", va="center", fontsize=8.5, color=GREEN)
    ax.text(0.024, 1, "CI chứa 0", va="center", fontsize=8.5, color=INK2)
    ax.text(0.024, 0, "CI nằm hẳn bên trái 0", va="center", fontsize=8.5, color=RED)
    ax.set_xlim(-0.03, 0.04); ax.set_ylim(-0.8, 2.8); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("ΔF1 so với B0 (với ΔDCF: thấp = tốt, nên màu đảo chiều)")
    ax.set_title("Cách đọc một dòng của Hình 7", loc="left", fontsize=11.5, fontweight="bold", x=-0.25)
    save(fig, "10_cach_doc_fig7.png")


if __name__ == "__main__":
    for f in (fig_workflow, fig_folders, fig_windows, fig_steps, fig_hysteresis, fig_order, fig_errors, fig_eval_steps,
              fig_tune_flow, fig_read_ci):
        f()
