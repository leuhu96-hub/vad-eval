"""Bảng so sánh trực quan các phương án hậu xử lý, từng bước -> docs/img/13_*.png … 21_*.png.

Mỗi bảng: mỗi phương án một hàng, có hình minh hoạ nhỏ trên cùng một file mẫu 10 s (tín hiệu tự tạo, chạy đúng các
hàm của vadlib), cách làm, ưu, nhược, chi phí và đánh giá. Số FP / FN trên hình chỉ đếm trên file mẫu để minh hoạ.
    python docs/make_compare_tables.py            # tiếng Việt -> docs/img/13_….png
    python docs/make_compare_tables.py --lang en  # tiếng Anh  -> docs/img/13_…_en.png (bản dịch: compare_tables_en.py)
"""
import re, sys, textwrap
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from make_figures import (OUT, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE, INK, INK2, MUTED, GRID, TINT, save)
from vadlib import (rescore_hop, smooth_scores, hysteresis, frames_to_segs, postprocess_segs, segs_to_bins_fast,
                    HOP, BIN, WIN, HALF_WIN)

DUR = 10.0
LANG = "en" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en" else "vi"
_VI = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
if LANG == "en":
    from compare_tables_en import EN
    MISSING = set()

def tr(t):
    """Chuỗi hiển thị theo LANG; bản tiếng Anh lấy từ compare_tables_en.EN (thiếu bản dịch -> báo lỗi khi chạy xong)."""
    if LANG == "vi" or not isinstance(t, str): return t
    if t in EN: return EN[t]
    if _VI.search(t): MISSING.add(t)
    return t
TAG = {"hiện tại": ("#52514e", "#ecebe7"), "đề xuất": ("#0d7a52", "#dcf3e9"), "thử thêm": ("#8a6a00", "#fbf0d2"),
       "tránh": ("#b42f2e", "#fbe1e0"), "chưa có": ("#4a3aa7", "#e8e5f6")}


# ---------------- file mẫu ----------------
def example(noise=0.03, seed=4):
    """Như make_figures.make_example, thêm tham số nhiễu: ngắt hơi 2.55 s, tiếng ho 6.75 s, nói nhỏ 8.6 s."""
    rng = np.random.default_rng(seed); FINE = 0.02; t = np.arange(0, DUR, FINE)
    sp = [(0.9, 2.3), (2.8, 4.0), (5.0, 5.8), (7.7, 9.4)]
    g = np.zeros(len(t))
    for a, b in sp: g[(t >= a) & (t < b)] = 1
    st = np.round(np.arange(0, DUR - WIN + 1e-9, HOP), 4); c = st + HALF_WIN
    frac = np.array([g[(t >= s) & (t < s + WIN)].mean() for s in st])
    sc = 0.06 + 0.9 / (1 + np.exp(-12 * (frac - 0.5)))
    for mu, sg, amp in ((2.55, 0.12, -0.15), (6.75, 0.16, 0.95), (8.6, 0.2, -0.62)):
        sc += amp * np.exp(-0.5 * ((c - mu) / sg) ** 2)
    sc = np.clip(sc + rng.normal(0, noise, len(st)), 0.01, 0.99)
    m = pd.DataFrame(dict(start=st, end=st + WIN, score=sc)); m["center"] = c
    gt = (np.add.reduceat(g, np.arange(0, len(g), int(BIN / FINE))) / (BIN / FINE) >= 0.5).astype(int)
    return m, sp, gt

M, SP, GT = example()
M_NOISY, _, _ = example(noise=0.09, seed=11)
TC = (np.arange(int(round(DUR / HOP))) + 0.5) * HOP
FR_CUR = rescore_hop(M, DUR, "tri", 5, "mean")                  # rebin hiện tại
FR_REC = rescore_hop(M, DUR, "tri", 3, "median")                # rebin đề xuất


def pipeline(fr=FR_REC, on=0.60, off=0.45, order="mdp", pre=0.0, post=0.0, gap=0.5, drop=0.32):
    segs = postprocess_segs(frames_to_segs(hysteresis(fr, on, off if off is not None and off < on else None), HOP),
                            pre, post, gap, drop, DUR, order)
    return segs, segs_to_bins_fast(segs, len(GT))


def counts(pb):
    return int(((pb == 1) & (GT == 0)).sum()), int(((pb == 0) & (GT == 1)).sum())


# ---------------- hình nhỏ ----------------
def mini_frame(ax):
    for a, b in SP: ax.axvspan(a, b, color=BLUE, alpha=0.08, lw=0)
    ax.set_xlim(0, DUR); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)

def mini_score(ax, fr, ref=FR_CUR, thr=0.6, thr2=None, color=BLUE, show_ref=True):
    mini_frame(ax); ax.set_ylim(-0.05, 1.05)
    if show_ref and ref is not None: ax.plot(TC, ref, color=MUTED, lw=0.9, ls="--")
    ax.plot(TC, fr, color=color, lw=1.6)
    ax.axhline(thr, color=INK, lw=0.7, ls=":")
    if thr2 is not None: ax.axhline(thr2, color=INK2, lw=0.6, ls=":")

def mini_bars(ax, segs, color=GREEN, show_counts=True, pb=None, extra=""):
    mini_frame(ax); ax.set_ylim(0, 1)
    for a, b in SP: ax.add_patch(Rectangle((a, 0.62), b - a, 0.26, fc=BLUE, alpha=0.75, ec="none"))
    for a, b in segs: ax.add_patch(Rectangle((a, 0.18), b - a, 0.26, fc=color, alpha=0.85, ec="none"))
    ax.text(-0.1, 0.75, tr("nhãn"), ha="right", va="center", fontsize=6.8, color=INK2)
    ax.text(-0.1, 0.31, tr("ra"), ha="right", va="center", fontsize=6.8, color=INK2)
    if show_counts:
        fp, fn = counts(pb if pb is not None else segs_to_bins_fast(segs, len(GT)))
        ax.text(DUR, 0.02, tr("{n} đoạn · FP {fp} bin · FN {fn} bin").format(n=len(segs), fp=fp, fn=fn) + extra, ha="right", va="bottom", fontsize=7, color=INK2)


# ---------------- khung bảng ----------------
COLW = [2.35, 3.9, 6.35, 1.35, 1.25]          # inch
HEAD = ["Phương án", "Minh hoạ trên file mẫu", "Cách làm · ưu (+) · nhược (−)", "Chi phí", "Đánh giá"]

def wrap(t, n): return "\n".join(textwrap.wrap(t, n, break_long_words=False))

def visual_table(fname, title, subtitle, rows, legend=None, row_h=1.3):
    title, subtitle, legend = tr(title), tr(subtitle), tr(legend)
    W = sum(COLW) + 0.3; H = 1.25 + 0.42 + row_h * len(rows) + (0.45 if legend else 0.2)
    fig = plt.figure(figsize=(W, H)); X = np.r_[0.15, 0.15 + np.cumsum(COLW)]
    def fx(x): return x / W
    def fy(y): return 1 - y / H                      # y tính từ trên xuống (inch)
    fig.text(fx(0.15), fy(0.38), title, fontsize=14, fontweight="bold", color=INK, va="center")
    fig.text(fx(0.15), fy(0.78), subtitle, fontsize=9, color=INK2, va="center")
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1); bg.axis("off"); bg.set_xlim(0, W); bg.set_ylim(H, 0)
    y0 = 1.25
    bg.add_patch(Rectangle((X[0], y0), X[-1] - X[0], 0.42, fc="#1f4e78", ec="none"))
    for j, h in enumerate(HEAD): bg.text(X[j] + 0.1, y0 + 0.21, tr(h), color="white", fontsize=9.5, fontweight="bold", va="center")
    for i, r in enumerate(rows):
        top = y0 + 0.42 + i * row_h; fg, fill = TAG[r["tag"]]
        bg.add_patch(Rectangle((X[0], top), X[-1] - X[0], row_h, fc=TINT if i % 2 else "white", ec="none"))
        if r["tag"] in ("đề xuất", "hiện tại"):
            bg.add_patch(Rectangle((X[0] + 0.01, top + 0.03), X[-1] - X[0] - 0.02, row_h - 0.06, fc="none", ec=fg, lw=1.6 if r["tag"] == "đề xuất" else 1.0,
                                   ls="-" if r["tag"] == "đề xuất" else "--"))
        bg.plot([X[0], X[-1]], [top + row_h] * 2, color=GRID, lw=0.8)
        bg.text(X[0] + 0.12, top + 0.33, wrap(tr(r["name"]), 24), fontsize=9.6, fontweight="bold", color=INK, va="center")
        if r.get("param"): bg.text(X[0] + 0.12, top + row_h - 0.3, wrap(tr(r["param"]), 30), fontsize=7.8, color=PURPLE, va="center")
        lines = [("", tr(r["how"]), INK)] + [("+ ", tr(p), "#0d7a52") for p in r.get("pros", [])] + [("− ", tr(c), "#b42f2e") for c in r.get("cons", [])]
        yy = top + 0.16
        for pre, t, col in lines:
            txt = wrap(pre + t, 86); bg.text(X[2] + 0.1, yy, txt, fontsize=8.1, color=col, va="top"); yy += 0.165 * (txt.count("\n") + 1) + 0.02
        bg.text(X[3] + 0.1, top + row_h / 2, wrap(tr(r["cost"]), 14), fontsize=8, color=INK2, va="center")
        bg.add_patch(FancyBboxPatch((X[4] + 0.12, top + row_h / 2 - 0.16), COLW[4] - 0.3, 0.32, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fill, ec=fg, lw=1))
        bg.text(X[4] + 0.12 + (COLW[4] - 0.3) / 2, top + row_h / 2, tr(r["tag"]), fontsize=8.6 if len(tr(r["tag"])) <= 9 else 7.4, fontweight="bold", color=fg, ha="center", va="center")
        ax = fig.add_axes([fx(X[1] + 0.15), fy(top + row_h - 0.1), fx(COLW[1] - 0.3), (row_h - 0.2) / H])
        ax.set_facecolor("none"); r["mini"](ax)
    if legend: fig.text(fx(0.15), fy(H - 0.22), legend, fontsize=8.2, color=INK2, va="center")
    save_lang(fig, fname)


# ======================= bảng tổng quan =======================
def table_overview():
    rows = [("1. Gộp cửa sổ (Rebin)", "tam giác, xác suất", "giữ tam giác (thử Hann, logit)", "--overlap, --domain", "điểm cửa sổ đại diện tốt nhất cho tâm cửa sổ", "AUC", "≈ 0"),
            ("2. Làm mượt", "mean 5 hop", "median 3 hop", "--smooth 3 --smooth-kind median", "kernel tam giác đã mượt; median giữ biên, bỏ gai", "AUC, biên", "≈ 0"),
            ("3. Ngưỡng", "ngưỡng đơn 0.6026", "ngưỡng kép 0.60 / 0.45", "--hyst 0.15", "không bật tắt liên tục khi điểm dao động", "MR, số đoạn vụn", "≈ 0"),
            ("4. Thứ tự", "pad → drop → merge", "merge → drop → pad", "--order mdp", "ghép mẩu speech vỡ trước khi lọc", "MR (hát, đám đông)", "≈ 0"),
            ("5. Merge", "< 500 ms", "< 500 ms (tune 0.3–0.8)", "--merge-gap", "khớp độ phân giải nhãn 0.5 s", "FA ở khoảng lặng", "≈ 0"),
            ("6. Drop", "0 (đo sau pad)", "0.32 s đo trước pad (tune 0.32–0.5)", "--drop 0.32", "bỏ gai do ho, gõ, nhạc", "FAR", "≈ 0"),
            ("7. Pad", "100 / 120 ms", "0 / 0 (tune −0.16…+0.16)", "--pad-pre 0 --pad-post 0", "YAMNet đã nở biên ~0.48 s", "FAR ở biên", "≈ 0"),
            ("8. Hop cửa sổ", "0.08 s", "thử 0.16 / 0.24 s", "R-h16, R-h24 (mục 6)", "số lần chạy model giảm 2–3×", "AUC giảm nhẹ", "CPU ÷2–3")]
    cols = ["Bước", "Hiện tại", "Đề xuất", "Tham số dòng lệnh", "Vì sao", "Ảnh hưởng chính", "Chi phí thêm"]
    cw = [2.1, 1.85, 2.6, 2.75, 3.6, 1.7, 1.3]; W = sum(cw) + 0.3; rh = 0.52; H = 1.2 + 0.45 + rh * len(rows) + 0.75
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    X = np.r_[0.15, 0.15 + np.cumsum(cw)]
    ax.text(0.15, 0.4, tr("Tổng quan: hậu xử lý hiện tại → cấu hình đề xuất (--preset de-xuat)"), fontsize=14, fontweight="bold")
    ax.text(0.15, 0.8, tr("Mỗi bước một hàng. Đề xuất là điểm khởi đầu; giá trị cuối cùng chọn bằng so sánh (mục 6) và --tune trên dev của dữ liệu thật."),
            fontsize=9, color=INK2)
    y0 = 1.2; ax.add_patch(Rectangle((X[0], y0), X[-1] - X[0], 0.45, fc="#1f4e78", ec="none"))
    for j, c in enumerate(cols): ax.text(X[j] + 0.1, y0 + 0.225, tr(c), color="white", fontsize=9.5, fontweight="bold", va="center")
    for i, r in enumerate(rows):
        top = y0 + 0.45 + i * rh
        ax.add_patch(Rectangle((X[0], top), X[-1] - X[0], rh, fc=TINT if i % 2 else "white", ec="none"))
        ax.plot([X[0], X[-1]], [top + rh] * 2, color=GRID, lw=0.8)
        for j, v in enumerate(r):
            v = tr(v); kw = dict(fontsize=8.8, va="center", color=INK)
            if j == 0: kw.update(fontweight="bold")
            if j == 1: kw.update(color=INK2)
            if j == 2: kw.update(color="#0d7a52", fontweight="bold")
            if j == 3: kw.update(family="DejaVu Sans Mono", fontsize=7.9, color=PURPLE)
            if j == 6: kw.update(color=ORANGE if "CPU" in v else INK2)
            ax.text(X[j] + 0.1, top + rh / 2, wrap(v, int(cw[j] * 12.5)), **kw)
        if r[1] != r[2]: ax.text(X[2] - 0.02, top + rh / 2, "→", fontsize=11, color="#0d7a52", ha="right", va="center")
    ax.text(0.15, H - 0.45, tr("Hậu xử lý chỉ tốn vài mili-giây cho cả giờ audio (một lượt O(N)); chi phí thật nằm ở số lần chạy model, tức hop cửa sổ."),
            fontsize=8.6, color=INK2)
    save_lang(fig, "13_bang_tong_quan_toi_uu.png")


# ======================= bước 1: gộp cửa sổ =======================
def table_aggregation():
    def fr(w, d="prob"): return rescore_hop(M, DUR, w, 1, "mean", d)
    rows = [
        dict(name="Trung bình đều", param="--overlap mean", tag="tránh", cost="12 phép cộng / hop",
             how="Mọi cửa sổ phủ hop có trọng số bằng nhau.", pros=["Đơn giản nhất."],
             cons=["Cửa sổ chỉ chạm speech ở mép cũng góp đủ → biên đoạn nở rộng, gai lan ra ±0.48 s."],
             mini=lambda ax: mini_score(ax, fr("mean"), ref=fr("tri"))),
        dict(name="Trọng số tam giác", param="--overlap tri", tag="hiện tại", cost="12 phép nhân / hop",
             how="Trọng số 1 ở tâm cửa sổ, giảm tuyến tính về 0 ở hai mép.",
             pros=["Đúng với nhận xét của Google: điểm YAMNet đại diện tốt nhất cho tâm patch.", "Giữ được trong cấu hình đề xuất."],
             cons=["Vẫn còn nở biên do bản chất cửa sổ 0.96 s."], mini=lambda ax: mini_score(ax, fr("tri"), ref=None)),
        dict(name="Trọng số Hann", param="--overlap hann", tag="thử thêm", cost="như tam giác (bảng tính sẵn)",
             how="0.5·(1 + cos(π·d/0.48)): giảm trọng số mép mạnh hơn tam giác.",
             pros=["Bù tốt hơn một chút cho hiện tượng nở biên."], cons=["Khác tam giác rất ít; chỉ giữ nếu dev cho thấy hơn."],
             mini=lambda ax: mini_score(ax, fr("hann"), ref=fr("tri"))),
        dict(name="Tam giác trong miền logit", param="--domain logit", tag="thử thêm", cost="+ log/exp mỗi cửa sổ",
             how="Đổi điểm sang log-odds, gộp có trọng số, đổi lại xác suất.",
             pros=["Điểm 'quyết đoán' hơn; có ích nếu model rụt rè (điểm dồn quanh 0.5)."],
             cons=["Có hại nếu model đã tự tin quá; 12 cửa sổ tương quan mạnh nên lợi ích không chắc."],
             mini=lambda ax: mini_score(ax, fr("tri", "logit"), ref=fr("tri"))),
        dict(name="Median 12 cửa sổ", param="--overlap median", tag="thử thêm", cost="sắp xếp 12 số / hop",
             how="Lấy trung vị điểm các cửa sổ phủ hop.",
             pros=["Bền với cửa sổ ngoại lai; MarbleNet: ngang mean ở overlap cao (AUROC 0.876 vs 0.866)."],
             cons=["Không ưu tiên tâm cửa sổ như tam giác."], mini=lambda ax: mini_score(ax, fr("median"), ref=fr("tri"))),
        dict(name="Max các cửa sổ", param="--overlap max", tag="tránh", cost="12 phép so sánh / hop",
             how="Lấy điểm lớn nhất trong các cửa sổ phủ hop.", pros=["Recall cao nhất."],
             cons=["Biên nở thêm, gai (ho, nhạc) thành đoạn dài → FAR tăng mạnh."], mini=lambda ax: mini_score(ax, fr("max"), ref=fr("tri")))]
    visual_table("14_so_sanh_gop_cua_so.png", "Bước 1 – Gộp điểm các cửa sổ chồng nhau (Rebin)",
                 "Mỗi hop 0.08 s nằm trong tối đa 12 cửa sổ 0.96 s. Hình nhỏ: đường xanh = phương án, nét đứt xám = tam giác (hiện tại), "
                 "chấm chấm = ngưỡng 0.6, nền xanh = speech theo nhãn. Chưa làm mượt.", rows,
                 legend="Bước này là bước duy nhất (cùng làm mượt và hop) làm thay đổi AUC: so các phương án bằng AUC_hop_bin trong pp_compare.csv.")


# ======================= bước 2: làm mượt =======================
def table_smoothing():
    base = rescore_hop(M_NOISY, DUR, "tri", 1)
    def sm(n, k): return smooth_scores(base, n, k)
    def mn(n, k, ref=True): return lambda ax: mini_score(ax, sm(n, k), ref=base if ref else None)
    rows = [
        dict(name="Không làm mượt", param="--smooth 1", tag="thử thêm", cost="0",
             how="Dùng thẳng điểm sau trọng số tam giác.",
             pros=["Kernel tam giác 0.96 s vốn đã là một bộ lọc mượt; ở overlap 92% làm mượt thêm lợi rất ít."],
             cons=["Còn gai nhỏ khi điểm model nhiễu."], mini=mn(1, "mean", False)),
        dict(name="Trung bình trượt 5 hop", param="--smooth 5 --smooth-kind mean", tag="hiện tại", cost="O(1) / hop (tổng trượt)",
             how="Trung bình 5 hop (0.40 s) quanh mỗi hop.", pros=["Bỏ nhiễu tốt."],
             cons=["Cộng dồn với kernel tam giác → mờ khoảng lặng ngắn giữa câu, biên tụt vào trong."], mini=mn(5, "mean")),
        dict(name="Median trượt 3 hop", param="--smooth 3 --smooth-kind median", tag="đề xuất", cost="sắp xếp 3 số / hop",
             how="Trung vị 3 hop (0.24 s).",
             pros=["Bỏ gai đơn lẻ mà không làm mờ biên (median giữ bậc thang)."], cons=["Gai dài ≥ 2 hop vẫn còn."], mini=mn(3, "median")),
        dict(name="Median trượt 5 hop", param="--smooth 5 --smooth-kind median", tag="thử thêm", cost="sắp xếp 5 số / hop",
             how="Trung vị 5 hop (0.40 s).", pros=["Bỏ gai tới 2 hop."],
             cons=["Dinkel & Yu: median có thể xoá dự đoán đúng ngắn và dịch biên."], mini=mn(5, "median")),
        dict(name="Gauss 5 hop", param="--smooth 5 --smooth-kind gauss", tag="thử thêm", cost="O(n) / hop",
             how="Trung bình có trọng số Gauss, sigma = n/4.", pros=["Mượt hơn mean, ít mờ hơn ở cùng độ dài."],
             cons=["Vẫn là lọc tuyến tính: gai thành bướu chứ không mất."], mini=mn(5, "gauss"))]
    visual_table("15_so_sanh_lam_muot.png", "Bước 2 – Làm mượt điểm theo hop",
                 "File mẫu có thêm nhiễu để thấy khác biệt. Hình nhỏ: đường xanh = sau làm mượt, nét đứt xám = trước làm mượt "
                 "(chỉ trọng số tam giác), chấm chấm = ngưỡng 0.6.", rows,
                 legend="Bằng chứng: MarbleNet (ICASSP 2021) – median và mean chênh nhau không đáng kể ở overlap 87.5%; overlap của YAMNet ở đây là 92%.")


# ======================= bước 3: ngưỡng =======================
def viterbi(p, stay=0.97):
    lp = np.log(np.clip(np.c_[1 - p, p], 1e-6, 1)); lt = np.log(np.array([[stay, 1 - stay], [1 - stay, stay]]))
    d = lp[0].copy(); bk = np.zeros((len(p), 2), int)
    for i in range(1, len(p)):
        c = d[:, None] + lt; bk[i] = c.argmax(0); d = c.max(0) + lp[i]
    s = np.zeros(len(p), int); s[-1] = d.argmax()
    for i in range(len(p) - 1, 0, -1): s[i - 1] = bk[i, s[i]]
    return s.astype(bool)

def mini_mask(ax, fr, mask, thr=0.6, thr2=None):
    mini_frame(ax); ax.set_ylim(-0.32, 1.05)
    ax.plot(TC, fr, color=BLUE, lw=1.3); ax.axhline(thr, color=INK, lw=0.7, ls=":")
    if thr2 is not None: ax.axhline(thr2, color=INK2, lw=0.6, ls=":")
    segs = frames_to_segs(mask, HOP)
    for a, b in segs: ax.add_patch(Rectangle((a, -0.28), b - a, 0.17, fc=PURPLE, alpha=0.85, ec="none"))
    ax.text(DUR, 1.02, tr("{n} đoạn").format(n=len(segs)), ha="right", va="top", fontsize=7, color=INK2)

def table_threshold():
    fr = FR_REC
    pct = float(np.percentile(fr, 60))
    rows = [
        dict(name="Ngưỡng đơn (Youden)", param="0.6026, chọn trên dữ liệu thiết kế", tag="hiện tại", cost="1 phép so sánh / hop",
             how="Speech khi điểm ≥ θ; θ tối đa TPR − FPR trên ROC.", pros=["Đơn giản, dễ giải thích."],
             cons=["Youden ngầm coi chi phí miss = FA và tỉ lệ speech 50%; không tối ưu F1/DCF trên phân phối thật.",
                   "Điểm dao động quanh θ → đoạn bị cắt vụn."], mini=lambda ax: mini_mask(ax, fr, fr >= 0.6)),
        dict(name="Ngưỡng đơn chọn trên dev", param="--criterion DCF-min | F1-max", tag="đề xuất", cost="quét ngưỡng trên dev (offline)",
             how="Chọn θ tối thiểu DCF (hoặc tối đa F1) qua toàn pipeline trên dev, đánh giá trên test (hình: ví dụ θ = 0.52).",
             pros=["Đúng mục tiêu thật (trọng số miss/FA), đúng phân phối dữ liệu.", "evaluate_real.py đã làm sẵn cho mọi cấu hình."],
             cons=["Cần dev đủ lớn, phân tầng theo category."], mini=lambda ax: mini_mask(ax, fr, fr >= 0.52, thr=0.52)),
        dict(name="Ngưỡng kép (hysteresis)", param="--hyst 0.15 → bật 0.60 / tắt 0.45", tag="đề xuất", cost="1 so sánh / hop",
             how="Bắt đầu đoạn khi điểm ≥ θ, chỉ kết thúc khi điểm < θ − Δ (Silero, pyannote, NeMo, SpeechBrain đều dùng).",
             pros=["Hết bật tắt liên tục ở hát, rap, đám đông; ít đoạn vụn."], cons=["Một tham số nữa (Δ) cần tune; gai ngắn bị nở rộng hơn."],
             mini=lambda ax: mini_mask(ax, fr, hysteresis(fr, 0.6, 0.45), thr2=0.45)),
        dict(name="HMM / Viterbi 2 trạng thái", param="(chưa có trong vad-eval)", tag="chưa có", cost="O(N) nhưng cần ước lượng xác suất chuyển",
             how="Giải mã chuỗi trạng thái speech / non-speech tối ưu với xác suất chuyển trạng thái.",
             pros=["TS-VAD (CHiME-6): tốt hơn ngưỡng + median khoảng 0.6–0.7 điểm DER."],
             cons=["Lợi ích thêm nhỏ khi đã có ngưỡng kép + min duration; cần dữ liệu để ước lượng."],
             mini=lambda ax: mini_mask(ax, fr, viterbi(fr))),
        dict(name="Ngưỡng theo từng file", param="θ = max(θ chung, phân vị điểm của file)", tag="tránh", cost="1 lượt phân vị / file",
             how="Nâng ngưỡng cho file có điểm nền cao (vd nhạc nền to).", pros=["Có thể giúp file nhiều nhạc nền."],
             cons=["File gần như toàn speech hoặc toàn non-speech bị cắt sai; chỉ dùng khi dev cho thấy lợi rõ."],
             mini=lambda ax: mini_mask(ax, fr, fr >= max(0.6, pct), thr=max(0.6, pct)))]
    visual_table("16_so_sanh_nguong.png", "Bước 3 – Ngưỡng: từ điểm sang speech / non-speech",
                 "Hình nhỏ: đường xanh = điểm mỗi hop (đã rebin + median 3), chấm chấm = ngưỡng, dải tím = các đoạn speech sau ngưỡng.",
                 rows, legend="Chọn ngưỡng bằng F1 / DCF / MR / FAR. AUC không đổi theo ngưỡng nên không dùng AUC để so bước này.")


# ======================= bước 4: thứ tự =======================
def table_order():
    segs = [[1.0, 1.16], [1.4, 1.56], [1.8, 1.96], [2.2, 2.36], [3.6, 3.76]]
    def mini_o(out, col):
        def f(ax):
            ax.set_xlim(0.6, 4.2); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values(): s_.set_visible(False)
            for a, b in segs: ax.add_patch(Rectangle((a, 0.62), b - a, 0.26, fc=MUTED, alpha=0.8, ec="none"))
            for a, b in out: ax.add_patch(Rectangle((a, 0.18), b - a, 0.26, fc=col, alpha=0.85, ec="none"))
            if not out: ax.text(2.4, 0.31, tr("không còn đoạn nào → mất cả câu"), ha="center", va="center", fontsize=7.6, color=RED)
            ax.text(0.58, 0.75, tr("vào"), ha="right", va="center", fontsize=6.8, color=INK2)
            ax.text(0.58, 0.31, tr("ra"), ha="right", va="center", fontsize=6.8, color=INK2)
            ax.text(1.68, 0.95, tr("câu hát ngắt quãng"), ha="center", va="top", fontsize=6.8, color=BLUE)
            ax.text(3.68, 0.95, tr("tiếng ho"), ha="center", va="top", fontsize=6.8, color=RED)
        return f
    rows = [
        dict(name="pad → drop → merge", param="--order pdm (pad 100/120, drop 0.5 sau pad)", tag="hiện tại", cost="O(số đoạn)",
             how="Nới đoạn, bỏ đoạn ngắn (độ dài đã gồm pad), rồi gộp khoảng lặng ngắn.", pros=["Như code hiện tại của lib.so."],
             cons=["Mẩu speech ngắn bị drop trước khi kịp gộp → mất cả câu hát/rap ngắt quãng.",
                   "Ngưỡng drop phụ thuộc pad: đổi pad là phải đổi drop."],
             mini=mini_o(postprocess_segs(segs, 0.10, 0.12, 0.5, 0.5, 4.6, "pdm"), ORANGE)),
        dict(name="merge → drop → pad", param="--order mdp (drop 0.32 trước pad, pad 0)", tag="đề xuất", cost="O(số đoạn)",
             how="Gộp khoảng lặng ngắn trên đoạn chưa pad, bỏ đoạn ngắn (đo trước pad), rồi pad.",
             pros=["Mẩu vỡ được ghép lại trước khi lọc; tiếng ho đứng riêng vẫn bị bỏ (pyannote, SpeechBrain).",
                   "Drop tách khỏi pad (như Silero) → tune từng tham số độc lập."], cons=["Khác thứ tự trong lib.so hiện tại: cần sửa code lib."],
             mini=mini_o(postprocess_segs(segs, 0, 0, 0.5, 0.32, 4.6, "mdp"), GREEN))]
    visual_table("17_so_sanh_thu_tu.png", "Bước 4 – Thứ tự pad / drop / merge",
                 "Ví dụ: câu hát vỡ thành 4 mẩu 0.16 s cách nhau 0.24 s, và một tiếng ho 0.16 s đứng riêng. Hàng 'vào' = đoạn sau ngưỡng, hàng 'ra' = kết quả.",
                 rows, row_h=1.55)


# ======================= bước 5–7: merge, drop, pad =======================
def table_merge():
    def mk(g): s, pb = pipeline(gap=g, drop=0.4); return lambda ax: mini_bars(ax, s, GREEN, pb=pb)
    rows = [
        dict(name="Không merge", param="--merge-gap 0", tag="tránh", cost="0", how="Giữ nguyên mọi khoảng lặng giữa các đoạn.",
             pros=["Không lấp nhầm khoảng nghỉ thật."], cons=["Câu có chỗ ngắt hơi bị tách thành nhiều đoạn vụn; event-F1 thấp."], mini=mk(0.0)),
        dict(name="Merge < 0.25 s", param="--merge-gap 0.25", tag="thử thêm", cost="O(số đoạn)", how="Chỉ gộp khoảng lặng rất ngắn.",
             pros=["Ít lấp nhầm (SpeechBrain dùng 0.25 s)."], cons=["Ngắt hơi 0.3–0.5 s vẫn tách đoạn."], mini=mk(0.25)),
        dict(name="Merge < 0.5 s", param="--merge-gap 0.5", tag="hiện tại", cost="O(số đoạn)",
             how="Gộp khoảng lặng ngắn hơn 0.5 s (giữ trong cấu hình đề xuất).",
             pros=["Khớp độ phân giải nhãn 0.5 s: khoảng lặng < 0.5 s gần như không được gán non-speech."],
             cons=["Với thứ tự pdm, cộng pad thành lấp khoảng lặng tới 0.72 s."], mini=mk(0.5)),
        dict(name="Merge < 1.0 s", param="--merge-gap 1.0", tag="tránh", cost="O(số đoạn)", how="Gộp cả khoảng nghỉ dài.",
             pros=["Rất ít đoạn; hợp nếu downstream (ASR) cần câu dài."],
             cons=["Nuốt khoảng nghỉ giữa hai người nói, giữa lời và nhạc → FA tăng."], mini=mk(1.0))]
    visual_table("18_so_sanh_merge.png", "Bước 5 – Merge: gộp đoạn cách nhau ngắn",
                 "Hình nhỏ chạy toàn pipeline đề xuất (ngưỡng kép, merge → drop 0.4 s → pad 0), chỉ đổi merge gap. Hàng trên = nhãn, hàng dưới = kết quả; "
                 "FP / FN đếm theo bin 0.5 s trên file mẫu.", rows,
                 legend="Tune trong {0.3, 0.4, 0.5, 0.6, 0.8} s. pyannote (VoxConverse) chọn 0.50 s; NeMo dùng 0.2–0.5 s.")

def table_drop():
    def mk(d, order="mdp", pre=0.0, post=0.0): s, pb = pipeline(order=order, pre=pre, post=post, drop=d); return lambda ax: mini_bars(ax, s, ORANGE, pb=pb)
    rows = [
        dict(name="Không drop", param="--drop 0", tag="hiện tại", cost="0", how="Giữ mọi đoạn, kể cả rất ngắn.",
             pros=["Không mất từ ngắn thật."], cons=["Tiếng ho, gõ, nốt nhạc thành đoạn speech giả → FAR tăng."], mini=mk(0.0)),
        dict(name="Drop < 0.32 s, đo trước pad", param="--order mdp --drop 0.32", tag="đề xuất", cost="O(số đoạn)",
             how="Bỏ đoạn ngắn hơn 4 hop (0.32 s), đo trước khi pad.",
             pros=["Silero 250 ms, SpeechBrain 0.25 s; độc lập với pad."],
             cons=["Cửa sổ 0.96 s + ngưỡng kép làm gai ngắn nở ra ~0.3–0.5 s: có thể chưa đủ để bỏ (file mẫu: tiếng ho còn)."],
             mini=mk(0.32)),
        dict(name="Drop < 0.4–0.5 s, đo trước pad", param="--order mdp --drop 0.4", tag="thử thêm", cost="O(số đoạn)",
             how="Ngưỡng drop cao hơn để bỏ gai đã bị nở rộng.", pros=["Bỏ được tiếng ho trong file mẫu."],
             cons=["Bắt đầu xoá từ đơn lẻ, câu trả lời rất ngắn ('ừ', 'vâng') → MR tăng."], mini=mk(0.4)),
        dict(name="Drop < 0.5 s, đo sau pad", param="--order pdm --drop 0.5", tag="tránh", cost="O(số đoạn)",
             how="Thứ tự hiện tại: độ dài đã cộng pad 0.22 s.", pros=["Không cần sửa thứ tự trong lib."],
             cons=["Ngưỡng thật = 0.5 − 0.22 = 0.28 s, đổi theo pad; mẩu speech vỡ bị drop trước khi merge."],
             mini=mk(0.5, "pdm", 0.10, 0.12))]
    visual_table("19_so_sanh_drop.png", "Bước 6 – Drop: bỏ đoạn speech quá ngắn",
                 "Hình nhỏ: toàn pipeline đề xuất (ngưỡng kép, merge → drop → pad 0), chỉ đổi drop; hàng cuối dùng thứ tự hiện tại (pad 100/120). Tiếng ho ở 6.75 s là đoạn cần bỏ; FP / FN theo bin 0.5 s trên file mẫu.",
                 rows, legend="Tune trong {0, 0.16, 0.24, 0.32, 0.40, 0.50} s, đo trước pad; theo dõi MR của category nhiều câu ngắn.")

def table_pad():
    def mk(pre, post): s, pb = pipeline(pre=pre, post=post, drop=0.4); return lambda ax: mini_bars(ax, s, YELLOW, pb=pb)
    rows = [
        dict(name="Pad 100 / 120 ms", param="--pad-pre 0.10 --pad-post 0.12", tag="hiện tại", cost="O(số đoạn)",
             how="Nới đầu 100 ms, cuối 120 ms cho mọi đoạn.", pros=["Biên 'an toàn' cho ASR / cắt audio: ít cắt mất đầu, đuôi câu."],
             cons=["YAMNet đã nở vùng speech ~0.48 s mỗi đầu → pad thêm thường là thừa, tăng FA ở bin biên."], mini=mk(0.10, 0.12)),
        dict(name="Pad 0 / 0", param="--pad-pre 0 --pad-post 0", tag="đề xuất", cost="0",
             how="Không nới; biên lấy đúng theo ngưỡng.", pros=["Bù cho hiện tượng nở biên; pyannote mặc định 0."],
             cons=["Nếu downstream cần biên an toàn thì pad lại sau, ngoài VAD."], mini=mk(0.0, 0.0)),
        dict(name="Pad âm −80 ms", param="--pad-pre -0.08 --pad-post -0.08", tag="thử thêm", cost="O(số đoạn)",
             how="Co mỗi đầu đoạn 80 ms (NeMo cho phép pad âm, có cấu hình pad_offset −0.09).",
             pros=["Bù nở biên mạnh hơn; giảm FA biên."], cons=["Dễ cắt mất phụ âm đầu / đuôi câu → MR biên tăng."], mini=mk(-0.08, -0.08)),
        dict(name="Pad 200 / 240 ms", param="--pad-pre 0.20 --pad-post 0.24", tag="tránh", cost="O(số đoạn)",
             how="Nới rộng gấp đôi hiện tại.", pros=["Gần như không cắt mất speech."],
             cons=["Lấp khoảng lặng ngắn, FA tăng rõ ở mọi biên."], mini=mk(0.20, 0.24))]
    visual_table("20_so_sanh_pad.png", "Bước 7 – Pad: nới / co biên đoạn",
                 "Hình nhỏ: toàn pipeline (ngưỡng kép, merge → drop 0.4 s → pad), chỉ đổi pad. FP / FN theo bin 0.5 s trên file mẫu.",
                 rows, legend="Tune pad trước và pad sau trong {−0.16, −0.08, 0, +0.08, +0.16} s. Pad nhỏ hơn 0.5 s chủ yếu đổi lỗi ở bin biên (FEC / OVER).")


# ======================= bước 8: hop =======================
def table_hop():
    def mk(k):
        fr = rescore_hop(M.iloc[::k], DUR, "tri", 3, "median")
        s, pb = pipeline(fr=fr, drop=0.4)
        def f(ax):
            mini_score(ax, fr, ref=FR_REC if k > 1 else None)
            st = M.start.to_numpy()[::k]
            for x in st[: int(3.0 / (HOP * k))]: ax.plot([x + HALF_WIN], [0.02], "|", color=ORANGE, ms=5)
            fp, fn = counts(pb)
            ax.text(DUR, 0.02, tr("{n} lần chạy / 10 s · FP {fp} · FN {fn}").format(n=len(st), fp=fp, fn=fn), ha="right", va="bottom", fontsize=7, color=INK2)
        return f
    rows = [
        dict(name="Hop 0.08 s", param="R (lib.so hiện tại)", tag="hiện tại", cost="12.5 lần chạy model / giây audio",
             how="Mỗi hop 0.08 s chạy YAMNet một lần trên cửa sổ 0.96 s (overlap 92%).", pros=["Điểm mịn nhất; AUC cao nhất."],
             cons=["Tốn CPU / pin nhất: mỗi đoạn audio được tính lại 12 lần."], mini=mk(1)),
        dict(name="Hop 0.16 s", param="R-h16 (mục 6)", tag="thử thêm", cost="6.25 lần / giây (÷2)",
             how="Giữ 1/2 cửa sổ; Rebin vẫn ra điểm mỗi 0.08 s.", pros=["CPU / pin giảm ~2×; MarbleNet: AUROC bão hoà khi overlap cao."],
             cons=["AUC giảm nhẹ; cần kiểm tra trên dữ liệu thật."], mini=mk(2)),
        dict(name="Hop 0.24 s", param="R-h24 (mục 6)", tag="thử thêm", cost="4.2 lần / giây (÷3)",
             how="Giữ 1/3 cửa sổ.", pros=["CPU / pin giảm ~3×."], cons=["Biên kém chính xác hơn; gai ngắn có thể lọt giữa hai cửa sổ."], mini=mk(3)),
        dict(name="Tính log-mel một lần cho cả file", param="(thay đổi trong lib.so)", tag="chưa có", cost="giảm phần tính STFT trùng lặp",
             how="Tính spectrogram một lần rồi cắt patch 96×64 chồng nhau, thay vì tính lại cho từng cửa sổ.",
             pros=["YAMNet gốc làm như vậy; không đổi kết quả, chỉ nhanh hơn."], cons=["Cần sửa code lib.so; không đo được trong vad-eval."],
             mini=lambda ax: (mini_frame(ax), ax.text(5, 0.5, tr("không đổi kết quả,\nchỉ đổi tốc độ"), ha="center", va="center", fontsize=8.5, color=INK2)))]
    visual_table("21_so_sanh_hop.png", "Bước 8 – Hop cửa sổ: đổi chất lượng lấy hiệu năng",
                 "Hình nhỏ: điểm mỗi hop sau Rebin khi chỉ giữ một phần cửa sổ (nét đứt xám = hop 0.08 s), vạch cam = vị trí các lần chạy model (3 s đầu).",
                 rows, legend="Chọn hop lớn nhất mà F1 giảm không quá ~0.5 điểm so với R trên dữ liệu thật; đo lại runtime / CPU / pin trên thiết bị.")


def save_lang(fig, name):
    save(fig, name if LANG == "vi" else name.replace(".png", "_en.png"))


if __name__ == "__main__":
    for f in (table_overview, table_aggregation, table_smoothing, table_threshold, table_order, table_merge, table_drop, table_pad, table_hop):
        f()
    if LANG == "en" and MISSING:
        raise SystemExit("Thiếu bản dịch trong compare_tables_en.py:\n" + "\n".join(sorted(MISSING)))
