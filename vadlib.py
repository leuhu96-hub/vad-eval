"""Thư viện đánh giá VAD: đọc file, gom bin, hậu xử lý, metric."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BIN, WIN, HOP, HALF_WIN, DUR = 0.5, 0.96, 0.08, 0.48, 10.0
RESCORE_WEIGHT, RESCORE_SMOOTH = "tri", 5   # cách gộp overlap mặc định của rescore_hop

# ---------------- I/O ----------------
def pair_files(root="."):
    root = Path(root)
    gt_index = {p.name.lower(): p for p in root.glob("Groundtruth/*/groundtruth/*.txt")}
    pairs, problems, used = [], [], set()
    for mp in sorted(root.glob("Model/*.txt")):
        m = re.fullmatch(r"([A-Za-z]+)_(\d+)\.txt", mp.name)
        if not m: problems.append(("ten_file_la", mp.name, "")); continue
        key = f"gt_{m.group(1).lower()}_{m.group(2)}.txt"
        gp = gt_index.get(key)
        if gp is None: problems.append(("Model thiếu GT", mp.name, "")); continue
        used.add(key); pairs.append(dict(category=m.group(1), idx=int(m.group(2)), file=mp.name, model=mp, gt=gp))
    for k, p in gt_index.items():
        if k not in used: problems.append(("GT thiếu model", p.name, str(p.parent.parent.name)))
    return pd.DataFrame(pairs), problems

def load_model(path, center=True):
    """Output model 'idx, start, end, score' (cũ) hoặc 'start end score' (mới); phân cách phẩy hoặc khoảng trắng."""
    df = pd.read_csv(path, header=None, sep=r"[,\s;]+", engine="python")
    df = df.iloc[:, -3:].set_axis(["start", "end", "score"], axis=1)
    df["center"] = df["start"] + (HALF_WIN if center else 0.0)
    return df

def load_gt(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["start", "end", "label"])
    return (df.label.str.lower() == "speech").astype(int).to_numpy()

# ---------------- gom bin / hậu xử lý ----------------
def bin_scores(m, n_bins, shift=0.0):
    c = m["center"].to_numpy() + shift; s = m["score"].to_numpy()
    b = np.floor(c / BIN).astype(int); out = np.full(n_bins, np.nan)
    for k in range(n_bins):
        mk = b == k
        if mk.any(): out[k] = s[mk].mean()
    ok = np.flatnonzero(~np.isnan(out))
    return np.interp(np.arange(n_bins), ok, out[ok]) if len(ok) else np.zeros(n_bins)

OVERLAP_DESC = {"tri": "trọng số tam giác theo tâm cửa sổ", "hann": "trọng số Hann theo tâm cửa sổ",
                "mean": "trung bình các cửa sổ phủ hop", "max": "max các cửa sổ phủ hop",
                "median": "median các cửa sổ phủ hop", "nearest": "cửa sổ có tâm gần hop nhất"}
SMOOTH_DESC = {"mean": "trung bình", "median": "median trượt", "gauss": "Gauss"}

def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def smooth_scores(x, n, kind="mean"):
    """Làm mượt chuỗi điểm theo hop, cửa sổ n hop (n = 1: giữ nguyên); đầu/cuối file chỉ dùng các hop có sẵn.
    mean  : trung bình trượt n hop
    median: median trượt n hop (giữ biên sắc hơn mean, bỏ được gai đơn lẻ)
    gauss : trung bình có trọng số Gauss trên n hop, sigma = n / 4"""
    n = int(n); x = np.asarray(x, float)
    if n <= 1 or len(x) == 0: return x
    if kind == "median":
        h = n // 2; xp = np.r_[np.full(h, np.nan), x, np.full(n - 1 - h, np.nan)]
        return np.nanmedian(np.lib.stride_tricks.sliding_window_view(xp, n), axis=1)
    if kind == "gauss":
        u = np.arange(n) - (n - 1) / 2; k = np.exp(-0.5 * (u / (n / 4)) ** 2)
    elif kind == "mean": k = np.ones(n)
    else: raise ValueError(f"smooth kind phải là một trong {list(SMOOTH_DESC)}")
    return np.convolve(x, k, "same") / np.convolve(np.ones(len(x)), k, "same")

def rescore_hop(m, dur=DUR, weight=RESCORE_WEIGHT, smooth=RESCORE_SMOOTH, smooth_kind="mean", domain="prob"):
    """Điểm mỗi hop 0.08s từ các cửa sổ 0.96s phủ tâm hop.
    weight="tri"    : trọng tâm tam giác - trọng số 1 - |t - tâm cửa sổ| / 0.48 (1 ở tâm, về 0 ở mép cửa sổ)
    weight="hann"   : trọng số Hann 0.5·(1 + cos(π·|t - tâm| / 0.48)) - giảm mép mạnh hơn tam giác
    weight="mean"   : trung bình đều (cách cũ)
    weight="max" / "median": max / median các cửa sổ phủ hop
    weight="nearest": chỉ lấy cửa sổ có tâm gần tâm hop nhất (không gộp)
    domain="logit"  : gộp tri / hann / mean trong miền log-odds rồi đổi lại xác suất (prob = gộp trực tiếp xác suất)
    smooth=5        : làm mượt 5 hop (2 trước, 2 sau và chính nó) kiểu smooth_kind (mean / median / gauss);
                      smooth=1 là không làm mượt.
    Cửa sổ không cần cách đều 0.08 s (vd giữ 1/2 cửa sổ để mô phỏng hop 0.16 s)."""
    if weight not in OVERLAP_DESC: raise ValueError(f"weight phải là một trong {list(OVERLAP_DESC)}")
    if domain not in ("prob", "logit"): raise ValueError("domain phải là prob hoặc logit")
    n = int(round(dur / HOP)); c = (np.arange(n) + 0.5) * HOP
    st = m["start"].to_numpy(); sc = m["score"].to_numpy(); out = np.full(n, np.nan)
    lg = domain == "logit" and weight in ("tri", "hann", "mean")
    if lg: sc = _logit(sc)
    for i, t in enumerate(c):
        mk = (st <= t) & (t < st + WIN)
        if not mk.any(): continue
        s, d = sc[mk], np.abs(t - (st[mk] + HALF_WIN))            # điểm và khoảng cách tới tâm từng cửa sổ
        if weight in ("tri", "hann"):
            w = 1 - d / HALF_WIN if weight == "tri" else 0.5 * (1 + np.cos(np.pi * np.minimum(d / HALF_WIN, 1)))
            out[i] = np.sum(w * s) / np.sum(w) if np.sum(w) > 0 else s.mean()
        elif weight == "mean": out[i] = s.mean()
        elif weight == "max": out[i] = s.max()
        elif weight == "median": out[i] = np.median(s)
        else: out[i] = s[np.argmin(d)]
    if lg: out = 1 / (1 + np.exp(-out))
    ok = np.flatnonzero(~np.isnan(out))
    out = np.interp(np.arange(n), ok, out[ok]) if len(ok) else np.zeros(n)
    return smooth_scores(out, smooth, smooth_kind)

def rescore_desc(weight=RESCORE_WEIGHT, smooth=RESCORE_SMOOTH, smooth_kind="mean", domain="prob"):
    """Mô tả ngắn cách gộp overlap, dùng trong báo cáo."""
    return (OVERLAP_DESC[weight] + (" (miền logit)" if domain == "logit" and weight in ("tri", "hann", "mean") else "")
            + (f" + làm mượt {SMOOTH_DESC[smooth_kind]} {smooth} hop" if smooth > 1 else ""))

# ---------------- quyết định + hậu xử lý đoạn (các phương án tối ưu) ----------------
def hysteresis(s, on, off=None):
    """Ngưỡng kép kiểu Silero / pyannote: bắt đầu speech khi s >= on, chỉ kết thúc khi s < off (off <= on).
    off = None hoặc off >= on: ngưỡng đơn s >= on. O(n), không vòng lặp Python."""
    s = np.asarray(s, float)
    if off is None or off >= on: return s >= on
    keep, start = s >= off, s >= on; idx = np.arange(len(s))
    last_break = np.maximum.accumulate(np.where(~keep, idx, -1))      # vị trí gần nhất điểm rơi dưới off
    last_on = np.maximum.accumulate(np.where(start, idx, -1))         # vị trí gần nhất điểm vượt on
    return keep & (last_on > last_break)

ORDER_DESC = {"pdm": "pad → drop → merge", "mdp": "merge → drop → pad"}

def postprocess_segs(segs, pre, post, gap, drop=0.0, dur=DUR, order="pdm"):
    """Hậu xử lý danh sách đoạn [start, end] đã sắp xếp, không chồng nhau.
    order="pdm": pad -> drop (độ dài sau pad) -> merge (cách hiện tại, = pad_merge)
    order="mdp": merge (gap đo trên đoạn chưa pad) -> drop (độ dài chưa pad) -> pad -> gộp đoạn chồng nhau sau pad
                 (pyannote / SpeechBrain merge trước drop; Silero đo min speech trước pad)
    pre / post được phép âm (co đoạn); đoạn co về độ dài <= 0 bị bỏ."""
    def merge(s, g):
        out = []
        for a, b in s:
            if out and a - out[-1][1] < g - 1e-9: out[-1][1] = max(out[-1][1], b)
            else: out.append([a, b])
        return out
    def pad(s): return [x for x in ([max(0.0, a - pre), min(dur, b + post)] for a, b in s) if x[1] - x[0] > 1e-9]
    s = sorted([float(a), float(b)] for a, b in segs)
    if order == "pdm":
        s = pad(s)
        if drop > 0: s = [x for x in s if x[1] - x[0] >= drop - 1e-9]
        return merge(s, gap)
    if order != "mdp": raise ValueError(f"order phải là một trong {list(ORDER_DESC)}")
    s = merge(s, gap)
    if drop > 0: s = [x for x in s if x[1] - x[0] >= drop - 1e-9]
    return merge(pad(s), 0.0)

def segs_to_bins_fast(segs, n_bins):
    """Như segs_to_bins (bin speech khi đoạn phủ >= 50% bin) nhưng O(số đoạn + số bin); đoạn phải không chồng nhau."""
    if not len(segs): return np.zeros(n_bins, int)
    s = np.asarray(segs, float); a, b = s[:, 0], s[:, 1]; cum = np.r_[0.0, np.cumsum(b - a)]
    def F(t):   # tổng thời lượng speech trong [0, t)
        j = np.searchsorted(a, t, side="right") - 1; jj = np.maximum(j, 0)
        return np.where(j >= 0, cum[jj] + np.clip(t - a[jj], 0, b[jj] - a[jj]), 0.0)
    lo = np.arange(n_bins) * BIN
    return ((F(lo + BIN) - F(lo)) / BIN >= 0.5 - 1e-9).astype(int)

def frames_to_bin_scores(fr, n_bins):
    """Điểm mỗi bin 0.5 s = trung bình điểm hop 0.08 s theo phần thời gian chồng lấp (lưới 0.02 s: hop 4 ô, bin 25 ô)."""
    g = np.repeat(np.asarray(fr, float), int(round(HOP / FINE))); need = n_bins * int(round(BIN / FINE))
    g = g[:need] if len(g) >= need else np.r_[g, np.full(need - len(g), g[-1] if len(g) else 0.0)]
    return g.reshape(n_bins, -1).mean(1)

def frames_to_segs(b, step):
    segs, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]: j += 1
            segs.append([i * step, j * step]); i = j
        else: i += 1
    return segs

def pad_merge(segs, pre, post, gap, dur=DUR, drop=0.0):
    """Hậu xử lý theo thứ tự padding -> drop -> merge: nới đoạn (cắt tại biên file), bỏ đoạn ngắn hơn drop giây,
    gộp đoạn cách nhau < gap giây."""
    s = [[max(0.0, a - pre), min(dur, b + post)] for a, b in segs]
    if drop > 0: s = [x for x in s if x[1] - x[0] >= drop - 1e-9]
    out = []
    for a, b in sorted(s):
        if out and a - out[-1][1] < gap - 1e-9: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def segs_to_bins(segs, n_bins):
    cov = np.zeros(n_bins)
    for a, b in segs:
        for k in range(n_bins):
            lo, hi = k * BIN, (k + 1) * BIN
            cov[k] += max(0.0, min(b, hi) - max(a, lo))
    return (cov / BIN >= 0.5 - 1e-9).astype(int)

def predict(m, n_bins, thr, mode="A4", pre=0.10, post=0.12, gap=0.50, drop=0.0):
    """A0: điểm bin thô>=thr | A1: rescore+rebin | A2: +pad | A3: +merge | A4: +pad+drop+merge"""
    if mode == "A0": return (bin_scores(m, n_bins) >= thr).astype(int)
    fr = rescore_hop(m) >= thr
    segs = frames_to_segs(fr, HOP)
    p, q, g, d = {"A1": (0, 0, 0, 0), "A2": (pre, post, 0, 0), "A3": (0, 0, gap, 0)}.get(mode, (pre, post, gap, drop))
    return segs_to_bins(pad_merge(segs, p, q, g, drop=d), n_bins)

# ---------------- mặt nạ biên & loại lỗi ----------------
def boundary_mask(y, k=1):
    m = np.zeros(len(y), bool)
    for c in np.flatnonzero(y[1:] != y[:-1]):
        m[max(0, c - k + 1): c + k + 1] = True
    return m

def error_types(y, p):
    """FEC*: miss ở bin đầu/cuối đoạn speech | MSC: miss giữa đoạn
       OVER*: FA ở bin non-speech sát đoạn speech | NDS: FA giữa vùng non-speech"""
    b = boundary_mask(y); fn = (y == 1) & (p == 0); fp = (y == 0) & (p == 1)
    return dict(FEC=int((fn & b).sum()), MSC=int((fn & ~b).sum()), OVER=int((fp & b).sum()), NDS=int((fp & ~b).sum()))

def rates(y, p, miss_w=0.75):
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    mr = fn / (tp + fn) if tp + fn else np.nan; far = fp / (fp + tn) if fp + tn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan; rec = 1 - mr if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, MR=mr, FAR=far, Precision=prec, Recall=rec, F1=f1,
                Specificity=1 - far if fp + tn else np.nan, DCF=miss_w * mr + (1 - miss_w) * far)

def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < np.mean(y) < 1 else np.nan

# ---------------- metric theo đoạn ----------------
def bins_to_segs(y): return frames_to_segs(y.astype(bool), BIN)

def event_f1(gt_segs, pr_segs, collar=0.5, onset_only=False):
    used, tp = set(), 0
    for a, b in gt_segs:
        for j, (c, d) in enumerate(pr_segs):
            if j in used: continue
            ok = abs(c - a) <= collar
            if not onset_only: ok = ok and abs(d - b) <= max(collar, 0.5 * (b - a))
            if ok: used.add(j); tp += 1; break
    fp = len(pr_segs) - tp; fn = len(gt_segs) - tp
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else np.nan, tp, fp, fn

def frag_merge(gt_segs, pr_segs):
    ov = lambda x, y: min(x[1], y[1]) - max(x[0], y[0]) > 0
    frag = sum(1 for g in gt_segs if sum(ov(g, p) for p in pr_segs) >= 2)
    merge = sum(1 for p in pr_segs if sum(ov(g, p) for g in gt_segs) >= 2)
    return frag, merge

def boundary_deltas(gt_segs, pr_segs):
    out = []
    for g in gt_segs:
        best, bo = None, 0
        for p in pr_segs:
            o = min(g[1], p[1]) - max(g[0], p[0])
            if o > bo: best, bo = p, o
        if best: out.append((best[0] - g[0], best[1] - g[1]))
    return out

# ---------------- đặc trưng tập theo độ phân giải model ----------------
FINE = 0.02   # lưới chung của nhãn và model: BIN = 25 ô, HOP = 4 ô, WIN = 48 ô -> mọi phép tính chính xác

def window_frac(m, y):
    """Tỉ lệ speech của nhãn (bin 0.5 s) trong từng cửa sổ [start, end) của output model. Phần cửa sổ vượt cuối nhãn
    không tính (chuẩn hoá theo phần có nhãn); cửa sổ nằm hẳn ngoài nhãn = NaN."""
    g = np.repeat(np.asarray(y, float), int(round(BIN / FINE))); cs = np.r_[0.0, np.cumsum(g)]
    i0, i1 = (np.clip(np.round(m[k].to_numpy() / FINE).astype(int), 0, len(g)) for k in ("start", "end"))
    return np.where(i1 > i0, (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1), np.nan)

def oracle_model(m, y):
    """Model 'hoàn hảo' cùng lưới cửa sổ: điểm = tỉ lệ speech thật trong cửa sổ; bỏ cửa sổ nằm ngoài nhãn."""
    f = window_frac(m, y); ok = ~np.isnan(f)
    o = m.loc[ok].copy(); o["score"] = f[ok]
    return o.reset_index(drop=True)

def run_lengths(y):
    """(độ dài các đoạn speech, độ dài các khoảng lặng nằm giữa hai đoạn speech), giây."""
    segs = frames_to_segs(np.asarray(y).astype(bool), BIN)
    return [b - a for a, b in segs], [c - b for (_, b), (c, _) in zip(segs[:-1], segs[1:])]

def dist_to_transition(y):
    """Khoảng cách (s) từ tâm mỗi bin tới chuyển trạng thái gần nhất của nhãn; inf nếu file chỉ có 1 lớp."""
    y = np.asarray(y); tr = (np.flatnonzero(y[1:] != y[:-1]) + 1) * BIN; c = (np.arange(len(y)) + 0.5) * BIN
    if not len(tr): return np.full(len(y), np.inf)
    k = np.searchsorted(tr, c)
    return np.minimum(np.abs(c - tr[np.maximum(k - 1, 0)]), np.abs(tr[np.minimum(k, len(tr) - 1)] - c))

def centers_per_bin(m, n_bins):
    """Số tâm cửa sổ rơi vào mỗi bin (điểm A0 của bin là trung bình các cửa sổ này; 0 -> nội suy)."""
    b = np.floor(m["center"].to_numpy() / BIN).astype(int)
    return np.bincount(b[(b >= 0) & (b < n_bins)], minlength=n_bins)
