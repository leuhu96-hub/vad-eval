"""Đánh giá output model trên audio THẬT (data/audio) theo quy trình của evaluate.py.

Khác evaluate.py (dữ liệu giả định 10 s, 20 bin):
  - file dài khác nhau (19–120 s), tên <Cat>_<youtube id>
  - không có meta.csv (SNR, nguồn train) -> bỏ bước rò rỉ / SNR
  - không có Annotator2 -> Silero (cache của auto_label.py) làm người gán nhãn thứ 2
  - lag từng file chỉ báo cáo, không tự dịch
  - dev/test chia theo file, phân tầng theo category; thêm ngưỡng chọn trên tập val TỔNG HỢP
    (syn_val_* trong --raw) để xem ngưỡng chọn không cần nhãn thật có dùng được không
Nhãn là nhãn BẠC (FireRedVAD AED): chỉ số đo mức khớp với FireRedVAD, không phải độ chính xác tuyệt đối.

    python evaluate_real.py --raw C:/vad_work/run1/raw_real --gt C:/vad_work/real_test/gt_policy \
        --synth-gt C:/vad_work/synth/clips/gt/val --out results_real
"""
import argparse, json, wave, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             det_curve, brier_score_loss, cohen_kappa_score)
from scipy.stats import norm
from vadlib import *
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--raw", type=Path, default=Path("C:/vad_work/run1/raw_real"), help="output thô index, start, end, score")
ap.add_argument("--gt", type=Path, default=Path("C:/vad_work/real_test/gt_policy"), help="nhãn chính: nói HOẶC hát = speech")
ap.add_argument("--gt-alt", type=Path, default=Path("data/auto_labels/Groundtruth"), help="nhãn gốc auto_label: hát = non-speech")
ap.add_argument("--audio", type=Path, default=Path("data/audio"))
ap.add_argument("--cache", type=Path, default=Path("data/auto_labels/cache"), help="xác suất AED/Silero của auto_label.py")
ap.add_argument("--synth-gt", type=Path, default=Path("C:/vad_work/synth/clips/gt/val"))
ap.add_argument("--dev-frac", type=float, default=0.3)
ap.add_argument("--out", type=Path, default=Path("results_real"))
A = ap.parse_args()
OUT = A.out; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True})
CATS = ["Asm", "Babble", "Clean", "Music", "Noise"]
COLS = dict(zip(CATS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))   # thứ tự cố định, không xoay vòng
C1, C2 = "#2a78d6", "#eb6834"
MODES = ["A0", "A1", "A2", "A3", "A4"]
rng = np.random.default_rng(0)
S = {}

# ---- bản vector hóa của vadlib (cùng kết quả): file thật tới 240 bin, quét ngưỡng bằng vòng lặp quá chậm ----
def frames_to_segs(b, step):
    d = np.diff(np.r_[0, np.asarray(b, int), 0])
    return [[i * step, j * step] for i, j in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

def segs_to_bins(segs, n_bins):
    if not segs: return np.zeros(n_bins, int)
    s = np.asarray(segs, float); lo = np.arange(n_bins) * BIN
    cov = np.clip(np.minimum(s[:, 1:], lo + BIN) - np.maximum(s[:, :1], lo), 0, None).sum(0)
    return (cov / BIN >= 0.5 - 1e-9).astype(int)

def bin_scores(m, n_bins, shift=0.0):
    b = np.floor((m["center"].to_numpy() + shift) / BIN).astype(int); ok = (b >= 0) & (b < n_bins)
    cnt = np.bincount(b[ok], minlength=n_bins)[:n_bins]
    out = np.bincount(b[ok], m["score"].to_numpy()[ok], n_bins)[:n_bins] / np.maximum(cnt, 1)
    has = np.flatnonzero(cnt)
    return np.interp(np.arange(n_bins), has, out[has]) if len(has) else np.zeros(n_bins)

# ---- I/O ----
def index_gt(root):
    """<Cat>_<id> -> file nhãn. Nhận <Cat>/<Cat>_<id>.txt và <Cat>/groundtruth/Gt_<cat>_<id>.txt."""
    out = {}
    for p in Path(root).rglob("*.txt"):
        cat = p.parent.parent.name if p.parent.name == "groundtruth" else p.parent.name
        out[f"{cat}_{p.stem.split('_', 2)[2]}" if p.stem.lower().startswith("gt_") else p.stem] = p
    return out

def load_gt_exact(path):
    """Nhãn thời điểm chính xác (tập tổng hợp) -> bin 0.5 s, speech khi >= 50% thời lượng bin."""
    iv = pd.read_csv(path, sep=r"\s+", header=None, names=["start", "end", "label"])
    end = iv.end.max(); n = int(np.ceil(end / BIN - 1e-6)); lo = np.arange(n) * BIN
    sp = iv[iv.label.str.lower() == "speech"]
    cov = np.clip(np.minimum(sp.end.to_numpy()[:, None], lo + BIN) - np.maximum(sp.start.to_numpy()[:, None], lo), 0, None).sum(0)
    return (cov / (np.minimum(lo + BIN, end) - lo) >= 0.5 - 1e-9).astype(int)

def wav_dur(p):
    try:
        with wave.open(str(p)) as w: return w.getnframes() / w.getframerate()
    except Exception: return np.nan

def silero_bins(key, n):
    """Như auto_label.py: bin là speech khi >= 50% chunk 32 ms có p_silero >= 0.5."""
    p = A.cache / f"{key}.npz"
    if not p.exists(): return None
    sp = np.load(p)["silero"]
    if not len(sp): return None
    k = np.floor((np.arange(len(sp)) + 0.5) * 512 / 16000 / BIN).astype(int); ok = k < n
    frac = np.bincount(k[ok], (sp[ok] >= 0.5).astype(float), n) / np.maximum(np.bincount(k[ok], minlength=n), 1)
    return (frac >= 0.5 - 1e-9).astype(int)

def auc(y, s): return roc_auc_score(y, s) if len(y) and 0 < np.mean(y) < 1 else np.nan
def cat_(files, key, src=None): return np.concatenate([(src or recs)[f][key] for f in files])

# =============== 5a. Ghép file audio <-> output model <-> nhãn ===============
audio = {p.stem: p for p in A.audio.glob("*/*.wav")}
raw = {p.stem: p for p in A.raw.glob("*.txt")}
real_raw = {k for k in raw if not k.startswith("syn_")}
gt, gt_alt = index_gt(A.gt), index_gt(A.gt_alt)
problems = []
for k in sorted(set(audio) | real_raw | set(gt)):
    miss = [n for n, d in (("audio", audio), ("raw", real_raw), ("gt", gt), ("gt_alt", gt_alt)) if k not in d]
    if miss: problems.append((k, "thiếu " + ", ".join(miss)))
keys = sorted(set(audio) & real_raw & set(gt))
S.update(n_audio=len(audio), n_raw_real=len(real_raw), n_gt=len(gt), n_pairs=len(keys), problems=problems)

recs = {}
for k in keys:
    y = load_gt(gt[k]); n = len(y); m = load_model(raw[k], center=True)
    ya = load_gt(gt_alt[k]) if k in gt_alt else y.copy()
    sil = silero_bins(k, n)
    recs[k] = dict(file=k, category=k.split("_")[0], y=y, y_alt=ya if len(ya) == n else y.copy(), alt_len_ok=len(ya) == n,
                   sil=sil if sil is not None else ya.copy(), has_sil=sil is not None,
                   m=m, m0=load_model(raw[k], center=False), n=n, dur_wav=wav_dur(audio[k]))

# tập val tổng hợp (syn_val_* trong --raw): nhãn thời điểm chính xác -> mốc căn chỉnh + ngưỡng không cần nhãn thật
syn = {}
for k in sorted(r for r in raw if r.startswith("syn_val_")):
    p = A.synth_gt / f"{k}.txt"
    if not p.exists() or raw[k].stat().st_size == 0: continue
    y = load_gt_exact(p); m = load_model(raw[k]); n = len(y)
    if n < 2 or len(m) == 0: continue
    syn[k] = dict(y=y, n=n, m=m, braw=bin_scores(m, n), fr=rescore_hop(m, dur=n * BIN))
S["n_synth_val"] = len(syn)

# =============== 5b/5d. Cấu trúc ===============
struct = []
for f, d in recs.items():
    m = d["m"]; dd = np.diff(m.start)
    n_exp = int((round(d["dur_wav"] * 16000) - 15600) // 1280) + 1   # số cửa sổ YAMNet: 15600 mẫu, hop 1280
    struct.append(dict(file=f, dur_wav=d["dur_wav"], n_bins=d["n"], n_win=len(m), n_win_expected=n_exp,
                       first_start=m.start.min(), last_end=m.end.max(),
                       hop_ok=bool(np.allclose(dd, HOP, atol=1e-3)), win_ok=bool(np.allclose(m.end - m.start, WIN, atol=1e-3)),
                       score_ok=bool(m.score.between(0, 1).all()), alt_len_ok=d["alt_len_ok"], has_silero=d["has_sil"]))
struct = pd.DataFrame(struct)
struct["bins_ok"] = struct.n_bins == (struct.dur_wav / BIN + 1e-9).astype(int)
struct["win_count_ok"] = (struct.n_win - struct.n_win_expected).abs() <= 1
struct["flag"] = ~(struct.hop_ok & struct.win_ok & struct.score_ok & struct.bins_ok & struct.win_count_ok
                   & struct.alt_len_ok) | (struct.first_start.abs() > 0.01) | (struct.last_end > struct.dur_wav + 0.01)
S.update(n_struct_flag=int(struct.flag.sum()), struct_flag_files=struct.file[struct.flag].tolist(),
         n_no_silero=int((~struct.has_silero).sum()))

# =============== 5e. Căn chỉnh: quét shift + lag từng file ===============
def pooled_f1_shift(key, shift, thr=0.5):
    Y, P = [], []
    for d in recs.values():
        s = bin_scores(d[key], d["n"], shift=shift); Y.append(d["y"][1:-1]); P.append((s >= thr)[1:-1])
    return rates(np.concatenate(Y), np.concatenate(P).astype(int))["F1"]
def pooled_auc_shift(src, yk, shift):
    """AUC không phụ thuộc ngưỡng: so độ dịch trên nhãn FireRedVAD, Silero và nhãn chính xác của tập tổng hợp."""
    return auc(np.concatenate([d[yk][1:-1] for d in src.values()]),
               np.concatenate([bin_scores(d["m"], d["n"], shift=shift)[1:-1] for d in src.values()]))
shifts = np.round(np.arange(-12, 13) * 0.08, 2)
sweep = pd.DataFrame(dict(shift_s=shifts, F1_start=[pooled_f1_shift("m0", s) for s in shifts],
                          F1_center=[pooled_f1_shift("m", s) for s in shifts],
                          AUC_real_fireredvad=[pooled_auc_shift(recs, "y", s) for s in shifts],
                          AUC_real_silero=[pooled_auc_shift(recs, "sil", s) for s in shifts],
                          AUC_synth_exact=[pooled_auc_shift(syn, "y", s) for s in shifts]))
S["best_shift_start"] = float(sweep.shift_s[sweep.F1_start.idxmax()]); S["best_shift_center"] = float(sweep.shift_s[sweep.F1_center.idxmax()])
S["best_shift_auc"] = {c: float(sweep.shift_s[sweep[c].idxmax()]) for c in ("AUC_real_fireredvad", "AUC_real_silero", "AUC_synth_exact")}
S["auc_gain_best_shift_real"] = float(sweep.AUC_real_fireredvad.max() - sweep.AUC_real_fireredvad[sweep.shift_s == 0].iloc[0])

def best_lag(m, y, grid=0.02, max_lag=1.5):
    t = np.arange(0, len(y) * BIN, grid); s = np.interp(t, m.center, m.score)
    g = y[np.minimum((t / BIN).astype(int), len(y) - 1)].astype(float)
    s = (s - s.mean()) / (s.std() + 1e-9); g = (g - g.mean()) / (g.std() + 1e-9)
    L = int(max_lag / grid); lags = np.arange(-L, L + 1)
    cc = [np.mean(s[max(0, -l):len(s) - max(0, l)] * g[max(0, l):len(g) - max(0, -l)]) for l in lags]
    return lags[int(np.argmax(cc))] * grid, float(max(cc))
lagrows = []
for f, d in recs.items():
    if 0 < d["y"].mean() < 1:
        lag, pc = best_lag(d["m"], d["y"]); lagrows.append(dict(file=f, category=d["category"], lag_s=round(lag, 2), peak_corr=pc))
lagdf = pd.DataFrame(lagrows)
# dữ liệu thật: lag lớn thường do nhãn/điểm nhiễu (corr thấp) chứ không phải lỗi xuất file -> chỉ báo cáo
lagdf["flag"] = np.where(lagdf.lag_s.abs() >= 0.4, "Lệch ~1 bin", np.where(lagdf.lag_s.abs() > 0.2, "Cần xem", ""))
S.update(lag_median=float(lagdf.lag_s.median()), lag_flag_files=lagdf.file[lagdf.flag == "Lệch ~1 bin"].tolist(),
         lag_flag_corr=lagdf.peak_corr[lagdf.flag == "Lệch ~1 bin"].round(3).tolist(),
         n_lag_review=int((lagdf.flag == "Cần xem").sum()))

# cache điểm
for d in recs.values():
    d["braw"] = bin_scores(d["m"], d["n"]); d["fr"] = rescore_hop(d["m"], dur=d["n"] * BIN); d["bmask"] = boundary_mask(d["y"])
    d["sure"] = (d["y"] == d["y_alt"]) & (d["y_alt"] == d["sil"])   # AED (2 quy tắc) và Silero cùng nhãn

def pred_cached(d, thr, mode="A4", pre=0.10, post=0.12, gap=0.50):
    if mode == "A0": return (d["braw"] >= thr).astype(int)
    p, q, g = {"A1": (0, 0, 0), "A2": (pre, post, 0), "A3": (0, 0, gap), "A4": (pre, post, gap)}.get(mode, (pre, post, gap))
    return segs_to_bins(pad_merge(frames_to_segs(d["fr"] >= thr, HOP), p, q, g, dur=d["n"] * BIN), d["n"])

# =============== 1. Phân bố ===============
rows = []
for f, d in recs.items():
    y = d["y"]; segs = bins_to_segs(y)
    rows.append(dict(file=f, category=d["category"], dur_s=d["n"] * BIN, speech_ratio=y.mean(), speech_ratio_alt=d["y_alt"].mean(),
                     bins_hat=int((y != d["y_alt"]).sum()), silero_ratio=d["sil"].mean(), agree_aed_silero=(d["y_alt"] == d["sil"]).mean(),
                     transitions=int((y[1:] != y[:-1]).sum()), n_speech_seg=len(segs),
                     mean_speech_seg_s=np.mean([b - a for a, b in segs]) if segs else 0,
                     boundary_frac=d["bmask"].mean(), auc_file=safe_auc(y, d["braw"]), mean_score=d["braw"].mean()))
pf = pd.DataFrame(rows)
pf["degenerate"] = (pf.speech_ratio == 0) | (pf.speech_ratio == 1)
pf["extreme"] = (pf.speech_ratio < 0.1) | (pf.speech_ratio > 0.9)
Y_all, S_all = cat_(pf.file, "y"), cat_(pf.file, "braw")
S.update(hours=float(pf.dur_s.sum() / 3600), pct_speech_bins=float(Y_all.mean()),
         n_degenerate=int(pf.degenerate.sum()), degenerate_files=pf.file[pf.degenerate].tolist(),
         pct_extreme=float(pf.extreme.mean()), median_transitions=float(pf.transitions.median()),
         mean_boundary_frac=float(pf.boundary_frac.mean()), auc_pooled_all=auc(Y_all, S_all),
         auc_macro=float(pf.auc_file.mean()), auc_macro_sd=float(pf.auc_file.std()),
         auc_worst=float(pf.auc_file.min()), auc_worst_file=pf.file[pf.auc_file.idxmin()],
         n_files_auc_lt_09=int((pf.auc_file < 0.9).sum()), n_files_auc_lt_07=int((pf.auc_file < 0.7).sum()))

# =============== 2. Chia dev/test theo file, phân tầng theo category ===============
split, rs = {}, np.random.default_rng(3)
for c in CATS:
    fs = sorted(pf.file[pf.category == c]); rs.shuffle(fs); nd = int(round(A.dev_frac * len(fs)))
    for i, f in enumerate(fs): split[f] = "dev" if i < nd else "test"
pf["split"] = pf.file.map(split)
dev = sorted(pf.file[pf.split == "dev"]); test = sorted(pf.file[pf.split == "test"])
S.update(n_dev=len(dev), n_test=len(test), pct_speech_dev=float(cat_(dev, "y").mean()), pct_speech_test=float(cat_(test, "y").mean()))

# =============== 3. Nhất quán nhãn: AED (quy tắc gốc) vs Silero, quy tắc hát ===============
has = [f for f in pf.file if recs[f]["has_sil"]]
Aa, Bs, BM = cat_(has, "y_alt"), cat_(has, "sil"), np.concatenate([boundary_mask(recs[f]["y_alt"]) for f in has])
S.update(kappa_aed_silero=cohen_kappa_score(Aa, Bs), kappa_aed_silero_boundary=cohen_kappa_score(Aa[BM], Bs[BM]),
         kappa_aed_silero_interior=cohen_kappa_score(Aa[~BM], Bs[~BM]), raw_agree_aed_silero=float((Aa == Bs).mean()),
         pct_bins_changed_by_singing=float((Y_all != cat_(pf.file, "y_alt")).mean()),
         pct_sure_bins=float(cat_(pf.file, "sure").mean()))
lab_cat = []
for c in CATS:
    fs = [f for f in has if recs[f]["category"] == c]
    a_, b_ = cat_(fs, "y_alt"), cat_(fs, "sil")
    lab_cat.append(dict(Category=c, n_file=len(fs), speech_policy=cat_(fs, "y").mean(), speech_auto=a_.mean(), speech_silero=b_.mean(),
                        bin_doi_do_hat=int((cat_(fs, "y") != a_).sum()), agree_aed_silero=(a_ == b_).mean(),
                        kappa_aed_silero=cohen_kappa_score(a_, b_) if len(set(a_)) > 1 or len(set(b_)) > 1 else np.nan,
                        pct_bin_nhan_chac=cat_(fs, "sure").mean()))
lab_cat = pd.DataFrame(lab_cat)

# =============== 4. Độ khó ===============
amb = (S_all > 0.1) & (S_all < 0.9); bm_all = cat_(pf.file, "bmask")
S.update(pct_ambiguous=float(amb.mean()), pct_ambiguous_in_boundary=float((amb & bm_all).sum() / max(amb.sum(), 1)))
thr_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
f1_thr = [rates(Y_all, (S_all >= t).astype(int))["F1"] for t in thr_grid]
S["f1_range_0.2_0.9"] = float(max(f1_thr[3:18]) - min(f1_thr[3:18]))

# =============== Tầng A (điểm thô) ===============
def tierA(files, name, yk="y"):
    y, s = cat_(files, yk), cat_(files, "braw")
    bm = np.concatenate([boundary_mask(recs[f][yk]) for f in files]); ok = cat_(files, "sure")
    fpr, tpr, _ = roc_curve(y, s) if 0 < y.mean() < 1 else (np.array([0, 1]), np.array([np.nan, np.nan]), None)
    return dict(Tập=name, n_file=len(files), n_bin=len(y), pct_speech=y.mean(), AUC_micro=auc(y, s),
                AUC_macro_file=np.nanmean([safe_auc(recs[f][yk], recs[f]["braw"]) for f in files]),
                AUC_khong_bien=auc(y[~bm], s[~bm]), AUC_chi_bien=auc(y[bm], s[bm]), AUC_nhan_chac=auc(y[ok], s[ok]),
                PR_AUC=average_precision_score(y, s) if y.any() else np.nan,
                TPR_at_FPR1=float(np.interp(0.01, fpr, tpr)), TPR_at_FPR5=float(np.interp(0.05, fpr, tpr)),
                Brier=brier_score_loss(y, s), mean_score_nonspeech=s[y == 0].mean() if (y == 0).any() else np.nan)
tA = [tierA(test, "Test (tất cả)")] + [tierA([f for f in test if recs[f]["category"] == c], c) for c in CATS]
tA += [tierA(dev, "Dev (tham khảo)"), tierA(list(pf.file), "Cả 144 file (tham khảo)")]
tA = pd.DataFrame(tA)
tA_alt = pd.DataFrame([tierA(test, "Test (tất cả)", "y_alt")] +
                      [tierA([f for f in test if recs[f]["category"] == c], c, "y_alt") for c in CATS])
aucs = []
for _ in range(1000):
    pick = rng.choice(test, len(test), replace=True); aucs.append(auc(cat_(pick, "y"), cat_(pick, "braw")))
S["AUC_test_CI"] = [float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))]

# =============== Chọn ngưỡng trên dev thật và val tổng hợp qua toàn pipeline ===============
def pooled(files, thr, mode="A4", mask=None, src=None, **kw):
    src = src or recs; Y, P = [], []
    for f in files:
        d = src[f]; p = pred_cached(d, thr, mode, **kw); y = d["y"]
        sel = slice(None) if mask is None else (~d["bmask"] if mask == "nb" else d["bmask"])
        Y.append(y[sel]); P.append(p[sel])
    return rates(np.concatenate(Y), np.concatenate(P))
thr_sweep = np.round(np.arange(0.01, 0.951, 0.01), 2)   # từ 0.01: trên dữ liệu thật DCF-min rơi xuống rất thấp
sweep_rows = []
for tap, files, src in (("dev", dev, recs), ("synth", list(syn), syn)):
    for mode in MODES:
        for t in thr_sweep:
            r = pooled(files, t, mode, src=src)
            sweep_rows.append(dict(tap=tap, config=mode, thr=t, TP=r["TP"], FN=r["FN"], FP=r["FP"], TN=r["TN"],
                                   MR=r["MR"], FAR=r["FAR"], F1=r["F1"], DCF=r["DCF"]))
thr_df = pd.DataFrame(sweep_rows)
best = {tap: {mode: float(g.thr[g.DCF.idxmin()]) for mode, g in gg.groupby("config")} for tap, gg in thr_df.groupby("tap")}
best_f1 = {tap: {mode: float(g.thr[g.F1.idxmax()]) for mode, g in gg.groupby("config")} for tap, gg in thr_df.groupby("tap")}
S["thr_at_grid_edge"] = sorted({f"{tap}/{m}" for tap in best for m, t in best[tap].items() if t in (thr_sweep[0], thr_sweep[-1])})
yd, sd_ = cat_(dev, "y"), cat_(dev, "braw"); fpr, tpr, th = roc_curve(yd, sd_); S["thr_youden_raw_dev"] = float(th[np.argmax(tpr - fpr)])
ys, ss = cat_(list(syn), "y", syn), cat_(list(syn), "braw", syn); fpr, tpr, th = roc_curve(ys, ss)
S["thr_youden_raw_synth"] = float(th[np.argmax(tpr - fpr)]); S["auc_synth_val"] = auc(ys, ss)
S["best_thr_dev"] = best["dev"]; S["best_thr_synth"] = best["synth"]; S["best_f1_thr_dev"] = best_f1["dev"]
S["best_f1_thr_synth"] = best_f1["synth"]; THR = best["dev"]["A4"]
transfer = []
for name, mode, t in [("Youden điểm thô – val tổng hợp", "A0", S["thr_youden_raw_synth"]),
                      ("DCF-min A0 – val tổng hợp", "A0", best["synth"]["A0"]), ("DCF-min A4 – val tổng hợp", "A4", best["synth"]["A4"]),
                      ("F1-max A4 – val tổng hợp", "A4", best_f1["synth"]["A4"]),
                      ("Youden điểm thô – dev thật", "A0", S["thr_youden_raw_dev"]), ("DCF-min A0 – dev thật", "A0", best["dev"]["A0"]),
                      ("F1-max A4 – dev thật", "A4", best_f1["dev"]["A4"]), ("DCF-min A4 – dev thật (dùng chính)", "A4", THR)]:
    r = pooled(test, t, mode)
    transfer.append(dict(Ngưỡng_chọn_từ=name, Cấu_hình=mode, thr=t, MR=r["MR"], FAR=r["FAR"], Precision=r["Precision"],
                         Recall=r["Recall"], F1=r["F1"], DCF=r["DCF"]))
transfer = pd.DataFrame(transfer)

# =============== Tầng B trên test ===============
tB = []
for mode_name, mk in [("Strict (mọi bin)", None), ("Collar ±1 bin (bỏ bin biên)", "nb"), ("Chỉ bin biên", "b")]:
    r = pooled(test, THR, "A4", mask=mk); r["Chế độ"] = mode_name; tB.append(r)
tB = pd.DataFrame(tB)
et = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}; ev_tp = ev_fp = ev_fn = on_tp = on_fp = on_fn = 0
frag = merge = n_gt_seg = n_pr_seg = 0; deltas = []; gap_filled = gap_total = 0
for f in test:
    d = recs[f]; p = pred_cached(d, THR); y = d["y"]
    for k, v in error_types(y, p).items(): et[k] += v
    gs, ps = bins_to_segs(y), bins_to_segs(p)
    _, a, b, c = event_f1(gs, ps); ev_tp += a; ev_fp += b; ev_fn += c
    _, a, b, c = event_f1(gs, ps, onset_only=True); on_tp += a; on_fp += b; on_fn += c
    fr_, mg_ = frag_merge(gs, ps); frag += fr_; merge += mg_; n_gt_seg += len(gs); n_pr_seg += len(ps)
    deltas += boundary_deltas(gs, ps)
    for (a1, b1), (a2, b2) in zip(gs[:-1], gs[1:]):      # khoảng lặng GT <= 1s bị lấp?
        if a2 - b1 <= 1.0:
            gap_total += 1; k0, k1 = int(round(b1 / BIN)), int(round(a2 / BIN))
            gap_filled += int(p[k0:k1].all())
deltas = np.array(deltas)
# chỉ số theo đoạn phụ thuộc mạnh vào ngưỡng: ngưỡng thấp -> đoạn dự đoán gộp nhiều đoạn GT
ev_thr = []
for name, t in [("DCF-min dev", THR), ("F1-max dev", best_f1["dev"]["A4"]), ("DCF-min val tổng hợp", best["synth"]["A4"])]:
    c_ = np.zeros(8)
    for f in test:
        gs, ps = bins_to_segs(recs[f]["y"]), bins_to_segs(pred_cached(recs[f], t))
        c_ += [*event_f1(gs, ps)[1:], event_f1(gs, ps, onset_only=True)[1], *frag_merge(gs, ps), len(ps), len(gs)]
    tp, fp, fn, otp, fr_, mg_, npr, ngt = c_
    ev_thr.append(dict(Ngưỡng=name, thr=t, event_F1=2 * tp / (2 * tp + fp + fn), onset_F1=2 * otp / (npr + ngt),
                       n_gt_seg=int(ngt), n_pred_seg=int(npr), n_fragmented=int(fr_), n_overmerged=int(mg_)))
ev_thr = pd.DataFrame(ev_thr)
S.update(thr_used=THR, err_types=et, pct_true_err=(et["MSC"] + et["NDS"]) / max(sum(et.values()), 1),
         event_f1=2 * ev_tp / (2 * ev_tp + ev_fp + ev_fn), event_f1_onset=2 * on_tp / (2 * on_tp + on_fp + on_fn),
         n_gt_seg=n_gt_seg, n_pred_seg=n_pr_seg, n_fragmented=frag, n_overmerged=merge,
         mean_d_onset=float(deltas[:, 0].mean()), mean_d_offset=float(deltas[:, 1].mean()),
         gap_short_total=gap_total, gap_short_filled=gap_filled)

def boot(files, fn, B=1000):
    vals = []; per = {f: fn([f]) for f in files}
    for _ in range(B):
        tot = pd.DataFrame([per[f] for f in rng.choice(files, len(files), replace=True)]).sum()
        tp, fp, fn_, tn = tot.TP, tot.FP, tot.FN, tot.TN
        mr = fn_ / (tp + fn_) if tp + fn_ else np.nan; far = fp / (fp + tn) if fp + tn else np.nan
        vals.append((2 * tp / (2 * tp + fp + fn_) if tp + fp + fn_ else np.nan, 0.75 * mr + 0.25 * far, mr, far))
    return np.nanpercentile(np.array(vals), [2.5, 97.5], axis=0)
ci = boot(test, lambda fs: pooled(fs, THR)); ci_nb = boot(test, lambda fs: pooled(fs, THR, mask="nb"))
S.update(F1_CI=ci[:, 0].tolist(), DCF_CI=ci[:, 1].tolist(), MR_CI=ci[:, 2].tolist(), FAR_CI=ci[:, 3].tolist(),
         F1nb_CI=ci_nb[:, 0].tolist())

# =============== Báo cáo theo category ===============
cat_rows = []
for c in CATS:
    fs = [f for f in test if recs[f]["category"] == c]
    r = pooled(fs, THR); rnb = pooled(fs, THR, mask="nb"); e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
    for f in fs:
        for k, v in error_types(recs[f]["y"], pred_cached(recs[f], THR)).items(): e[k] += v
    cib = boot(fs, lambda x: pooled(x, THR), B=500)
    cat_rows.append(dict(Category=c, n_file=len(fs), pct_speech=cat_(fs, "y").mean(), MR=r["MR"], FAR=r["FAR"],
                         Precision=r["Precision"], **e, F1_strict=r["F1"], F1_nb=rnb["F1"], F1_CI_lo=cib[0, 0], F1_CI_hi=cib[1, 0],
                         FAR_CI_lo=cib[0, 3], FAR_CI_hi=cib[1, 3]))
cat_df = pd.DataFrame(cat_rows)

# =============== Ablation (ngưỡng chọn riêng cho từng cấu hình trên dev) ===============
abl = []
for mode, desc in [("A0", "Điểm bin thô, không hậu xử lý"), ("A1", "Rescore + rebin"), ("A2", "+ pad 100/120ms"),
                   ("A3", "+ merge <500ms"), ("A4", "+ pad + merge (hiện tại)")]:
    t = best["dev"][mode]; r = pooled(test, t, mode); rnb = pooled(test, t, mode, mask="nb"); e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
    for f in test:
        for k, v in error_types(recs[f]["y"], pred_cached(recs[f], t, mode)).items(): e[k] += v
    abl.append(dict(Cấu_hình=mode, Mô_tả=desc, Ngưỡng_dev=t, MR=r["MR"], FAR=r["FAR"], F1_strict=r["F1"], F1_nb=rnb["F1"],
                    DCF=r["DCF"], **e))
abl = pd.DataFrame(abl)
grid = []
for pre in [0, 0.05, 0.10, 0.20]:
    for post in [0, 0.06, 0.12, 0.24]:
        for gap in [0, 0.25, 0.5, 0.75, 1.0, 1.5]:
            r = pooled(dev, THR, "grid", pre=pre, post=post, gap=gap)
            grid.append(dict(pre_ms=int(pre * 1000), post_ms=int(post * 1000), gap_ms=int(gap * 1000),
                             TP=r["TP"], FN=r["FN"], FP=r["FP"], TN=r["TN"], F1=r["F1"], DCF=r["DCF"]))
grid = pd.DataFrame(grid)
S["grid_best"] = grid.loc[grid.DCF.idxmin()].to_dict()

# =============== Nhãn đáng ngờ: đoạn bin liền nhau (không sát biên) mà điểm và nhãn trái ngược mạnh ===============
sus = []
for f, d in recs.items():
    s_, y_ = d["braw"], d["y"]
    bad = ~d["bmask"] & (((s_ > 0.9) & (y_ == 0)) | ((s_ < 0.1) & (y_ == 1)))
    for a0, b0 in frames_to_segs(bad, 1):
        a0, b0 = int(a0), int(b0)
        sus.append(dict(file=f, category=d["category"], split=split[f], start=a0 * BIN, end=b0 * BIN, n_bins=b0 - a0,
                        GT="speech" if y_[a0] else "non-speech", score=round(float(s_[a0:b0].mean()), 3),
                        silero_speech=round(float(d["sil"][a0:b0].mean()), 2), nhan_do_hat=bool((y_[a0:b0] != d["y_alt"][a0:b0]).any())))
sus = pd.DataFrame(sus).sort_values("n_bins", ascending=False) if sus else pd.DataFrame()
S["n_suspicious_seg"] = len(sus); S["n_suspicious_bins"] = int(sus.n_bins.sum()) if len(sus) else 0
if len(sus):
    S["suspicious_by_type"] = sus.groupby(["category", "GT"]).n_bins.sum().unstack(fill_value=0).to_dict()
    S["suspicious_silero_agrees_model"] = float(((sus.GT == "non-speech") & (sus.silero_speech >= 0.5) |
                                                 (sus.GT == "speech") & (sus.silero_speech < 0.5)).mul(sus.n_bins).sum() / sus.n_bins.sum())

# =============== Số đếm thô cho Excel: make_excel_real.py tính mọi chỉ số bằng công thức từ đây ===============
XL = OUT / "excel"; XL.mkdir(exist_ok=True)
def counts(y, p):
    return dict(TP=int(((y == 1) & (p == 1)).sum()), FN=int(((y == 1) & (p == 0)).sum()),
                FP=int(((y == 0) & (p == 1)).sum()), TN=int(((y == 0) & (p == 0)).sum()))
# quét ngưỡng: dev / val tổng hợp gộp; test theo category × vùng bin (all / nb = bỏ bin biên / b = chỉ bin biên)
rows = []
for mode in MODES:
    for t in thr_sweep:
        P = {f: pred_cached(recs[f], t, mode) for f in test}
        for c in CATS:
            fs = [f for f in test if recs[f]["category"] == c]; e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
            for f in fs:
                for k, v in error_types(recs[f]["y"], P[f]).items(): e[k] += v
            for mk in ("all", "nb", "b"):
                sel = {f: slice(None) if mk == "all" else (~recs[f]["bmask"] if mk == "nb" else recs[f]["bmask"]) for f in fs}
                rows.append(dict(tap="test", config=mode, category=c, mask=mk, thr=t,
                                 **counts(np.concatenate([recs[f]["y"][sel[f]] for f in fs]), np.concatenate([P[f][sel[f]] for f in fs])),
                                 **(e if mk == "all" else {})))
sw = pd.concat([thr_df.assign(category="Tất cả", mask="all")[["tap", "config", "category", "mask", "thr", "TP", "FN", "FP", "TN"]],
                pd.DataFrame(rows)], ignore_index=True)
chk = sw[(sw.tap == "test") & (sw.config == "A4") & (sw["mask"] == "all") & (sw.thr == THR)][["TP", "FN", "FP", "TN"]].sum()
assert (chk.to_numpy() == tB.iloc[0][["TP", "FN", "FP", "TN"]].to_numpy()).all(), "số đếm quét ngưỡng lệch tầng B"
sw.to_csv(XL / "sweep_counts.csv", index=False)
# ROC trên điểm thô: lưới ngưỡng = phân vị điểm của cả 144 file (+ 2 ngưỡng chặn) -> Excel tính AUC bằng hình thang
roc_thr = np.r_[-1e-6, np.unique(np.quantile(S_all, np.linspace(0, 1, 401))), 1 + 1e-6]
roc = pd.DataFrame(dict(thr=roc_thr))
for g, fs in [(c, [f for f in test if recs[f]["category"] == c]) for c in CATS] + [("Dev", dev)]:
    y, s = cat_(fs, "y"), cat_(fs, "braw"); pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = len(pos) - np.searchsorted(pos, roc_thr, "left"); fp = len(neg) - np.searchsorted(neg, roc_thr, "left")
    roc[f"{g}_TP"], roc[f"{g}_FP"], roc[f"{g}_FN"], roc[f"{g}_TN"] = tp, fp, len(pos) - tp, len(neg) - fp
roc.to_csv(XL / "roc_counts.csv", index=False)
tpr = sum(roc[f"{c}_TP"] for c in CATS).to_numpy(); fpr = sum(roc[f"{c}_FP"] for c in CATS).to_numpy()
tpr, fpr = tpr / tpr[0], fpr / fpr[0]
S["auc_test_trapezoid"] = float(np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:])) / 2)
# nhãn: FireRedVAD quy tắc gốc (a) vs Silero (b), bảng 2×2 theo category × vùng bin
lab_rows = []
for c in CATS:
    fs = [f for f in pf.file if recs[f]["category"] == c]
    a_, b_, y_, ok = cat_(fs, "y_alt"), cat_(fs, "sil"), cat_(fs, "y"), cat_(fs, "sure")
    bm = np.concatenate([boundary_mask(recs[f]["y_alt"]) for f in fs])
    for mk, sel in (("all", np.ones(len(a_), bool)), ("trong", ~bm), ("bien", bm)):
        A_, B_ = a_[sel], b_[sel]
        lab_rows.append(dict(category=c, mask=mk, n11=int(((A_ == 1) & (B_ == 1)).sum()), n10=int(((A_ == 1) & (B_ == 0)).sum()),
                             n01=int(((A_ == 0) & (B_ == 1)).sum()), n00=int(((A_ == 0) & (B_ == 0)).sum()),
                             bin_doi_do_hat=int((y_[sel] != A_).sum()), n_nhan_chac=int(ok[sel].sum()), n_speech_policy=int(y_[sel].sum())))
pd.DataFrame(lab_rows).to_csv(XL / "label_counts.csv", index=False)
pd.DataFrame(deltas, columns=["d_onset", "d_offset"]).to_csv(XL / "deltas.csv", index=False)

# =============== Lưu ===============
pf = pf.merge(lagdf[["file", "lag_s", "peak_corr"]], on="file", how="left")
pf.to_csv(OUT / "per_file.csv", index=False); struct.to_csv(OUT / "structure.csv", index=False)
sweep.to_csv(OUT / "shift_sweep.csv", index=False); lagdf.to_csv(OUT / "lag_per_file.csv", index=False)
lab_cat.to_csv(OUT / "label_agreement.csv", index=False)
tA.to_csv(OUT / "tierA.csv", index=False); tA_alt.to_csv(OUT / "tierA_nhan_goc.csv", index=False); tB.to_csv(OUT / "tierB.csv", index=False)
thr_df.to_csv(OUT / "thr_sweep.csv", index=False); transfer.to_csv(OUT / "threshold_transfer.csv", index=False)
ev_thr.to_csv(OUT / "event_by_threshold.csv", index=False)
cat_df.to_csv(OUT / "category.csv", index=False); abl.to_csv(OUT / "ablation.csv", index=False); grid.to_csv(OUT / "grid_dev.csv", index=False)
sus.to_csv(OUT / "suspicious.csv", index=False, encoding="utf-8-sig")
json.dump(S, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

# =============== Hình ===============
def jitter(n): return rng.uniform(-0.18, 0.18, n)
# Hình 1: kiểm tra dữ liệu
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for i, c in enumerate(CATS):
    g = pf[pf.category == c]
    ax[0, 0].scatter(i + jitter(len(g)), g.speech_ratio, s=22, color=COLS[c], edgecolor="white", linewidth=0.6)
    g = g[g.auc_file.notna()]
    ax[0, 1].scatter(i + jitter(len(g)), g.auc_file, s=22, color=COLS[c], edgecolor="white", linewidth=0.6)
    ax[0, 1].plot([i - 0.28, i + 0.28], [g.auc_file.median()] * 2, c="#0b0b0b", lw=1.5)
for a_ in ax[0, :2]: a_.set_xticks(range(len(CATS))); a_.set_xticklabels(CATS)
ax[0, 0].set(title="1a. Tỉ lệ speech mỗi file (nhãn chính sách)", ylabel="speech ratio")
ax[0, 1].axhline(0.9, ls=":", c="#52514e", lw=0.8)
ax[0, 1].set(title=f"1d. AUC từng file (vạch = trung vị); {S['n_degenerate']} file 1 lớp không có AUC", ylabel="AUC")
for col, c_, lab in (("AUC_synth_exact", C1, "val tổng hợp (nhãn chính xác)"), ("AUC_real_fireredvad", C2, "thật – nhãn FireRedVAD"),
                      ("AUC_real_silero", "#1baf7a", "thật – nhãn Silero")):
    ax[0, 2].plot(sweep.shift_s, sweep[col] - sweep[col][sweep.shift_s == 0].iloc[0], "o-", ms=3, lw=1.8, c=c_, label=lab)
ax[0, 2].axvline(0, ls=":", c="#52514e", lw=0.8)
ax[0, 2].set(title="5e. ΔAUC theo độ dịch điểm (score tại tâm cửa sổ)", xlabel="shift (s)", ylabel="AUC − AUC(shift 0)")
ax[0, 2].legend(fontsize=7)
ax[1, 0].hist(lagdf.lag_s, bins=np.arange(-1.55, 1.6, 0.1), edgecolor="white", color=C1)
ax[1, 0].set(title=f"5e. Lag từng file: trung vị {S['lag_median']:+.2f}s, {len(S['lag_flag_files'])} file |lag|≥0.4s", xlabel="lag (s)", ylabel="#file")
ax[1, 1].hist(S_all[Y_all == 0], bins=40, alpha=.65, color=C1, label="non-speech", density=True)
ax[1, 1].hist(S_all[Y_all == 1], bins=40, alpha=.65, color=C2, label="speech", density=True)
ax[1, 1].set_yscale("log"); ax[1, 1].legend(fontsize=7); ax[1, 1].set(title="4a. Phân bố điểm theo lớp (144 file)", xlabel="score")
ax[1, 2].bar(lab_cat.Category, lab_cat.agree_aed_silero, color=C1, width=0.6)
for i, r in enumerate(lab_cat.itertuples()):
    ax[1, 2].text(i, r.agree_aed_silero + 0.01, f"{r.agree_aed_silero:.0%}\nκ={r.kappa_aed_silero:.2f}", ha="center", fontsize=7)
ax[1, 2].set_ylim(0, 1.12); ax[1, 2].set(title="3. Nhất quán nhãn: AED vs Silero (tỉ lệ bin khớp)")
plt.tight_layout(); plt.savefig(OUT / "fig1_kiem_tra_du_lieu.png", dpi=130); plt.close()
# Hình 2: tầng A trên test
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for c in CATS:
    fs = [f for f in test if recs[f]["category"] == c]; yy, ss = cat_(fs, "y"), cat_(fs, "braw")
    if not 0 < yy.mean() < 1: continue
    fpr, tpr, _ = roc_curve(yy, ss); ax[0].plot(fpr, tpr, c=COLS[c], lw=2, label=f"{c} AUC={auc(yy, ss):.3f}")
    pr, rc, _ = precision_recall_curve(yy, ss); ax[1].plot(rc, pr, c=COLS[c], lw=2, label=f"{c} AP={average_precision_score(yy, ss):.3f}")
    fp_, fn_, _ = det_curve(yy, ss); ax[2].plot(norm.ppf(np.clip(fp_, 1e-4, 1 - 1e-4)), norm.ppf(np.clip(fn_, 1e-4, 1 - 1e-4)), c=COLS[c], lw=2, label=c)
ax[0].plot([0, 1], [0, 1], ":", c="#52514e"); ax[0].set(title="ROC (test, tầng A)", xlabel="FPR", ylabel="TPR"); ax[0].legend(fontsize=7)
ax[1].set(title="Precision–Recall", xlabel="Recall", ylabel="Precision"); ax[1].legend(fontsize=7)
tk = [0.01, 0.05, 0.2, 0.5]; ax[2].set_xticks(norm.ppf(tk)); ax[2].set_xticklabels(tk); ax[2].set_yticks(norm.ppf(tk)); ax[2].set_yticklabels(tk)
ax[2].set(title="DET", xlabel="FAR", ylabel="Miss rate"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig2_tang_A.png", dpi=130); plt.close()
# Hình 3: ngưỡng + tầng B
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for tap, ls in (("dev", "-"), ("synth", "--")):
    g = thr_df[(thr_df.tap == tap) & (thr_df.config == "A4")]
    ax[0, 0].plot(g.thr, g.MR, ls, c=C2, lw=1.8, label=f"MR {tap}"); ax[0, 0].plot(g.thr, g.FAR, ls, c=C1, lw=1.8, label=f"FAR {tap}")
ax[0, 0].axvline(THR, c="#0b0b0b", lw=.8); ax[0, 0].axvline(best["synth"]["A4"], c="#52514e", ls=":")
ax[0, 0].set(title=f"MR/FAR theo ngưỡng (A4). DCF-min: dev thật={THR}, val tổng hợp={best['synth']['A4']}", xlabel="ngưỡng")
ax[0, 0].legend(fontsize=7)
x = np.arange(len(cat_df)); bnd = cat_df.FEC + cat_df.OVER; tru = cat_df.MSC + cat_df.NDS
ax[0, 1].bar(x, bnd, label="lỗi biên (FEC+OVER)", color=C1, width=0.6, edgecolor="white", linewidth=2)
ax[0, 1].bar(x, tru, bottom=bnd, label="lỗi thật (MSC+NDS)", color=C2, width=0.6, edgecolor="white", linewidth=2)
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(cat_df.Category); ax[0, 1].set(title=f"Loại lỗi theo category (test, A4, thr={THR})", ylabel="#bin")
ax[0, 1].legend(loc="upper left", fontsize=7)
ax[1, 0].hist(deltas[:, 0], bins=np.arange(-3.25, 3.3, 0.5), alpha=.65, color=C1, label=f"Δonset (TB {deltas[:, 0].mean():+.2f}s)")
ax[1, 0].hist(deltas[:, 1], bins=np.arange(-3.25, 3.3, 0.5), alpha=.65, color=C2, label=f"Δoffset (TB {deltas[:, 1].mean():+.2f}s)")
ax[1, 0].set(title="Độ lệch biên đoạn (dự đoán − GT), test", xlabel="giây"); ax[1, 0].legend(fontsize=7)
hm = grid.groupby(["pre_ms", "gap_ms"]).DCF.min().unstack()
im = ax[1, 1].imshow(hm.values, cmap="Blues_r", aspect="auto"); plt.colorbar(im, ax=ax[1, 1]); ax[1, 1].grid(False)
ax[1, 1].set_xticks(range(hm.shape[1])); ax[1, 1].set_xticklabels(hm.columns); ax[1, 1].set_yticks(range(hm.shape[0])); ax[1, 1].set_yticklabels(hm.index)
mid = np.nanmean(hm.values)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        ax[1, 1].text(j, i, f"{hm.values[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if hm.values[i, j] < mid else "#0b0b0b")
ax[1, 1].set(title="DCF (dev) theo pad trước × merge gap (min theo pad sau; thấp = tốt)", xlabel="merge gap (ms)", ylabel="pad trước (ms)")
plt.tight_layout(); plt.savefig(OUT / "fig3_tang_B.png", dpi=130); plt.close()
# Hình 4: timeline mẫu (test): file AUC thấp nhất + 1 file mỗi category có AUC gần trung vị
pt = pf[(pf.split == "test") & pf.auc_file.notna()]
show = [pt.sort_values("auc_file").file.iloc[0]]
for c in CATS:
    g = pt[(pt.category == c) & ~pt.file.isin(show)]
    if len(g): show.append(g.iloc[(g.auc_file - g.auc_file.median()).abs().argsort()].file.iloc[0])
fig, axs = plt.subplots(len(show), 1, figsize=(14, 1.9 * len(show)))
for a_, f in zip(axs, show):
    d = recs[f]; y = d["y"]; p = pred_cached(d, THR); tt = np.arange(d["n"] + 1) * BIN
    a_.fill_between(tt, np.r_[y, y[-1]], step="post", color=C1, alpha=.18, lw=0, label="GT speech")
    a_.plot(d["m"].center, d["m"].score, lw=0.8, c=C1, label="score tại tâm cửa sổ")
    a_.fill_between(tt, -0.12 * np.r_[p, p[-1]], step="post", color="#0b0b0b", alpha=.55, lw=0, label="dự đoán A4")
    a_.axhline(THR, ls="--", c="#52514e", lw=.7); a_.set_ylim(-0.15, 1.05); a_.set_xlim(0, d["n"] * BIN)
    a_.set_title(f"{f}  AUC={safe_auc(y, d['braw']):.3f}  speech={y.mean():.0%}", fontsize=8, loc="left")
axs[0].legend(fontsize=7, loc="upper right", ncol=3); axs[-1].set_xlabel("giây")
plt.tight_layout(); plt.savefig(OUT / "fig4_timeline.png", dpi=130); plt.close()

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(json.dumps({k: S[k] for k in ["n_pairs", "problems", "n_struct_flag", "struct_flag_files", "best_shift_start", "best_shift_center",
      "best_shift_auc", "auc_gain_best_shift_real", "lag_median", "hours", "pct_speech_bins", "n_degenerate", "auc_pooled_all", "auc_macro", "auc_worst", "auc_worst_file",
      "n_files_auc_lt_09", "n_files_auc_lt_07", "kappa_aed_silero", "kappa_aed_silero_boundary", "kappa_aed_silero_interior",
      "pct_bins_changed_by_singing", "pct_sure_bins", "n_dev", "n_test", "AUC_test_CI", "n_synth_val", "auc_synth_val",
      "thr_youden_raw_synth", "thr_youden_raw_dev", "best_thr_dev", "best_thr_synth", "best_f1_thr_dev", "best_f1_thr_synth",
      "thr_at_grid_edge", "event_f1", "event_f1_onset", "err_types",
      "pct_true_err", "F1_CI", "DCF_CI", "MR_CI", "FAR_CI", "n_gt_seg", "n_pred_seg", "n_fragmented", "n_overmerged",
      "mean_d_onset", "mean_d_offset", "gap_short_total", "gap_short_filled", "grid_best", "n_suspicious_seg", "n_suspicious_bins", "suspicious_silero_agrees_model"]},
      ensure_ascii=False, indent=1, default=float))
for name, df in [("Nhất quán nhãn", lab_cat), ("Tầng A (nhãn chính sách)", tA), ("Tầng A (nhãn gốc: hát = non-speech)", tA_alt),
                 ("Ngưỡng: chọn ở đâu -> kết quả trên test", transfer), ("Tầng B (A4)", tB), ("Chỉ số theo đoạn (test, A4)", ev_thr), ("Ablation", abl), ("Category (test, A4)", cat_df)]:
    print(f"\n=== {name} ==="); print(df.round(4).to_string(index=False))
print("\nTop đoạn nghi nhãn sai:"); print(sus.head(15).to_string(index=False) if len(sus) else "(không có)")
