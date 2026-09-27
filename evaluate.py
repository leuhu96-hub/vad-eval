"""Chạy toàn bộ quy trình: kiểm tra dữ liệu (Bước 8) -> tầng A -> chọn ngưỡng -> tầng B -> ablation -> báo cáo."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             det_curve, brier_score_loss, cohen_kappa_score)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import norm
from vadlib import *
warnings.filterwarnings("ignore")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
rng = np.random.default_rng(0)
S = {}                                   # summary
meta = pd.read_csv("meta.csv", keep_default_na=False)

# =============== 5a. Ghép file ===============
pairs, problems = pair_files(".")
S["n_pairs"] = len(pairs); S["problems"] = problems
recs = {}
for r in pairs.itertuples():
    y = load_gt(r.gt); m = load_model(r.model, center=True); m0 = load_model(r.model, center=False)
    recs[r.file] = dict(file=r.file, category=r.category, y=y, m=m, m0=m0)

# =============== 5b/5d. Cấu trúc ===============
struct = []
for f, d in recs.items():
    m = d["m"]; dd = np.diff(m.start)
    struct.append(dict(file=f, n_bins=len(d["y"]), last_end=m.end.max(), first_start=m.start.min(),
                       hop_ok=bool(np.allclose(dd, HOP, atol=1e-3)), win_ok=bool(np.allclose(m.end - m.start, WIN, atol=1e-3)),
                       score_ok=bool(m.score.between(0, 1).all())))
struct = pd.DataFrame(struct)
struct["dur_flag"] = (struct.n_bins != 20) | ((struct.last_end - DUR).abs() > 0.1) | (struct.first_start.abs() > 0.01)
S["n_dur_flag"] = int(struct.dur_flag.sum()); S["dur_flag_files"] = struct.loc[struct.dur_flag, "file"].tolist()
S["n_hop_bad"] = int((~struct.hop_ok).sum()); S["n_win_bad"] = int((~struct.win_ok).sum()); S["n_score_bad"] = int((~struct.score_ok).sum())

# =============== 5e. Căn chỉnh: quét shift + lag từng file ===============
def pooled_f1_shift(key, shift, thr=0.5):
    Y, P = [], []
    for d in recs.values():
        s = bin_scores(d[key], len(d["y"]), shift=shift); Y.append(d["y"][1:-1]); P.append((s >= thr)[1:-1])
    return rates(np.concatenate(Y), np.concatenate(P).astype(int))["F1"]
shifts = np.round(np.arange(-12, 13) * 0.08, 2)
sweep = pd.DataFrame(dict(shift_s=shifts,
                          F1_start=[pooled_f1_shift("m0", s) for s in shifts],
                          F1_center=[pooled_f1_shift("m", s) for s in shifts]))
S["best_shift_start"] = float(sweep.shift_s[sweep.F1_start.idxmax()]); S["best_shift_center"] = float(sweep.shift_s[sweep.F1_center.idxmax()])
S["F1_start_at0"] = float(sweep.F1_start[sweep.shift_s == 0].iloc[0]); S["F1_center_at0"] = float(sweep.F1_center[sweep.shift_s == 0].iloc[0])

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
        lag, pc = best_lag(d["m"], d["y"]); lagrows.append(dict(file=f, lag_s=round(lag, 2), peak_corr=pc))
lagdf = pd.DataFrame(lagrows)
# |lag| >= 0.4s (~1 bin) -> lỗi lệch thật cần sửa; 0.2–0.4s -> chỉ "cần xem" (thường do SNR thấp làm cross-corr nhiễu)
lagdf["flag"] = np.where(lagdf.lag_s.abs() >= 0.4, "Lệch ~1 bin", np.where(lagdf.lag_s.abs() > 0.2, "Cần xem", ""))
lag_flag = lagdf[lagdf.flag == "Lệch ~1 bin"]
S["lag_flag_files"] = lag_flag.file.tolist(); S["lag_flag_values"] = lag_flag.lag_s.tolist()
S["lag_review_files"] = lagdf.file[lagdf.flag == "Cần xem"].tolist()
# Sửa: file bị gắn cờ được dịch lại theo lag ước lượng (làm tròn theo hop) -> mô phỏng việc sửa script xuất file
for r in lag_flag.itertuples():
    corr = np.round(r.lag_s / BIN) * BIN   # sau khi kiểm tra: lỗi xuất file lệch đúng 1 bin
    for k in ("m", "m0"):
        recs[r.file][k] = recs[r.file][k].assign(start=lambda x: x.start + corr, end=lambda x: x.end + corr,
                                                 center=lambda x: x.center + corr)
S["lag_correction_applied"] = {r.file: float(np.round(r.lag_s / BIN) * BIN) for r in lag_flag.itertuples()}

# cache điểm
for d in recs.values():
    d["braw"] = bin_scores(d["m"], len(d["y"])); d["fr"] = rescore_hop(d["m"]); d["bmask"] = boundary_mask(d["y"])

def pred_cached(d, thr, mode="A4", pre=0.10, post=0.12, gap=0.50):
    if mode == "A0": return (d["braw"] >= thr).astype(int)
    segs = frames_to_segs(d["fr"] >= thr, HOP)
    p, q, g = {"A1": (0, 0, 0), "A2": (pre, post, 0), "A3": (0, 0, gap), "A4": (pre, post, gap)}.get(mode, (pre, post, gap))
    return segs_to_bins(pad_merge(segs, p, q, g), len(d["y"]))

# =============== 1. Phân bố ===============
rows = []
for f, d in recs.items():
    y = d["y"]; segs = bins_to_segs(y)
    rows.append(dict(file=f, category=d["category"], speech_ratio=y.mean(), transitions=int((y[1:] != y[:-1]).sum()),
                     n_speech_seg=len(segs), mean_speech_seg_s=np.mean([b - a for a, b in segs]) if segs else 0,
                     boundary_frac=d["bmask"].mean(), auc_file=safe_auc(y, d["braw"])))
pf = pd.DataFrame(rows).merge(meta[["file", "snr_db", "seen_in_train", "speech_src", "noise_src", "injected_shift_s"]], on="file", how="left")
pf["degenerate"] = (pf.speech_ratio == 0) | (pf.speech_ratio == 1)
pf["extreme"] = (pf.speech_ratio < 0.1) | (pf.speech_ratio > 0.9)
Y_all = np.concatenate([recs[f]["y"] for f in pf.file]); S_all = np.concatenate([recs[f]["braw"] for f in pf.file])
keep = np.concatenate([np.full(20, not dg) for dg in pf.degenerate])
S.update(pct_degenerate=float(pf.degenerate.mean()), pct_extreme=float(pf.extreme.mean()),
         median_transitions=float(pf.transitions.median()), mean_boundary_frac=float(pf.boundary_frac.mean()),
         auc_pooled_all=roc_auc_score(Y_all, S_all), auc_pooled_nodeg=roc_auc_score(Y_all[keep], S_all[keep]),
         auc_macro=float(pf.auc_file.mean()), auc_macro_sd=float(pf.auc_file.std()),
         auc_worst=float(pf.auc_file.min()), n_files_auc_lt_09=int((pf.auc_file < 0.9).sum()))

# =============== 2. Rò rỉ + chia dev/test theo nhóm ===============
train_src = set(pd.read_csv("train_sources.csv").source)
pf["seen"] = pf.speech_src.isin(train_src) | pf.noise_src.isin(train_src)
S["pct_seen"] = float(pf.seen.mean())
mask_ok = ~pf.degenerate
auc_seen = roc_auc_score(np.concatenate([recs[f]["y"] for f in pf.file[pf.seen & mask_ok]]),
                         np.concatenate([recs[f]["braw"] for f in pf.file[pf.seen & mask_ok]]))
auc_unseen = roc_auc_score(np.concatenate([recs[f]["y"] for f in pf.file[~pf.seen & mask_ok]]),
                           np.concatenate([recs[f]["braw"] for f in pf.file[~pf.seen & mask_ok]]))
S.update(auc_seen=auc_seen, auc_unseen=auc_unseen, auc_seen_gap=auc_seen - auc_unseen)
# union-find: clip dùng chung speech_src hoặc noise_src -> cùng nhóm
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for r in pf.itertuples():
    for v in (r.speech_src, r.noise_src):
        if v: parent[find(f"clip:{r.file}")] = find(f"src:{v}")
pf["group_full"] = [find(f"clip:{f}") for f in pf.file]
S["n_groups_full"] = int(pf.group_full.nunique()); S["largest_group_full"] = int(pf.group_full.value_counts().max())
# Nhóm khổng lồ (câu nói dùng lại ở nhiều category nối mọi thứ lại) -> không chia được.
# Dự phòng: nhóm theo câu nói gốc (file không có câu nói -> theo nhiễu), rồi đếm nhiễu dùng chung giữa dev/test.
pf["group"] = np.where(pf.speech_src != "", "sp:" + pf.speech_src, "nz:" + pf.noise_src)
S["n_groups"] = int(pf.group.nunique()); S["largest_group"] = int(pf.group.value_counts().max())
# Chính sách: dùng file 'unseen' làm tập đánh giá chính; chia dev/test theo nhóm
ev = pf[~pf.seen].copy()
# gán nhóm vào dev cho tới khi đủ ~30% số file; ưu tiên category đang thiếu dev
ev["split"] = "test"; target = 0.3
groups = ev.group.unique().copy(); np.random.default_rng(3).shuffle(groups)
for gr in groups:
    need = {c: target * (ev.category == c).sum() - ((ev.category == c) & (ev.split == "dev")).sum() for c in ev.category.unique()}
    cats = ev.loc[ev.group == gr, "category"]
    if all(need[c] > 0 for c in cats): ev.loc[ev.group == gr, "split"] = "dev"
pf = pf.merge(ev[["file", "split"]], on="file", how="left"); pf["split"] = pf.split.fillna("seen (loại)")
dev = pf.file[pf.split == "dev"].tolist(); test = pf.file[pf.split == "test"].tolist()
S.update(n_dev=len(dev), n_test=len(test), n_seen_excluded=int(pf.seen.sum()),
         groups_crossing=int(len(set(pf.group[pf.split == "dev"]) & set(pf.group[pf.split == "test"]))),
         noise_shared_dev_test=int(len(set(pf.noise_src[pf.split == "dev"]) & set(pf.noise_src[pf.split == "test"]))))

# =============== 3. Nhất quán nhãn ===============
A, B, BM, offs = [], [], [], []
for p2 in sorted(Path("Annotator2").glob("Gt_*.txt")):
    cat, i = p2.stem.split("_")[1:]; f = f"{cat.capitalize()}_{i}.txt"
    if f not in recs: continue
    y1 = recs[f]["y"]; y2 = load_gt(p2); A.append(y1); B.append(y2); BM.append(recs[f]["bmask"])
    f1 = np.load(next(Path("Groundtruth").glob(f"*/groundtruth/fine_{cat}_{i}.npy"))); f2 = np.load(Path("Annotator2") / f"fine_{cat}_{i}.npy")
    c1 = np.flatnonzero(np.diff(f1)) * 0.01; c2 = np.flatnonzero(np.diff(f2)) * 0.01
    for t in c1:
        if len(c2) and np.min(np.abs(c2 - t)) <= 1.0: offs.append(c2[np.argmin(np.abs(c2 - t))] - t)
A, B, BM, offs = np.concatenate(A), np.concatenate(B), np.concatenate(BM), np.array(offs)
S.update(n_ann2_files=int(len(A) / 20), kappa_all=cohen_kappa_score(A, B), kappa_boundary=cohen_kappa_score(A[BM], B[BM]),
         kappa_interior=cohen_kappa_score(A[~BM], B[~BM]) if len(set(A[~BM])) > 1 else 1.0,
         raw_agree=float((A == B).mean()), pabak=float(2 * (A == B).mean() - 1),
         median_abs_offset_ms=float(np.median(np.abs(offs)) * 1000), pct_within_250ms=float((np.abs(offs) <= 0.25).mean()))

# =============== 4. Độ khó ===============
amb = (S_all[keep] > 0.1) & (S_all[keep] < 0.9)
bm_all = np.concatenate([recs[f]["bmask"] for f in pf.file])[keep]
S.update(pct_ambiguous=float(amb.mean()), pct_ambiguous_in_boundary=float((amb & bm_all).sum() / max(amb.sum(), 1)))
thr_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
f1_thr = [rates(Y_all[keep], (S_all[keep] >= t).astype(int))["F1"] for t in thr_grid]
S["f1_range_0.2_0.9"] = float(max(f1_thr[3:18]) - min(f1_thr[3:18]))
pf["snr_bucket"] = pd.cut(pf.snr_db, [-99, 0, 5, 10, 20, 99], labels=["<0", "0–5", "5–10", "10–20", ">20"])
snr_auc = []
for b, g in pf[~pf.degenerate].groupby("snr_bucket", observed=True):
    yy = np.concatenate([recs[f]["y"] for f in g.file]); ss = np.concatenate([recs[f]["braw"] for f in g.file])
    snr_auc.append(dict(snr_bucket=str(b), n_file=len(g), auc=roc_auc_score(yy, ss)))
snr_auc = pd.DataFrame(snr_auc); S["pct_snr_gt15"] = float((pf.snr_db > 15).mean())

# =============== Tầng A (điểm thô) trên test ===============
def tierA(files, name):
    y = np.concatenate([recs[f]["y"] for f in files]); s = np.concatenate([recs[f]["braw"] for f in files])
    bm = np.concatenate([recs[f]["bmask"] for f in files])
    fpr, tpr, _ = roc_curve(y, s)
    macro = np.nanmean([safe_auc(recs[f]["y"], recs[f]["braw"]) for f in files])
    return dict(Tập=name, n_file=len(files), n_bin=len(y), pct_speech=y.mean(),
                AUC_micro=roc_auc_score(y, s), AUC_macro_file=macro,
                AUC_khong_bien=roc_auc_score(y[~bm], s[~bm]) if len(set(y[~bm])) > 1 else np.nan,
                AUC_chi_bien=roc_auc_score(y[bm], s[bm]) if len(set(y[bm])) > 1 else np.nan,
                PR_AUC=average_precision_score(y, s), TPR_at_FPR1=float(np.interp(0.01, fpr, tpr)),
                TPR_at_FPR5=float(np.interp(0.05, fpr, tpr)), Brier=brier_score_loss(y, s))
tA = [tierA(test, "Test (tất cả)")]
for c in sorted(pf.category.unique()):
    fs = [f for f in test if recs[f]["category"] == c]
    if fs: tA.append(tierA(fs, c))
tA.append(tierA(pf.file[pf.seen].tolist(), "File 'seen' (tham khảo)"))
tA = pd.DataFrame(tA)
# Youden trên dev (điểm thô)
yd = np.concatenate([recs[f]["y"] for f in dev]); sd_ = np.concatenate([recs[f]["braw"] for f in dev])
fpr, tpr, th = roc_curve(yd, sd_); S["thr_youden_raw_dev"] = float(th[np.argmax(tpr - fpr)])

# =============== Chọn ngưỡng trên dev qua toàn pipeline ===============
def pooled(files, thr, mode="A4", mask=None, **kw):
    Y, P = [], []
    for f in files:
        d = recs[f]; p = pred_cached(d, thr, mode, **kw); y = d["y"]
        sel = np.ones(len(y), bool) if mask is None else (~d["bmask"] if mask == "nb" else d["bmask"])
        Y.append(y[sel]); P.append(p[sel])
    return rates(np.concatenate(Y), np.concatenate(P))
thr_sweep = np.round(np.arange(0.05, 0.951, 0.01), 2)
sweep_rows, best_thr = [], {}
for mode in ["A0", "A1", "A2", "A3", "A4"]:
    for t in thr_sweep:
        r = pooled(dev, t, mode); sweep_rows.append(dict(config=mode, thr=t, MR=r["MR"], FAR=r["FAR"], F1=r["F1"], DCF=r["DCF"]))
thr_df = pd.DataFrame(sweep_rows)
for mode, g in thr_df.groupby("config"): best_thr[mode] = float(g.thr[g.DCF.idxmin()])
S["best_thr_dev"] = best_thr; THR = best_thr["A4"]
S["dcf_A4_at_youden"] = pooled(test, S["thr_youden_raw_dev"])["DCF"]; S["dcf_A4_at_best"] = pooled(test, THR)["DCF"]

# =============== Tầng B trên test ===============
tB = []
for mode_name, mk in [("Strict (mọi bin)", None), ("Collar ±1 bin (bỏ bin biên)", "nb"), ("Chỉ bin biên", "b")]:
    r = pooled(test, THR, "A4", mask=mk); r["Chế độ"] = mode_name; tB.append(r)
tB = pd.DataFrame(tB)
# loại lỗi, event, fragmentation, delta biên
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
S.update(thr_used=THR, err_types=et, pct_true_err=(et["MSC"] + et["NDS"]) / max(sum(et.values()), 1),
         event_f1=2 * ev_tp / (2 * ev_tp + ev_fp + ev_fn), event_f1_onset=2 * on_tp / (2 * on_tp + on_fp + on_fn),
         n_gt_seg=n_gt_seg, n_pred_seg=n_pr_seg, n_fragmented=frag, n_overmerged=merge,
         mean_d_onset=float(deltas[:, 0].mean()), mean_d_offset=float(deltas[:, 1].mean()),
         gap_short_total=gap_total, gap_short_filled=gap_filled)

# Bootstrap CI theo file
def boot(files, fn, B=1000):
    vals = []
    per = {f: fn([f]) for f in files}
    for _ in range(B):
        pick = rng.choice(files, len(files), replace=True)
        tot = pd.DataFrame([per[f] for f in pick]).sum()
        tp, fp, fn_ = tot.TP, tot.FP, tot.FN; tn = tot.TN
        f1 = 2 * tp / (2 * tp + fp + fn_); mr = fn_ / (tp + fn_); far = fp / (fp + tn)
        vals.append((f1, 0.75 * mr + 0.25 * far))
    v = np.array(vals); return np.percentile(v, [2.5, 97.5], axis=0)
ci = boot(test, lambda fs: pooled(fs, THR))
ci_nb = boot(test, lambda fs: pooled(fs, THR, mask="nb"))
S.update(F1_CI=[float(ci[0, 0]), float(ci[1, 0])], DCF_CI=[float(ci[0, 1]), float(ci[1, 1])],
         F1nb_CI=[float(ci_nb[0, 0]), float(ci_nb[1, 0])])
# CI của AUC (bootstrap theo file)
aucs = []
for _ in range(500):
    pick = rng.choice(test, len(test), replace=True)
    yy = np.concatenate([recs[f]["y"] for f in pick]); ss = np.concatenate([recs[f]["braw"] for f in pick]); aucs.append(roc_auc_score(yy, ss))
S["AUC_test_CI"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]

# =============== Báo cáo theo category ===============
cat_rows = []
for c in sorted(pf.category.unique()):
    fs = [f for f in test if recs[f]["category"] == c]
    if not fs: continue
    r = pooled(fs, THR); rnb = pooled(fs, THR, mask="nb"); e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
    for f in fs:
        for k, v in error_types(recs[f]["y"], pred_cached(recs[f], THR)).items(): e[k] += v
    cib = boot(fs, lambda x: pooled(x, THR), B=500)
    cat_rows.append(dict(Category=c, n_file=len(fs), pct_speech=np.mean([recs[f]["y"].mean() for f in fs]),
                         MR=r["MR"], FAR=r["FAR"], **e, F1_strict=r["F1"], F1_nb=rnb["F1"],
                         F1_CI_lo=cib[0, 0], F1_CI_hi=cib[1, 0]))
cat_df = pd.DataFrame(cat_rows)

# =============== Ablation (ngưỡng chọn riêng cho từng cấu hình trên dev) ===============
abl = []
for mode, desc in [("A0", "Điểm bin thô, không hậu xử lý"), ("A1", "Rescore + rebin"), ("A2", "+ pad 100/120ms"),
                   ("A3", "+ merge <500ms"), ("A4", "+ pad + merge (hiện tại)")]:
    t = best_thr[mode]; r = pooled(test, t, mode); rnb = pooled(test, t, mode, mask="nb"); e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
    for f in test:
        for k, v in error_types(recs[f]["y"], pred_cached(recs[f], t, mode)).items(): e[k] += v
    abl.append(dict(Cấu_hình=mode, Mô_tả=desc, Ngưỡng_dev=t, MR=r["MR"], FAR=r["FAR"], F1_strict=r["F1"], F1_nb=rnb["F1"], **e))
abl = pd.DataFrame(abl)
# Lưới pad x merge trên dev
grid = []
for pre in [0, 0.05, 0.10, 0.20]:
    for post in [0, 0.06, 0.12, 0.24]:
        for gap in [0, 0.25, 0.5, 0.75, 1.0]:
            r = pooled(dev, THR, "grid", pre=pre, post=post, gap=gap)
            grid.append(dict(pre_ms=int(pre * 1000), post_ms=int(post * 1000), gap_ms=int(gap * 1000), F1=r["F1"], DCF=r["DCF"]))
grid = pd.DataFrame(grid)
bg = grid.loc[grid.DCF.idxmin()]; S["grid_best"] = bg.to_dict()

# =============== Nhãn đáng ngờ ===============
sus = []
for f in pf.file:
    d = recs[f]
    for k in range(len(d["y"])):
        s_, y_ = d["braw"][k], d["y"][k]
        if not d["bmask"][k] and ((s_ > 0.9 and y_ == 0) or (s_ < 0.1 and y_ == 1)):
            sus.append(dict(file=f, bin=k, t_start=k * BIN, GT="speech" if y_ else "non-speech", score=round(float(s_), 3),
                            category=d["category"], nghi_van=abs(s_ - y_)))
sus = pd.DataFrame(sus).sort_values("nghi_van", ascending=False) if sus else pd.DataFrame()
S["n_suspicious"] = len(sus)

# =============== Lưu ===============
pf.to_csv(OUT / "per_file.csv", index=False); struct.to_csv(OUT / "structure.csv", index=False)
sweep.to_csv(OUT / "shift_sweep.csv", index=False); lagdf.to_csv(OUT / "lag_per_file.csv", index=False)
tA.to_csv(OUT / "tierA.csv", index=False); tB.to_csv(OUT / "tierB.csv", index=False); thr_df.to_csv(OUT / "thr_sweep_dev.csv", index=False)
cat_df.to_csv(OUT / "category.csv", index=False); abl.to_csv(OUT / "ablation.csv", index=False); grid.to_csv(OUT / "grid_dev.csv", index=False)
snr_auc.to_csv(OUT / "snr_auc.csv", index=False); sus.to_csv(OUT / "suspicious.csv", index=False)
json.dump(S, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=float)

# =============== Hình ===============
cols = dict(Asm="#1f77b4", Babble="#d62728", Clean="#2ca02c", Music="#9467bd", Noise="#ff7f0e")
# Hình 1: kiểm tra dữ liệu
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
ax[0, 0].hist(pf.speech_ratio, bins=np.linspace(0, 1, 21), edgecolor="k", color="#9ecae1")
ax[0, 0].axvspan(0, .1, color="r", alpha=.12); ax[0, 0].axvspan(.9, 1, color="r", alpha=.12)
ax[0, 0].set(title="1a/1b. Tỉ lệ speech mỗi file", xlabel="speech ratio", ylabel="#file")
ax[0, 1].hist(pf.transitions, bins=range(0, pf.transitions.max() + 2), edgecolor="k", color="#9ecae1")
ax[0, 1].set(title="1c. Số lần chuyển trạng thái / file", xlabel="#transitions")
ax[0, 2].plot(sweep.shift_s, sweep.F1_start, "o-", ms=3, label="score đặt tại start (sai)")
ax[0, 2].plot(sweep.shift_s, sweep.F1_center, "s-", ms=3, label="score đặt tại tâm (+0.48s)")
for v in (0, 0.48): ax[0, 2].axvline(v, ls=":", c="gray")
ax[0, 2].set(title="5e. F1 theo độ dịch thời gian", xlabel="shift (s)", ylabel="F1 (thr 0.5)"); ax[0, 2].legend(fontsize=7)
ax[1, 0].hist(lagdf.lag_s, bins=np.arange(-1.5, 1.55, 0.1), edgecolor="k", color="#fdae6b")
ax[1, 0].axvspan(-0.4, -0.2, color="y", alpha=.15); ax[1, 0].axvspan(0.2, 0.4, color="y", alpha=.15)
ax[1, 0].set(title=f"5e. Lag từng file: {len(lag_flag)} lệch ~1 bin, {len(S['lag_review_files'])} cần xem", xlabel="lag (s)")
ax[1, 1].hist(S_all[keep][Y_all[keep] == 0], bins=40, alpha=.6, label="non-speech", density=True)
ax[1, 1].hist(S_all[keep][Y_all[keep] == 1], bins=40, alpha=.6, label="speech", density=True)
ax[1, 1].set_yscale("log"); ax[1, 1].legend(); ax[1, 1].set(title="4a. Phân bố điểm theo lớp", xlabel="score")
ax[1, 2].bar(["seen", "unseen"], [auc_seen, auc_unseen], color=["#fb6a4a", "#6baed6"])
ax[1, 2].set_ylim(min(auc_seen, auc_unseen) - 0.02, 1); ax[1, 2].set(title="2c. AUC: nguồn đã thấy vs chưa thấy")
for i, v in enumerate([auc_seen, auc_unseen]): ax[1, 2].text(i, v + 0.002, f"{v:.4f}", ha="center")
plt.tight_layout(); plt.savefig(OUT / "fig1_kiem_tra_du_lieu.png", dpi=130); plt.close()
# Hình 2: tầng A
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for c in sorted(cols):
    fs = [f for f in test if recs[f]["category"] == c]
    if not fs: continue
    yy = np.concatenate([recs[f]["y"] for f in fs]); ss = np.concatenate([recs[f]["braw"] for f in fs])
    fpr, tpr, _ = roc_curve(yy, ss); ax[0].plot(fpr, tpr, c=cols[c], label=f"{c} AUC={roc_auc_score(yy, ss):.3f}")
    pr, rc, _ = precision_recall_curve(yy, ss); ax[1].plot(rc, pr, c=cols[c], label=f"{c} AP={average_precision_score(yy, ss):.3f}")
    fp_, fn_, _ = det_curve(yy, ss); ax[2].plot(norm.ppf(np.clip(fp_, 1e-4, 1 - 1e-4)), norm.ppf(np.clip(fn_, 1e-4, 1 - 1e-4)), c=cols[c], label=c)
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC (test, tầng A)", xlabel="FPR", ylabel="TPR"); ax[0].legend(fontsize=7)
ax[1].set(title="Precision–Recall", xlabel="Recall", ylabel="Precision"); ax[1].legend(fontsize=7)
tk = [0.01, 0.05, 0.2, 0.5]; ax[2].set_xticks(norm.ppf(tk)); ax[2].set_xticklabels(tk); ax[2].set_yticks(norm.ppf(tk)); ax[2].set_yticklabels(tk)
ax[2].set(title="DET", xlabel="FAR", ylabel="Miss rate"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig2_tang_A.png", dpi=130); plt.close()
# Hình 3: tầng B
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for mode, ls in (("A0", "--"), ("A4", "-")):
    g = thr_df[thr_df.config == mode]; ax[0, 0].plot(g.thr, g.MR, ls, c="#d62728", label=f"MR {mode}"); ax[0, 0].plot(g.thr, g.FAR, ls, c="#1f77b4", label=f"FAR {mode}")
ax[0, 0].axvline(THR, c="k", lw=.8); ax[0, 0].axvline(S["thr_youden_raw_dev"], c="gray", ls=":")
ax[0, 0].set(title=f"MR/FAR theo ngưỡng (dev). Chọn DCF-min={THR}; Youden thô={S['thr_youden_raw_dev']:.2f}", xlabel="ngưỡng"); ax[0, 0].legend(fontsize=7)
x = np.arange(len(cat_df)); bnd = cat_df.FEC + cat_df.OVER; tru = cat_df.MSC + cat_df.NDS
ax[0, 1].bar(x, bnd, label="lỗi biên (FEC+OVER)", color="#fdae6b"); ax[0, 1].bar(x, tru, bottom=bnd, label="lỗi thật (MSC+NDS)", color="#de2d26")
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(cat_df.Category); ax[0, 1].set(title="Loại lỗi theo category (test, A4)", ylabel="#bin"); ax[0, 1].legend(loc="upper left")
ax[1, 0].hist(deltas[:, 0], bins=np.arange(-2.25, 2.3, 0.5), alpha=.6, label=f"Δonset (TB {deltas[:,0].mean():+.2f}s)")
ax[1, 0].hist(deltas[:, 1], bins=np.arange(-2.25, 2.3, 0.5), alpha=.6, label=f"Δoffset (TB {deltas[:,1].mean():+.2f}s)")
ax[1, 0].set(title="Độ lệch biên đoạn (dự đoán − GT)", xlabel="giây"); ax[1, 0].legend()
hm = grid.groupby(["pre_ms", "gap_ms"]).F1.max().unstack()
im = ax[1, 1].imshow(hm.values, cmap="viridis", aspect="auto"); plt.colorbar(im, ax=ax[1, 1])
ax[1, 1].set_xticks(range(hm.shape[1])); ax[1, 1].set_xticklabels(hm.columns); ax[1, 1].set_yticks(range(hm.shape[0])); ax[1, 1].set_yticklabels(hm.index)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]): ax[1, 1].text(j, i, f"{hm.values[i,j]:.3f}", ha="center", va="center", color="w", fontsize=7)
ax[1, 1].set(title="F1 (dev) theo pad trước × merge gap (max theo pad sau)", xlabel="merge gap (ms)", ylabel="pad trước (ms)")
plt.tight_layout(); plt.savefig(OUT / "fig3_tang_B.png", dpi=130); plt.close()
# Hình 4: timeline mẫu
show = [pf.file[pf.injected_shift_s > 0].iloc[0], pf.sort_values("auc_file").file.iloc[0]]
show += [f for f in test if recs[f]["category"] == "Music"][:1] + [f for f in test if recs[f]["category"] == "Clean"][:1]
show += pf.file[pf.degenerate].tolist()[:1] + [f for f in test if recs[f]["category"] == "Babble"][:1]
fig, axs = plt.subplots(len(show), 1, figsize=(12, 1.9 * len(show)), sharex=True)
for a, f in zip(axs, show):
    d = recs[f]; y = d["y"]; p = pred_cached(d, THR)
    a.fill_between(np.arange(21) * BIN, np.r_[y, y[-1]], step="post", alpha=.25, label="GT speech")
    a.plot(d["m"].center, d["m"].score, lw=1.2, label="score tại tâm cửa sổ")
    a.fill_between(np.arange(21) * BIN, -0.12 * np.r_[p, p[-1]], step="post", color="k", alpha=.5, label="dự đoán A4")
    a.axhline(THR, ls="--", c="k", lw=.7)
    tag = " (đã sửa lag)" if f in S["lag_correction_applied"] else ""
    a.set_ylim(-0.15, 1.05); a.set_title(f"{f}{tag}  AUC={safe_auc(y, d['braw']):.3f}", fontsize=8, loc="left")
axs[0].legend(fontsize=7, loc="upper right", ncol=3); axs[-1].set_xlabel("giây")
plt.tight_layout(); plt.savefig(OUT / "fig4_timeline.png", dpi=130); plt.close()
print(json.dumps({k: S[k] for k in ["problems", "lag_flag_files", "best_shift_start", "best_shift_center", "auc_pooled_all", "auc_pooled_nodeg",
      "auc_macro", "auc_seen", "auc_unseen", "kappa_boundary", "kappa_interior", "thr_youden_raw_dev", "best_thr_dev", "event_f1",
      "err_types", "pct_true_err", "F1_CI", "DCF_CI", "gap_short_total", "gap_short_filled", "grid_best", "n_dev", "n_test"]},
      ensure_ascii=False, indent=1, default=float))
print(tA.round(4).to_string()); print(tB.round(4).to_string()); print(abl.round(4).to_string()); print(cat_df.round(3).to_string())
