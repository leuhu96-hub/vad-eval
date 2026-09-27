"""Sinh bộ dữ liệu VAD giả định + đầu ra 'model YAMNet giả định'.
Cấu trúc giống dữ liệu thật:
  Model/<Cat>_<i>.txt                        : index,start,end,score  (cửa sổ 0.96s, hop 0.08s)
  Groundtruth/<Cat>/groundtruth/Gt_<cat>_<i>.txt : start<TAB>end<TAB>label  (bin 0.5s)
  Annotator2/Gt_<cat>_<i>.txt               : nhãn của người gán thứ 2 (15 file)
  meta.csv, train_sources.csv               : nguồn âm (để kiểm tra rò rỉ)
Các "lỗi cài sẵn" để bước kiểm tra phát hiện:
  - file suy biến (toàn speech / toàn non-speech)
  - 3 file bị lệch timestamp +0.5s
  - 1 file model thiếu GT, 1 GT thiếu model
  - ~40% file dùng nguồn âm đã có trong train (model 'nhớ' -> điểm đẹp hơn)
  - nhạc có lời (Music) và babble gây false alarm
"""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
DUR, RES, BIN, WIN, HOP = 10.0, 0.01, 0.5, 0.96, 0.08
N_FINE, N_BIN = int(DUR / RES), int(DUR / BIN)
ROOT = Path(".")

CATS = {  # tham số mô phỏng mỗi category
    "Asm":    dict(n=30, snr=(15, 30), bias=0.0, fp_events=0.0, deg_speech=3, deg_non=0),
    "Clean":  dict(n=30, snr=(20, 35), bias=0.0, fp_events=0.0, deg_speech=0, deg_non=0),
    "Noise":  dict(n=30, snr=(-5, 10), bias=0.0, fp_events=0.1, deg_speech=1, deg_non=0),
    "Music":  dict(n=30, snr=(0, 15),  bias=0.0, fp_events=0.6, deg_speech=0, deg_non=5),
    "Babble": dict(n=30, snr=(0, 12),  bias=1.2, fp_events=0.3, deg_speech=0, deg_non=0),
}
SHIFT_BUG = {("Clean", 7), ("Noise", 12), ("Babble", 3)}

def make_fine(kind):
    y = np.zeros(N_FINE, dtype=int)
    if kind == "speech": return y + 1
    if kind == "non": return y
    t, state = 0.0, rng.random() < 0.5
    while t < DUR:
        d = rng.uniform(0.6, 3.5) if state else rng.uniform(0.3, 2.5)
        a, b = int(round(t / RES)), int(round(min(DUR, t + d) / RES))
        y[a:b] = int(state); t += d; state = not state
    return y

def fine_to_bins(yf):
    return (yf.reshape(N_BIN, -1).mean(1) >= 0.5).astype(int)

def jitter_fine(yf, sd=0.12):
    """Người gán 2: dịch mỗi biên một lượng ngẫu nhiên N(0, sd)."""
    ch = np.flatnonzero(np.diff(yf)) + 1
    y2 = yf.copy()
    for c in ch:
        new = int(np.clip(c + round(rng.normal(0, sd) / RES), 1, N_FINE - 1))
        a, b = sorted((c, new)); y2[a:b] = yf[c] if new < c else yf[c - 1]
    return y2

def ar1(n, sd, rho=0.8):
    e = rng.normal(0, sd * np.sqrt(1 - rho**2), n); x = np.zeros(n); x[0] = rng.normal(0, sd)
    for i in range(1, n): x[i] = rho * x[i - 1] + e[i]
    return x

def model_scores(yf, p, snr, seen):
    starts = np.round(np.arange(0, DUR - WIN + 1e-9, HOP), 2)
    fp = np.zeros(N_FINE)                         # sự kiện 'giống giọng' trong vùng non-speech
    t = 0.0
    while t < DUR:
        t += rng.exponential(2.0)
        if rng.random() < p["fp_events"]:
            a = int(t / RES); b = int(min(DUR, t + rng.uniform(0.5, 1.5)) / RES); fp[a:b] = 3.0
    fp *= (yf == 0)
    a_gain = 9.0 if snr > 10 else 6.0 + 0.3 * max(snr, -5) / 1.0 if snr > 0 else 5.0
    sd = 0.8 + 0.12 * max(0.0, 15 - snr)
    if seen: sd *= 0.5; a_gain *= 1.4           # model đã 'nhớ' nguồn âm -> điểm đẹp hơn
    noise = ar1(len(starts), sd)
    sc = []
    for k, s in enumerate(starts):
        a, b = int(round(s / RES)), int(round((s + WIN) / RES))
        w = np.hanning(b - a + 2)[1:-1]; w /= w.sum()   # tâm cửa sổ quan trọng hơn
        f = float((yf[a:b] * w).sum()); ev = float((fp[a:b] * w).sum())
        logit = a_gain * (f - 0.45) + p["bias"] * (1 - f) + ev + noise[k]
        sc.append(1 / (1 + np.exp(-logit)))
    return starts, np.array(sc)

# ---- nguồn âm & train_sources (mô phỏng rò rỉ) ----
speech_pool = [f"utt_{i:03d}" for i in range(80)]
train_speech = set(rng.choice(speech_pool, 30, replace=False))
noise_pool = {c: [f"{c.lower()}_noise_{i:02d}" for i in range(25)] for c in CATS}
train_noise = {n for c in CATS for n in rng.choice(noise_pool[c], 8, replace=False)}
pd.DataFrame({"source": sorted(train_speech | train_noise)}).to_csv(ROOT / "train_sources.csv", index=False)

meta = []
(ROOT / "Model").mkdir(exist_ok=True); (ROOT / "Annotator2").mkdir(exist_ok=True)
ann2_pick = set()
for cat, p in CATS.items():
    gdir = ROOT / "Groundtruth" / cat / "groundtruth"; gdir.mkdir(parents=True, exist_ok=True)
    kinds = ["speech"] * p["deg_speech"] + ["non"] * p["deg_non"]
    kinds += ["mix"] * (p["n"] - len(kinds)); rng.shuffle(kinds)
    ann_idx = set(rng.choice(np.arange(1, p["n"] + 1), 3, replace=False))
    for i in range(1, p["n"] + 1):
        yf = make_fine(kinds[i - 1]); yb = fine_to_bins(yf)
        sp = str(rng.choice(speech_pool)) if kinds[i-1] != "non" else ""
        ns = str(rng.choice(noise_pool[cat]))
        seen = (sp in train_speech) or (ns in train_noise)
        snr = float(np.round(rng.uniform(*p["snr"]), 1))
        starts, sc = model_scores(yf, p, snr, seen)
        shift = 0.5 if (cat, i) in SHIFT_BUG else 0.0
        mname = f"{cat}_{i}.txt"; gname = f"Gt_{cat.lower()}_{i}.txt"
        if not (cat == "Babble" and i == p["n"]):          # lỗi cài sẵn: GT không có model
            with open(ROOT / "Model" / mname, "w") as fh:
                for k, (s, v) in enumerate(zip(starts, sc)):
                    fh.write(f"{k},{s + shift:.2f},{s + shift + WIN:.2f},{v:.4f}\n")
        with open(gdir / gname, "w") as fh:
            for k, l in enumerate(yb):
                fh.write(f"{k*BIN:.1f}\t{(k+1)*BIN:.1f}\t{'speech' if l else 'non-speech'}\n")
        np.save(gdir / f"fine_{cat.lower()}_{i}.npy", yf)   # nhãn mốc 10ms (dùng đo lệch biên)
        if i in ann_idx and kinds[i - 1] == "mix":
            yf2 = jitter_fine(yf); yb2 = fine_to_bins(yf2)
            np.save(ROOT / "Annotator2" / f"fine_{cat.lower()}_{i}.npy", yf2)
            with open(ROOT / "Annotator2" / gname, "w") as fh:
                for k, l in enumerate(yb2):
                    fh.write(f"{k*BIN:.1f}\t{(k+1)*BIN:.1f}\t{'speech' if l else 'non-speech'}\n")
        meta.append(dict(file=mname, category=cat, idx=i, speech_src=sp, noise_src=ns, snr_db=snr,
                         seen_in_train=seen, injected_shift_s=shift, kind=kinds[i - 1]))
# lỗi cài sẵn: model thừa không có GT
starts, sc = model_scores(make_fine("mix"), CATS["Music"], 5.0, False)
with open(ROOT / "Model" / "Music_31.txt", "w") as fh:
    for k, (s, v) in enumerate(zip(starts, sc)): fh.write(f"{k},{s:.2f},{s+WIN:.2f},{v:.4f}\n")
pd.DataFrame(meta).to_csv(ROOT / "meta.csv", index=False)
print("Đã sinh", len(meta), "clip GT;", len(list((ROOT/'Model').glob('*.txt'))), "file model")
