"""Đánh giá output model VAD trên một bộ audio có nhãn, theo quy trình của evaluate.py.

Dùng cho dữ liệu thật hoặc bộ dữ liệu mới bất kỳ có cấu trúc (tên file bất kỳ):
    <root>/Model/[<quality>/]<tên>.txt     output model, mỗi cửa sổ 0.96 s một dòng: 'idx, start, end, score'
                                           hoặc 'start end score' (start = đầu cửa sổ)
    <root>/.../audio/[<category>/]<tên>.wav                          audio (wav/flac/mp3...; tuỳ chọn)
    <root>/.../Ground_truth|GT/[<category>/][ground_truth/][Gt_]<tên>.txt     nhãn
                                           mỗi dòng 'start end speech|non-speech', theo bin 0.5 s hoặc theo thời điểm
Ghép theo <tên>: không phân biệt hoa thường, bỏ tiền tố / hậu tố nhãn (Gt_, label_, _gt...), vd Gt_asm_x <-> Asm_x.
Nhóm (cột category: chia dev/test, báo cáo) theo --group; auto = thư mục <quality> của model > thư mục <category>
của audio / nhãn > phần trước dấu '_' đầu tiên > một nhóm. Nhóm < 2 file gộp vào "Khác".

Tuỳ chọn (có thì dùng, không có thì bỏ qua):
  --label2        nhãn của người gán thứ 2 (cùng định dạng) -> đồng thuận nhãn, bin "nhãn chắc"
  --silero-cache  cache của auto_label.py -> Silero làm người gán thứ 2 khi không có --label2
  --gt-alt        nhãn theo quy tắc khác (vd hát = non-speech) -> bảng tầng A phụ; người gán 2 so với nhãn này
  syn_val_* trong thư mục output model + --synth-gt -> chọn ngưỡng trên val tổng hợp, mốc căn chỉnh nhãn chính xác
Khác evaluate.py (dữ liệu giả định 10 s): file dài khác nhau; không có meta.csv nên bỏ bước rò rỉ / SNR;
lag từng file chỉ báo cáo, không tự dịch; dev/test chia theo file, phân tầng theo category.

    python evaluate_real.py --root "C:/Users/leuhu/OneDrive/Máy tính/test" --out results_test
    python evaluate_real.py --root <root> --out results_thu --overlap mean --smooth 1 --pad-pre 0.2 --merge-gap 1.0 \
        --drop 0.1 --criterion F1-max          # tinh chỉnh hậu xử lý / chọn ngưỡng; xem --help
    python evaluate_real.py --root <root> --out results_dexuat --preset de-xuat   # chạy cả báo cáo với cấu hình đề xuất
    python evaluate_real.py --root <root> --out results_test --tune 300          # + tìm cấu hình tốt nhất trên dev
    python evaluate_real.py --raw C:/vad_work/run1/raw_real --gt C:/vad_work/real_test/gt_policy --audio <audio> \
        --gt-alt <auto_labels/Groundtruth> --silero-cache <auto_labels/cache> --out results_real
"""
import argparse, json, re, wave, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             det_curve, brier_score_loss, cohen_kappa_score)
from scipy.stats import norm
from vadlib import *
try:
    import soundfile as sf
except ImportError:
    sf = None
warnings.filterwarnings("ignore")

# cấu hình đề xuất (vad_post_processing.md, báo cáo tối ưu hậu xử lý): ngưỡng kép, merge -> drop -> pad, pad 0, median 3 hop
PRESETS = {"hien-tai": {}, "de-xuat": dict(overlap="tri", smooth=3, smooth_kind="median", domain="prob", hyst=0.15, order="mdp",
                                           pad_pre=0.0, pad_post=0.0, merge_gap=0.5, drop=0.32)}
_pp = argparse.ArgumentParser(add_help=False); _pp.add_argument("--preset", choices=list(PRESETS), default="hien-tai")
_preset = _pp.parse_known_args()[0].preset
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", type=Path, help="thư mục bộ dữ liệu: tự tìm Model/, audio/, Ground_truth/")
ap.add_argument("--raw", type=Path, help="output model (mặc định thư mục Model trong --root)")
ap.add_argument("--gt", type=Path, help="nhãn chính (mặc định thư mục Ground_truth trong --root)")
ap.add_argument("--audio", type=Path, help="audio (mặc định thư mục audio trong --root); không có thì bỏ kiểm tra thời lượng")
ap.add_argument("--gt-alt", type=Path, help="nhãn theo quy tắc khác (tuỳ chọn)")
ap.add_argument("--label2", type=Path, help="nhãn người gán thứ 2 (tuỳ chọn)")
ap.add_argument("--silero-cache", "--cache", dest="silero_cache", type=Path, help="cache auto_label.py (tuỳ chọn)")
ap.add_argument("--synth-gt", type=Path, default=Path("C:/vad_work/synth/clips/gt/val"), help="nhãn của syn_val_* (tuỳ chọn)")
ap.add_argument("--group", choices=["auto", "quality", "category", "prefix", "none"], default="auto",
                help="nhóm chính: quality = thư mục con của model, category = thư mục con của audio / nhãn, "
                     "prefix = phần trước '_', none = một nhóm; auto = cái đầu tiên mọi file đều có (mặc định %(default)s)")
ap.add_argument("--mo-ta-nhan", help="mô tả nhãn chính cho báo cáo")
ap.add_argument("--mo-ta-model", help="mô tả model cho báo cáo")
ap.add_argument("--dev-frac", type=float, default=0.3, help="tỉ lệ file vào dev (chọn ngưỡng), còn lại là test")
ap.add_argument("--out", type=Path, default=Path("results_real"))
g = ap.add_argument_group("tinh chỉnh hậu xử lý (A1–A4)")
g.add_argument("--preset", choices=list(PRESETS), default="hien-tai",
               help="bộ tham số hậu xử lý: hien-tai = cấu hình hiện tại, de-xuat = cấu hình đề xuất; tham số ghi rõ vẫn ghi đè (mặc định %(default)s)")
g.add_argument("--overlap", choices=list(OVERLAP_DESC), default=RESCORE_WEIGHT,
               help="gộp điểm các cửa sổ chồng nhau thành điểm mỗi hop 0.08 s (mặc định %(default)s)")
g.add_argument("--smooth", type=int, default=RESCORE_SMOOTH, help="làm mượt N hop; 1 = không (mặc định %(default)s)")
g.add_argument("--smooth-kind", choices=list(SMOOTH_DESC), default="mean", help="kiểu làm mượt: mean / median / gauss (mặc định %(default)s)")
g.add_argument("--domain", choices=["prob", "logit"], default="prob", help="gộp tri/hann/mean trong miền xác suất hay log-odds (mặc định %(default)s)")
g.add_argument("--hyst", type=float, default=0.0,
               help="ngưỡng kép: bắt đầu speech khi điểm >= ngưỡng, kết thúc khi < ngưỡng − N; 0 = ngưỡng đơn (mặc định %(default)s)")
g.add_argument("--order", choices=list(ORDER_DESC), default="pdm",
               help="thứ tự ở A4: pdm = pad → drop → merge (hiện tại), mdp = merge → drop → pad (mặc định %(default)s)")
g.add_argument("--pad-pre", type=float, default=0.10, help="nới đầu đoạn speech, giây; âm = co (mặc định %(default)s)")
g.add_argument("--pad-post", type=float, default=0.12, help="nới cuối đoạn speech, giây; âm = co (mặc định %(default)s)")
g.add_argument("--merge-gap", type=float, default=0.50, help="gộp hai đoạn cách nhau < N giây (mặc định %(default)s)")
g.add_argument("--drop", type=float, default=0.32,
               help="bỏ đoạn ngắn hơn N giây; pdm: đo sau pad, mdp: đo trước pad; chỉ ở A4 (mặc định %(default)s)")
g = ap.add_argument_group("so sánh phương án hậu xử lý (mục 6)")
g.add_argument("--no-compare", action="store_true", help="bỏ mục so sánh các phương án hậu xử lý")
g.add_argument("--tune", type=int, default=0, help="số lần thử khi tìm cấu hình hậu xử lý tốt nhất trên dev; 0 = không tìm (mặc định %(default)s)")
g.add_argument("--tune-seed", type=int, default=0, help="seed tìm cấu hình (mặc định %(default)s)")
g = ap.add_argument_group("tinh chỉnh chọn ngưỡng và thống kê")
g.add_argument("--criterion", choices=["DCF-min", "F1-max"], default="DCF-min", help="tiêu chí chọn ngưỡng trên dev (mặc định %(default)s)")
g.add_argument("--dcf-miss", type=float, default=0.75, help="trọng số miss trong DCF; FA = 1 − giá trị này (mặc định %(default)s)")
g.add_argument("--thr-min", type=float, default=0.01, help="ngưỡng nhỏ nhất khi quét (mặc định %(default)s)")
g.add_argument("--thr-max", type=float, default=0.95, help="ngưỡng lớn nhất khi quét (mặc định %(default)s)")
g.add_argument("--thr-step", type=float, default=0.01, help="bước quét ngưỡng (mặc định %(default)s)")
g.add_argument("--grid-pre", default="0,50,100,200", help="lưới pad trước khi quét trên dev, ms (mặc định %(default)s)")
g.add_argument("--grid-post", default="0,60,120,240", help="lưới pad sau, ms (mặc định %(default)s)")
g.add_argument("--grid-gap", default="0,250,500,750,1000,1500", help="lưới merge gap, ms (mặc định %(default)s)")
g.add_argument("--seed", type=int, default=3, help="seed chia dev/test (mặc định %(default)s)")
g.add_argument("--n-boot", type=int, default=1000, help="số lần bootstrap cho CI (mặc định %(default)s)")
ap.set_defaults(**PRESETS[_preset])
A = ap.parse_args()
def ms_list(s): return [float(x) / 1000 for x in s.split(",") if x.strip()]
GRID = dict(pre=ms_list(A.grid_pre), post=ms_list(A.grid_post), gap=ms_list(A.grid_gap))
for cond, msg in [(A.smooth >= 1, "--smooth phải >= 1"), (0 < A.dev_frac < 1, "--dev-frac phải trong (0, 1)"),
                  (0 <= A.dcf_miss <= 1, "--dcf-miss phải trong [0, 1]"), (A.thr_step > 0 and A.thr_min < A.thr_max, "dải ngưỡng sai"),
                  (min(A.merge_gap, A.drop, A.hyst) >= 0, "merge / drop / hyst phải >= 0"),
                  (min(A.pad_pre, A.pad_post) > -WIN / 2, "pad âm quá lớn"), (A.tune >= 0, "--tune phải >= 0"),
                  (all(GRID.values()), "--grid-* không được rỗng"), (A.n_boot >= 100, "--n-boot phải >= 100")]:
    if not cond: raise SystemExit(msg)
ms = lambda x: f"{x * 1000:g}"
RS = dict(weight=A.overlap, smooth=A.smooth, smooth_kind=A.smooth_kind, domain=A.domain)   # cách rescore dùng ở A1–A4
HYST_DESC = f", ngưỡng kép Δ {A.hyst:g}" if A.hyst > 0 else ""
CONFIG_DESC = ["Điểm bin thô, không hậu xử lý", f"Rescore ({rescore_desc(**RS)}) + rebin{HYST_DESC}",
               f"+ pad {ms(A.pad_pre)}/{ms(A.pad_post)} ms", f"+ merge < {ms(A.merge_gap)} ms",
               (f"+ merge < {ms(A.merge_gap)} ms" + (f" + drop {ms(A.drop)} ms" if A.drop else "") + f" + pad {ms(A.pad_pre)}/{ms(A.pad_post)} ms"
                if A.order == "mdp" else "+ pad" + (f" + drop {ms(A.drop)} ms" if A.drop else "") + " + merge")
               + (" (hiện tại)" if A.preset == "hien-tai" else f" (preset {A.preset})")]
PARAMS = dict(preset=A.preset, overlap=A.overlap, smooth=A.smooth, smooth_kind=A.smooth_kind, domain=A.domain, hyst=A.hyst, order=A.order,
              pad_pre=A.pad_pre, pad_post=A.pad_post, merge_gap=A.merge_gap, drop=A.drop, tune=A.tune, tune_seed=A.tune_seed,
              criterion=A.criterion, dcf_miss=A.dcf_miss, thr_min=A.thr_min, thr_max=A.thr_max, thr_step=A.thr_step,
              grid_pre=A.grid_pre, grid_post=A.grid_post, grid_gap=A.grid_gap, seed=A.seed, n_boot=A.n_boot, dev_frac=A.dev_frac)

def norm_name(s): return re.sub(r"[_\-\s]", "", s.lower())
def find_dir(root, names):
    """Thư mục nông nhất dưới root có tên thuộc names (không phân biệt hoa thường, bỏ '_', '-', dấu cách)."""
    hits = [p for p in [root, *root.rglob("*")] if p.is_dir() and norm_name(p.name) in names]
    return min(hits, key=lambda p: len(p.parts)) if hits else None
if A.root:
    A.raw = A.raw or find_dir(A.root, {"model", "models", "raw"})
    A.gt = A.gt or find_dir(A.root, {"groundtruth", "gt", "labels", "label"})
    A.audio = A.audio or find_dir(A.root, {"audio", "audios", "wav"})
for k in ("raw", "gt"):
    if getattr(A, k) is None or not getattr(A, k).exists():
        raise SystemExit(f"Không tìm thấy thư mục --{k} ({getattr(A, k)}). Chỉ rõ bằng --{k} hoặc --root.")

OUT = A.out; OUT.mkdir(exist_ok=True); XL = OUT / "excel"; XL.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True})
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # thứ tự cố định
C1, C2, C3 = PALETTE[:3]
MODES = ["A0", "A1", "A2", "A3", "A4"]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".aac", ".wma", ".opus"}
rng = np.random.default_rng(0)
S = {}

# ---- bản vector hóa của vadlib (cùng kết quả): file dài tới hàng trăm bin, quét ngưỡng bằng vòng lặp quá chậm ----
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
LABEL_DIRS = {"groundtruth", "gt", "labels", "label"}
LABEL_TAG = re.compile(r"^(gt|ground[_\-\s]?truth|labels?)[_\-\s]+|[_\-\s]+(gt|ground[_\-\s]?truth|labels?)$", re.I)
def key_of(stem):
    """Khoá ghép, như nhau cho model / audio / nhãn: bỏ tiền tố / hậu tố nhãn, không phân biệt hoa thường (Gt_asm_x = Asm_x)."""
    return (LABEL_TAG.sub("", stem) or stem).casefold()

def index_files(root, exts):
    """key_of(tên) -> file, ở mọi thư mục con. Khoá trùng (>= 2 file) không biết ghép file nào -> bỏ, trả riêng trong dup."""
    out, dup = {}, {}
    if root is None or not Path(root).exists(): return out, dup
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in exts or not p.is_file(): continue
        k = key_of(p.stem)
        if k in out or k in dup: dup.setdefault(k, [out.pop(k)] if k in out else []).append(p)
        else: out[k] = p
    return out, dup

def sub_dir(p, root):
    """Thư mục con đầu tiên chứa p dưới root, bỏ qua thư mục kiểu groundtruth; None nếu p nằm thẳng trong root."""
    parts = [q for q in Path(p).relative_to(root).parts[:-1] if norm_name(q) not in LABEL_DIRS]
    return parts[0] if parts else None

LAB = {"speech": 1, "sp": 1, "1": 1, "non-speech": 0, "nonspeech": 0, "non_speech": 0, "non speech": 0, "ns": 0, "0": 0,
       "silence": 0, "noise": 0}
def load_labels(path):
    """Nhãn 'start end nhãn' (tab / dấu cách / phẩy) theo bin 0.5 s hoặc theo thời điểm -> bin 0.5 s;
    bin là speech khi >= 50% thời lượng là speech (nhãn theo bin 0.5 s cho kết quả y hệt)."""
    iv = []
    for i, ln in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        t = [x for x in re.split(r"[\s,;]+", ln.strip()) if x]
        j = len(t)
        while j and not re.fullmatch(r"-?\d+(\.\d*)?", t[j - 1]): j -= 1   # nhãn chữ ở cuối, có thể 2 từ ("non speech")
        lab, nums = " ".join(t[j:]).lower(), t[:j]
        if not lab and len(nums) >= 3: lab, nums = nums[-1], nums[:-1]     # nhãn dạng số 0/1
        if len(nums) < 2 or not lab: continue                              # dòng tiêu đề / rỗng
        if lab not in LAB: raise ValueError(f"{path}:{i}: nhãn không đọc được: {ln!r}")
        iv.append((float(nums[-2]), float(nums[-1]), LAB[lab]))
    if not iv: raise ValueError(f"{path}: không có dòng nhãn nào")
    iv = np.array(iv); end = iv[:, 1].max(); n = int(np.ceil(end / BIN - 1e-6)); lo = np.arange(n) * BIN
    sp = iv[iv[:, 2] == 1]
    cov = np.clip(np.minimum(sp[:, 1:2], lo + BIN) - np.maximum(sp[:, 0:1], lo), 0, None).sum(0)
    return (cov / (np.minimum(lo + BIN, end) - lo) >= 0.5 - 1e-9).astype(int)

def audio_info(p):
    """(thời lượng s, sample rate, số kênh) – NaN/None nếu không đọc được."""
    try:
        if sf is not None:
            i = sf.info(str(p)); return i.duration, i.samplerate, i.channels
        with wave.open(str(p)) as w: return w.getnframes() / w.getframerate(), w.getframerate(), w.getnchannels()
    except Exception:
        return np.nan, None, None

def silero_bins(key, n):
    """Như auto_label.py: bin là speech khi >= 50% chunk 32 ms có p_silero >= 0.5."""
    p = A.silero_cache / f"{key}.npz"
    if not p.exists(): return None
    sp = np.load(p)["silero"]
    if not len(sp): return None
    k = np.floor((np.arange(len(sp)) + 0.5) * 512 / 16000 / BIN).astype(int); ok = k < n
    frac = np.bincount(k[ok], (sp[ok] >= 0.5).astype(float), n) / np.maximum(np.bincount(k[ok], minlength=n), 1)
    return (frac >= 0.5 - 1e-9).astype(int)

def fit(v, n): return v[:n] if len(v) >= n else np.r_[v, np.zeros(n - len(v), int)]
def auc(y, s): return roc_auc_score(y, s) if len(y) and 0 < np.mean(y) < 1 else np.nan
def cat_(files, key, src=None): return np.concatenate([(src or recs)[f][key] for f in files])

# =============== 5a. Ghép file audio <-> output model <-> nhãn ===============
# khoá ghép = key_of(tên file); tên hiển thị (cột file) = tên file output model
SOURCES = dict(raw=(A.raw, {".txt"}), audio=(A.audio, AUDIO_EXTS), gt=(A.gt, {".txt"}),
               gt_alt=(A.gt_alt, {".txt"}), label2=(A.label2, {".txt"}))
idx = {n: index_files(r, ex) for n, (r, ex) in SOURCES.items()}
(raw, _), (audio, _), (gt, _), (gt_alt, _), (lab2, _) = idx.values()
real_raw = {k for k in raw if not k.startswith("syn_")}
checks = [("raw", real_raw), ("gt", gt)] + ([("audio", audio)] if audio else []) + \
         ([("gt_alt", gt_alt)] if A.gt_alt else []) + ([("label2", lab2)] if A.label2 else [])
problems = [(p.relative_to(SOURCES[n][0]).as_posix(), f"tên trùng trong --{n}, không biết ghép file nào")
            for n, (_, dup) in idx.items() for ps in dup.values() for p in ps]
for k in sorted(set(real_raw) | set(gt) | set(audio)):
    miss = [n for n, d in checks if k not in d]
    if miss: problems.append(((raw.get(k) or audio.get(k) or gt[k]).stem, "thiếu " + ", ".join(miss)))
keys = sorted(set(real_raw) & set(gt) & (set(audio) if audio else set(gt)), key=lambda k: raw[k].stem)
if not keys:
    ex = lambda d, ks: [d[k].stem for k in sorted(ks)[:3]]
    raise SystemExit("Không ghép được file nào giữa output model, nhãn" + (" và audio" if audio else "") +
                     f". Ví dụ tên: model {ex(raw, real_raw)}, nhãn {ex(gt, gt)}, audio {ex(audio, audio)}")

# nhóm chính (cột category): thư mục quality của model / thư mục category của audio (không có thì của nhãn) / tiền tố tên
qual = {k: sub_dir(raw[k], A.raw) for k in keys}
fcat = {k: (sub_dir(audio[k], A.audio) if k in audio else None) or sub_dir(gt[k], A.gt) for k in keys}
GROUPS = dict(quality=qual, category=fcat, none=dict.fromkeys(keys, "Tất cả"),
              prefix={k: raw[k].stem.split("_")[0] if "_" in raw[k].stem else None for k in keys})
GROUP_BY = A.group if A.group != "auto" else next(h for h in ("quality", "category", "prefix", "none") if all(GROUPS[h].values()))
grp = pd.Series({k: GROUPS[GROUP_BY][k] or "Khác" for k in keys})
small = sorted(set(grp) - set(grp.value_counts()[lambda c: c >= 2].index))   # nhóm 1 file: không chia dev/test được
grp[grp.isin(small)] = "Khác"
S.update(n_audio=len(audio), n_raw_real=len(real_raw), n_gt=len(gt), n_pairs=len(keys), problems=problems,
         group_by=GROUP_BY, groups_merged=small)

recs = {}
for k in keys:
    f = raw[k].stem
    y = load_labels(gt[k]); n = len(y); m = load_model(raw[k], center=True)
    ya = load_labels(gt_alt[k]) if k in gt_alt else None
    y2 = load_labels(lab2[k]) if k in lab2 else \
        (silero_bins(audio.get(k, raw[k]).stem, n) if A.silero_cache and not A.label2 else None)   # cache đặt theo tên audio
    dur, sr_, ch_ = audio_info(audio[k]) if k in audio else (np.nan, None, None)
    recs[f] = dict(file=f, category=grp[k], quality_dir=qual[k], category_dir=fcat[k],
                   y=y, n=n, m=m, m0=load_model(raw[k], center=False), dur=dur, sr=sr_, ch=ch_,
                   has_alt=ya is not None, alt_len_ok=ya is None or len(ya) == n, y_alt=fit(ya, n) if ya is not None else y.copy(),
                   has_l2=y2 is not None, l2_len_ok=y2 is None or len(y2) == n, y2=fit(y2, n) if y2 is not None else None)
HAS_ALT = any(d["has_alt"] for d in recs.values()); HAS_L2 = any(d["has_l2"] for d in recs.values())
CATS = sorted({d["category"] for d in recs.values()})
COLS = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(CATS)}
L2_NAME = f"Annotator 2 ({A.label2.name})" if A.label2 else ("Silero (cache auto_label)" if A.silero_cache else None)
for d in recs.values():   # người gán 2 so với nhãn cùng quy tắc: --gt-alt nếu có, không thì nhãn chính
    d["y_ref"] = d["y_alt"] if HAS_ALT else d["y"]

# tập val tổng hợp (syn_val_* trong output model): nhãn thời điểm chính xác -> mốc căn chỉnh + ngưỡng không cần nhãn thật
syn = {}
if A.synth_gt and A.synth_gt.exists():
    for k in sorted(r for r in raw if r.startswith("syn_val_")):
        p = A.synth_gt / f"{raw[k].stem}.txt"
        if not p.exists() or raw[k].stat().st_size == 0: continue
        y = load_labels(p); m = load_model(raw[k]); n = len(y)
        if n < 2 or len(m) == 0: continue
        syn[raw[k].stem] = dict(y=y, n=n, m=m, braw=bin_scores(m, n), fr=rescore_hop(m, dur=n * BIN, **RS))
S["n_synth_val"] = len(syn)

# =============== 5b/5c/5d. Cấu trúc, định dạng audio ===============
struct = []
for f, d in recs.items():
    m = d["m"]; dd = np.diff(m.start); known = not np.isnan(d["dur"])
    n_exp = int((round(d["dur"] * 16000) - 15600) // 1280) + 1 if known else np.nan   # cửa sổ YAMNet: 15600 mẫu, hop 1280
    struct.append(dict(file=f, dur_audio=d["dur"], sample_rate=d["sr"], channels=d["ch"], n_bins=d["n"], n_win=len(m),
                       n_win_expected=n_exp, first_start=m.start.min(), last_end=m.end.max(),
                       hop_ok=bool(np.allclose(dd, HOP, atol=1e-3)), win_ok=bool(np.allclose(m.end - m.start, WIN, atol=1e-3)),
                       score_ok=bool(m.score.between(0, 1).all()), alt_len_ok=d["alt_len_ok"], l2_len_ok=d["l2_len_ok"],
                       bins_ok=(not known) or d["n"] == int(d["dur"] / BIN + 1e-9),
                       win_count_ok=(not known) or abs(len(m) - n_exp) <= 1, end_ok=(not known) or m.end.max() <= d["dur"] + 0.01))
struct = pd.DataFrame(struct)
struct["flag"] = ~(struct.hop_ok & struct.win_ok & struct.score_ok & struct.bins_ok & struct.win_count_ok & struct.end_ok
                   & struct.alt_len_ok & struct.l2_len_ok) | (struct.first_start.abs() > 0.01)
S.update(n_struct_flag=int(struct.flag.sum()), struct_flag_files=struct.file[struct.flag].tolist(),
         n_hop_bad=int((~struct.hop_ok).sum()), n_win_bad=int((~struct.win_ok).sum()), n_score_bad=int((~struct.score_ok).sum()),
         n_audio_checked=int(struct.dur_audio.notna().sum()),
         n_audio_16k_mono=int(((struct.sample_rate == 16000) & (struct.channels == 1)).sum()),
         windows_drop_tail=bool(struct.dur_audio.notna().any() and struct.win_count_ok.all()))

# =============== 5e. Căn chỉnh: quét shift + lag từng file ===============
def pooled_f1_shift(key, shift, thr=0.5):
    Y, P = [], []
    for d in recs.values():
        s = bin_scores(d[key], d["n"], shift=shift); Y.append(d["y"][1:-1]); P.append((s >= thr)[1:-1])
    return rates(np.concatenate(Y), np.concatenate(P).astype(int))["F1"]
def pooled_auc_shift(src, yk, shift):
    """AUC không phụ thuộc ngưỡng: so độ dịch trên nhãn chính, người gán 2 và nhãn chính xác của tập tổng hợp."""
    ds = [d for d in src.values() if d.get(yk) is not None]
    if not ds: return np.nan
    return auc(np.concatenate([d[yk][1:-1] for d in ds]), np.concatenate([bin_scores(d["m"], d["n"], shift=shift)[1:-1] for d in ds]))
shifts = np.round(np.arange(-12, 13) * 0.08, 2)
sweep = pd.DataFrame(dict(shift_s=shifts, F1_start=[pooled_f1_shift("m0", s) for s in shifts],
                          F1_center=[pooled_f1_shift("m", s) for s in shifts],
                          AUC_gt=[pooled_auc_shift(recs, "y", s) for s in shifts],
                          AUC_l2=[pooled_auc_shift(recs, "y2", s) if HAS_L2 else np.nan for s in shifts],
                          AUC_synth=[pooled_auc_shift(syn, "y", s) if syn else np.nan for s in shifts]))
S["best_shift_start"] = float(sweep.shift_s[sweep.F1_start.idxmax()]); S["best_shift_center"] = float(sweep.shift_s[sweep.F1_center.idxmax()])
S["best_shift_auc"] = {c: float(sweep.shift_s[sweep[c].idxmax()]) for c in ("AUC_gt", "AUC_l2", "AUC_synth") if sweep[c].notna().any()}
S["auc_gain_best_shift_real"] = float(sweep.AUC_gt.max() - sweep.AUC_gt[sweep.shift_s == 0].iloc[0])

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
lagdf = pd.DataFrame(lagrows, columns=["file", "category", "lag_s", "peak_corr"])
# dữ liệu thật: lag lớn thường do nhãn/điểm nhiễu (corr thấp) chứ không phải lỗi xuất file -> chỉ báo cáo
lagdf["flag"] = np.where(lagdf.lag_s.abs() >= 0.4, "Lệch ~1 bin", np.where(lagdf.lag_s.abs() > 0.2, "Cần xem", ""))
S.update(lag_median=float(lagdf.lag_s.median()) if len(lagdf) else None, lag_flag_files=lagdf.file[lagdf.flag == "Lệch ~1 bin"].tolist(),
         lag_flag_corr=lagdf.peak_corr[lagdf.flag == "Lệch ~1 bin"].round(3).tolist(),
         n_lag_review=int((lagdf.flag == "Cần xem").sum()))

# cache điểm
for d in recs.values():
    d["braw"] = bin_scores(d["m"], d["n"]); d["fr"] = rescore_hop(d["m"], dur=d["n"] * BIN, **RS); d["bmask"] = boundary_mask(d["y"])
    d["sure"] = (d["y"] == d["y_alt"]) & (d["y_ref"] == d["y2"]) if d["has_l2"] else None   # các nhãn cùng đồng ý

def pred_cached(d, thr, mode="A4", pre=None, post=None, gap=None):
    """A0: điểm bin thô | A1: rescore + rebin | A2: + pad | A3: + merge | A4: + pad + drop + merge theo --order
    | khác: A4 với pre/post/gap cho trước. A1–A4 dùng ngưỡng kép khi --hyst > 0."""
    if mode == "A0": return (d["braw"] >= thr).astype(int)
    pre, post, gap = (A.pad_pre if pre is None else pre), (A.pad_post if post is None else post), (A.merge_gap if gap is None else gap)
    p, q, g, dr = {"A1": (0, 0, 0, 0), "A2": (pre, post, 0, 0), "A3": (0, 0, gap, 0)}.get(mode, (pre, post, gap, A.drop))
    od = A.order if mode not in ("A1", "A2", "A3") else "pdm"
    fr = hysteresis(d["fr"], thr, thr - A.hyst if A.hyst > 0 else None)
    return segs_to_bins(postprocess_segs(frames_to_segs(fr, HOP), p, q, g, dr, dur=d["n"] * BIN, order=od), d["n"])

# =============== 1. Phân bố ===============
rows = []
for f, d in recs.items():
    y = d["y"]; segs = bins_to_segs(y)
    rows.append(dict(file=f, category=d["category"], quality_dir=d["quality_dir"], category_dir=d["category_dir"],
                     dur_s=d["n"] * BIN, speech_ratio=y.mean(),
                     speech_ratio_alt=d["y_alt"].mean() if d["has_alt"] else np.nan,
                     bins_khac_alt=int((y != d["y_alt"]).sum()) if d["has_alt"] else np.nan,
                     l2_ratio=d["y2"].mean() if d["has_l2"] else np.nan,
                     agree_l2=(d["y_ref"] == d["y2"]).mean() if d["has_l2"] else np.nan,
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
         auc_worst=float(pf.auc_file.min()), auc_worst_file=pf.file[pf.auc_file.idxmin()] if pf.auc_file.notna().any() else None,
         n_files_auc_lt_09=int((pf.auc_file < 0.9).sum()), n_files_auc_lt_07=int((pf.auc_file < 0.7).sum()))

# =============== 2. Chia dev/test theo file, phân tầng theo category ===============
split, rs = {}, np.random.default_rng(A.seed)
for c in CATS:
    fs = sorted(pf.file[pf.category == c]); rs.shuffle(fs); nd = int(round(A.dev_frac * len(fs)))
    for i, f in enumerate(fs): split[f] = "dev" if i < nd else "test"
pf["split"] = pf.file.map(split)
dev = sorted(pf.file[pf.split == "dev"]); test = sorted(pf.file[pf.split == "test"])
if not dev or not test:
    raise SystemExit(f"Chia dev/test ra {len(dev)} / {len(test)} file ({len(pf)} file, --dev-frac {A.dev_frac}): "
                     "cần ít nhất 1 file mỗi tập. Thêm file hoặc đổi --dev-frac.")
CATS_T = [c for c in CATS if any(recs[f]["category"] == c for f in test)]          # category có file test
S.update(n_dev=len(dev), n_test=len(test), pct_speech_dev=float(cat_(dev, "y").mean()), pct_speech_test=float(cat_(test, "y").mean()))

# =============== 2b. Đặc trưng tập theo độ phân giải model (chỉ từ nhãn + lưới cửa sổ, không dùng điểm model) ===============
# thang thời gian: nhãn bin 0.5 s, cửa sổ 0.96 s, bin + cửa sổ 1.46 s; lưới chung 0.02 s (vadlib.FINE)
# khoảng lặng ngắn hơn -> pad + merge lấp kể cả khi model đúng (mdp: merge trên đoạn chưa pad, rồi pad gộp chồng lấn)
FILL_GAP = A.merge_gap + max(A.pad_pre, 0) + max(A.pad_post, 0) if A.order == "pdm" else max(A.merge_gap, A.pad_pre + A.pad_post)
DROP_ADD = A.pad_pre + A.pad_post if A.order == "pdm" else 0.0   # drop đo sau pad (pdm) hay trước pad (mdp)
LEN_EDGES, LEN_CLS = [WIN, WIN + BIN, 2.0], ["< 0.96 s", "0.96–1.46 s", "1.46–2 s", "≥ 2 s"]
LEN_KEY = ["lt_win", "win_146", "146_2", "ge_2"]
DIST_EDGES, DIST_CLS = [HALF_WIN, WIN, WIN + HALF_WIN], ["≤ 0.48 s", "0.48–0.96 s", "0.96–1.44 s", "> 1.44 s"]
DIST_KEY = ["le_048", "048_096", "096_144", "gt_144"]
FRAC_CLS = ["0", "(0, .25]", "(.25, .5)", "[.5, .75)", "[.75, 1)", "1"]
HIST_LEN = np.arange(1, 11) * BIN                           # cột bảng phân bố độ dài: 0.5 … 4.5 s, cột cuối = ≥ 5 s
MIN_COV = int(BIN / HOP + 1e-9)                             # bin đủ cửa sổ: >= 6 tâm cửa sổ
def len_cls(L): return np.searchsorted(LEN_EDGES, np.asarray(L, float) + 1e-9, side="right")
def frac_cls(f): return np.select([f <= 0, f <= .25, f < .5, f < .75, f < 1], [0, 1, 2, 3, 4], 5)
def counts4(y, p): return [int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum()),
                           int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())]
prof_rows, runs, fracs = [], {}, {}
for f, d in recs.items():
    y, n, m = d["y"], d["n"], d["m"]
    sp, gp = (np.array(v, float) for v in run_lengths(y)); runs[f] = (sp, gp)
    wf = window_frac(m, y); wf = wf[~np.isnan(wf)]; fracs[f] = wf
    om = oracle_model(m, y)
    orc = dict(n=n, braw=bin_scores(om, n), fr=rescore_hop(om, dur=n * BIN, **RS))
    r = dict(file=f, category=d["category"], split=split[f], n_bin=n, dur_s=n * BIN, speech_bins=int(y.sum()), speech_s=y.sum() * BIN,
             n_trans=int((y[1:] != y[:-1]).sum()), n_seg=len(sp), n_gap=len(gp), n_win=len(wf),
             gap_filled=int((gp < FILL_GAP - 1e-9).sum()), seg_dropped=int((sp + DROP_ADD < A.drop - 1e-9).sum()),
             win_ns=int((wf == 0).sum()), win_mixed=int(((wf > 0) & (wf < 1)).sum()), win_sp=int((wf == 1).sum()),
             low_cov=int((centers_per_bin(m, n) < MIN_COV).sum()))
    for i, k in enumerate(LEN_KEY):
        r[f"seg_{k}"] = int((len_cls(sp) == i).sum()); r[f"gap_{k}"] = int((len_cls(gp) == i).sum())
    r["seg_s_lt_win"] = float(sp[len_cls(sp) == 0].sum())
    dc = np.searchsorted(DIST_EDGES, dist_to_transition(y) - 1e-9)
    for i, k in enumerate(DIST_KEY): r[f"bin_{k}"] = int((dc == i).sum())
    for mode in ("A0", "A4"):
        r.update(zip([f"o{mode}_{x}" for x in ("TP", "FN", "FP", "TN")], counts4(y, pred_cached(orc, 0.5, mode))))
    prof_rows.append(r)
prof = pd.DataFrame(prof_rows)
assert (prof.win_ns + prof.win_mixed + prof.win_sp == prof.n_win).all() and np.allclose(prof.speech_s, [runs[f][0].sum() for f in prof.file])

def prof_summary(g, cat, tap):
    """Đặc trưng gộp (cộng số đếm, không trung bình theo file) của nhóm file g."""
    s = g.drop(columns=["file", "category", "split"]).sum()
    def pc(a, b): return s[a] / s[b] if s[b] else np.nan
    sp = np.concatenate([runs[f][0] for f in g.file]); gp = np.concatenate([runs[f][1] for f in g.file])
    r = dict(Category=cat, split=tap, n_file=len(g), minutes=s.dur_s / 60, n_bin=int(s.n_bin), n_win=int(s.n_win),
             n_win_indep=s.dur_s / WIN, pct_speech=pc("speech_bins", "n_bin"), trans_per_min=s.n_trans / (s.dur_s / 60),
             n_seg=int(s.n_seg), **{f"seg_{k}": pc(f"seg_{k}", "n_seg") for k in LEN_KEY},
             speech_time_in_short_seg=pc("seg_s_lt_win", "speech_s"), median_seg_s=float(np.median(sp)) if len(sp) else np.nan,
             n_gap=int(s.n_gap), **{f"gap_{k}": pc(f"gap_{k}", "n_gap") for k in LEN_KEY},
             gap_filled_pp=pc("gap_filled", "n_gap"), median_gap_s=float(np.median(gp)) if len(gp) else np.nan,
             seg_dropped_pp=pc("seg_dropped", "n_seg"),
             win_pure_speech=pc("win_sp", "n_win"), win_pure_ns=pc("win_ns", "n_win"), win_mixed=pc("win_mixed", "n_win"),
             **{f"bin_dist_{k}": pc(f"bin_{k}", "n_bin") for k in DIST_KEY}, bin_low_cov=pc("low_cov", "n_bin"))
    for mode in ("A0", "A4"):
        tp, fn, fp, tn = (s[f"o{mode}_{x}"] for x in ("TP", "FN", "FP", "TN"))
        mr = fn / (tp + fn) if tp + fn else np.nan; far = fp / (fp + tn) if fp + tn else np.nan
        r.update({f"oracle_{mode}_MR": mr, f"oracle_{mode}_FAR": far, f"oracle_{mode}_F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                  f"oracle_{mode}_DCF": A.dcf_miss * mr + (1 - A.dcf_miss) * far, f"oracle_{mode}_err": (fp + fn) / s.n_bin})
    return r
TAPS = [("dev", "dev"), ("test", "test"), ("tất cả", None)]
profile = pd.DataFrame([prof_summary(g_, c, tap) for c, gc in [(c, prof[prof.category == c]) for c in CATS] + [("Tất cả", prof)]
                        for tap, sv in TAPS for g_ in [gc if sv is None else gc[gc.split == sv]] if len(g_)])
# phân bố độ dài đoạn / khoảng lặng (số đoạn theo độ dài, cột cuối = ≥ 5 s) và tỉ lệ speech trong cửa sổ, toàn bộ file
def len_bucket(L): return np.minimum(np.round(L / BIN).astype(int), len(HIST_LEN)) - 1   # cột 0.5 s … ≥ 5 s
hist_rows, frac_rows, hist_len = [], [], {}                 # hist_len[(category, kind)] = độ dài từng đoạn (s), cho Hình 6
for c, fs in [(c, list(prof.file[prof.category == c])) for c in CATS] + [("Tất cả", list(prof.file))]:
    for i, kind in enumerate(("speech", "khoảng lặng")):
        L = hist_len[(c, kind)] = np.concatenate([runs[f][i] for f in fs]); k = len_bucket(L)
        hist_rows.append(dict(Category=c, kind=kind, n=len(L), **{("≥ 5" if j == len(HIST_LEN) - 1 else f"{v:g}"): int((k == j).sum())
                                                                 for j, v in enumerate(HIST_LEN)}))
    fc = frac_cls(np.concatenate([fracs[f] for f in fs]))
    frac_rows.append(dict(Category=c, n_win=len(fc), **{lab: int((fc == j).sum()) for j, lab in enumerate(FRAC_CLS)}))
seg_hist, frac_hist = pd.DataFrame(hist_rows), pd.DataFrame(frac_rows)
# thêm cột % theo file vào per_file.csv
q = prof.set_index("file"); nz = lambda s: s.where(s > 0)
pf_prof = pd.DataFrame(dict(pct_seg_lt_win=q.seg_lt_win / nz(q.n_seg), pct_gap_lt_win=q.gap_lt_win / nz(q.n_gap),
                            pct_gap_filled_pp=q.gap_filled / nz(q.n_gap), pct_win_mixed=q.win_mixed / nz(q.n_win),
                            pct_bin_near_trans=q.bin_le_048 / q.n_bin, trans_per_min=q.n_trans / (q.dur_s / 60),
                            **{f"oracle_F1_{mode}": 2 * q[f"o{mode}_TP"] / nz(2 * q[f"o{mode}_TP"] + q[f"o{mode}_FP"] + q[f"o{mode}_FN"])
                               for mode in ("A0", "A4")},
                            oracle_err_A4=(q.oA4_FP + q.oA4_FN) / q.n_bin, pct_bin_low_cov=q.low_cov / q.n_bin))
pf = pf.merge(pf_prof, left_on="file", right_index=True, how="left")
# dev vs test: đặc trưng lệch > 10 điểm %
FLAG_KEYS = ["pct_speech", "seg_lt_win", "gap_lt_win", "gap_filled_pp", "win_mixed", "bin_dist_le_048", "oracle_A4_F1"]
flags = []
for c in CATS + ["Tất cả"]:
    g = profile[profile.Category == c].set_index("split")
    if not {"dev", "test"} <= set(g.index): continue
    for k in FLAG_KEYS:
        dv, tv = g.at["dev", k], g.at["test", k]
        if pd.notna(dv) and pd.notna(tv) and abs(dv - tv) > 0.10: flags.append(f"{c}: {k} dev {dv:.0%} / test {tv:.0%}")
PROF_KEYS = ["pct_speech", "trans_per_min", "seg_lt_win", "speech_time_in_short_seg", "gap_lt_win", "gap_filled_pp", "win_mixed",
             "bin_dist_le_048", "oracle_A0_F1", "oracle_A4_F1", "oracle_A4_DCF", "oracle_A4_err"]
S.update(profile={r.Category: {k: getattr(r, k) for k in PROF_KEYS} for r in profile[profile.split == "tất cả"].itertuples()},
         profile_fill_gap_s=FILL_GAP, profile_devtest_flags=flags,
         profile_len_classes=LEN_CLS, profile_dist_classes=DIST_CLS, profile_frac_classes=FRAC_CLS)

# =============== 3. Nhất quán nhãn: người gán 2 vs nhãn cùng quy tắc ===============
lab_cat = pd.DataFrame()
if HAS_L2:
    has = [f for f in pf.file if recs[f]["has_l2"]]
    Aa, Bs = cat_(has, "y_ref"), cat_(has, "y2"); BM = np.concatenate([boundary_mask(recs[f]["y_ref"]) for f in has])
    S.update(kappa_l2=cohen_kappa_score(Aa, Bs), kappa_l2_boundary=cohen_kappa_score(Aa[BM], Bs[BM]),
             kappa_l2_interior=cohen_kappa_score(Aa[~BM], Bs[~BM]), raw_agree_l2=float((Aa == Bs).mean()),
             pct_sure_bins=float(np.concatenate([recs[f]["sure"] for f in has]).mean()))
    rows = []
    for c in CATS:
        fs = [f for f in has if recs[f]["category"] == c]
        if not fs: continue
        a_, b_ = cat_(fs, "y_ref"), cat_(fs, "y2")
        rows.append(dict(Category=c, n_file=len(fs), speech_gt=cat_(fs, "y").mean(), speech_ref=a_.mean(), speech_l2=b_.mean(),
                         agree_l2=(a_ == b_).mean(), kappa_l2=cohen_kappa_score(a_, b_) if len(set(a_)) > 1 or len(set(b_)) > 1 else np.nan,
                         pct_bin_nhan_chac=cat_(fs, "sure").mean()))
    lab_cat = pd.DataFrame(rows)
else:
    S.update(kappa_l2=None, kappa_l2_boundary=None, kappa_l2_interior=None, raw_agree_l2=None, pct_sure_bins=None)
S["pct_bins_khac_alt"] = float((Y_all != cat_(pf.file, "y_alt")).mean()) if HAS_ALT else None

# =============== 4. Độ khó ===============
amb = (S_all > 0.1) & (S_all < 0.9); bm_all = cat_(pf.file, "bmask")
S.update(pct_ambiguous=float(amb.mean()), pct_ambiguous_in_boundary=float((amb & bm_all).sum() / max(amb.sum(), 1)))
thr_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
f1_thr = [rates(Y_all, (S_all >= t).astype(int))["F1"] for t in thr_grid]
S["f1_range_0.2_0.9"] = float(np.nanmax(f1_thr[3:18]) - np.nanmin(f1_thr[3:18]))

# =============== Tầng A (điểm thô) ===============
def tierA(files, name, yk="y"):
    y, s = cat_(files, yk), cat_(files, "braw")
    bm = np.concatenate([boundary_mask(recs[f][yk]) for f in files])
    ok = np.concatenate([recs[f]["sure"] for f in files]) if all(recs[f]["sure"] is not None for f in files) else None
    fpr, tpr, _ = roc_curve(y, s) if 0 < y.mean() < 1 else (np.array([0, 1]), np.array([np.nan, np.nan]), None)
    return dict(Tập=name, n_file=len(files), n_bin=len(y), pct_speech=y.mean(), AUC_micro=auc(y, s),
                AUC_macro_file=np.nanmean([safe_auc(recs[f][yk], recs[f]["braw"]) for f in files]),
                AUC_khong_bien=auc(y[~bm], s[~bm]), AUC_chi_bien=auc(y[bm], s[bm]),
                AUC_nhan_chac=auc(y[ok], s[ok]) if ok is not None else np.nan,
                PR_AUC=average_precision_score(y, s) if y.any() else np.nan,
                TPR_at_FPR1=float(np.interp(0.01, fpr, tpr)), TPR_at_FPR5=float(np.interp(0.05, fpr, tpr)),
                Brier=brier_score_loss(y, s), mean_score_nonspeech=s[y == 0].mean() if (y == 0).any() else np.nan)
def by_cat(c, files): return [f for f in files if recs[f]["category"] == c]
tA = [tierA(test, "Test (tất cả)")] + [tierA(by_cat(c, test), c) for c in CATS_T]
tA += [tierA(dev, "Dev (tham khảo)"), tierA(list(pf.file), f"Cả {len(pf)} file (tham khảo)")]
tA = pd.DataFrame(tA)
tA_alt = pd.DataFrame([tierA(test, "Test (tất cả)", "y_alt")] + [tierA(by_cat(c, test), c, "y_alt") for c in CATS_T]) if HAS_ALT else None
aucs = []
for _ in range(A.n_boot):
    pick = rng.choice(test, len(test), replace=True); aucs.append(auc(cat_(pick, "y"), cat_(pick, "braw")))
S["AUC_test_CI"] = [float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))]

# =============== Chọn ngưỡng trên dev (và val tổng hợp nếu có) qua toàn pipeline ===============
def pooled(files, thr, mode="A4", mask=None, src=None, **kw):
    src = src or recs; Y, P = [], []
    for f in files:
        d = src[f]; p = pred_cached(d, thr, mode, **kw); y = d["y"]
        sel = slice(None) if mask is None else (~d["bmask"] if mask == "nb" else d["bmask"])
        Y.append(y[sel]); P.append(p[sel])
    return rates(np.concatenate(Y), np.concatenate(P), A.dcf_miss)
thr_sweep = np.round(np.arange(A.thr_min, A.thr_max + A.thr_step / 2, A.thr_step), 4)   # trên dữ liệu thật DCF-min có thể rơi xuống rất thấp
sweep_rows = []
for tap, files, src in [("dev", dev, recs)] + ([("synth", list(syn), syn)] if syn else []):
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
if syn:
    ys, ss = cat_(list(syn), "y", syn), cat_(list(syn), "braw", syn); fpr, tpr, th = roc_curve(ys, ss)
    S["thr_youden_raw_synth"] = float(th[np.argmax(tpr - fpr)]); S["auc_synth_val"] = auc(ys, ss)
else:
    S["thr_youden_raw_synth"] = S["auc_synth_val"] = None
S["best_thr_dev"] = best["dev"]; S["best_thr_synth"] = best.get("synth"); S["best_f1_thr_dev"] = best_f1["dev"]
S["best_f1_thr_synth"] = best_f1.get("synth")
CHOSEN = best if A.criterion == "DCF-min" else best_f1                  # ngưỡng theo tiêu chí dùng cho tầng B, ablation
OTHER, OTHER_NAME = (best_f1, "F1-max") if A.criterion == "DCF-min" else (best, "DCF-min")
THR = CHOSEN["dev"]["A4"]
tr_rows = ([("Youden điểm thô – val tổng hợp", "A0", S["thr_youden_raw_synth"]), ("DCF-min A0 – val tổng hợp", "A0", best["synth"]["A0"]),
            ("DCF-min A4 – val tổng hợp", "A4", best["synth"]["A4"]), ("F1-max A4 – val tổng hợp", "A4", best_f1["synth"]["A4"])] if syn else []) + \
          [("Youden điểm thô – dev", "A0", S["thr_youden_raw_dev"]), ("DCF-min A0 – dev", "A0", best["dev"]["A0"]),
           (f"{OTHER_NAME} A4 – dev", "A4", OTHER["dev"]["A4"]), (f"{A.criterion} A4 – dev (dùng chính)", "A4", THR)]
transfer = []
for name, mode, t in tr_rows:
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
deltas = np.array(deltas, float).reshape(-1, 2)
def f1_(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
# chỉ số theo đoạn phụ thuộc mạnh vào ngưỡng: ngưỡng thấp -> đoạn dự đoán gộp nhiều đoạn GT
ev_thr = []
for name, t in [(f"{A.criterion} dev (dùng chính)", THR), (f"{OTHER_NAME} dev", OTHER["dev"]["A4"])] + ([(f"{A.criterion} val tổng hợp", CHOSEN["synth"]["A4"])] if syn else []):
    c_ = np.zeros(8)
    for f in test:
        gs, ps = bins_to_segs(recs[f]["y"]), bins_to_segs(pred_cached(recs[f], t))
        c_ += [*event_f1(gs, ps)[1:], event_f1(gs, ps, onset_only=True)[1], *frag_merge(gs, ps), len(ps), len(gs)]
    tp, fp, fn, otp, fr_, mg_, npr, ngt = c_
    ev_thr.append(dict(Ngưỡng=name, thr=t, event_F1=f1_(tp, fp, fn), onset_F1=2 * otp / (npr + ngt) if npr + ngt else np.nan,
                       n_gt_seg=int(ngt), n_pred_seg=int(npr), n_fragmented=int(fr_), n_overmerged=int(mg_)))
ev_thr = pd.DataFrame(ev_thr)
S.update(thr_used=THR, err_types=et, pct_true_err=(et["MSC"] + et["NDS"]) / max(sum(et.values()), 1),
         event_f1=f1_(ev_tp, ev_fp, ev_fn), event_f1_onset=f1_(on_tp, on_fp, on_fn),
         n_gt_seg=n_gt_seg, n_pred_seg=n_pr_seg, n_fragmented=frag, n_overmerged=merge,
         mean_d_onset=float(deltas[:, 0].mean()) if len(deltas) else None, mean_d_offset=float(deltas[:, 1].mean()) if len(deltas) else None,
         gap_short_total=gap_total, gap_short_filled=gap_filled)

def boot(files, fn, B=None):
    B = B or A.n_boot
    vals = []; per = {f: fn([f]) for f in files}
    for _ in range(B):
        tot = pd.DataFrame([per[f] for f in rng.choice(files, len(files), replace=True)]).sum()
        tp, fp, fn_, tn = tot.TP, tot.FP, tot.FN, tot.TN
        mr = fn_ / (tp + fn_) if tp + fn_ else np.nan; far = fp / (fp + tn) if fp + tn else np.nan
        vals.append((2 * tp / (2 * tp + fp + fn_) if tp + fp + fn_ else np.nan, A.dcf_miss * mr + (1 - A.dcf_miss) * far, mr, far))
    return np.nanpercentile(np.array(vals), [2.5, 97.5], axis=0)
ci = boot(test, lambda fs: pooled(fs, THR)); ci_nb = boot(test, lambda fs: pooled(fs, THR, mask="nb"))
S.update(F1_CI=ci[:, 0].tolist(), DCF_CI=ci[:, 1].tolist(), MR_CI=ci[:, 2].tolist(), FAR_CI=ci[:, 3].tolist(),
         F1nb_CI=ci_nb[:, 0].tolist())

# =============== Báo cáo theo category ===============
cat_rows = []
for c in CATS_T:
    fs = by_cat(c, test)
    r = pooled(fs, THR); rnb = pooled(fs, THR, mask="nb"); e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
    for f in fs:
        for k, v in error_types(recs[f]["y"], pred_cached(recs[f], THR)).items(): e[k] += v
    cib = boot(fs, lambda x: pooled(x, THR), B=max(100, A.n_boot // 2))
    cat_rows.append(dict(Category=c, n_file=len(fs), pct_speech=cat_(fs, "y").mean(), MR=r["MR"], FAR=r["FAR"],
                         Precision=r["Precision"], **e, F1_strict=r["F1"], F1_nb=rnb["F1"], F1_CI_lo=cib[0, 0], F1_CI_hi=cib[1, 0],
                         FAR_CI_lo=cib[0, 3], FAR_CI_hi=cib[1, 3]))
cat_df = pd.DataFrame(cat_rows)

# =============== Ablation (ngưỡng chọn riêng cho từng cấu hình trên dev) ===============
abl = []
for mode, desc in zip(MODES, CONFIG_DESC):
    t = CHOSEN["dev"][mode]; r = pooled(test, t, mode); rnb = pooled(test, t, mode, mask="nb"); e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
    for f in test:
        for k, v in error_types(recs[f]["y"], pred_cached(recs[f], t, mode)).items(): e[k] += v
    abl.append(dict(Cấu_hình=mode, Mô_tả=desc, Ngưỡng_dev=t, MR=r["MR"], FAR=r["FAR"], F1_strict=r["F1"], F1_nb=rnb["F1"],
                    DCF=r["DCF"], **e))
abl = pd.DataFrame(abl)
grid = []
for pre in GRID["pre"]:
    for post in GRID["post"]:
        for gap in GRID["gap"]:
            r = pooled(dev, THR, "grid", pre=pre, post=post, gap=gap)
            grid.append(dict(pre_ms=int(pre * 1000), post_ms=int(post * 1000), gap_ms=int(gap * 1000),
                             TP=r["TP"], FN=r["FN"], FP=r["FP"], TN=r["TN"], F1=r["F1"], DCF=r["DCF"]))
grid = pd.DataFrame(grid)
S["grid_best"] = grid.loc[grid.DCF.idxmin()].to_dict()

# =============== 6. So sánh phương án hậu xử lý ===============
# Mỗi cấu hình: điểm hop (gộp cửa sổ + làm mượt) -> ngưỡng (đơn / kép) -> đoạn -> merge / drop / pad -> bin 0.5 s.
# Ngưỡng chọn riêng cho từng cấu hình trên dev theo --criterion, đánh giá một lần trên test; Δ so với B0 có CI bootstrap
# ghép cặp theo file. B0 luôn là cấu hình hiện tại (không theo --preset / tham số dòng lệnh); C1–C7 = B0 + một thay đổi;
# CLI = tham số dòng lệnh nếu khác B0 và R. AUC chỉ phụ thuộc cách gộp / làm mượt; ngưỡng kép, pad, drop, merge chỉ đổi MR / FAR / F1 / DCF.
PP_KEYS = ("overlap", "smooth", "smooth_kind", "domain", "hyst", "order", "pre", "post", "gap", "drop", "hop_sub")
CLI = dict(overlap=A.overlap, smooth=A.smooth, smooth_kind=A.smooth_kind, domain=A.domain, hyst=A.hyst, order=A.order,
           pre=A.pad_pre, post=A.pad_post, gap=A.merge_gap, drop=A.drop, hop_sub=1)
BASE = dict(overlap=RESCORE_WEIGHT, smooth=RESCORE_SMOOTH, smooth_kind="mean", domain="prob", hyst=0.0, order="pdm",
            pre=0.10, post=0.12, gap=0.50, drop=0.32, hop_sub=1)   # B0 luôn là cấu hình hiện tại để các lần chạy so sánh được
_rec = PRESETS["de-xuat"]
REC = dict(overlap=_rec["overlap"], smooth=_rec["smooth"], smooth_kind=_rec["smooth_kind"], domain=_rec["domain"], hyst=_rec["hyst"],
           order=_rec["order"], pre=_rec["pad_pre"], post=_rec["pad_post"], gap=_rec["merge_gap"], drop=_rec["drop"], hop_sub=1)
def cfg_desc(c):
    return (f"{rescore_desc(c['overlap'], c['smooth'], c['smooth_kind'], c['domain'])}; "
            + (f"ngưỡng kép Δ {c['hyst']:g}" if c["hyst"] > 0 else "ngưỡng đơn")
            + f"; {ORDER_DESC[c['order']]}: pad {ms(c['pre'])}/{ms(c['post'])} ms, merge < {ms(c['gap'])} ms, drop {ms(c['drop'])} ms"
            + (f"; hop {HOP * c['hop_sub']:g} s (giữ 1/{c['hop_sub']} cửa sổ)" if c["hop_sub"] > 1 else ""))
COMPARE = [("B0", "Hiện tại (pad 100/120, drop 0.32, merge 500, mean 5 hop)", BASE),
           ("C1", "B0 + ngưỡng kép Δ 0.15", {**BASE, "hyst": 0.15}),
           ("C2", "B0 + thứ tự merge → drop → pad (drop 0.32 đo trước pad)", {**BASE, "order": "mdp"}),
           ("C3", "B0 + pad 0 / 0", {**BASE, "pre": 0.0, "post": 0.0}),
           ("C4", "B0 + làm mượt median 3 hop", {**BASE, "smooth": 3, "smooth_kind": "median"}),
           ("C5", "B0 + không làm mượt", {**BASE, "smooth": 1}),
           ("C6", "B0 + trọng số Hann", {**BASE, "overlap": "hann"}),
           ("C7", "B0 + gộp miền logit", {**BASE, "domain": "logit"}),
           ("R", "Đề xuất (preset de-xuat)", REC),
           ("R-h16", "Đề xuất, hop 0.16 s (giảm 2× số lần chạy model)", {**REC, "hop_sub": 2}),
           ("R-h24", "Đề xuất, hop 0.24 s (giảm 3× số lần chạy model)", {**REC, "hop_sub": 3})]
if CLI not in (BASE, REC): COMPARE.insert(1, ("CLI", "Cấu hình chính (tham số dòng lệnh)", CLI))
_fr_cache = {}
def fr_of(f, c):
    """Điểm hop của file f theo cách gộp / làm mượt / hop của cấu hình c (cache)."""
    k = (f, c["overlap"], c["smooth"], c["smooth_kind"], c["domain"], c["hop_sub"])
    if k not in _fr_cache:
        d = recs[f]; m = d["m"].iloc[::c["hop_sub"]]
        _fr_cache[k] = rescore_hop(m, dur=d["n"] * BIN, weight=c["overlap"], smooth=c["smooth"], smooth_kind=c["smooth_kind"], domain=c["domain"])
    return _fr_cache[k]
def pred_cfg(f, c, thr):
    d = recs[f]; b = hysteresis(fr_of(f, c), thr, thr - c["hyst"] if c["hyst"] > 0 else None)
    segs = postprocess_segs(frames_to_segs(b, HOP), c["pre"], c["post"], c["gap"], c["drop"], dur=d["n"] * BIN, order=c["order"])
    return segs_to_bins_fast(segs, d["n"])
def cnt_of(y, p): return np.array([((y == 1) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 0) & (p == 0)).sum()])
def met(c4):
    tp, fn, fp, tn = c4; mr = fn / (tp + fn) if tp + fn else np.nan; far = fp / (fp + tn) if fp + tn else np.nan
    return dict(MR=mr, FAR=far, Precision=tp / (tp + fp) if tp + fp else np.nan, Recall=1 - mr if tp + fn else np.nan,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan, DCF=A.dcf_miss * mr + (1 - A.dcf_miss) * far)
def objective(m_): return m_["DCF"] if A.criterion == "DCF-min" else -m_["F1"]
def pick_thr(c, files, thrs):
    """Ngưỡng tốt nhất trên files theo --criterion; trả (ngưỡng, metric tại ngưỡng đó)."""
    best_ = None
    for t in thrs:
        m_ = met(sum(cnt_of(recs[f]["y"], pred_cfg(f, c, t)) for f in files))
        if best_ is None or objective(m_) < objective(best_[1]) - 1e-12: best_ = (float(t), m_)
    return best_

# tìm cấu hình tốt nhất trên dev (--tune): Optuna TPE nếu có, không thì tìm ngẫu nhiên; ngưỡng quét trong từng lần thử
SPACE = dict(overlap=["tri", "hann"], domain=["prob", "logit"], smooth=["mean:1", "median:3", "median:5", "gauss:5", "mean:5"],
             hyst=[float(round(x, 2)) for x in np.arange(0, 0.36, 0.05)], order=["pdm", "mdp"],
             pre=[float(round(x, 2)) for x in np.arange(-0.16, 0.17, 0.04)], post=[float(round(x, 2)) for x in np.arange(-0.16, 0.17, 0.04)],
             gap=[float(round(x, 1)) for x in np.arange(0, 1.01, 0.1)], drop=[float(round(x, 2)) for x in np.arange(0, 0.57, 0.08)])
def space_to_cfg(v):
    k, n = v["smooth"].split(":")
    return dict(overlap=v["overlap"], smooth=int(n), smooth_kind=k, domain=v["domain"], hyst=float(v["hyst"]), order=v["order"],
                pre=float(v["pre"]), post=float(v["post"]), gap=float(v["gap"]), drop=float(v["drop"]), hop_sub=1)
tune_rows, tuner = [], None
if A.tune and not A.no_compare:
    thr_tune = thr_sweep[::2] if len(thr_sweep) > 40 else thr_sweep
    def run_trial(v):
        c = space_to_cfg(v); t, m_ = pick_thr(c, dev, thr_tune)
        tune_rows.append(dict(trial=len(tune_rows), **v, thr=t, **{f"dev_{k}": m_[k] for k in ("F1", "DCF", "MR", "FAR")}))
        return objective(m_)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st_ = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=A.tune_seed)); seen_ = []
        for c0 in (BASE, REC, CLI):    # bắt đầu từ cấu hình hiện tại, đề xuất, dòng lệnh (nếu nằm trong không gian tìm)
            v0 = dict(overlap=c0["overlap"], domain=c0["domain"], smooth=f"{c0['smooth_kind']}:{c0['smooth']}", hyst=c0["hyst"], order=c0["order"],
                      pre=c0["pre"], post=c0["post"], gap=c0["gap"], drop=c0["drop"])
            if all(v0[k] in SPACE[k] for k in SPACE) and v0 not in seen_: st_.enqueue_trial(v0); seen_.append(v0)
        st_.optimize(lambda tr: run_trial({k: tr.suggest_categorical(k, SPACE[k]) for k in SPACE}), n_trials=A.tune)
        tuner = "Optuna TPE"
    except ImportError:
        rs_t = np.random.default_rng(A.tune_seed)
        for _ in range(A.tune): run_trial({k: SPACE[k][rs_t.integers(len(SPACE[k]))] for k in SPACE})
        tuner = "tìm ngẫu nhiên (cài optuna để dùng TPE)"
    tune_df = pd.DataFrame(tune_rows)
    bt = tune_df.loc[tune_df.dev_DCF.idxmin() if A.criterion == "DCF-min" else tune_df.dev_F1.idxmax()]
    COMPARE.append(("T", f"Tốt nhất trên dev ({tuner}, {A.tune} lần thử)", space_to_cfg(bt)))
    tune_df.to_csv(OUT / "pp_tune_trials.csv", index=False)
else:
    (OUT / "pp_tune_trials.csv").unlink(missing_ok=True)

pp_rows, pp_cat_rows, pp_cnt = [], [], {}
if not A.no_compare:
    for code, name, c in COMPARE:
        t, mdev = pick_thr(c, dev, thr_sweep)
        P = {f: pred_cfg(f, c, t) for f in test}
        pp_cnt[code] = {f: (cnt_of(recs[f]["y"], P[f]), cnt_of(recs[f]["y"][~recs[f]["bmask"]], P[f][~recs[f]["bmask"]])) for f in test}
        tot = sum(v[0] for v in pp_cnt[code].values()); tot_nb = sum(v[1] for v in pp_cnt[code].values()); mt = met(tot)
        ev = np.zeros(6)
        for f in test:
            gs, ps = bins_to_segs(recs[f]["y"]), bins_to_segs(P[f])
            ev += [*event_f1(gs, ps)[1:], *frag_merge(gs, ps), len(ps)]
        tp_, fp_, fn_, fr_, mg_, npr = ev
        yt = cat_(test, "y"); st = np.concatenate([frames_to_bin_scores(fr_of(f, c), recs[f]["n"]) for f in test])
        pp_rows.append(dict(Mã=code, Cấu_hình=name, Chi_tiết=cfg_desc(c), **{k: c[k] for k in PP_KEYS}, Ngưỡng_dev=t,
                            Ngưỡng_off=round(t - c["hyst"], 4) if c["hyst"] > 0 else t, dev_F1=mdev["F1"], dev_DCF=mdev["DCF"],
                            **mt, F1_nb=met(tot_nb)["F1"], AUC_hop_bin=auc(yt, st), event_F1=f1_(tp_, fp_, fn_),
                            n_pred_seg=int(npr), n_fragmented=int(fr_), n_overmerged=int(mg_),
                            so_lan_chay_model=f"1/{c['hop_sub']}" if c["hop_sub"] > 1 else "1"))
        for cc in CATS_T:
            fs = by_cat(cc, test); mc = met(sum(pp_cnt[code][f][0] for f in fs))
            pp_cat_rows.append(dict(Mã=code, Category=cc, n_file=len(fs), **mc, F1_nb=met(sum(pp_cnt[code][f][1] for f in fs))["F1"]))
    # Δ so với B0: bootstrap ghép cặp theo file test (cùng file lấy mẫu cho cả hai cấu hình)
    picks = [rng.choice(test, len(test), replace=True) for _ in range(A.n_boot)]
    for r_ in pp_rows:
        dd = []
        for pk in picks:
            a_ = met(sum(pp_cnt[r_["Mã"]][f][0] for f in pk)); b_ = met(sum(pp_cnt["B0"][f][0] for f in pk))
            dd.append((a_["F1"] - b_["F1"], a_["DCF"] - b_["DCF"]))
        dd = np.array(dd, float); lo_, hi_ = np.nanpercentile(dd, [2.5, 97.5], axis=0)
        r_.update(dF1=r_["F1"] - pp_rows[0]["F1"], dF1_CI_lo=lo_[0], dF1_CI_hi=hi_[0], P_dF1_gt0=float(np.nanmean(dd[:, 0] > 0)),
                  dDCF=r_["DCF"] - pp_rows[0]["DCF"], dDCF_CI_lo=lo_[1], dDCF_CI_hi=hi_[1])
pp = pd.DataFrame(pp_rows); pp_cat = pd.DataFrame(pp_cat_rows)
if len(pp):
    pp.to_csv(OUT / "pp_compare.csv", index=False, encoding="utf-8-sig"); pp_cat.to_csv(OUT / "pp_compare_category.csv", index=False, encoding="utf-8-sig")
    bestr = pp.loc[pp.DCF.idxmin() if A.criterion == "DCF-min" else pp.F1.idxmax()]
    S.update(pp_compare=pp[["Mã", "Cấu_hình", "Ngưỡng_dev", "F1", "F1_nb", "DCF", "MR", "FAR", "AUC_hop_bin", "dF1", "dF1_CI_lo", "dF1_CI_hi",
                            "dDCF", "dDCF_CI_lo", "dDCF_CI_hi"]].to_dict("records"),
             pp_best_on_test=str(bestr["Mã"]), pp_tuner=tuner, pp_tune_best=(COMPARE[-1][2] if tuner else None))
else:
    for n_ in ("pp_compare.csv", "pp_compare_category.csv"): (OUT / n_).unlink(missing_ok=True)

# =============== Nhãn đáng ngờ: đoạn bin liền nhau (không sát biên) mà điểm và nhãn trái ngược mạnh ===============
sus = []
for f, d in recs.items():
    s_, y_ = d["braw"], d["y"]
    bad = ~d["bmask"] & (((s_ > 0.9) & (y_ == 0)) | ((s_ < 0.1) & (y_ == 1)))
    for a0, b0 in frames_to_segs(bad, 1):
        a0, b0 = int(a0), int(b0)
        sus.append(dict(file=f, category=d["category"], split=split[f], start=a0 * BIN, end=b0 * BIN, n_bins=b0 - a0,
                        GT="speech" if y_[a0] else "non-speech", score=round(float(s_[a0:b0].mean()), 3),
                        l2_speech=round(float(d["y2"][a0:b0].mean()), 2) if d["has_l2"] else np.nan,
                        khac_alt=bool((y_[a0:b0] != d["y_alt"][a0:b0]).any()) if d["has_alt"] else False))
sus = pd.DataFrame(sus).sort_values("n_bins", ascending=False) if sus else \
    pd.DataFrame(columns=["file", "category", "split", "start", "end", "n_bins", "GT", "score", "l2_speech", "khac_alt"])
S["n_suspicious_seg"] = len(sus); S["n_suspicious_bins"] = int(sus.n_bins.sum()) if len(sus) else 0
S["suspicious_l2_agrees_model"] = float(((sus.GT == "non-speech") & (sus.l2_speech >= 0.5) | (sus.GT == "speech") & (sus.l2_speech < 0.5))
                                        .mul(sus.n_bins).sum() / sus.n_bins.sum()) if HAS_L2 and len(sus) else None

# =============== Số đếm thô cho Excel: make_excel_real.py tính mọi chỉ số bằng công thức từ đây ===============
def counts(y, p):
    return dict(TP=int(((y == 1) & (p == 1)).sum()), FN=int(((y == 1) & (p == 0)).sum()),
                FP=int(((y == 0) & (p == 1)).sum()), TN=int(((y == 0) & (p == 0)).sum()))
# quét ngưỡng: dev / val tổng hợp gộp; test theo category × vùng bin (all / nb = bỏ bin biên / b = chỉ bin biên)
rows = []
for mode in MODES:
    for t in thr_sweep:
        P = {f: pred_cached(recs[f], t, mode) for f in test}
        for c in CATS_T:
            fs = by_cat(c, test); e = {"FEC": 0, "MSC": 0, "OVER": 0, "NDS": 0}
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
# ROC trên điểm thô: lưới ngưỡng = phân vị điểm của mọi file (+ 2 ngưỡng chặn) -> Excel tính AUC bằng hình thang
roc_thr = np.r_[-1e-6, np.unique(np.quantile(S_all, np.linspace(0, 1, 401))), 1 + 1e-6]
roc = pd.DataFrame(dict(thr=roc_thr))
for g, fs in [(c, by_cat(c, test)) for c in CATS_T] + [("Dev", dev)]:
    y, s = cat_(fs, "y"), cat_(fs, "braw"); pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = len(pos) - np.searchsorted(pos, roc_thr, "left"); fp = len(neg) - np.searchsorted(neg, roc_thr, "left")
    roc[f"{g}_TP"], roc[f"{g}_FP"], roc[f"{g}_FN"], roc[f"{g}_TN"] = tp, fp, len(pos) - tp, len(neg) - fp
roc.to_csv(XL / "roc_counts.csv", index=False)
tpr = sum(roc[f"{c}_TP"] for c in CATS_T).to_numpy(); fpr = sum(roc[f"{c}_FP"] for c in CATS_T).to_numpy()
tpr, fpr = tpr / max(tpr[0], 1), fpr / max(fpr[0], 1)
S["auc_test_trapezoid"] = float(np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:])) / 2)
# nhãn: nhãn cùng quy tắc (a) vs người gán 2 (b), bảng 2×2 theo category × vùng bin
if HAS_L2:
    lab_rows = []
    for c in CATS:
        fs = [f for f in pf.file if recs[f]["category"] == c and recs[f]["has_l2"]]
        if not fs: continue
        a_, b_, y_, ya_, ok = cat_(fs, "y_ref"), cat_(fs, "y2"), cat_(fs, "y"), cat_(fs, "y_alt"), cat_(fs, "sure")
        bm = np.concatenate([boundary_mask(recs[f]["y_ref"]) for f in fs])
        for mk, sel in (("all", np.ones(len(a_), bool)), ("trong", ~bm), ("bien", bm)):
            A_, B_ = a_[sel], b_[sel]
            lab_rows.append(dict(category=c, mask=mk, n11=int(((A_ == 1) & (B_ == 1)).sum()), n10=int(((A_ == 1) & (B_ == 0)).sum()),
                                 n01=int(((A_ == 0) & (B_ == 1)).sum()), n00=int(((A_ == 0) & (B_ == 0)).sum()),
                                 bin_khac_alt=int((y_[sel] != ya_[sel]).sum()), n_nhan_chac=int(ok[sel].sum()), n_speech_gt=int(y_[sel].sum())))
    pd.DataFrame(lab_rows).to_csv(XL / "label_counts.csv", index=False)
else:
    (XL / "label_counts.csv").unlink(missing_ok=True)
pd.DataFrame(deltas, columns=["d_onset", "d_offset"]).to_csv(XL / "deltas.csv", index=False)

# =============== Lưu ===============
S["meta"] = dict(root=str(A.root) if A.root else None, raw=str(A.raw), gt=str(A.gt), audio=str(A.audio) if audio else None,
                 gt_alt=str(A.gt_alt) if HAS_ALT else None, label2=L2_NAME if HAS_L2 else None, synth=bool(syn),
                 mo_ta_nhan=A.mo_ta_nhan or f"Nhãn trong {A.gt.as_posix()}", mo_ta_model=A.mo_ta_model or f"Output model: {A.raw.as_posix()}",
                 categories=CATS_T, all_categories=CATS, group_by=GROUP_BY, rescore=rescore_desc(A.overlap, A.smooth), rescore_weight=A.overlap,
                 config_desc=CONFIG_DESC, params=PARAMS)
pf = pf.merge(lagdf[["file", "lag_s", "peak_corr"]], on="file", how="left")
pf.to_csv(OUT / "per_file.csv", index=False); struct.to_csv(OUT / "structure.csv", index=False)
sweep.to_csv(OUT / "shift_sweep.csv", index=False); lagdf.to_csv(OUT / "lag_per_file.csv", index=False)
for name, df in (("label_agreement.csv", lab_cat if HAS_L2 else None), ("tierA_nhan_goc.csv", tA_alt)):
    df.to_csv(OUT / name, index=False) if df is not None else (OUT / name).unlink(missing_ok=True)
tA.to_csv(OUT / "tierA.csv", index=False); tB.to_csv(OUT / "tierB.csv", index=False)
thr_df.to_csv(OUT / "thr_sweep.csv", index=False); transfer.to_csv(OUT / "threshold_transfer.csv", index=False)
ev_thr.to_csv(OUT / "event_by_threshold.csv", index=False)
profile.to_csv(OUT / "dataset_profile.csv", index=False); seg_hist.to_csv(OUT / "seg_gap_hist.csv", index=False)
frac_hist.to_csv(OUT / "window_frac_hist.csv", index=False)
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
    if len(g): ax[0, 1].plot([i - 0.28, i + 0.28], [g.auc_file.median()] * 2, c="#0b0b0b", lw=1.5)
for a_ in ax[0, :2]: a_.set_xticks(range(len(CATS))); a_.set_xticklabels(CATS)
ax[0, 0].set(title="1a. Tỉ lệ speech mỗi file (nhãn chính)", ylabel="speech ratio")
ax[0, 1].axhline(0.9, ls=":", c="#52514e", lw=0.8)
ax[0, 1].set(title=f"1d. AUC từng file (vạch = trung vị); {S['n_degenerate']} file 1 lớp không có AUC", ylabel="AUC")
for col, c_, lab in (("AUC_synth", C1, "val tổng hợp (nhãn chính xác)"), ("AUC_gt", C2, "nhãn chính"), ("AUC_l2", C3, f"người gán 2 ({L2_NAME})")):
    if sweep[col].notna().any():
        ax[0, 2].plot(sweep.shift_s, sweep[col] - sweep[col][sweep.shift_s == 0].iloc[0], "o-", ms=3, lw=1.8, c=c_, label=lab)
ax[0, 2].axvline(0, ls=":", c="#52514e", lw=0.8)
ax[0, 2].set(title="5e. ΔAUC theo độ dịch điểm (score tại tâm cửa sổ)", xlabel="shift (s)", ylabel="AUC − AUC(shift 0)")
ax[0, 2].legend(fontsize=7)
ax[1, 0].hist(lagdf.lag_s, bins=np.arange(-1.55, 1.6, 0.1), edgecolor="white", color=C1)
ax[1, 0].set(title=f"5e. Lag từng file: trung vị {S['lag_median'] or 0:+.2f}s, {len(S['lag_flag_files'])} file |lag|≥0.4s", xlabel="lag (s)", ylabel="#file")
ax[1, 1].hist(S_all[Y_all == 0], bins=40, alpha=.65, color=C1, label="non-speech", density=True)
ax[1, 1].hist(S_all[Y_all == 1], bins=40, alpha=.65, color=C2, label="speech", density=True)
ax[1, 1].set_yscale("log"); ax[1, 1].legend(fontsize=7); ax[1, 1].set(title=f"4a. Phân bố điểm theo lớp ({len(pf)} file)", xlabel="score")
if HAS_L2:
    ax[1, 2].bar(lab_cat.Category, lab_cat.agree_l2, color=C1, width=0.6)
    for i, r in enumerate(lab_cat.itertuples()):
        ax[1, 2].text(i, r.agree_l2 + 0.01, f"{r.agree_l2:.0%}\nκ={r.kappa_l2:.2f}", ha="center", fontsize=7)
    ax[1, 2].set_ylim(0, 1.12); ax[1, 2].set(title=f"3. Nhất quán nhãn: nhãn chính vs {L2_NAME}")
else:
    ax[1, 2].axis("off"); ax[1, 2].text(0.5, 0.5, "3. Nhất quán nhãn: không có người gán nhãn thứ 2\n(--label2 hoặc --silero-cache)",
                                        ha="center", va="center", fontsize=9, color="#52514e")
plt.tight_layout(); plt.savefig(OUT / "fig1_kiem_tra_du_lieu.png", dpi=130); plt.close()
# Hình 2: tầng A trên test
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for c in CATS_T:
    fs = by_cat(c, test); yy, ss = cat_(fs, "y"), cat_(fs, "braw")
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
for tap, ls in [("dev", "-")] + ([("synth", "--")] if syn else []):
    g = thr_df[(thr_df.tap == tap) & (thr_df.config == "A4")]
    ax[0, 0].plot(g.thr, g.MR, ls, c=C2, lw=1.8, label=f"MR {tap}"); ax[0, 0].plot(g.thr, g.FAR, ls, c=C1, lw=1.8, label=f"FAR {tap}")
ax[0, 0].axvline(THR, c="#0b0b0b", lw=.8)
if syn: ax[0, 0].axvline(CHOSEN["synth"]["A4"], c="#52514e", ls=":")
ax[0, 0].set(title=f"MR/FAR theo ngưỡng (A4). {A.criterion}: dev={THR}" + (f", val tổng hợp={CHOSEN['synth']['A4']}" if syn else ""), xlabel="ngưỡng")
ax[0, 0].legend(fontsize=7)
x = np.arange(len(cat_df)); bnd = cat_df.FEC + cat_df.OVER; tru = cat_df.MSC + cat_df.NDS
ax[0, 1].bar(x, bnd, label="lỗi biên (FEC+OVER)", color=C1, width=0.6, edgecolor="white", linewidth=2)
ax[0, 1].bar(x, tru, bottom=bnd, label="lỗi thật (MSC+NDS)", color=C2, width=0.6, edgecolor="white", linewidth=2)
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(cat_df.Category); ax[0, 1].set(title=f"Loại lỗi theo category (test, A4, thr={THR})", ylabel="#bin")
ax[0, 1].legend(loc="upper left", fontsize=7)
if len(deltas):
    ax[1, 0].hist(deltas[:, 0], bins=np.arange(-3.25, 3.3, 0.5), alpha=.65, color=C1, label=f"Δonset (TB {deltas[:, 0].mean():+.2f}s)")
    ax[1, 0].hist(deltas[:, 1], bins=np.arange(-3.25, 3.3, 0.5), alpha=.65, color=C2, label=f"Δoffset (TB {deltas[:, 1].mean():+.2f}s)")
    ax[1, 0].legend(fontsize=7)
ax[1, 0].set(title="Độ lệch biên đoạn (dự đoán − GT), test", xlabel="giây")
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
show = [pt.sort_values("auc_file").file.iloc[0]] if len(pt) else []
for c in CATS_T:
    g = pt[(pt.category == c) & ~pt.file.isin(show)]
    if len(g): show.append(g.iloc[(g.auc_file - g.auc_file.median()).abs().argsort()].file.iloc[0])
if show:
    fig, axs = plt.subplots(len(show), 1, figsize=(14, 1.9 * len(show)), squeeze=False); axs = axs[:, 0]
    for a_, f in zip(axs, show):
        d = recs[f]; y = d["y"]; p = pred_cached(d, THR); tt = np.arange(d["n"] + 1) * BIN
        a_.fill_between(tt, np.r_[y, y[-1]], step="post", color=C1, alpha=.18, lw=0, label="GT speech")
        a_.plot(d["m"].center, d["m"].score, lw=0.8, c=C1, label="score tại tâm cửa sổ")
        a_.fill_between(tt, -0.12 * np.r_[p, p[-1]], step="post", color="#0b0b0b", alpha=.55, lw=0, label="dự đoán A4")
        a_.axhline(THR, ls="--", c="#52514e", lw=.7); a_.set_ylim(-0.15, 1.05); a_.set_xlim(0, d["n"] * BIN)
        a_.set_title(f"{f}  AUC={safe_auc(y, d['braw']):.3f}  speech={y.mean():.0%}", fontsize=8, loc="left")
    axs[0].legend(fontsize=7, loc="upper right", ncol=3); axs[-1].set_xlabel("giây")
    plt.tight_layout(); plt.savefig(OUT / "fig4_timeline.png", dpi=130); plt.close()
# Hình 5: đặc trưng tập theo độ phân giải model (toàn bộ file)
pa = profile[profile.split == "tất cả"].set_index("Category"); GC = CATS + ["Tất cả"]
gcol = {**COLS, "Tất cả": "#52514e"}; bw = 0.8 / len(GC)
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for a_, kind in ((ax[0, 0], "seg"), (ax[0, 1], "gap")):
    for i, c in enumerate(GC):
        lab = f"{c} (n={pa.at[c, f'n_{kind}']}" + (f", bị lấp {pa.at[c, 'gap_filled_pp']:.0%}" if kind == "gap" and pd.notna(pa.at[c, "gap_filled_pp"]) else "") + ")"
        a_.bar(np.arange(4) + (i - (len(GC) - 1) / 2) * bw, [pa.at[c, f"{kind}_{k}"] for k in LEN_KEY], bw, color=gcol[c],
               edgecolor="white", linewidth=0.6, label=lab)
    a_.set_xticks(range(4)); a_.set_xticklabels(LEN_CLS); a_.legend(fontsize=7, loc="upper left")
    a_.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax[0, 0].set(title="Độ dài đoạn speech so với cửa sổ 0.96 s", ylabel="% số đoạn")
ax[0, 1].set(title=f"Độ dài khoảng lặng giữa hai đoạn speech (< {FILL_GAP:g} s: pad + merge lấp)", ylabel="% số khoảng lặng")
bot = np.zeros(len(GC))
for k, lab, c_ in (("win_pure_ns", "thuần non-speech", C1), ("win_mixed", "lẫn (0 < tỉ lệ < 1)", C2), ("win_pure_speech", "thuần speech", C3)):
    v = pa.loc[GC, k].to_numpy(float); ax[1, 0].bar(range(len(GC)), v, 0.6, bottom=bot, color=c_, edgecolor="white", linewidth=1.5, label=lab)
    if k == "win_mixed":
        for i, x in enumerate(v):
            if x >= 0.03: ax[1, 0].text(i, bot[i] + x / 2, f"{x:.0%}", ha="center", va="center", fontsize=7, color="white")
    bot += v
ax[1, 0].set_xticks(range(len(GC))); ax[1, 0].set_xticklabels(GC); ax[1, 0].set_ylim(0, 1.13)
ax[1, 0].legend(fontsize=7, loc="upper center", ncol=3); ax[1, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax[1, 0].set(title="Cửa sổ 0.96 s theo tỉ lệ speech của nhãn", ylabel="% cửa sổ")
for j, (mode, c_) in enumerate((("A0", C1), ("A4", C2))):
    v = pa.loc[GC, f"oracle_{mode}_F1"].to_numpy(float)
    ax[1, 1].bar(np.arange(len(GC)) + (j - 0.5) * 0.36, v, 0.36, color=c_, edgecolor="white", linewidth=1, label=f"oracle {mode}")
    for i, x in enumerate(v):
        if np.isfinite(x): ax[1, 1].text(i + (j - 0.5) * 0.36, x + 0.01, f"{x:.2f}", ha="center", fontsize=7)
ax[1, 1].set_xticks(range(len(GC))); ax[1, 1].set_xticklabels(GC); ax[1, 1].set_ylim(0, 1.2)
ax[1, 1].legend(fontsize=7, loc="upper center", ncol=2)
ax[1, 1].set(title="Trần F1: model hoàn hảo cửa sổ 0.96 s (điểm = tỉ lệ speech, ngưỡng 0.5)", ylabel="F1 trên bin 0.5 s")
plt.tight_layout(); plt.savefig(OUT / "fig5_dac_trung_tap.png", dpi=130); plt.close()
# Hình 6: độ dài đoạn speech / khoảng lặng theo giây; nhãn theo bin nên mỗi bước 0.5 s = 1 frame (toàn bộ file)
HCOL = [c for c in seg_hist.columns if c not in ("Category", "kind", "n")]; hx = np.arange(1, len(HCOL) + 1) * BIN
fig, ax = plt.subplots(2, len(GC), figsize=(3.2 * len(GC), 8), sharex=True, sharey=True, squeeze=False)
for i, (kind, ylab) in enumerate((("speech", "% đoạn speech"), ("khoảng lặng", "% khoảng lặng"))):
    for j, c in enumerate(GC):
        a_ = ax[i, j]; L = hist_len[(c, kind)]; n_, tot = len(L), L.sum()
        p_ = np.bincount(len_bucket(L), minlength=len(HCOL)) / max(n_, 1)            # tỉ lệ số đoạn theo độ dài
        t_ = np.bincount(len_bucket(L), L, len(HCOL)) / max(tot, 1e-9)               # tỉ lệ thời lượng theo độ dài
        a_.bar(hx, p_, 0.4, color=gcol[c], edgecolor="white", linewidth=0.6)
        if n_:
            a_.plot(hx, np.cumsum(p_), "o-", c="#0b0b0b", lw=1, ms=2.5, label="% tích luỹ theo số đoạn")
            a_.plot(hx, np.cumsum(t_), "s--", c="#4a3aa7", lw=1.2, ms=2.5, label="% tích luỹ theo thời lượng")
        a_.axvline(BIN, ls="--", c="#8a8984", lw=1, label=f"1 frame = {BIN:g} s")
        a_.axvline(WIN, ls=":", c="#e34948", lw=1.5, label=f"cửa sổ model {WIN:g} s")
        sh = lambda thr: f"{(L < thr - 1e-9).mean():.0%} số đoạn · {L[L < thr - 1e-9].sum() / tot:.0%} thời lượng"
        sub = f"< {WIN:g} s: {sh(WIN)}" if n_ else "không có"
        if kind != "speech":
            a_.axvline(FILL_GAP, ls="-.", c="#eda100", lw=1.2, label=f"pad + merge lấp (< {FILL_GAP:g} s)")
            if n_: sub += f"\nbị lấp: {sh(FILL_GAP)}"
        a_.set_title(f"{c} (n={n_})\n{sub}", fontsize=8.5)
        if i == 0:
            top = a_.secondary_xaxis("top", functions=(lambda s: s / BIN, lambda f: f * BIN))
            top.set_xticks(hx / BIN); top.set_xticklabels([f"{k:g}" for k in hx[:-1] / BIN] + [f"≥{hx[-1] / BIN:g}"], fontsize=7)
            top.set_xlabel(f"số frame {BIN:g} s", fontsize=7)
    ax[i, 0].set_ylabel(ylab)
for a_ in ax[1]:
    a_.set_xticks(hx); a_.set_xticklabels(HCOL, fontsize=7); a_.set_xlabel("độ dài (s)")
ax[0, 0].set_xlim(0.1, hx[-1] + 0.4); ax[0, 0].set_ylim(0, 1.08); ax[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
fig.suptitle(f"Độ dài đoạn speech / khoảng lặng so với frame nhãn {BIN:g} s (1 cột = 1 frame; cột cuối = ≥ {hx[-1]:g} s). "
             "Cột = % số đoạn; thời lượng tính trên tổng thời lượng các đoạn cùng loại", fontsize=11)
fig.legend(*ax[1, -1].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(OUT / "fig6_do_dai_doan.png", dpi=130); plt.close()

# Hình 7: so sánh phương án hậu xử lý (test): Δ F1 và Δ DCF so với B0, CI 95% bootstrap ghép cặp theo file
if len(pp):
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.42 * len(pp) + 1.6), sharey=True)
    yy_ = np.arange(len(pp))[::-1]; lab_ = [f"{r.Mã}  {r.Cấu_hình}" for r in pp.itertuples()]
    for a_, k, good, ttl in ((ax[0], "dF1", 1, "Δ F1 so với B0 (test, cao = tốt)"), (ax[1], "dDCF", -1, f"Δ DCF so với B0 (test, thấp = tốt; miss {A.dcf_miss:g})")):
        v, lo_, hi_ = pp[k].to_numpy(float), pp[f"{k}_CI_lo"].to_numpy(float), pp[f"{k}_CI_hi"].to_numpy(float)
        better, worse = (lo_ > 0, hi_ < 0) if good > 0 else (hi_ < 0, lo_ > 0)      # CI nằm hẳn một phía của 0
        col = np.where(better, C3, np.where(worse, PALETTE[7], "#8a8984"))          # xanh = tốt hơn B0, đỏ = kém hơn, xám = chưa rõ
        a_.hlines(yy_, lo_, hi_, color=col, lw=2.2); a_.scatter(v, yy_, color=col, s=28, zorder=3)
        a_.axvline(0, c="#52514e", lw=0.8); a_.set_title(ttl, fontsize=9)
        for y0, x0 in zip(yy_, v): a_.text(x0, y0 + 0.22, f"{x0:+.3f}", ha="center", fontsize=7, color="#3b3a37")
    ax[0].set_yticks(yy_); ax[0].set_yticklabels(lab_, fontsize=8)
    fig.suptitle(f"So sánh phương án hậu xử lý trên test ({len(test)} file); ngưỡng chọn riêng cho từng cấu hình trên dev ({A.criterion}). "
                 f"B0: F1 {pp.F1.iloc[0]:.3f}, DCF {pp.DCF.iloc[0]:.3f}\n"
                 "Vạch = CI 95% bootstrap ghép cặp theo file. Xanh = tốt hơn B0, đỏ = kém hơn B0, xám = chưa phân biệt được", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "fig7_so_sanh_hau_xu_ly.png", dpi=130); plt.close()

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("Dữ liệu:", json.dumps(S["meta"], ensure_ascii=False, indent=1))
print(json.dumps({k: S[k] for k in ["n_pairs", "problems", "n_struct_flag", "struct_flag_files", "best_shift_center", "best_shift_auc",
      "lag_median", "hours", "pct_speech_bins", "n_degenerate", "auc_pooled_all", "auc_macro", "auc_worst", "auc_worst_file",
      "n_files_auc_lt_09", "kappa_l2", "kappa_l2_boundary", "kappa_l2_interior", "pct_bins_khac_alt", "n_dev", "n_test", "AUC_test_CI",
      "n_synth_val", "auc_synth_val", "thr_youden_raw_dev", "best_thr_dev", "best_thr_synth", "best_f1_thr_dev", "thr_at_grid_edge",
      "event_f1", "err_types", "pct_true_err", "F1_CI", "DCF_CI", "mean_d_onset", "mean_d_offset", "n_suspicious_seg"]},
      ensure_ascii=False, indent=1, default=float)[:6000])
for name, df in [("Nhất quán nhãn", lab_cat), ("Tầng A (nhãn chính)", tA), ("Tầng A (nhãn --gt-alt)", tA_alt),
                 ("Ngưỡng: chọn ở đâu -> kết quả trên test", transfer), ("Tầng B (A4)", tB), ("Chỉ số theo đoạn (test, A4)", ev_thr),
                 ("Ablation", abl), ("Category (test, A4)", cat_df),
                 ("So sánh phương án hậu xử lý (test; Δ so với B0, CI 95% bootstrap)",
                  pp[["Mã", "Cấu_hình", "Ngưỡng_dev", "MR", "FAR", "F1", "F1_nb", "DCF", "AUC_hop_bin", "event_F1", "n_pred_seg",
                      "dF1", "dF1_CI_lo", "dF1_CI_hi", "dDCF"]] if len(pp) else None),
                 ("Đặc trưng tập theo độ phân giải model (tất cả file)", profile[profile.split == "tất cả"][["Category", "n_file", "minutes"] + PROF_KEYS])]:
    if df is not None and len(df): print(f"\n=== {name} ==="); print(df.round(4).to_string(index=False))
print("\nĐặc trưng lệch dev/test > 10 điểm %:", "; ".join(flags) if flags else "(không có)")
print("\nTop đoạn nghi nhãn sai:"); print(sus.head(15).to_string(index=False) if len(sus) else "(không có)")
