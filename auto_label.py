"""Gán nhãn speech/non-speech tự động (bin 0.5s) cho data/audio/<Cat>/*.wav.

Model:
  - FireRedVAD AED (nhãn chính): xác suất speech / singing / music mỗi frame 10ms
  - Silero VAD (đối chiếu)     : xác suất speech mỗi chunk 32ms
Quy tắc nhãn:
  - bin là speech nếu >=50% thời lượng bin nằm trong đoạn speech của AED (giống segs_to_bins)
  - nói trên nền nhạc -> speech; singing mà không có speech -> non-speech
  - thì thầm (ASMR) -> speech
Cần nghe lại (review.csv; các bin liền nhau được gom thành 1 đoạn):
  prio 1  babble                 : file Babble có bin speech hoặc nghi ngờ speech
          speech_va_singing      : bin speech nhưng AED cũng thấy singing (có thể là hát bị nhận nhầm)
          lech_silero            : AED và Silero cho nhãn khác nhau
  prio 2  silero_nghe_hat        : Silero báo speech, AED báo singing (theo quy ước là non-speech)
          silero_bo_sot_thi_tham : Asm, AED báo speech, Silero không (Silero hay bỏ sót giọng thì thầm)
          aed_khong_chac         : xác suất speech trung bình của bin sát ngưỡng
Đầu ra (--out, mặc định data/auto_labels):
  Groundtruth/<Cat>/groundtruth/Gt_<cat>_<id>.txt : start<TAB>end<TAB>speech|non-speech
  review.csv, summary.csv
  cache/*.npz : xác suất theo frame -> đổi ngưỡng chạy lại không cần chạy model

Cài đặt:
    pip install fireredvad silero-vad huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download('FireRedTeam/FireRedVAD', local_dir='pretrained_models/FireRedVAD')"

Ví dụ:
    python auto_label.py                        # toàn bộ data/audio
    python auto_label.py -c Babble Music --limit 3
    python auto_label.py --speech-thr 0.5       # dùng lại cache, chỉ tính lại nhãn
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).parent
SR, BIN, FRAME, SIL_CHUNK = 16000, 0.5, 0.01, 512
EVENTS = ("speech", "singing", "music")
WHISPER_CATS = {"asm"}  # thì thầm tính là speech; Silero hay bỏ sót nên không dùng làm trọng tài


# ---------------- model ----------------
class Models:
    """Chỉ nạp model khi có file chưa có cache."""
    def __init__(self, aed_dir, use_silero, gpu):
        self.aed_dir, self.use_silero, self.gpu = aed_dir, use_silero, gpu
        self.aed = self.sil = None

    def run(self, wav):
        import torch
        if self.aed is None:
            from fireredvad import FireRedAed, FireRedAedConfig
            self.aed = FireRedAed.from_pretrained(str(self.aed_dir), FireRedAedConfig(use_gpu=self.gpu))
            if self.use_silero:
                from silero_vad import load_silero_vad
                self.sil = load_silero_vad()
        _, probs = self.aed.detect((wav, SR))
        sil = np.zeros(0, np.float32)
        if self.sil is not None:
            self.sil.reset_states()
            sil = self.sil.audio_forward(torch.from_numpy(wav.astype(np.float32) / 32768), SR)[0].numpy()
        return probs.numpy().astype(np.float32), sil.astype(np.float32)


def aed_decisions(p, thr):
    """Hậu xử lý của FireRedVAD (làm mượt + min speech/silence 0.2s), không cắt đoạn dài."""
    from fireredvad.core.vad_postprocessor import VadPostprocessor
    pp = VadPostprocessor(smooth_window_size=5, prob_threshold=thr, min_speech_frame=20,
                          max_speech_frame=10**9, min_silence_frame=20,
                          merge_silence_frame=0, extend_speech_frame=0)
    return np.asarray(pp.process(p.tolist()), float)


def per_bin(x, n_bins, step):
    """Trung bình giá trị theo frame (bước `step` giây) trong mỗi bin 0.5s, theo tâm frame."""
    if len(x) == 0: return np.full(n_bins, np.nan)
    k = np.floor((np.arange(len(x)) + 0.5) * step / BIN).astype(int); ok = k < n_bins
    return np.bincount(k[ok], x[ok], n_bins) / np.maximum(np.bincount(k[ok], minlength=n_bins), 1)


# ---------------- gán nhãn + lý do cần nghe lại ----------------
def label_file(aed_p, sil_p, dur, cat, a):
    n = int(dur / BIN + 1e-9)
    cov = per_bin(aed_decisions(aed_p[:, 0], a.speech_thr), n, FRAME)
    sing = per_bin(aed_decisions(aed_p[:, 1], a.singing_thr), n, FRAME) >= 0.5
    music = per_bin(aed_decisions(aed_p[:, 2], a.music_thr), n, FRAME) >= 0.5
    ps, pg, pm = (per_bin(aed_p[:, i], n, FRAME) for i in range(3))
    y = cov >= 0.5 - 1e-9
    has_sil = len(sil_p) > 0
    sil = per_bin((sil_p >= a.silero_thr).astype(float), n, SIL_CHUNK / SR) >= 0.5 - 1e-9 if has_sil else y

    reasons = [[] for _ in range(n)]
    for k in range(n):
        r = reasons[k]
        if cat.lower() == "babble" and (y[k] or ps[k] >= a.suspect_thr or sil[k]): r.append("babble")
        if y[k] and sing[k]: r.append("speech_va_singing")
        if y[k] != sil[k]:
            if sil[k] and sing[k]: r.append("silero_nghe_hat")
            elif y[k] and cat.lower() in WHISPER_CATS: r.append("silero_bo_sot_thi_tham")
            else: r.append("lech_silero")
        if abs(ps[k] - a.speech_thr) < a.margin: r.append("aed_khong_chac")
    bins = pd.DataFrame(dict(start=np.arange(n) * BIN, end=(np.arange(n) + 1) * BIN, speech=y, silero=sil,
                             singing=sing, music=music, aed_speech=ps, aed_singing=pg, aed_music=pm,
                             silero_p=per_bin(sil_p, n, SIL_CHUNK / SR), reasons=reasons))
    return bins


PRIO2 = {"silero_nghe_hat", "silero_bo_sot_thi_tham", "aed_khong_chac"}

def review_segments(bins, file, cat):
    """Gom các bin cần nghe lại liền nhau thành đoạn."""
    out, cur = [], None
    for k, row in bins.iterrows():
        if not row.reasons:
            cur = None; continue
        if cur is None:
            cur = dict(file=file, category=cat, start=row.start, idx=[], reasons=set())
            out.append(cur)
        cur["end"] = row.end; cur["idx"].append(k); cur["reasons"] |= set(row.reasons)
    rows = []
    for s in out:
        b = bins.loc[s["idx"]]
        rows.append(dict(file=s["file"], category=s["category"], start=s["start"], end=s["end"],
                         n_bins=len(b), n_speech=int(b.speech.sum()),
                         prio=1 if s["reasons"] - PRIO2 else 2, reasons="+".join(sorted(s["reasons"])),
                         aed_speech=b.aed_speech.mean(), aed_singing=b.aed_singing.mean(),
                         aed_music=b.aed_music.mean(), silero_p=b.silero_p.mean()))
    return rows


def file_note(bins, cat):
    y, notes = bins.speech.to_numpy(), []
    if y.all(): notes.append("toàn speech")
    if not y.any(): notes.append("toàn non-speech")
    if cat.lower() == "babble":
        k = sum("babble" in r for r in bins.reasons)
        if k: notes.append(f"Babble: {k} bin speech/nghi ngờ -> nghe lại")
    agree = (bins.speech == bins.silero).mean()
    if agree < 0.8: notes.append(f"AED-Silero chỉ khớp {agree:.0%}")
    return "; ".join(notes)


def write_gt(bins, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{s:.1f}\t{e:.1f}\t{'speech' if y else 'non-speech'}"
             for s, e, y in zip(bins.start, bins.end, bins.speech)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser(description="Gán nhãn VAD tự động bằng FireRedVAD AED + Silero")
    p.add_argument("-a", "--audio", type=Path, default=HERE / "data" / "audio")
    p.add_argument("-o", "--out", type=Path, default=HERE / "data" / "auto_labels")
    p.add_argument("-c", "--categories", nargs="*", help="Chỉ chạy các category này")
    p.add_argument("--limit", type=int, default=0, help="Tối đa N file mỗi category (để thử)")
    p.add_argument("--aed-dir", type=Path, default=HERE / "pretrained_models" / "FireRedVAD" / "AED")
    p.add_argument("--speech-thr", type=float, default=0.4)
    p.add_argument("--singing-thr", type=float, default=0.5)
    p.add_argument("--music-thr", type=float, default=0.5)
    p.add_argument("--silero-thr", type=float, default=0.5)
    p.add_argument("--suspect-thr", type=float, default=0.2, help="Babble: xác suất speech >= ngưỡng này là nghi ngờ")
    p.add_argument("--margin", type=float, default=0.1, help="aed_khong_chac: |p_speech - speech_thr| < margin")
    p.add_argument("--no-silero", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--force", action="store_true", help="Bỏ cache, chạy lại model")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    logging.getLogger("fireredvad").setLevel(logging.ERROR)

    cats = [d for d in sorted(a.audio.iterdir()) if d.is_dir()]
    if a.categories: cats = [d for d in cats if d.name in a.categories]
    files = [(d.name, f) for d in cats for f in sorted(d.glob("*.wav"))[: a.limit or None]]
    models, cache = Models(a.aed_dir, not a.no_silero, a.gpu), a.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    reviews, summary = [], []
    for i, (cat, f) in enumerate(files, 1):
        cp = cache / f"{f.stem}.npz"
        if cp.exists() and not a.force:
            z = np.load(cp); aed_p, sil_p, dur = z["aed"], z["silero"], float(z["dur"])
        else:
            wav, sr = sf.read(f, dtype="int16")
            if sr != SR or wav.ndim != 1:
                print(f"[BỎ QUA] {f.name}: cần 16kHz mono (đang {sr}Hz, {wav.ndim} kênh)"); continue
            dur = len(wav) / SR
            aed_p, sil_p = models.run(wav)
            np.savez_compressed(cp, aed=aed_p, silero=sil_p, dur=dur)

        bins = label_file(aed_p, sil_p, dur, cat, a)
        uid = f.stem.split("_", 1)[1] if "_" in f.stem else f.stem
        write_gt(bins, a.out / "Groundtruth" / cat / "groundtruth" / f"Gt_{cat.lower()}_{uid}.txt")
        rv = review_segments(bins, f.name, cat); reviews += rv
        n_rev = sum(r["n_bins"] for r in rv)
        summary.append(dict(file=f.name, category=cat, dur=round(dur, 2), n_bins=len(bins),
                            speech_ratio=bins.speech.mean(), silero_ratio=bins.silero.mean(),
                            agree=(bins.speech == bins.silero).mean(),
                            singing_ratio=bins.singing.mean(), music_ratio=bins.music.mean(),
                            review_bins=n_rev, review_prio1_bins=sum(r["n_bins"] for r in rv if r["prio"] == 1),
                            note=file_note(bins, cat)))
        print(f"[{i}/{len(files)}] {f.name}: speech {bins.speech.mean():.0%}, "
              f"silero {bins.silero.mean():.0%}, cần nghe lại {n_rev}/{len(bins)} bin")

    rv = pd.DataFrame(reviews)
    if len(rv): rv = rv.sort_values(["prio", "category", "file", "start"])
    rv.to_csv(a.out / "review.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    sm = pd.DataFrame(summary)
    sm.to_csv(a.out / "summary.csv", index=False, float_format="%.3f", encoding="utf-8-sig")

    if len(sm):
        g = sm.groupby("category").agg(files=("file", "size"), speech=("speech_ratio", "mean"),
                                       agree=("agree", "mean"), bins=("n_bins", "sum"),
                                       review=("review_bins", "sum"), prio1=("review_prio1_bins", "sum"))
        g["review_%"] = g.review / g.bins
        print("\n" + g.to_string(float_format=lambda x: f"{x:.2f}"))
        print(f"\nTổng: {g.review.sum()}/{g.bins.sum()} bin cần nghe lại ({g.prio1.sum()} bin prio 1)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
