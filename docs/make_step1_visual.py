"""Hình giải thích trực quan Bước 1 (tri / hann / logit / median). Chạy: python docs/make_step1_visual.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from vadlib import rescore_hop
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
H, W = 0.08, 0.96; DUR = 7.0
# điểm cửa sổ mô phỏng: lời nói 1.0–3.2 s, ho 4.3–4.55 s (ngắn), từ ngắn "ok" 5.2–5.5 s; mô hình lệch 1 cửa sổ sai ở giữa lời nói
st = np.arange(0, DUR - W + 1e-9, H); ctr = st + W / 2
def truth(t): return ((t > 1.0) & (t < 3.2)) | ((t > 4.3) & (t < 4.55)) | ((t > 5.2) & (t < 5.5))
rng = np.random.default_rng(3)
raw = np.where(truth(ctr), 0.93, 0.05) + rng.normal(0, 0.03, len(ctr))
# cửa sổ có tâm nằm trong 0.24 s cũng "thấy" tiếng ho/từ ngắn kém mạnh hơn
raw[(ctr > 4.3) & (ctr < 4.55)] = 0.80; raw[(ctr > 5.2) & (ctr < 5.5)] = 0.85
bad = np.argmin(abs(ctr - 2.1)); raw[bad] = 0.004      # 1 cửa sổ cực tự tin nhưng sai
raw = np.clip(raw, 0.002, 0.998)
m = pd.DataFrame({"start": st, "score": raw}); T = (np.arange(int(round(DUR / H))) + .5) * H
kw = dict(dur=DUR, smooth=1)
res = {"Trung bình đều (tham chiếu)": rescore_hop(m, weight="mean", **kw),
       "Tam giác (hiện tại)": rescore_hop(m, weight="tri", **kw),
       "Hann": rescore_hop(m, weight="hann", **kw),
       "Tam giác – miền logit": rescore_hop(m, weight="tri", domain="logit", **kw),
       "Median 12 cửa sổ": rescore_hop(m, weight="median", **kw)}
col = ["#9aa0a6", "#1a73e8", "#34a853", "#f29900", "#d93025"]
fig = plt.figure(figsize=(13, 11)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.05, 1.5], hspace=.5, wspace=.25)
# A. trọng số
ax = fig.add_subplot(gs[0, 0]); d = np.linspace(0, .48, 200)
ax.plot(d, 1 - d / .48, color=col[1], lw=2.5, label="Tam giác  1 − d/0.48")
ax.plot(d, .5 * (1 + np.cos(np.pi * d / .48)), color=col[2], lw=2.5, label="Hann  0.5·(1+cos(πd/0.48))")
ax.axhline(1, color=col[0], ls="--", lw=1.5, label="Trung bình đều")
ax.set(xlabel="d = khoảng cách từ tâm cửa sổ tới hop (s)", ylabel="trọng số", title="A. Trọng số của từng cửa sổ"); ax.legend(frameon=False, fontsize=8.5)
ax.annotate("Hann phẳng hơn\nở gần tâm", (.07, .93), (.02, .52), arrowprops=dict(arrowstyle="->"), fontsize=8.5)
ax.annotate("Hann dập đuôi\nmạnh hơn", (.38, .15), (.31, .62), arrowprops=dict(arrowstyle="->"), fontsize=8.5)
# B. logit
ax = fig.add_subplot(gs[0, 1]); p = np.linspace(.001, .999, 400)
ax.plot(p, np.log(p / (1 - p)), color=col[3], lw=2.5); ax.axhline(0, color="#ccc", lw=1)
ax.set(xlabel="xác suất p", ylabel="logit z = ln(p/(1−p))", title="B. Miền logit kéo giãn hai đầu")
for pp, c in [(.9, col[1]), (.01, col[4])]:
    ax.plot(pp, np.log(pp / (1 - pp)), "o", color=c, ms=8)
ax.annotate("p=0.01 → z=−4.6\n(rất 'nặng')", (.01, -4.6), (.12, -4.9), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
ax.annotate("p=0.90 → z=+2.2", (.9, 2.2), (.5, 1.2), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
ax.text(.03, 5.9, "Trung bình xác suất của 0.90 và 0.01 = 0.455\nTrung bình logit rồi đổi lại = 0.23", fontsize=8.5, va="top", bbox=dict(fc="#fff8e1", ec="#f29900"))
ax.set_ylim(-6.5, 6.2)
# C. 12 cửa sổ quanh 1 hop
ax = fig.add_subplot(gs[1, :]); t0 = 4.40; k = (st <= t0) & (t0 < st + W)
cc, ss = ctr[k], raw[k]; dd = abs(t0 - cc)
ax.bar(cc, ss, width=.07, color=[col[1] if v > .5 else "#bbb" for v in ss])
ax.axvline(t0, color="k", ls=":"); ax.text(t0+.01, 1.12, "hop đang xét", fontsize=9)
ax.axvspan(4.3, 4.55, color="#fde7e9", zorder=0); ax.text(4.425, .88, "tiếng ho", ha="center", fontsize=9, color=col[4])
ax.set(ylim=(0, 1.2), xlabel="tâm cửa sổ (s)", ylabel="điểm cửa sổ", title="C. Ví dụ: 12 cửa sổ cùng phủ 1 hop trong tiếng ho 0.25 s")
vals = {"mean": ss.mean(), "tam giác": np.sum((1 - dd / .48) * ss) / np.sum(1 - dd / .48), "median": np.median(ss)}
ax.text(.995, .97, "  ".join(f"{a} = {v:.2f}" for a, v in vals.items()) + "\n→ median xóa tiếng ho (nhưng cũng xóa từ ngắn thật)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(fc="white", ec="#999"))
# D. kết quả theo thời gian
ax = fig.add_subplot(gs[2, :])
ax.axhline(.6, color="k", ls="--", lw=1); ax.text(.02, .615, "ngưỡng 0.60", fontsize=8.5)
for y in [(1.0, 3.2, "lời nói"), (4.3, 4.55, "ho (không phải speech)"), (5.2, 5.5, "từ 'ok' (speech thật)")]:
    ax.axvspan(y[0], y[1], color="#e8f0fe" if "ok" in y[2] or y[2] == "lời nói" else "#fde7e9", zorder=0)
    ax.text((y[0] + y[1]) / 2, 1.07 if "ok" not in y[2] else 1.0, y[2].replace(" (","\n("), ha="center", fontsize=8.5)
ax.plot(ctr, raw, ".", color="#bbb", ms=4, label="điểm gốc từng cửa sổ")
for (n, v), c in zip(res.items(), col): ax.plot(T, v, color=c, lw=2.3 if "Tam giác (" in n else 1.8, label=n)
ax.set(ylim=(-.03, 1.15), xlabel="thời gian (s)", ylabel="điểm mỗi hop 0.08 s", title="D. Cùng dữ liệu, 5 cách gộp (chưa làm mượt bước 2)")
ax.legend(ncol=3, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(.5,-.18))
fig.suptitle("Bước 1 – Gộp điểm các cửa sổ chồng lấn: so sánh trực quan", fontsize=14, fontweight="bold", y=.97)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "22_giai_thich_buoc1.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print(out)
for n, v in res.items(): print(f"{n:32s} ho={v[int(4.4/H)]:.2f}  ok={v[int(5.35/H)]:.2f}  noi_2.1s={v[int(2.1/H)]:.2f}")
