"""Điền kết quả evaluate_real.py vào template tổng hợp VAD; mọi chỉ số tính bằng CÔNG THỨC Excel.

evaluate_real.py xuất số đếm thô (results_real/excel/): TP/FN/FP/TN theo ngưỡng × cấu hình × category × vùng bin,
số đếm ROC, bảng nhãn 2×2, độ lệch biên. Script này:
  - ghi số đếm vào các sheet "DL ..." (chữ xanh = số nhập từ Python, không sửa tay)
  - sheet "KQ du lieu that": MR, FAR, F1, DCF, AUC, κ, chọn ngưỡng... là công thức; tham số ở ô vàng B6:B10
    (trọng số DCF, cấu hình, tiêu chí chọn ngưỡng, mốc FPR) -> đổi là mọi bảng tính lại
  - sheet "Hinh du lieu that": 4 hình vẽ bằng Python (results_real/fig*.png của evaluate_real.py)
  - các sheet của template lấy số bằng công thức từ KQ du lieu that; chỉ ghi ô vàng/xám, không sửa công thức
    sẵn có; ô xám (ví dụ) đã thay bằng số thật đổi sang nền vàng
Giữ là số từ Python (chữ xanh, có ghi chú): AUC chính xác, CI bootstrap, event-F1, Δbiên, lag.

    python make_excel_real.py   # results/Tong_hop_danh_gia_VAD_template.xlsx -> results/Tong_hop_danh_gia_VAD_baseline_real.xlsx
"""
import argparse, hashlib, json, wave
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.worksheet.datavalidation import DataValidation

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--template", type=Path, default=Path("results/Tong_hop_danh_gia_VAD_template.xlsx"), help="template trống (không bị ghi)")
ap.add_argument("--out", type=Path, default=Path("results/Tong_hop_danh_gia_VAD_baseline_real.xlsx"))
ap.add_argument("--res", type=Path, default=Path("results_real"))
ap.add_argument("--audio", type=Path, default=Path("data/audio"))
ap.add_argument("--run", type=Path, default=Path("C:/vad_work/run1"))
ap.add_argument("--yamnet", type=Path, default=Path("C:/vad_work/models/yamnet_embed_float32.tflite"))
a = ap.parse_args()
assert a.template.resolve() != a.out.resolve(), "--out phải khác --template (template luôn giữ trống)"
R, XD = a.res, a.res / "excel"
S = json.load(open(R / "summary.json", encoding="utf-8"))
pf = pd.read_csv(R / "per_file.csv").sort_values(["category", "file"]).reset_index(drop=True)
sw, roc, labc = pd.read_csv(XD / "sweep_counts.csv"), pd.read_csv(XD / "roc_counts.csv"), pd.read_csv(XD / "label_counts.csv")
dls, grid, shift = pd.read_csv(XD / "deltas.csv"), pd.read_csv(R / "grid_dev.csv"), pd.read_csv(R / "shift_sweep.csv")
tA, cat, evt = pd.read_csv(R / "tierA.csv"), pd.read_csv(R / "category.csv"), pd.read_csv(R / "event_by_threshold.csv")
sus, struct = pd.read_csv(R / "suspicious.csv"), pd.read_csv(R / "structure.csv")
THR = S["thr_used"]; TODAY = date.today().strftime("%d/%m/%Y")
SRC = f"Nguồn: {R.as_posix()}/ (evaluate_real.py, {TODAY}). Python reference trên 144 file audio thật, nhãn bạc FireRedVAD; chưa phải output lib."

F = "Arial"; NAVY, BLUE = "1F4E78", "0000FF"
YEL, HDR = PatternFill("solid", fgColor="FFF2CC"), PatternFill("solid", fgColor=NAVY)
thin = Side(style="thin", color="BFBFBF"); BD = Border(left=thin, right=thin, top=thin, bottom=thin)
FN_, FB_ = Font(name=F, size=10), Font(name=F, size=10, color=BLUE)          # công thức / số nhập từ Python
CATS = ["Asm", "Babble", "Clean", "Music", "Noise"]
P, D3 = "0.0%", "0.000"

wb = load_workbook(a.template)
NEW = ["KQ du lieu that", "Hinh du lieu that", "DL quet nguong", "DL ROC", "DL theo file", "DL nhan va phan bo", "DL do lech bien"]
for n in NEW + ["Bieu do du lieu that", "DL doan va timeline"]:
    if n in wb.sheetnames: del wb[n]
kq, hd, dq, dr, dfs, dn, dt = (wb.create_sheet(n) for n in NEW)
KQ, DQ, DR, DF, DN, DT = (f"'{s.title}'" for s in (kq, dq, dr, dfs, dn, dt))   # tên sheet có dấu cách -> để trong nháy


# ---------------- helpers ----------------
def val(v):
    if isinstance(v, (float, np.floating)): return None if np.isnan(v) else float(v)
    return v.item() if isinstance(v, np.generic) else v

def w(ws, r, c, v, fmt=None, blue=False, bold=False, fill=None, left=False):
    """Ô bảng kết quả: có viền."""
    x = ws.cell(r, c, val(v))
    x.font = Font(name=F, size=10, bold=True, color=BLUE if blue else None) if bold else (FB_ if blue else FN_)
    if fmt: x.number_format = fmt
    if fill: x.fill = fill
    x.border = BD; x.alignment = Alignment(horizontal="left" if left else "center", vertical="top", wrap_text=left)
    return x

def raw(ws, r, c, v, fmt=None, blue=True):
    """Ô dữ liệu thô: không viền, để sheet DL nhẹ."""
    x = ws.cell(r, c, val(v)); x.font = FB_ if blue else FN_
    if fmt: x.number_format = fmt
    return x

def head(ws, r, names, c0=1):
    for j, n in enumerate(names):
        x = ws.cell(r, c0 + j, n); x.font = Font(name=F, size=10, bold=True, color="FFFFFF"); x.fill = HDR; x.border = BD
        x.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def title(ws, t, sub):
    ws["A1"] = t; ws["A1"].font = Font(name=F, size=14, bold=True, color=NAVY)
    ws["A2"] = sub; ws["A2"].font = Font(name=F, size=9, color="595959"); ws.sheet_view.showGridLines = False

def sec(ws, r, t): ws.cell(r, 1, t).font = Font(name=F, size=11, bold=True, color=NAVY)

def note(x, t, wd=320, ht=140): x.comment = Comment(t, "make_excel_real.py"); x.comment.width, x.comment.height = wd, ht

def rg(sh, col, r0, r1): return f"{sh}!${col}${r0}:${col}${r1}"

def rates(tp, fn, fp, tn):
    """Công thức tỉ lệ từ 4 ô số đếm; DCF dùng trọng số ở KQ!B6."""
    return dict(MR=f'=IF({tp}+{fn}=0,"",{fn}/({tp}+{fn}))', FAR=f'=IF({fp}+{tn}=0,"",{fp}/({fp}+{tn}))',
                Prec=f'=IF({tp}+{fp}=0,"",{tp}/({tp}+{fp}))', Rec=f'=IF({tp}+{fn}=0,"",{tp}/({tp}+{fn}))',
                F1=f'=IF(2*{tp}+{fn}+{fp}=0,"",2*{tp}/(2*{tp}+{fn}+{fp}))', Spec=f'=IF({fp}+{tn}=0,"",{tn}/({fp}+{tn}))')

def dcf(mr, far): return f'=IF(OR({mr}="",{far}=""),"",{KQ}!$B$6*{mr}+(1-{KQ}!$B$6)*{far})'


# ======================= DL theo file =======================
title(dfs, "Số liệu theo file (144 file)", "Chữ xanh = số nhập từ evaluate_real.py (per_file.csv). Cột P–Q là công thức.")
fcols = [("file", "File", None), ("category", "Category", None), ("split", "Tập", None), ("dur_s", "Thời lượng (s)", "0.0"),
         ("speech_ratio", "% speech (chính sách)", P), ("speech_ratio_alt", "% speech (FireRedVAD gốc)", P),
         ("silero_ratio", "% speech (Silero)", P), ("agree_aed_silero", "Khớp FireRedVAD–Silero", P),
         ("transitions", "#chuyển trạng thái", None), ("n_speech_seg", "#đoạn speech", None), ("boundary_frac", "% bin biên", P),
         ("auc_file", "AUC file", D3), ("lag_s", "Lag (s)", "0.00"), ("peak_corr", "Tương quan đỉnh", D3), ("mean_score", "Điểm TB", D3)]
head(dfs, 4, [c[1] for c in fcols] + ["1 lớp? (CT)", "Cực đoan? (CT)"])
DF0, DF1 = 5, 4 + len(pf)
for i, row in enumerate(pf.itertuples(index=False), DF0):
    for j, (k, _, fmt) in enumerate(fcols, 1): raw(dfs, i, j, getattr(row, k), fmt)
    raw(dfs, i, 16, f"=IF(OR(E{i}=0,E{i}=1),1,0)", blue=False); raw(dfs, i, 17, f"=IF(OR(E{i}<0.1,E{i}>0.9),1,0)", blue=False)
dfs.column_dimensions["A"].width = 24
for c in range(2, 18): dfs.column_dimensions[L(c)].width = 12
dfs.freeze_panes = "B5"
def DFc(col): return rg(DF, col, DF0, DF1)

# ======================= DL quet nguong =======================
title(dq, "Số đếm theo ngưỡng", "Tập: dev = dev thật (43 file), synth = val tổng hợp (500 file), test = test thật theo category. "
      "Vùng bin: all = mọi bin, nb = bỏ bin biên ±1, b = chỉ bin biên. Chữ xanh = số đếm từ Python; MR…DCF (chỉ dev/synth) là công thức.")
head(dq, 4, ["Tập", "Cấu hình", "Category", "Vùng bin", "Ngưỡng", "TP", "FN", "FP", "TN", "FEC", "MSC", "OVER", "NDS",
             "MR", "FAR", "Precision", "F1", "DCF"])
sw["o1"] = sw.tap.map({"dev": 0, "synth": 1, "test": 2}); sw["o2"] = sw["mask"].map({"all": 0, "nb": 1, "b": 2})
sw["o3"] = sw.category.map({c: i for i, c in enumerate(["Tất cả"] + CATS)})
sw = sw.sort_values(["o1", "config", "o3", "o2", "thr"]).reset_index(drop=True)
blocks = {}
for i, row in enumerate(sw.itertuples(index=False), 5):
    for j, k in enumerate(["tap", "config", "category", "mask", "thr", "TP", "FN", "FP", "TN", "FEC", "MSC", "OVER", "NDS"], 1):
        raw(dq, i, j, getattr(row, k), "0.00" if k == "thr" else None)
    if row.tap in ("dev", "synth"):
        f = rates(f"F{i}", f"G{i}", f"H{i}", f"I{i}")
        for j, k in zip(range(14, 18), ["MR", "FAR", "Prec", "F1"]): raw(dq, i, j, f[k], D3 if k == "F1" else P, blue=False)
        raw(dq, i, 18, dcf(f"N{i}", f"O{i}"), D3, blue=False)
        blocks[(row.tap, row.config)] = (blocks.get((row.tap, row.config), (i, i))[0], i)
DQ0, DQ1 = 5, 4 + len(sw)
for c in range(1, 19): dq.column_dimensions[L(c)].width = 10
dq.freeze_panes = "F5"

def sif(col, config, thr, mask, cat=None):
    """SUMIFS trên số đếm test; config/thr/mask/cat là địa chỉ ô hoặc chuỗi đã có nháy."""
    s = (f'=SUMIFS({rg(DQ, col, DQ0, DQ1)},{rg(DQ, "A", DQ0, DQ1)},"test",{rg(DQ, "B", DQ0, DQ1)},{config},'
         f'{rg(DQ, "D", DQ0, DQ1)},{mask},{rg(DQ, "E", DQ0, DQ1)},{thr}')
    return s + (f',{rg(DQ, "C", DQ0, DQ1)},{cat})' if cat else ")")

# ======================= DL ROC =======================
RAWG = CATS + ["Dev"]; GROUPS = RAWG + ["Test", "Tất cả"]
cc = {g: 2 + 4 * i for i, g in enumerate(GROUPS)}                     # cột TP của nhóm (TP, FP, FN, TN)
mc = {g: 2 + 4 * len(GROUPS) + 3 * i for i, g in enumerate(GROUPS)}   # cột TPR của nhóm (TPR, FPR, Precision)
title(dr, "Số đếm ROC trên điểm thô (tầng A)", f"{len(roc)} ngưỡng = phân vị điểm của 144 file + 2 ngưỡng chặn; speech khi điểm ≥ ngưỡng. "
      "Test theo category và Dev gộp là số từ Python (chữ xanh); nhóm Test, Tất cả và mọi tỉ lệ là công thức.")
head(dr, 4, ["Ngưỡng"] + [f"{g} {k}" for g in GROUPS for k in ("TP", "FP", "FN", "TN")]
     + [f"{g} {k}" for g in GROUPS for k in ("TPR", "FPR", "Precision")])
DR0, DR1 = 5, 4 + len(roc)
for i, row in enumerate(roc.itertuples(index=False), DR0):
    raw(dr, i, 1, row.thr, "0.00000")
    for g in RAWG:
        for k, st in enumerate(("TP", "FP", "FN", "TN")): raw(dr, i, cc[g] + k, getattr(row, f"{g}_{st}"))
    for k in range(4):
        raw(dr, i, cc["Test"] + k, "=" + "+".join(f"{L(cc[c] + k)}{i}" for c in CATS), blue=False)
        raw(dr, i, cc["Tất cả"] + k, f"={L(cc['Test'] + k)}{i}+{L(cc['Dev'] + k)}{i}", blue=False)
    for g in GROUPS:
        tp, fp, fn, tn = (f"{L(cc[g] + k)}{i}" for k in range(4))
        for k, fml in enumerate([f"={tp}/({tp}+{fn})", f"={fp}/({fp}+{tn})", f"=IF({tp}+{fp}=0,1,{tp}/({tp}+{fp}))"]):
            raw(dr, i, mc[g] + k, fml, D3, blue=False)
dr.column_dimensions["A"].width = 10; dr.freeze_panes = "B5"
def rocc(g, k, r0=DR0, r1=DR1): return rg(DR, L(mc[g] + k), r0, r1)   # k: 0 TPR, 1 FPR, 2 Precision

# ======================= DL nhan va phan bo =======================
title(dn, "Nhãn, độ dịch thời gian, lưới hậu xử lý", "Chữ xanh = số từ Python; chữ đen = công thức.")
r = 4; sec(dn, r, "a. Bảng 2×2 FireRedVAD (quy tắc gốc) vs Silero, theo category × vùng bin (144 file)")
head(dn, r + 1, ["Category", "Vùng bin", "n11 (cả hai speech)", "n10 (chỉ FireRedVAD)", "n01 (chỉ Silero)", "n00",
                 "#bin đổi do hát", "#bin nhãn chắc", "#bin speech (chính sách)"])
LB0 = r + 2; LB1 = LB0 + len(labc) - 1
for i, row in enumerate(labc.itertuples(index=False), LB0):
    for j, k in enumerate(["category", "mask", "n11", "n10", "n01", "n00", "bin_doi_do_hat", "n_nhan_chac", "n_speech_policy"], 1):
        raw(dn, i, j, getattr(row, k))
r = LB1 + 2; sec(dn, r, "b. Quét độ dịch điểm (score đặt tại tâm cửa sổ)")
head(dn, r + 1, ["Shift (s)", "F1 start (thr 0.5)", "F1 tâm (thr 0.5)", "AUC thật – FireRedVAD", "AUC thật – Silero",
                 "AUC val tổng hợp", "ΔAUC FireRedVAD", "ΔAUC Silero", "ΔAUC tổng hợp"])
SH0 = r + 2; SH1 = SH0 + len(shift) - 1
for i, row in enumerate(shift.itertuples(index=False), SH0):
    for j, k in enumerate(["shift_s", "F1_start", "F1_center", "AUC_real_fireredvad", "AUC_real_silero", "AUC_synth_exact"], 1):
        raw(dn, i, j, getattr(row, k), "0.00" if j == 1 else "0.0000")
    for j, src in zip((7, 8, 9), "DEF"):
        raw(dn, i, j, f"={src}{i}-INDEX(${src}${SH0}:${src}${SH1},MATCH(0,$A${SH0}:$A${SH1},0))", "0.0000", blue=False)
r = SH1 + 2; sec(dn, r, "c. Lưới pad × merge trên dev (ngưỡng Python, A4 với pad/merge thay đổi)")
head(dn, r + 1, ["Pad trước (ms)", "Pad sau (ms)", "Merge gap (ms)", "TP", "FN", "FP", "TN", "MR", "FAR", "F1", "DCF"])
GR0 = r + 2; GR1 = GR0 + len(grid) - 1
for i, row in enumerate(grid.itertuples(index=False), GR0):
    for j, k in enumerate(["pre_ms", "post_ms", "gap_ms", "TP", "FN", "FP", "TN"], 1): raw(dn, i, j, getattr(row, k))
    f = rates(f"D{i}", f"E{i}", f"F{i}", f"G{i}")
    for j, k in zip((8, 9, 10), ("MR", "FAR", "F1")): raw(dn, i, j, f[k], D3 if k == "F1" else P, blue=False)
    raw(dn, i, 11, dcf(f"H{i}", f"I{i}"), D3, blue=False)
for c in range(1, 12): dn.column_dimensions[L(c)].width = 13

# ======================= DL do lech bien =======================
title(dt, "Độ lệch biên mỗi đoạn GT (test, A4, ngưỡng Python)", "Chữ xanh = số từ Python (dự đoán − GT, giây).")
head(dt, 4, ["Δonset (s)", "Δoffset (s)"])
DL0, DL1 = 5, 4 + len(dls)
for i, row in enumerate(dls.itertuples(index=False), DL0): raw(dt, i, 1, row.d_onset, "0.00"); raw(dt, i, 2, row.d_offset, "0.00")
for c in "AB": dt.column_dimensions[c].width = 12

# ======================= KQ du lieu that =======================
title(kq, "Kết quả trên audio thật – chỉ số tính bằng công thức", SRC)
kq["A3"] = ("Nền vàng = tham số bạn chỉnh (B6:B10) · chữ đen = công thức · chữ xanh = số nhập từ Python (bootstrap, event, "
            "AUC chính xác…; không tính lại trong Excel). Đổi tham số thì mọi bảng và sheet template liên kết tự tính lại; hình vẽ bằng Python thì không.")
kq["A3"].font = Font(name=F, size=9, italic=True, color="595959")
kq.column_dimensions["A"].width = 46
for c in range(2, 26): kq.column_dimensions[L(c)].width = 12

sec(kq, 5, "Tham số")
params = [(6, "Trọng số miss trong DCF (FA = 1 − ô này)", 0.75, "0.00"), (7, "Cấu hình hậu xử lý cho tầng B, category", "A4", None),
          (8, "Tiêu chí chọn ngưỡng trên dev thật", "DCF-min", None), (9, "Mốc FPR 1 (TPR@FPR)", 0.01, P), (10, "Mốc FPR 2 (TPR@FPR)", 0.05, P)]
for r_, lab, v, fmt in params:
    w(kq, r_, 1, lab, left=True); w(kq, r_, 2, v, fmt, fill=YEL)
for ref, opts in (("B7", '"A0,A1,A2,A3,A4"'), ("B8", '"DCF-min,F1-max"')):
    dv = DataValidation(type="list", formula1=opts, allow_blank=False); kq.add_data_validation(dv); dv.add(ref)
w(kq, 11, 1, "Ngưỡng dùng (tự chọn trên dev theo B7, B8)", left=True, bold=True)
w(kq, 11, 2, "=INDEX($I$17:$I$21,MATCH($B$7,$A$17:$A$21,0))", "0.00", bold=True)
w(kq, 12, 1, "Ngưỡng Python đã dùng cho số theo đoạn, CI, Δbiên, hình", left=True); w(kq, 12, 2, THR, "0.00", blue=True)
w(kq, 13, 1, "Kiểm tra", left=True)
w(kq, 13, 2, '=IF(AND(ABS(B11-B12)<0.0000001,B7="A4"),"Khớp","KHÁC: số theo đoạn, CI, Δbiên, hình vẫn ở ngưỡng Python / A4")', left=True)

# A. chọn ngưỡng
sec(kq, 15, "A. Chọn ngưỡng trên dev thật / val tổng hợp (qua toàn pipeline; DCF theo trọng số B6)")
head(kq, 16, ["Cấu hình", "Mô tả", "Ngưỡng DCF-min (dev)", "DCF min (dev)", "Ngưỡng F1-max (dev)", "F1 max (dev)",
              "Ngưỡng DCF-min (val tổng hợp)", "Ngưỡng F1-max (val tổng hợp)", "Ngưỡng dùng (theo B8)"])
MODES = ["A0", "A1", "A2", "A3", "A4"]
DESC = ["Điểm bin thô, không hậu xử lý", "Rescore + rebin", "+ pad 100/120 ms", "+ merge < 500 ms", "+ pad + merge (hiện tại)"]
ROW_A = {m: 17 + i for i, m in enumerate(MODES)}
def arg(b, col_key, col_val, fn):   # ngưỡng tại MIN/MAX của một khối dev/synth
    r0, r1 = b; return f"=INDEX({rg(DQ, col_key, r0, r1)},MATCH({fn}({rg(DQ, col_val, r0, r1)}),{rg(DQ, col_val, r0, r1)},0))"
for m, desc in zip(MODES, DESC):
    r_, bdev, bsyn = ROW_A[m], blocks[("dev", m)], blocks[("synth", m)]
    w(kq, r_, 1, m); w(kq, r_, 2, desc, left=True)
    w(kq, r_, 3, arg(bdev, "E", "R", "MIN"), "0.00"); w(kq, r_, 4, f"=MIN({rg(DQ, 'R', *bdev)})", D3)
    w(kq, r_, 5, arg(bdev, "E", "Q", "MAX"), "0.00"); w(kq, r_, 6, f"=MAX({rg(DQ, 'Q', *bdev)})", D3)
    w(kq, r_, 7, arg(bsyn, "E", "R", "MIN"), "0.00"); w(kq, r_, 8, arg(bsyn, "E", "Q", "MAX"), "0.00")
    w(kq, r_, 9, f'=IF($B$8="F1-max",E{r_},C{r_})', "0.00", bold=True)
w(kq, 22, 1, "Tham khảo (Python): ngưỡng Youden trên điểm thô – val tổng hợp / dev thật", left=True)
w(kq, 22, 3, S["thr_youden_raw_synth"], D3, blue=True); w(kq, 22, 5, S["thr_youden_raw_dev"], D3, blue=True)

# B. tầng B
sec(kq, 24, "B. Tầng B trên test (cấu hình B7, ngưỡng B11)")
head(kq, 25, ["Chế độ", "Vùng bin", "TP", "FN", "FP", "TN", "MR", "FAR", "Precision", "Recall", "F1", "Specificity", "DCF",
              "CI95 F1 (Python)", "CI95 DCF (Python)", "CI95 MR (Python)", "CI95 FAR (Python)"])
def ci(k, pct=False): lo, hi = S[k]; return f"{lo:.1%}–{hi:.1%}" if pct else f"{lo:.3f}–{hi:.3f}"
for r_, (lab, mk) in zip((26, 27, 28), [("Strict (mọi bin)", "all"), ("Collar ±1 bin (bỏ bin biên)", "nb"), ("Chỉ bin biên", "b")]):
    w(kq, r_, 1, lab, left=True); w(kq, r_, 2, mk)
    for k, col in enumerate("FGHI"): w(kq, r_, 3 + k, sif(col, "$B$7", "$B$11", f"$B{r_}"))
    f = rates(f"C{r_}", f"D{r_}", f"E{r_}", f"F{r_}")
    for k, key in enumerate(["MR", "FAR", "Prec", "Rec", "F1", "Spec"]): w(kq, r_, 7 + k, f[key], D3 if key == "F1" else P)
    w(kq, r_, 13, dcf(f"G{r_}", f"H{r_}"), D3)
w(kq, 26, 14, ci("F1_CI"), blue=True); w(kq, 26, 15, ci("DCF_CI"), blue=True); w(kq, 26, 16, ci("MR_CI", True), blue=True)
w(kq, 26, 17, ci("FAR_CI", True), blue=True); w(kq, 27, 14, ci("F1nb_CI"), blue=True)
note(kq["N26"], f"Bootstrap 1000 lần theo file, ở ngưỡng Python {THR} và A4 (xem B12, B13).")

# C. theo category
sec(kq, 30, "C. Theo category (test; cấu hình B7, ngưỡng B11)")
head(kq, 31, ["Category", "#file", "TP", "FN", "FP", "TN", "FEC", "MSC", "OVER", "NDS", "TP ±1", "FN ±1", "FP ±1", "TN ±1",
              "% speech", "MR", "FAR", "Precision", "F1 strict", "F1 ±1 bin", "DCF", "Lỗi biên (FEC+OVER)", "Lỗi thật (MSC+NDS)",
              "F1 CI95 thấp (Python)", "F1 CI95 cao (Python)"])
ROW_C = {c: 32 + i for i, c in enumerate(CATS)}
def cat_rates(r_):
    f = rates(f"C{r_}", f"D{r_}", f"E{r_}", f"F{r_}"); fnb = rates(f"K{r_}", f"L{r_}", f"M{r_}", f"N{r_}")
    w(kq, r_, 15, f'=IF(SUM(C{r_}:F{r_})=0,"",(C{r_}+D{r_})/SUM(C{r_}:F{r_}))', P)
    for k, v in enumerate([f["MR"], f["FAR"], f["Prec"], f["F1"], fnb["F1"]]): w(kq, r_, 16 + k, v, D3 if k >= 3 else P)
    w(kq, r_, 21, dcf(f"P{r_}", f"Q{r_}"), D3); w(kq, r_, 22, f"=G{r_}+I{r_}"); w(kq, r_, 23, f"=H{r_}+J{r_}")
for c in CATS:
    r_ = ROW_C[c]; w(kq, r_, 1, c, left=True)
    w(kq, r_, 2, f'=COUNTIFS({DFc("B")},$A{r_},{DFc("C")},"test")')
    for k, col in enumerate("FGHIJKLM"): w(kq, r_, 3 + k, sif(col, "$B$7", "$B$11", '"all"', f"$A{r_}"))
    for k, col in enumerate("FGHI"): w(kq, r_, 11 + k, sif(col, "$B$7", "$B$11", '"nb"', f"$A{r_}"))
    cat_rates(r_)
    cr = cat[cat.Category == c].iloc[0]; w(kq, r_, 24, cr.F1_CI_lo, D3, blue=True); w(kq, r_, 25, cr.F1_CI_hi, D3, blue=True)
w(kq, 37, 1, "Tổng test (gộp)", left=True, bold=True)
for k in range(2, 15): w(kq, 37, k, f"=SUM({L(k)}32:{L(k)}36)")
cat_rates(37); w(kq, 37, 24, S["F1_CI"][0], D3, blue=True); w(kq, 37, 25, S["F1_CI"][1], D3, blue=True)
w(kq, 38, 1, "Macro (TB 5 category)", left=True, bold=True)
for k in range(15, 22): w(kq, 38, k, f"=AVERAGE({L(k)}32:{L(k)}36)", D3 if k >= 19 else P)
note(kq["X32"], f"CI 95% bootstrap theo file (500 lần) ở ngưỡng Python {THR}, A4.")

# D. ablation
sec(kq, 40, "D. Ablation hậu xử lý (test; ngưỡng chọn riêng cho từng cấu hình trên dev theo B8)")
head(kq, 41, ["Cấu hình", "Mô tả", "Ngưỡng", "TP", "FN", "FP", "TN", "TP ±1", "FN ±1", "FP ±1", "TN ±1", "FEC", "MSC", "OVER", "NDS",
              "MR", "FAR", "F1 strict", "F1 ±1 bin", "DCF"])
ROW_D = {m: 42 + i for i, m in enumerate(MODES)}
for m, desc in zip(MODES, DESC):
    r_ = ROW_D[m]; w(kq, r_, 1, m); w(kq, r_, 2, desc, left=True); w(kq, r_, 3, f"=$I${ROW_A[m]}", "0.00")
    for k, col in enumerate("FGHI"): w(kq, r_, 4 + k, sif(col, f"$A{r_}", f"$C{r_}", '"all"'))
    for k, col in enumerate("FGHI"): w(kq, r_, 8 + k, sif(col, f"$A{r_}", f"$C{r_}", '"nb"'))
    for k, col in enumerate("JKLM"): w(kq, r_, 12 + k, sif(col, f"$A{r_}", f"$C{r_}", '"all"'))
    f = rates(f"D{r_}", f"E{r_}", f"F{r_}", f"G{r_}"); fnb = rates(f"H{r_}", f"I{r_}", f"J{r_}", f"K{r_}")
    w(kq, r_, 16, f["MR"], P); w(kq, r_, 17, f["FAR"], P); w(kq, r_, 18, f["F1"], D3); w(kq, r_, 19, fnb["F1"], D3)
    w(kq, r_, 20, dcf(f"P{r_}", f"Q{r_}"), D3)

# E. so sánh ngưỡng
sec(kq, 48, "E. Chọn ngưỡng ở đâu → kết quả trên test")
head(kq, 49, ["Ngưỡng chọn từ", "Cấu hình", "Ngưỡng", "TP", "FN", "FP", "TN", "MR", "FAR", "Precision", "Recall", "F1", "DCF"])
for r_, (lab, m, ref) in enumerate([("DCF-min – val tổng hợp", "A0", "$G$17"), ("DCF-min – val tổng hợp", "A4", "$G$21"),
                                    ("F1-max – val tổng hợp", "A4", "$H$21"), ("DCF-min – dev thật", "A0", "$C$17"),
                                    ("F1-max – dev thật", "A4", "$E$21"), ("DCF-min – dev thật", "A4", "$C$21")], 50):
    w(kq, r_, 1, lab, left=True); w(kq, r_, 2, m); w(kq, r_, 3, f"={ref}", "0.00")
    for k, col in enumerate("FGHI"): w(kq, r_, 4 + k, sif(col, f"$B{r_}", f"$C{r_}", '"all"'))
    f = rates(f"D{r_}", f"E{r_}", f"F{r_}", f"G{r_}")
    for k, key in enumerate(["MR", "FAR", "Prec", "Rec", "F1"]): w(kq, r_, 8 + k, f[key], D3 if key == "F1" else P)
    w(kq, r_, 13, dcf(f"H{r_}", f"I{r_}"), D3)

# F. tầng A
sec(kq, 57, f"F. Tầng A – điểm thô (AUC hình thang trên {len(roc)} ngưỡng ở sheet DL ROC)")
head(kq, 58, ["Tập", "#bin", "% speech", "AUC (công thức)", "AUC Python (chính xác)", "Chênh", "PR-AUC (công thức)",
              "TPR @ FPR mốc 1", "TPR @ FPR mốc 2", "AUC macro theo file", "AUC không biên (Python)", "AUC chỉ biên (Python)",
              "AUC nhãn chắc (Python)", "Brier (Python)"])
tiers = [("Test (tất cả)", "Test", '"test"', None)] + [(c, c, '"test"', c) for c in CATS] + \
        [("Dev (tham khảo)", "Dev", '"dev"', None), ("Cả 144 file (tham khảo)", "Tất cả", None, None)]
ROW_F = {}
for r_, (lab, g, split, c) in enumerate(tiers, 59):
    ROW_F[lab] = r_; tpc, fpc = L(cc[g]), L(cc[g] + 1)
    t0, t1, f0, f1 = rocc(g, 0, DR0, DR1 - 1), rocc(g, 0, DR0 + 1, DR1), rocc(g, 1, DR0, DR1 - 1), rocc(g, 1, DR0 + 1, DR1)
    w(kq, r_, 1, lab, left=True); w(kq, r_, 2, f"={DR}!{tpc}{DR0}+{DR}!{fpc}{DR0}"); w(kq, r_, 3, f"={DR}!{tpc}{DR0}/B{r_}", P)
    w(kq, r_, 4, f"=SUMPRODUCT(({f0}-{f1})*({t0}+{t1}))/2", "0.0000")
    tr_ = tA[tA["Tập"] == lab].iloc[0]
    w(kq, r_, 5, tr_.AUC_micro, "0.0000", blue=True); w(kq, r_, 6, f"=D{r_}-E{r_}", "0.0000")
    w(kq, r_, 7, f"=SUMPRODUCT(({t0}-{t1})*{rocc(g, 2, DR0, DR1 - 1)})", D3)
    for k, m_ in ((8, "$B$9"), (9, "$B$10")): w(kq, r_, k, f'=_xlfn.MAXIFS({rocc(g, 0)},{rocc(g, 1)},"<="&{m_})', P)
    crit = (f',{DFc("C")},{split}' if split else "") + (f',{DFc("B")},"{c}"' if c else "")
    w(kq, r_, 10, f"=AVERAGEIFS({DFc('L')}{crit})" if crit else f"=AVERAGE({DFc('L')})", D3)
    for k, col in zip(range(11, 15), ["AUC_khong_bien", "AUC_chi_bien", "AUC_nhan_chac", "Brier"]): w(kq, r_, k, tr_[col], D3, blue=True)
w(kq, 67, 1, "CI95 AUC test (Python, bootstrap 1000 lần theo file)", left=True)
w(kq, 67, 2, S["AUC_test_CI"][0], D3, blue=True); w(kq, 67, 3, S["AUC_test_CI"][1], D3, blue=True)
w(kq, 68, 1, "AUC val tổng hợp 500 file (Python)", left=True); w(kq, 68, 2, S["auc_synth_val"], D3, blue=True)

# G. nhãn
sec(kq, 70, "G. Nhất quán nhãn: FireRedVAD (quy tắc gốc) vs Silero – κ tính từ bảng 2×2")
head(kq, 71, ["Category", "Vùng bin", "n11", "n10", "n01", "n00", "#bin", "% speech FireRedVAD", "% speech Silero", "Khớp", "κ",
              "#bin đổi do hát", "#bin speech (chính sách)", "#bin nhãn chắc", "% speech (chính sách)", "% bin nhãn chắc"])
lab_rows = [(c, "all") for c in CATS] + [("Tất cả", "all"), ("Tất cả", "trong"), ("Tất cả", "bien")]
for r_, (c, mk) in enumerate(lab_rows, 72):
    w(kq, r_, 1, c, left=True); w(kq, r_, 2, mk)
    crit = f',{rg(DN, "B", LB0, LB1)},$B{r_}' + (f',{rg(DN, "A", LB0, LB1)},$A{r_}' if c != "Tất cả" else "")
    for k, col in zip((3, 4, 5, 6, 12, 13, 14), "CDEFGIH"): w(kq, r_, k, f"=SUMIFS({rg(DN, col, LB0, LB1)}{crit})")
    w(kq, r_, 7, f"=SUM(C{r_}:F{r_})"); w(kq, r_, 8, f"=(C{r_}+D{r_})/G{r_}", P); w(kq, r_, 9, f"=(C{r_}+E{r_})/G{r_}", P)
    w(kq, r_, 10, f"=(C{r_}+F{r_})/G{r_}", P)
    pe = f"((C{r_}+D{r_})*(C{r_}+E{r_})+(E{r_}+F{r_})*(D{r_}+F{r_}))/G{r_}^2"
    w(kq, r_, 11, f'=IFERROR((J{r_}-{pe})/(1-{pe}),"")', D3)
    w(kq, r_, 15, f"=M{r_}/G{r_}", P); w(kq, r_, 16, f"=N{r_}/G{r_}", P)

# H. phân bố
sec(kq, 81, "H. Phân bố và kiểm tra theo file (từ DL theo file)")
dist = [(82, "Số file", f"=COUNTA({DFc('A')})", None), (83, "Số giờ audio", f"=SUM({DFc('D')})/3600", "0.00"),
        (84, "Số file chỉ có 1 lớp", f"=SUM({DFc('P')})", None), (85, "% file chỉ có 1 lớp", "=B84/B82", P),
        (86, "% file cực đoan (speech < 0.1 hoặc > 0.9)", f"=SUM({DFc('Q')})/B82", P),
        (87, "Trung vị số lần chuyển trạng thái / file", f"=MEDIAN({DFc('I')})", "0.0"),
        (88, "AUC gộp 144 file (công thức, mục F)", f"=D{ROW_F['Cả 144 file (tham khảo)']}", D3),
        (89, "AUC macro theo file", f"=AVERAGE({DFc('L')})", D3), (90, "SD AUC theo file", f"=STDEV({DFc('L')})", D3),
        (91, "Chênh AUC gộp − macro", "=B88-B89", D3), (92, "Số file AUC < 0.9", f'=COUNTIF({DFc("L")},"<0.9")', None),
        (93, "Số file AUC < 0.7", f'=COUNTIF({DFc("L")},"<0.7")', None), (94, "AUC file tệ nhất", f"=MIN({DFc('L')})", D3),
        (95, "File tệ nhất", f"=INDEX({DFc('A')},MATCH(B94,{DFc('L')},0))", None),
        (96, "Trung vị lag (s)", f"=MEDIAN({DFc('M')})", "0.00"),
        (97, "Số file |lag| ≥ 0.4 s", f'=COUNTIF({DFc("M")},">=0.4")+COUNTIF({DFc("M")},"<=-0.4")', None)]
for r_, lab, fml, fmt in dist: w(kq, r_, 1, lab, left=True); w(kq, r_, 2, fml, fmt)

# I. theo đoạn (Python)
sec(kq, 99, "I. Chỉ số theo đoạn (Python, test, A4) – cần ghép từng đoạn nên không tính lại trong Excel")
head(kq, 100, ["Ngưỡng", "Giá trị", "Event-F1", "Onset-F1", "#đoạn GT", "#đoạn dự đoán", "#GT bị phân mảnh", "#dự đoán gộp ≥2 GT"])
for r_, row in enumerate(evt.itertuples(index=False), 101):
    for k, (v, fmt) in enumerate([(row.Ngưỡng, None), (row.thr, "0.00"), (row.event_F1, D3), (row.onset_F1, D3), (row.n_gt_seg, None),
                                  (row.n_pred_seg, None), (row.n_fragmented, None), (row.n_overmerged, None)], 1):
        w(kq, r_, k, v, fmt, blue=True, left=k == 1)
w(kq, 104, 1, "Δonset TB (s), ngưỡng Python", left=True); w(kq, 104, 2, f"=AVERAGE({rg(DT, 'A', DL0, DL1)})", "0.00")
w(kq, 105, 1, "Δoffset TB (s), ngưỡng Python", left=True); w(kq, 105, 2, f"=AVERAGE({rg(DT, 'B', DL0, DL1)})", "0.00")
w(kq, 106, 1, "Khoảng lặng GT ≤ 1 s bị lấp (Python)", left=True)
w(kq, 106, 2, f"{S['gap_short_filled']}/{S['gap_short_total']}", blue=True)

# J. nhãn đáng ngờ
sec(kq, 108, f"J. 30 đoạn nghi nhãn sai dài nhất (tổng {S['n_suspicious_seg']} đoạn; đủ danh sách: results_real/suspicious.csv) – nghe lại")
scol = [("file", "File", None), ("category", "Category", None), ("split", "Tập", None), ("start", "Bắt đầu (s)", "0.0"),
        ("end", "Kết thúc (s)", "0.0"), ("n_bins", "#bin", None), ("GT", "Nhãn", None), ("score", "Điểm TB", D3),
        ("silero_speech", "Silero speech", "0%"), ("nhan_do_hat", "Nhãn do hát", None)]
head(kq, 109, [c[1] for c in scol])
for r_, row in enumerate(sus.head(30).itertuples(index=False), 110):
    for k, (col, _, fmt) in enumerate(scol, 1): w(kq, r_, k, getattr(row, col), fmt, blue=True, left=k == 1)
kq.freeze_panes = "B5"

# ======================= Hinh du lieu that (vẽ bằng Python) =======================
title(hd, "Hình – đánh giá trên audio thật (vẽ bằng Python)",
      f"{SRC} Hình cố định ở ngưỡng Python {THR} (A4); đổi tham số ở KQ du lieu that không đổi hình – chạy lại evaluate_real.py.")
r = 4
for f, t in [("fig1_kiem_tra_du_lieu.png", "Hình 1. Kiểm tra dữ liệu"), ("fig2_tang_A.png", "Hình 2. Tầng A (test)"),
             ("fig3_tang_B.png", "Hình 3. Ngưỡng và tầng B"), ("fig4_timeline.png", "Hình 4. Timeline mẫu (test)")]:
    sec(hd, r, t)
    img = XLImage(str(R / f)); k = 1100 / img.width; img.width, img.height = 1100, int(img.height * k)
    hd.add_image(img, f"A{r + 1}"); r += int(img.height / 20) + 4

# ======================= các sheet của template =======================
def put(ws, ref, v, fmt=None, text=None, yellow=False):
    """Ghi một ô nhập liệu của template; không bao giờ ghi lên công thức sẵn có."""
    c = ws[ref]
    assert not (isinstance(c.value, str) and c.value.startswith("=")), f"{ws.title}!{ref} là công thức của template"
    c.value = val(v)
    if fmt: c.number_format = fmt
    if yellow: c.fill = YEL
    if text: note(c, text)

# Bao cao category <- KQ mục C
ws = wb["Bao cao category"]
for i, c in enumerate(CATS):
    r_, kr = 5 + i, ROW_C[c]
    put(ws, f"A{r_}", c, yellow=r_ == 5)
    for col, kc in zip("BCDEFGHIJKLM", ["B", "O", "P", "Q", "G", "H", "I", "J", "S", "T", "X", "Y"]):
        put(ws, f"{col}{r_}", f"={KQ}!{kc}{kr}", yellow=r_ == 5)
note(ws["A5"], f"Công thức lấy từ KQ du lieu that, mục C (test, cấu hình KQ!B7, ngưỡng KQ!B11). CI là số Python ở ngưỡng {THR}. {SRC}", 320, 160)

# Ablation <- KQ mục D
ws = wb["Ablation"]
for m in MODES:
    r_, kr = 5 + MODES.index(m), ROW_D[m]
    for col, kc in zip("GHIJ", "PQRS"): put(ws, f"{col}{r_}", f"={KQ}!{kc}{kr}", yellow=True)
    put(ws, f"M{r_}", f'="Ngưỡng "&{KQ}!$B$8&" trên dev = "&FIXED({KQ}!C{kr},2)')
for c in "FGHIJ": put(ws, f"{c}16", None, yellow=True)          # O0 chưa đo -> bỏ số ví dụ
put(ws, "M16", "Chưa đo")
seg_ratio = S["n_pred_seg"] / S["n_gt_seg"]
for r_, txt in ((17, "Giống A4 (rescore_hop: mean các cửa sổ phủ hop)"), (24, "Giống A4 (không drop)"), (32, "Giống A4: padding → merge, drop = 0")):
    for col, kc in zip("GHIJ", "PQRS"): put(ws, f"{col}{r_}", f"={KQ}!{kc}{ROW_D['A4']}", yellow=True)
    put(ws, f"M{r_}", txt)
    if r_ != 17: put(ws, f"F{r_}", seg_ratio, yellow=True, text=f"Python: {S['n_pred_seg']} đoạn dự đoán / {S['n_gt_seg']} đoạn GT ở ngưỡng {THR}.")

# Bien va timestamp (mục C)
ws = wb["Bien va timestamp"]
put(ws, "C24", round(S["best_shift_center"] * 1000), yellow=True,
    text=f"Python: quét F1 (thr 0.5) trên nhãn FireRedVAD, đỉnh +{S['best_shift_center']*1000:.0f} ms. Theo AUC: nhãn chính xác (val tổng hợp) "
         f"đỉnh tại {S['best_shift_auc']['AUC_synth_exact']*1000:.0f} ms; nhãn thật chỉ tăng {S['auc_gain_best_shift_real']:.4f} -> lệch do nhãn.")
put(ws, "C25", int((~struct.win_count_ok).sum()), text="Python: số cửa sổ = floor((N_mẫu − 15600)/1280) + 1. Đo trên output Python (raw_real).")

# KPI va Gate
ws = wb["KPI va Gate"]
put(ws, "F10", f"={KQ}!$D${ROW_F['Test (tất cả)']}")
put(ws, "F11", f"={KQ}!$C$101", text=f"Python: event-F1 ở ngưỡng {THR} (KQ du lieu that, mục I).")
kpi_notes = {5: f'="F1 gộp toàn test = "&FIXED({KQ}!$S$37,3)&"; ô Baseline = TB 5 category. Ngưỡng "&FIXED({KQ}!$B$11,2)&" ("&{KQ}!$B$8&" dev, "&{KQ}!$B$7&")"',
             6: f'="F1 ±1 bin gộp = "&FIXED({KQ}!$T$37,3)',
             7: f'="DCF gộp = "&FIXED({KQ}!$U$37,3)&" (trọng số miss "&FIXED({KQ}!$B$6,2)&")"',
             8: f'="MR gộp = "&FIXED({KQ}!$P$37*100,1)&"%; Asm = "&FIXED({KQ}!$P$32*100,1)&"%"',
             9: f'="FAR gộp = "&FIXED({KQ}!$Q$37*100,1)&"%; Babble = "&FIXED({KQ}!$Q$33*100,1)&"%"',
             10: f'="AUC hình thang (công thức); Python chính xác = "&FIXED({KQ}!$E$59,3)&", CI95 "&FIXED({KQ}!$B$67,3)&"–"&FIXED({KQ}!$C$67,3)',
             11: f'="Python ở ngưỡng "&FIXED({KQ}!$B$12,2)&"; thấp vì nhãn FireRedVAD cắt ở mọi quãng nghỉ 0.5 s"',
             15: "Lệch do nhãn FireRedVAD (xem comment Bien va timestamp!C24); trên nhãn chính xác = 0 ms."}
for r_, t in kpi_notes.items(): put(ws, f"S{r_}", t)

# Kiem tra du lieu (Bước 8): mục đo được bằng công thức; còn lại là mô tả
ws = wb["Kiem tra du lieu"]
wav = [wave.open(str(p)) for p in sorted(a.audio.glob("*/*.wav"))]
fmt_ok = sum(x.getframerate() == 16000 and x.getnchannels() == 1 and x.getsampwidth() == 2 for x in wav)
for x in wav: x.close()
checks = {
 "5a": (f"{S['n_pairs']}/{S['n_audio']} file ghép đủ audio ↔ output ↔ nhãn; 0 lỗi", "Đạt", "Tên <Cat>_<youtube id>"),
 "5b": (f"{int(struct.flag.sum())} file lệch số bin / số cửa sổ / last_end", "Đạt", "n_bins = floor(dur/0.5)"),
 "5c": (f"{fmt_ok}/{len(wav)} file 16 kHz, mono, PCM 16-bit", "Đạt" if fmt_ok == len(wav) else "Không đạt", None),
 "5d": ("hop 0.08 s, window 0.96 s, score ∈ [0,1] ở mọi file", "Đạt", None),
 "5e": (f"Đỉnh F1 (thr 0.5) +{S['best_shift_center']*1000:.0f} ms trên nhãn FireRedVAD; đỉnh AUC trên nhãn chính xác = "
        f"{S['best_shift_auc']['AUC_synth_exact']*1000:.0f} ms", "Đạt",
        f"Trên nhãn thật AUC chỉ tăng {S['auc_gain_best_shift_real']:.4f} khi dịch -> lệch do nhãn, không sửa. "
        f"File |lag| ≥ 0.4 s có tương quan đỉnh thấp (ước lượng nhiễu)."),
 "1a": (f'={KQ}!$B$84&"/"&{KQ}!$B$82&" file ("&FIXED({KQ}!$B$85*100,1)&"%) chỉ có 1 lớp"',
        f'=IF({KQ}!$B$85>0.1,"Không đạt","Đạt")', "Loại khỏi AUC theo file; giữ trong chỉ số gộp (quan trọng cho FAR). Quy tắc: > 10% -> Không đạt."),
 "1b": (f'=FIXED({KQ}!$B$86*100,1)&"% file có tỉ lệ speech < 0.1 hoặc > 0.9"', f'=IF({KQ}!$B$86>0.3,"Không đạt","Đạt")',
        "Do thiết kế category: Noise gần 0%, Clean ~90% speech."),
 "1c": (f'="Trung vị "&FIXED({KQ}!$B$87,1)&" lần chuyển trạng thái / file"', f'=IF({KQ}!$B$87<=1,"Không đạt","Đạt")', "File dài 20–120 s"),
 "1d": (f'="AUC gộp "&FIXED({KQ}!$B$88,3)&" vs macro theo file "&FIXED({KQ}!$B$89,3)&"; "&{KQ}!$B$92&" file < 0.9, tệ nhất "'
        f'&FIXED({KQ}!$B$94,3)&" ("&{KQ}!$B$95&")"', f'=IF(OR(ABS({KQ}!$B$91)>0.005,{KQ}!$B$92>0),"Không đạt","Đạt")',
        "File tệ tập trung ở Asm, Babble (Hinh du lieu that, hình 1 và 4)."),
 "2a": ("Không có metadata kênh / người nói; dev/test chia theo file, phân tầng category", "Chưa làm", None),
 "2b": (None, "Chưa làm", None),
 "2c": ("Model chỉ train trên dữ liệu tổng hợp (bộ dữ liệu công khai); audio thật chưa dùng để train", "Không áp dụng", None),
 "3a": (f'="κ FireRedVAD–Silero = "&FIXED({KQ}!$K$77,2)&" (trong "&FIXED({KQ}!$K$78,2)&", biên "&FIXED({KQ}!$K$79,2)&")"',
        f'=IF({KQ}!$K$78<0.9,"Không đạt","Đạt")', "Silero bỏ sót thì thầm (Asm) và không tính hát là speech; Silero là người gán thứ 2."),
 "3b": (f"{S['n_suspicious_seg']} đoạn / {S['n_suspicious_bins']} bin trái ngược mạnh; Silero đồng ý với model ở "
        f"{S['suspicious_silero_agrees_model']:.0%} số bin", "Đang làm", "KQ du lieu that mục J; đủ danh sách: results_real/suspicious.csv"),
 "4a": (f"{S['pct_ambiguous']:.1%} bin có 0.1<p<0.9; F1 thay đổi {S['f1_range_0.2_0.9']:.3f} trong ngưỡng 0.2–0.9", "Đạt", "Dữ liệu không quá dễ"),
 "4b": ("Có nhạc có lời / hát (Music), đám đông (Babble), nhiễu (Noise), thì thầm (Asm)", "Đạt", "Chưa có riêng tiếng cười / TV"),
 "4c": ("Không có metadata SNR cho audio thật", "Chưa làm", None),
 "4d": (f'="AUC val tổng hợp "&FIXED({KQ}!$B$68,3)&" vs test thật "&FIXED({KQ}!$D$59,3)&"; Asm "&FIXED({KQ}!$D$60,3)&", Noise "&FIXED({KQ}!$D$64,3)',
        "Đạt", "Đã báo cả hai; ưu tiên số thật"),
}
for r_ in range(5, 23):
    code = ws[f"A{r_}"].value
    if code in checks:
        g, h, i_ = checks[code]; put(ws, f"G{r_}", g); put(ws, f"H{r_}", h); put(ws, f"I{r_}", i_)

# Quy trinh 8 buoc
ws = wb["Quy trinh 8 buoc"]
steps = {5: ("Đạt", "Dev 43 / test 101 file, chia theo file, phân tầng category (seed 3). Không có metadata nguồn để chia theo nhóm."),
         6: ("Đạt", "Bin 0.5 s từ nhãn FireRedVAD (gt_policy); cờ biên ±1 bin; bin 'nhãn chắc' = FireRedVAD 2 quy tắc và Silero cùng nhãn."),
         7: ("Đang làm", "Alignment Đạt (5a–5e). Rò rỉ 2a/2b chưa làm (thiếu metadata)."),
         8: ("Đạt", "KQ du lieu that mục F: AUC, PR-AUC, TPR@FPR bằng công thức; tất cả / không biên / nhãn chắc. Chưa vẽ reliability diagram."),
         9: ("Đang làm", "KQ du lieu that mục A, E: ngưỡng chọn bằng công thức theo tiêu chí ở B8. Cần chốt tiêu chí (DCF hay FAR cố định)."),
         10: ("Đạt", "KQ du lieu that mục B, C, I: MR/FAR/F1/DCF 3 chế độ, FEC/MSC/OVER/NDS, event-F1, Δbiên. Chưa tính DetER."),
         11: ("Đạt", "KQ du lieu that mục D + heatmap DCF (Hinh du lieu that, hình 3): hậu xử lý ít tác động."),
         12: ("Đang làm", "Có CI + per-category + số tổng hợp và số thật. Chưa nghe kiểm tra top-K (mục J).")}
for r_, (g, h) in steps.items(): put(ws, f"G{r_}", g); put(ws, f"H{r_}", h)

# Ke hoach baseline: bằng chứng nhóm Q
ws = wb["Ke hoach baseline"]
ev = {8: "DL theo file (cột Tập)", 9: "DL theo file, DL nhan va phan bo", 10: "KQ du lieu that mục H; DL nhan va phan bo mục b; Hinh du lieu that, hình 1",
      11: "KQ du lieu that mục F; DL ROC", 12: "KQ du lieu that mục A, E; DL quet nguong", 13: "KQ du lieu that mục B, C, I",
      14: "KQ du lieu that mục D; DL nhan va phan bo mục c; hình 3", 15: "KQ du lieu that; Hinh du lieu that; results_real/"}
for r_, t in ev.items(): put(ws, f"O{r_}", t)

# Snapshot (cột Baseline)
ws = wb["Snapshot"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
snap = {"C5": "v2 – head run1/best (seed 4)", "C8": f"{a.yamnet.name} + {a.run.name}/best/head_weights.npz",
        "C9": f"yamnet: {sha(a.yamnet)}\nhead: {sha(a.run / 'best' / 'head_weights.npz')}", "C11": "float32",
        "C22": "Bỏ phần dư (chỉ cửa sổ đủ 15 600 mẫu)", "C23": "Mean các cửa sổ phủ hop", "C24": f"={KQ}!$B$11", "C27": 0,
        "C31": "real_test/gt_policy (FireRedVAD, nói hoặc hát = speech); chia theo file, seed 3",
        "C32": "real_test/gt_policy (FireRedVAD, nói hoặc hát = speech); chia theo file, seed 3",
        "C33": f'=COUNTIFS({DFc("C")},"dev")&" / "&COUNTIFS({DFc("C")},"test")',
        "C34": f'=SUMIFS({DFc("D")},{DFc("C")},"test")/3600',
        "C37": "evaluate_real.py + make_excel_real.py (vad_demo, chưa có git)", "C38": "0 / 1000"}
for ref, v in snap.items(): put(ws, ref, v, fmt="0.00" if ref == "C34" else None)
note(ws["C24"], f"Công thức = KQ du lieu that!B11 (tiêu chí ở B8, trọng số DCF ở B6). Ngưỡng cũ 0.6026 thuộc model cũ. "
                f"Cần chốt tiêu chí trước khi đóng băng.", 320, 120)

# Bieu do can ve
ws = wb["Bieu do can ve"]
drawn = {5: "hình 2", 6: "hình 2", 8: "hình 3 – chỉ sau hậu xử lý A4, dev thật vs val tổng hợp", 9: "hình 3", 10: "hình 3",
         11: "hình 3 – heatmap DCF (không phải F1)", 13: "hình 1 – chấm theo category thay histogram + boxplot",
         15: "hình 1 – chưa có F1 theo ngưỡng", 16: "hình 1 – ΔAUC theo shift + histogram lag", 17: "hình 4"}
for r_, t in drawn.items(): put(ws, f"E{r_}", "Rồi", text=f"Hinh du lieu that – {t}")

# Tong quan: nguồn số liệu
ws = wb["Tong quan"]
r0 = ws.max_row + 2
ws.cell(r0, 1, f"Số liệu đã điền ({TODAY})").font = Font(name=F, size=11, bold=True, color="FFFFFF"); ws.cell(r0, 1).fill = HDR
info = [("Model", f"{a.run.as_posix()}/best – YAMNet đóng băng + head 1024→64→32→2; output: {a.run.as_posix()}/raw_real"),
        ("Dữ liệu", f"{S['n_pairs']} file audio thật data/audio ({S['hours']:.2f} giờ, 5 category); dev {S['n_dev']} / test {S['n_test']} file"),
        ("Nhãn", "Nhãn bạc FireRedVAD AED (C:/vad_work/real_test/gt_policy: nói hoặc hát = speech); Silero là người gán nhãn thứ 2"),
        ("Loại số liệu", "Python reference – CHƯA phải output lib.so; Parity, hiệu năng, robustness chưa đo"),
        ("Cách đọc", "Chỉ số đo mức khớp với FireRedVAD, không phải độ chính xác tuyệt đối"),
        ("Ngưỡng dùng", f'=FIXED({KQ}!$B$11,2)&" ("&{KQ}!$B$8&" trên dev thật, "&{KQ}!$B$7&") – chỉnh ở KQ du lieu that, mục Tham số"'),
        ("Sheet thêm", "KQ du lieu that (chỉ số = công thức), Hinh du lieu that (hình vẽ bằng Python), DL … (số đếm thô từ Python, chữ xanh)")]
for i, (k, v) in enumerate(info, 1):
    ws.cell(r0 + i, 1, k).font = Font(name=F, size=11, bold=True)
    c = ws.cell(r0 + i, 2, v); c.font = Font(name=F, size=11); c.alignment = Alignment(wrap_text=True, vertical="top")

wb.calculation.fullCalcOnLoad = True
wb.save(a.out); print("saved", a.out)
