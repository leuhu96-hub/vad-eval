"""Điền kết quả evaluate_real.py vào template tổng hợp VAD; mọi chỉ số tính bằng CÔNG THỨC Excel.

evaluate_real.py xuất số đếm thô (<res>/excel/): TP/FN/FP/TN theo ngưỡng × cấu hình × category × vùng bin,
số đếm ROC, bảng nhãn 2×2 (nếu có người gán 2), độ lệch biên. Script này:
  - ghi số đếm vào các sheet "DL ..." (chữ xanh = số nhập từ Python, không sửa tay)
  - sheet "KQ du lieu that": MR, FAR, F1, DCF, AUC, κ, chọn ngưỡng... là công thức; tham số ở ô vàng B6:B10
    (trọng số DCF, cấu hình, tiêu chí chọn ngưỡng, mốc FPR) -> đổi là mọi bảng tính lại
  - sheet "Hinh du lieu that": 4 hình vẽ bằng Python (<res>/fig*.png của evaluate_real.py)
  - các sheet của template lấy số bằng công thức từ KQ du lieu that; chỉ ghi ô vàng/xám, không sửa công thức
    sẵn có; ô xám (ví dụ) đã thay bằng số thật đổi sang nền vàng
Số category, người gán nhãn 2, nhãn --gt-alt, val tổng hợp: lấy theo những gì evaluate_real.py tìm thấy.
Giữ là số từ Python (chữ xanh, có ghi chú): AUC chính xác, CI bootstrap, event-F1, Δbiên, lag.

    python make_excel_real.py --res results_test --out results/Tong_hop_danh_gia_VAD_test.xlsx
    python make_excel_real.py --res results_real --run C:/vad_work/run1     # --run: ghi hash model vào Snapshot
"""
import argparse, hashlib, json
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
ap.add_argument("--res", type=Path, default=Path("results_real"), help="thư mục kết quả của evaluate_real.py")
ap.add_argument("--run", type=Path, help="thư mục run của model (<run>/best/head_weights.npz) để ghi hash vào Snapshot")
ap.add_argument("--yamnet", type=Path, default=Path("C:/vad_work/models/yamnet_embed_float32.tflite"))
a = ap.parse_args()
assert a.template.resolve() != a.out.resolve(), "--out phải khác --template (template luôn giữ trống)"
R, XD = a.res, a.res / "excel"
S = json.load(open(R / "summary.json", encoding="utf-8")); META = S["meta"]
CATS = META["categories"]; HAS_L2, HAS_ALT, HAS_SYN = META["label2"] is not None, META["gt_alt"] is not None, META["synth"]
RESCORE = META.get("rescore", "trung bình các cửa sổ phủ hop")   # cách gộp overlap evaluate_real.py đã dùng
PARAMS = META.get("params", {})                                    # tham số dòng lệnh evaluate_real.py đã dùng
def PR(k, d): return PARAMS.get(k, d)
NB, SEED = PR("n_boot", 1000), PR("seed", 3)
pf = pd.read_csv(R / "per_file.csv").sort_values(["category", "file"]).reset_index(drop=True)
sw, roc = pd.read_csv(XD / "sweep_counts.csv"), pd.read_csv(XD / "roc_counts.csv")
labc = pd.read_csv(XD / "label_counts.csv") if HAS_L2 else None
dls, grid, shift = pd.read_csv(XD / "deltas.csv"), pd.read_csv(R / "grid_dev.csv"), pd.read_csv(R / "shift_sweep.csv")
tA, cat, evt = pd.read_csv(R / "tierA.csv"), pd.read_csv(R / "category.csv"), pd.read_csv(R / "event_by_threshold.csv")
sus, struct = pd.read_csv(R / "suspicious.csv"), pd.read_csv(R / "structure.csv")
HAS_PROF = (R / "dataset_profile.csv").exists()     # đặc trưng tập theo độ phân giải model (kết quả cũ không có)
prof, seg_hist, frac_hist = ((pd.read_csv(R / n) for n in ("dataset_profile.csv", "seg_gap_hist.csv", "window_frac_hist.csv"))
                             if HAS_PROF else (None, None, None))
THR = S["thr_used"]; TODAY = date.today().strftime("%d/%m/%Y")
SRC = (f"Nguồn: {R.as_posix()}/ (evaluate_real.py, {TODAY}). {S['n_pairs']} file; {META['mo_ta_nhan']}. "
       "Python reference, chưa phải output lib.")

F = "Arial"; NAVY, BLUE = "1F4E78", "0000FF"
YEL, HDR = PatternFill("solid", fgColor="FFF2CC"), PatternFill("solid", fgColor=NAVY)
thin = Side(style="thin", color="BFBFBF"); BD = Border(left=thin, right=thin, top=thin, bottom=thin)
FN_, FB_ = Font(name=F, size=10), Font(name=F, size=10, color=BLUE)          # công thức / số nhập từ Python
P, D3, NA = "0.0%", "0.000", "—"

wb = load_workbook(a.template)
NEW = ["KQ du lieu that", "Hinh du lieu that", "DL quet nguong", "DL ROC", "DL theo file", "DL nhan va phan bo", "DL do lech bien"]
for n in NEW + ["Bieu do du lieu that", "DL doan va timeline", "Dac trung tap danh gia"]:
    if n in wb.sheetnames: del wb[n]
kq, hd, dq, dr, dfs, dn, dt = (wb.create_sheet(n) for n in NEW)
dct = wb.create_sheet("Dac trung tap danh gia") if HAS_PROF else None
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
title(dfs, f"Số liệu theo file ({len(pf)} file)", "Chữ xanh = số nhập từ evaluate_real.py (per_file.csv). Cột P–Q là công thức.")
fcols = [("file", "File", None), ("category", "Category", None), ("split", "Tập", None), ("dur_s", "Thời lượng (s)", "0.0"),
         ("speech_ratio", "% speech (nhãn chính)", P), ("speech_ratio_alt", "% speech (nhãn --gt-alt)", P),
         ("l2_ratio", "% speech (người gán 2)", P), ("agree_l2", "Khớp với người gán 2", P),
         ("transitions", "#chuyển trạng thái", None), ("n_speech_seg", "#đoạn speech", None), ("boundary_frac", "% bin biên", P),
         ("auc_file", "AUC file", D3), ("lag_s", "Lag (s)", "0.00"), ("peak_corr", "Tương quan đỉnh", D3), ("mean_score", "Điểm TB", D3)]
head(dfs, 4, [c[1] for c in fcols] + ["1 lớp? (CT)", "Cực đoan? (CT)"])
DF0, DF1 = 5, 4 + len(pf)
for i, row in enumerate(pf.itertuples(index=False), DF0):
    for j, (k, _, fmt) in enumerate(fcols, 1): raw(dfs, i, j, getattr(row, k), fmt)
    raw(dfs, i, 16, f"=IF(OR(E{i}=0,E{i}=1),1,0)", blue=False); raw(dfs, i, 17, f"=IF(OR(E{i}<0.1,E{i}>0.9),1,0)", blue=False)
# cột R trở đi: đặc trưng theo độ phân giải model (per_file.csv mới); để sau P–Q để không đổi địa chỉ cột cũ
pcols = [c for c in [("pct_seg_lt_win", "% đoạn speech < 0.96 s", P), ("pct_gap_lt_win", "% khoảng lặng < 0.96 s", P),
                     ("pct_gap_filled_pp", "% khoảng lặng bị pad + merge lấp", P), ("pct_win_mixed", "% cửa sổ lẫn", P),
                     ("pct_bin_near_trans", "% bin cách biên ≤ 0.48 s", P), ("trans_per_min", "Chuyển trạng thái / phút", "0.0"),
                     ("oracle_F1_A0", "Trần F1 oracle A0", D3), ("oracle_F1_A4", "Trần F1 oracle A4", D3),
                     ("oracle_err_A4", "% bin oracle A4 sai", P), ("pct_bin_low_cov", "% bin < 6 cửa sổ", P)] if c[0] in pf.columns]
if pcols:
    head(dfs, 4, [c[1] for c in pcols], c0=18)
    for i, row in enumerate(pf.itertuples(index=False), DF0):
        for j, (k, _, fmt) in enumerate(pcols, 18): raw(dfs, i, j, getattr(row, k), fmt)
dfs.column_dimensions["A"].width = 24
for c in range(2, 18 + len(pcols)): dfs.column_dimensions[L(c)].width = 12
dfs.freeze_panes = "B5"
def DFc(col): return rg(DF, col, DF0, DF1)

# ======================= DL quet nguong =======================
title(dq, "Số đếm theo ngưỡng", "Tập: dev = dev (chia từ dữ liệu), synth = val tổng hợp (nếu có), test = test theo category. "
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
title(dr, "Số đếm ROC trên điểm thô (tầng A)", f"{len(roc)} ngưỡng = phân vị điểm của mọi file + 2 ngưỡng chặn; speech khi điểm ≥ ngưỡng. "
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
        for k, fml in enumerate([f'=IF({tp}+{fn}=0,"",{tp}/({tp}+{fn}))', f'=IF({fp}+{tn}=0,"",{fp}/({fp}+{tn}))',
                                 f"=IF({tp}+{fp}=0,1,{tp}/({tp}+{fp}))"]):
            raw(dr, i, mc[g] + k, fml, D3, blue=False)
dr.column_dimensions["A"].width = 10; dr.freeze_panes = "B5"
def rocc(g, k, r0=DR0, r1=DR1): return rg(DR, L(mc[g] + k), r0, r1)   # k: 0 TPR, 1 FPR, 2 Precision

# ======================= DL nhan va phan bo =======================
title(dn, "Nhãn, độ dịch thời gian, lưới hậu xử lý", "Chữ xanh = số từ Python; chữ đen = công thức.")
r = 4
if HAS_L2:
    sec(dn, r, f"a. Bảng 2×2 nhãn so sánh vs người gán 2 ({META['label2']}), theo category × vùng bin")
    head(dn, r + 1, ["Category", "Vùng bin", "n11 (cả hai speech)", "n10 (chỉ nhãn so sánh)", "n01 (chỉ người gán 2)", "n00",
                     "#bin khác nhãn --gt-alt", "#bin nhãn chắc", "#bin speech (nhãn chính)"])
    LB0 = r + 2; LB1 = LB0 + len(labc) - 1
    for i, row in enumerate(labc.itertuples(index=False), LB0):
        for j, k in enumerate(["category", "mask", "n11", "n10", "n01", "n00", "bin_khac_alt", "n_nhan_chac", "n_speech_gt"], 1):
            raw(dn, i, j, getattr(row, k))
    r = LB1 + 2
else:
    sec(dn, r, "a. Không có người gán nhãn thứ 2 (--label2 / --silero-cache) – bỏ qua bảng đồng thuận nhãn"); r += 2
sec(dn, r, "b. Quét độ dịch điểm (score đặt tại tâm cửa sổ)")
head(dn, r + 1, ["Shift (s)", "F1 start (thr 0.5)", "F1 tâm (thr 0.5)", "AUC nhãn chính", "AUC người gán 2", "AUC val tổng hợp",
                 "ΔAUC nhãn chính", "ΔAUC người gán 2", "ΔAUC val tổng hợp"])
SH0 = r + 2; SH1 = SH0 + len(shift) - 1
for i, row in enumerate(shift.itertuples(index=False), SH0):
    for j, k in enumerate(["shift_s", "F1_start", "F1_center", "AUC_gt", "AUC_l2", "AUC_synth"], 1):
        raw(dn, i, j, getattr(row, k), "0.00" if j == 1 else "0.0000")
    for j, src, col in zip((7, 8, 9), "DEF", ("AUC_gt", "AUC_l2", "AUC_synth")):
        if shift[col].notna().any():
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
DL0, DL1 = 5, max(5, 4 + len(dls))
for i, row in enumerate(dls.itertuples(index=False), DL0): raw(dt, i, 1, row.d_onset, "0.00"); raw(dt, i, 2, row.d_offset, "0.00")
for c in "AB": dt.column_dimensions[c].width = 12

# ======================= KQ du lieu that =======================
title(kq, "Kết quả đánh giá – chỉ số tính bằng công thức", SRC)
kq["A3"] = ("Nền vàng = tham số bạn chỉnh (B6:B10) · chữ đen = công thức · chữ xanh = số nhập từ Python (bootstrap, event, "
            "AUC chính xác…; không tính lại trong Excel). Đổi tham số thì mọi bảng và sheet template liên kết tự tính lại; hình vẽ bằng Python thì không.")
kq["A3"].font = Font(name=F, size=9, italic=True, color="595959")
kq.column_dimensions["A"].width = 46
for c in range(2, 26): kq.column_dimensions[L(c)].width = 12

sec(kq, 5, "Tham số")
for r_, lab, v, fmt in [(6, "Trọng số miss trong DCF (FA = 1 − ô này)", PR("dcf_miss", 0.75), "0.00"), (7, "Cấu hình hậu xử lý cho tầng B, category", "A4", None),
                        (8, "Tiêu chí chọn ngưỡng trên dev", PR("criterion", "DCF-min"), None), (9, "Mốc FPR 1 (TPR@FPR)", 0.01, P), (10, "Mốc FPR 2 (TPR@FPR)", 0.05, P)]:
    w(kq, r_, 1, lab, left=True); w(kq, r_, 2, v, fmt, fill=YEL)
for ref, opts in (("B7", '"A0,A1,A2,A3,A4"'), ("B8", '"DCF-min,F1-max"')):
    dv = DataValidation(type="list", formula1=opts, allow_blank=False); kq.add_data_validation(dv); dv.add(ref)
w(kq, 11, 1, "Ngưỡng dùng (tự chọn trên dev theo B7, B8)", left=True, bold=True)
w(kq, 11, 2, "=INDEX($I$17:$I$21,MATCH($B$7,$A$17:$A$21,0))", "0.00", bold=True)
w(kq, 12, 1, "Ngưỡng Python đã dùng cho số theo đoạn, CI, Δbiên, hình", left=True); w(kq, 12, 2, THR, "0.00", blue=True)
w(kq, 13, 1, "Kiểm tra", left=True)
w(kq, 13, 2, '=IF(AND(ABS(B11-B12)<0.0000001,B7="A4"),"Khớp","KHÁC: số theo đoạn, CI, Δbiên, hình vẫn ở ngưỡng Python / A4")', left=True)

# A. chọn ngưỡng (hàng cố định 15–22)
sec(kq, 15, "A. Chọn ngưỡng trên dev" + (" / val tổng hợp" if HAS_SYN else "") + " (qua toàn pipeline; DCF theo trọng số B6)")
head(kq, 16, ["Cấu hình", "Mô tả", "Ngưỡng DCF-min (dev)", "DCF min (dev)", "Ngưỡng F1-max (dev)", "F1 max (dev)",
              "Ngưỡng DCF-min (val tổng hợp)", "Ngưỡng F1-max (val tổng hợp)", "Ngưỡng dùng (theo B8)"])
MODES = ["A0", "A1", "A2", "A3", "A4"]
DESC = META.get("config_desc", ["Điểm bin thô, không hậu xử lý", "Rescore + rebin", "+ pad 100/120 ms", "+ merge < 500 ms",
                                "+ pad + merge (hiện tại)"])
ROW_A = {m: 17 + i for i, m in enumerate(MODES)}
def arg(b, col_key, col_val, fn):   # ngưỡng tại MIN/MAX của một khối dev/synth
    r0, r1 = b; return f"=INDEX({rg(DQ, col_key, r0, r1)},MATCH({fn}({rg(DQ, col_val, r0, r1)}),{rg(DQ, col_val, r0, r1)},0))"
for m, desc in zip(MODES, DESC):
    r_, bdev = ROW_A[m], blocks[("dev", m)]
    w(kq, r_, 1, m); w(kq, r_, 2, desc, left=True)
    w(kq, r_, 3, arg(bdev, "E", "R", "MIN"), "0.00"); w(kq, r_, 4, f"=MIN({rg(DQ, 'R', *bdev)})", D3)
    w(kq, r_, 5, arg(bdev, "E", "Q", "MAX"), "0.00"); w(kq, r_, 6, f"=MAX({rg(DQ, 'Q', *bdev)})", D3)
    if HAS_SYN:
        bsyn = blocks[("synth", m)]; w(kq, r_, 7, arg(bsyn, "E", "R", "MIN"), "0.00"); w(kq, r_, 8, arg(bsyn, "E", "Q", "MAX"), "0.00")
    else:
        w(kq, r_, 7, NA); w(kq, r_, 8, NA)
    w(kq, r_, 9, f'=IF($B$8="F1-max",E{r_},C{r_})', "0.00", bold=True)
w(kq, 22, 1, "Tham khảo (Python): ngưỡng Youden trên điểm thô – val tổng hợp / dev", left=True)
w(kq, 22, 3, S["thr_youden_raw_synth"] if HAS_SYN else NA, D3, blue=True); w(kq, 22, 5, S["thr_youden_raw_dev"], D3, blue=True)

# B. tầng B (hàng cố định 24–28)
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
note(kq["N26"], f"Bootstrap {NB} lần theo file, ở ngưỡng Python {THR} và A4 (xem B12, B13).")

# C. theo category (từ đây số hàng phụ thuộc số category)
r = 30; sec(kq, r, "C. Theo category (test; cấu hình B7, ngưỡng B11)")
head(kq, r + 1, ["Category", "#file", "TP", "FN", "FP", "TN", "FEC", "MSC", "OVER", "NDS", "TP ±1", "FN ±1", "FP ±1", "TN ±1",
                 "% speech", "MR", "FAR", "Precision", "F1 strict", "F1 ±1 bin", "DCF", "Lỗi biên (FEC+OVER)", "Lỗi thật (MSC+NDS)",
                 "F1 CI95 thấp (Python)", "F1 CI95 cao (Python)"])
ROW_C = {c: r + 2 + i for i, c in enumerate(CATS)}; R_TOT = r + 2 + len(CATS); R_MAC = R_TOT + 1
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
c0, c1 = ROW_C[CATS[0]], ROW_C[CATS[-1]]
w(kq, R_TOT, 1, "Tổng test (gộp)", left=True, bold=True)
for k in range(2, 15): w(kq, R_TOT, k, f"=SUM({L(k)}{c0}:{L(k)}{c1})")
cat_rates(R_TOT); w(kq, R_TOT, 24, S["F1_CI"][0], D3, blue=True); w(kq, R_TOT, 25, S["F1_CI"][1], D3, blue=True)
w(kq, R_MAC, 1, f"Macro (TB {len(CATS)} category)", left=True, bold=True)
for k in range(15, 22): w(kq, R_MAC, k, f"=AVERAGE({L(k)}{c0}:{L(k)}{c1})", D3 if k >= 19 else P)
note(kq[f"X{c0}"], f"CI 95% bootstrap theo file ({max(100, NB // 2)} lần) ở ngưỡng Python {THR}, A4.")

# D. ablation
r = R_MAC + 2; sec(kq, r, f"D. Ablation hậu xử lý (test; ngưỡng chọn riêng cho từng cấu hình trên dev theo B8; rescore: {RESCORE})")
head(kq, r + 1, ["Cấu hình", "Mô tả", "Ngưỡng", "TP", "FN", "FP", "TN", "TP ±1", "FN ±1", "FP ±1", "TN ±1", "FEC", "MSC", "OVER", "NDS",
                 "MR", "FAR", "F1 strict", "F1 ±1 bin", "DCF"])
ROW_D = {m: r + 2 + i for i, m in enumerate(MODES)}
for m, desc in zip(MODES, DESC):
    r_ = ROW_D[m]; w(kq, r_, 1, m); w(kq, r_, 2, desc, left=True); w(kq, r_, 3, f"=$I${ROW_A[m]}", "0.00")
    for k, col in enumerate("FGHI"): w(kq, r_, 4 + k, sif(col, f"$A{r_}", f"$C{r_}", '"all"'))
    for k, col in enumerate("FGHI"): w(kq, r_, 8 + k, sif(col, f"$A{r_}", f"$C{r_}", '"nb"'))
    for k, col in enumerate("JKLM"): w(kq, r_, 12 + k, sif(col, f"$A{r_}", f"$C{r_}", '"all"'))
    f = rates(f"D{r_}", f"E{r_}", f"F{r_}", f"G{r_}"); fnb = rates(f"H{r_}", f"I{r_}", f"J{r_}", f"K{r_}")
    w(kq, r_, 16, f["MR"], P); w(kq, r_, 17, f["FAR"], P); w(kq, r_, 18, f["F1"], D3); w(kq, r_, 19, fnb["F1"], D3)
    w(kq, r_, 20, dcf(f"P{r_}", f"Q{r_}"), D3)

# E. so sánh ngưỡng
r = ROW_D["A4"] + 2; sec(kq, r, "E. Chọn ngưỡng ở đâu → kết quả trên test")
head(kq, r + 1, ["Ngưỡng chọn từ", "Cấu hình", "Ngưỡng", "TP", "FN", "FP", "TN", "MR", "FAR", "Precision", "Recall", "F1", "DCF"])
e_rows = ([("DCF-min – val tổng hợp", "A0", "$G$17"), ("DCF-min – val tổng hợp", "A4", "$G$21"), ("F1-max – val tổng hợp", "A4", "$H$21")]
          if HAS_SYN else []) + [("DCF-min – dev", "A0", "$C$17"), ("F1-max – dev", "A4", "$E$21"), ("DCF-min – dev", "A4", "$C$21")]
for r_, (lab, m, ref) in enumerate(e_rows, r + 2):
    w(kq, r_, 1, lab, left=True); w(kq, r_, 2, m); w(kq, r_, 3, f"={ref}", "0.00")
    for k, col in enumerate("FGHI"): w(kq, r_, 4 + k, sif(col, f"$B{r_}", f"$C{r_}", '"all"'))
    f = rates(f"D{r_}", f"E{r_}", f"F{r_}", f"G{r_}")
    for k, key in enumerate(["MR", "FAR", "Prec", "Rec", "F1"]): w(kq, r_, 8 + k, f[key], D3 if key == "F1" else P)
    w(kq, r_, 13, dcf(f"H{r_}", f"I{r_}"), D3)

# F. tầng A
r = r + 2 + len(e_rows) + 1; sec(kq, r, f"F. Tầng A – điểm thô (AUC hình thang trên {len(roc)} ngưỡng ở sheet DL ROC)")
head(kq, r + 1, ["Tập", "#bin", "% speech", "AUC (công thức)", "AUC Python (chính xác)", "Chênh", "PR-AUC (công thức)",
                 "TPR @ FPR mốc 1", "TPR @ FPR mốc 2", "AUC macro theo file", "AUC không biên (Python)", "AUC chỉ biên (Python)",
                 "AUC nhãn chắc (Python)", "Brier (Python)"])
ROW_F = {}
for i, (r_, tr_) in enumerate(zip(range(r + 2, r + 2 + len(tA)), tA.to_dict("records"))):
    lab = tr_["Tập"]
    g = "Test" if i == 0 else ("Dev" if lab.startswith("Dev") else ("Tất cả" if lab.startswith("Cả ") else lab))
    split = '"test"' if g not in ("Dev", "Tất cả") else ('"dev"' if g == "Dev" else None)
    ROW_F[g] = r_; tpc, fpc = L(cc[g]), L(cc[g] + 1)
    t0, t1, f0, f1 = rocc(g, 0, DR0, DR1 - 1), rocc(g, 0, DR0 + 1, DR1), rocc(g, 1, DR0, DR1 - 1), rocc(g, 1, DR0 + 1, DR1)
    w(kq, r_, 1, lab, left=True); w(kq, r_, 2, f"={DR}!{tpc}{DR0}+{DR}!{fpc}{DR0}"); w(kq, r_, 3, f"={DR}!{tpc}{DR0}/B{r_}", P)
    w(kq, r_, 4, f"=SUMPRODUCT(({f0}-{f1})*({t0}+{t1}))/2", "0.0000")
    w(kq, r_, 5, tr_["AUC_micro"], "0.0000", blue=True); w(kq, r_, 6, f"=D{r_}-E{r_}", "0.0000")
    w(kq, r_, 7, f"=SUMPRODUCT(({t0}-{t1})*{rocc(g, 2, DR0, DR1 - 1)})", D3)
    for k, m_ in ((8, "$B$9"), (9, "$B$10")): w(kq, r_, k, f'=_xlfn.MAXIFS({rocc(g, 0)},{rocc(g, 1)},"<="&{m_})', P)
    crit = (f",{DFc('C')},{split}" if split else "") + (f',{DFc("B")},"{g}"' if g in CATS else "")
    w(kq, r_, 10, f"=AVERAGEIFS({DFc('L')}{crit})" if crit else f"=AVERAGE({DFc('L')})", D3)
    for k, col in zip(range(11, 15), ["AUC_khong_bien", "AUC_chi_bien", "AUC_nhan_chac", "Brier"]): w(kq, r_, k, tr_[col], D3, blue=True)
R_AUCCI = r + 2 + len(tA); R_AUCSYN = R_AUCCI + 1
w(kq, R_AUCCI, 1, f"CI95 AUC test (Python, bootstrap {NB} lần theo file)", left=True)
w(kq, R_AUCCI, 2, S["AUC_test_CI"][0], D3, blue=True); w(kq, R_AUCCI, 3, S["AUC_test_CI"][1], D3, blue=True)
w(kq, R_AUCSYN, 1, f"AUC val tổng hợp ({S['n_synth_val']} file, Python)", left=True)
w(kq, R_AUCSYN, 2, S["auc_synth_val"] if HAS_SYN else NA, D3, blue=True)

# G. nhãn
r = R_AUCSYN + 2
R_G = {}
if HAS_L2:
    sec(kq, r, f"G. Nhất quán nhãn: {'nhãn --gt-alt' if HAS_ALT else 'nhãn chính'} vs {META['label2']} – κ tính từ bảng 2×2")
    head(kq, r + 1, ["Category", "Vùng bin", "n11", "n10", "n01", "n00", "#bin", "% speech nhãn so sánh", "% speech người gán 2", "Khớp", "κ",
                     "#bin khác --gt-alt", "#bin speech (nhãn chính)", "#bin nhãn chắc", "% speech (nhãn chính)", "% bin nhãn chắc"])
    lcats = [c for c in META["all_categories"] if c in set(labc.category)]
    for r_, (c, mk) in enumerate([(c, "all") for c in lcats] + [("Tất cả", "all"), ("Tất cả", "trong"), ("Tất cả", "bien")], r + 2):
        if c == "Tất cả": R_G[mk] = r_
        w(kq, r_, 1, c, left=True); w(kq, r_, 2, mk)
        crit = f',{rg(DN, "B", LB0, LB1)},$B{r_}' + (f',{rg(DN, "A", LB0, LB1)},$A{r_}' if c != "Tất cả" else "")
        for k, col in zip((3, 4, 5, 6, 12, 13, 14), "CDEFGIH"): w(kq, r_, k, f"=SUMIFS({rg(DN, col, LB0, LB1)}{crit})")
        w(kq, r_, 7, f"=SUM(C{r_}:F{r_})")
        for k, fml in ((8, f"(C{r_}+D{r_})/G{r_}"), (9, f"(C{r_}+E{r_})/G{r_}"), (10, f"(C{r_}+F{r_})/G{r_}"), (15, f"M{r_}/G{r_}"), (16, f"N{r_}/G{r_}")):
            w(kq, r_, k, f'=IF(G{r_}=0,"",{fml})', P)
        pe = f"((C{r_}+D{r_})*(C{r_}+E{r_})+(E{r_}+F{r_})*(D{r_}+F{r_}))/G{r_}^2"
        w(kq, r_, 11, f'=IFERROR((J{r_}-{pe})/(1-{pe}),"")', D3)
    R_G_first = r + 2; r = R_G["bien"] + 2
else:
    sec(kq, r, "G. Nhất quán nhãn: không có người gán nhãn thứ 2 (--label2 hoặc --silero-cache) – bỏ qua"); r += 2

# H. phân bố
sec(kq, r, "H. Phân bố và kiểm tra theo file (từ DL theo file)")
dist = [("n_file", "Số file", f"=COUNTA({DFc('A')})", None), ("hours", "Số giờ audio", f"=SUM({DFc('D')})/3600", "0.00"),
        ("n_deg", "Số file chỉ có 1 lớp", f"=SUM({DFc('P')})", None), ("pct_deg", "% file chỉ có 1 lớp", "=B{n_deg}/B{n_file}", P),
        ("pct_ext", "% file cực đoan (speech < 0.1 hoặc > 0.9)", f"=SUM({DFc('Q')})/B{{n_file}}", P),
        ("med_tr", "Trung vị số lần chuyển trạng thái / file", f"=MEDIAN({DFc('I')})", "0.0"),
        ("auc_pool", f"AUC gộp {len(pf)} file (công thức, mục F)", f"=D{ROW_F['Tất cả']}", D3),
        ("auc_mac", "AUC macro theo file", f"=AVERAGE({DFc('L')})", D3), ("auc_sd", "SD AUC theo file", f"=STDEV({DFc('L')})", D3),
        ("auc_gap", "Chênh AUC gộp − macro", "=B{auc_pool}-B{auc_mac}", D3), ("n_lt09", "Số file AUC < 0.9", f'=COUNTIF({DFc("L")},"<0.9")', None),
        ("n_lt07", "Số file AUC < 0.7", f'=COUNTIF({DFc("L")},"<0.7")', None), ("auc_min", "AUC file tệ nhất", f"=MIN({DFc('L')})", D3),
        ("worst", "File tệ nhất", f"=IFERROR(INDEX({DFc('A')},MATCH(B{{auc_min}},{DFc('L')},0)),\"\")", None),
        ("lag_med", "Trung vị lag (s)", f'=IFERROR(MEDIAN({DFc("M")}),"")', "0.00"),
        ("n_lag", "Số file |lag| ≥ 0.4 s", f'=COUNTIF({DFc("M")},">=0.4")+COUNTIF({DFc("M")},"<=-0.4")', None)]
ROW_H = {k: r + 1 + i for i, (k, *_) in enumerate(dist)}
for k, lab, fml, fmt in dist:
    w(kq, ROW_H[k], 1, lab, left=True); w(kq, ROW_H[k], 2, fml.format(**ROW_H), fmt)
def H(k): return f"{KQ}!$B${ROW_H[k]}"

# I. theo đoạn (Python)
r = ROW_H["n_lag"] + 2; sec(kq, r, "I. Chỉ số theo đoạn (Python, test, A4) – cần ghép từng đoạn nên không tính lại trong Excel")
head(kq, r + 1, ["Ngưỡng", "Giá trị", "Event-F1", "Onset-F1", "#đoạn GT", "#đoạn dự đoán", "#GT bị phân mảnh", "#dự đoán gộp ≥2 GT"])
R_EVT0 = r + 2
for r_, row in enumerate(evt.itertuples(index=False), R_EVT0):
    for k, (v, fmt) in enumerate([(row.Ngưỡng, None), (row.thr, "0.00"), (row.event_F1, D3), (row.onset_F1, D3), (row.n_gt_seg, None),
                                  (row.n_pred_seg, None), (row.n_fragmented, None), (row.n_overmerged, None)], 1):
        w(kq, r_, k, v, fmt, blue=True, left=k == 1)
r = R_EVT0 + len(evt)
w(kq, r, 1, "Δonset TB (s), ngưỡng Python", left=True); w(kq, r, 2, f"=IFERROR(AVERAGE({rg(DT, 'A', DL0, DL1)}),\"\")", "0.00")
w(kq, r + 1, 1, "Δoffset TB (s), ngưỡng Python", left=True); w(kq, r + 1, 2, f"=IFERROR(AVERAGE({rg(DT, 'B', DL0, DL1)}),\"\")", "0.00")
w(kq, r + 2, 1, "Khoảng lặng GT ≤ 1 s bị lấp (Python)", left=True); w(kq, r + 2, 2, f"{S['gap_short_filled']}/{S['gap_short_total']}", blue=True)

# J. nhãn đáng ngờ
r += 4; sec(kq, r, f"J. 30 đoạn nghi nhãn sai dài nhất (tổng {S['n_suspicious_seg']} đoạn; đủ danh sách: {R.as_posix()}/suspicious.csv) – nghe lại")
scol = [("file", "File", None), ("category", "Category", None), ("split", "Tập", None), ("start", "Bắt đầu (s)", "0.0"),
        ("end", "Kết thúc (s)", "0.0"), ("n_bins", "#bin", None), ("GT", "Nhãn", None), ("score", "Điểm TB", D3)]
scol += ([("l2_speech", "Người gán 2: speech", "0%")] if HAS_L2 else []) + ([("khac_alt", "Khác nhãn --gt-alt", None)] if HAS_ALT else [])
head(kq, r + 1, [c[1] for c in scol])
for r_, row in enumerate(sus.head(30).itertuples(index=False), r + 2):
    for k, (col, _, fmt) in enumerate(scol, 1): w(kq, r_, k, getattr(row, col), fmt, blue=True, left=k == 1)
kq.freeze_panes = "B5"

# ======================= Dac trung tap danh gia (nếu evaluate_real.py có xuất) =======================
if HAS_PROF:
    from openpyxl.formatting.rule import FormulaRule
    FG = S.get("profile_fill_gap_s", PR("merge_gap", 0.5) + PR("pad_pre", 0.1) + PR("pad_post", 0.12))
    title(dct, "Đặc trưng tập đánh giá theo độ phân giải model",
          f"{SRC} Nhãn bin 0.5 s; model cửa sổ 0.96 s, hop 0.08 s (lưới chung 0.02 s). Chỉ tính từ nhãn và vị trí cửa sổ, không dùng "
          "điểm model. Chữ xanh = số từ Python; cột dev − test là công thức, chữ đỏ khi lệch > 10 điểm %.")
    FEAT = [  # (cột dataset_profile.csv, tên, ý nghĩa, định dạng); "" = dòng tiêu đề nhóm
        ("", "Cỡ mẫu", "", None),
        ("n_file", "Số file", "", None), ("minutes", "Thời lượng (phút)", "", "0.0"), ("n_bin", "Số bin 0.5 s", "", None),
        ("n_win_indep", "Số cửa sổ 0.96 s không chồng nhau", "thời lượng / 0.96 s: cỡ mẫu độc lập ở độ phân giải model", "0"),
        ("pct_speech", "% bin speech", "", P),
        ("trans_per_min", "Số lần chuyển trạng thái / phút", "càng cao càng nhiều bin sát biên", "0.0"),
        ("", "Đoạn speech so với cửa sổ", "", None),
        ("n_seg", "Số đoạn speech", "cỡ mẫu cho event-F1", None),
        ("seg_lt_win", "% đoạn < 0.96 s", "ngắn hơn cửa sổ: không cửa sổ nào nằm trọn trong đoạn", P),
        ("seg_win_146", "% đoạn 0.96–1.46 s", "chứa được 1 cửa sổ, chưa đủ cửa sổ + 1 bin", P),
        ("seg_146_2", "% đoạn 1.46–2 s", "", P), ("seg_ge_2", "% đoạn ≥ 2 s", "đủ dài: có bin ở giữa không bị biên ảnh hưởng", P),
        ("speech_time_in_short_seg", "% thời gian speech trong đoạn < 0.96 s", "phần speech model khó thấy trọn", P),
        ("median_seg_s", "Trung vị độ dài đoạn (s)", "", "0.00"),
        ("seg_dropped_pp", "% đoạn bị drop", "đoạn + pad ngắn hơn --drop (0 khi không dùng drop)", P),
        ("", "Khoảng lặng giữa hai đoạn speech", "", None),
        ("n_gap", "Số khoảng lặng", "", None),
        ("gap_lt_win", "% khoảng lặng < 0.96 s", "ngắn hơn cửa sổ", P), ("gap_win_146", "% khoảng lặng 0.96–1.46 s", "", P),
        ("gap_146_2", "% khoảng lặng 1.46–2 s", "", P), ("gap_ge_2", "% khoảng lặng ≥ 2 s", "", P),
        ("gap_filled_pp", f"% khoảng lặng < {FG:g} s (pad + merge lấp)", "model đúng hoàn toàn vẫn bị tính FA ở A2–A4", P),
        ("median_gap_s", "Trung vị độ dài khoảng lặng (s)", "", "0.00"),
        ("", "Cửa sổ 0.96 s theo tỉ lệ speech của nhãn", "", None),
        ("win_pure_speech", "% cửa sổ thuần speech", "", P), ("win_pure_ns", "% cửa sổ thuần non-speech", "", P),
        ("win_mixed", "% cửa sổ lẫn", "chứa cả speech lẫn non-speech: điểm tuỳ cách model hiểu nhãn cửa sổ", P),
        ("", "Khoảng cách từ tâm bin tới chuyển trạng thái gần nhất", "", None),
        ("bin_dist_le_048", "% bin ≤ 0.48 s", "bin biên: cửa sổ đặt tâm trong bin (A0) đã chứa biên", P),
        ("bin_dist_048_096", "% bin 0.48–0.96 s", "A0 không chạm biên; A1–A4 (gộp cửa sổ quanh hop) còn chạm", P),
        ("bin_dist_096_144", "% bin 0.96–1.44 s", "chỉ A1–A4 có làm mượt còn chạm biên", P),
        ("bin_dist_gt_144", "% bin > 1.44 s", "xa biên: không cửa sổ nào góp vào điểm bin chứa biên", P),
        ("bin_low_cov", "% bin < 6 tâm cửa sổ", "đầu / cuối file: điểm A0 dựa trên ít cửa sổ hoặc nội suy", P),
        ("", "Trần oracle: model hoàn hảo cùng cửa sổ 0.96 s (điểm = tỉ lệ speech, ngưỡng 0.5)", "", None)]
    for mode, desc in (("A0", "điểm bin thô"), ("A4", "hậu xử lý hiện tại")):
        FEAT += [(f"oracle_{mode}_F1", f"Oracle {mode} F1", f"{desc}; < 1 là phần mất do độ phân giải, không phải lỗi model", D3),
                 (f"oracle_{mode}_MR", f"Oracle {mode} MR", "", P), (f"oracle_{mode}_FAR", f"Oracle {mode} FAR", "", P),
                 (f"oracle_{mode}_DCF", f"Oracle {mode} DCF", f"trọng số miss {PR('dcf_miss', 0.75)} (Python, không theo ô B6)", D3),
                 (f"oracle_{mode}_err", f"Oracle {mode} % bin sai", "", P)]
    pv = prof.set_index(["Category", "split"]); PC = list(dict.fromkeys(prof.Category)); TAPS = ["dev", "test", "tất cả"]
    r = 4; sec(dct, r, "a. Đặc trưng theo category × tập (số gộp từ số đếm, không trung bình theo file)")
    head(dct, r + 1, ["Đặc trưng", "Ý nghĩa"] + [f"{c}\n{t}" for c in PC for t in TAPS + ["dev − test"]])
    red = Font(name=F, size=10, bold=True, color="C00000")
    for i, (k, name, desc, fmt) in enumerate(FEAT, r + 2):
        if not k:
            sec(dct, i, name); continue
        w(dct, i, 1, name, left=True); w(dct, i, 2, desc, left=True)
        for j, c in enumerate(PC):
            c0 = 3 + 4 * j
            for t_, tap in enumerate(TAPS):
                w(dct, i, c0 + t_, pv.at[(c, tap), k] if (c, tap) in pv.index else None, fmt, blue=True)
            if k in ("n_file", "minutes", "n_bin", "n_win_indep", "n_seg", "n_gap"):
                w(dct, i, c0 + 3, None); continue
            dv, tv = f"{L(c0)}{i}", f"{L(c0 + 1)}{i}"
            w(dct, i, c0 + 3, f'=IF(OR({dv}="",{tv}=""),"",{dv}-{tv})', fmt)
            if fmt == P:
                dct.conditional_formatting.add(f"{L(c0 + 3)}{i}", FormulaRule(formula=[f"AND(ISNUMBER({L(c0 + 3)}{i}),ABS({L(c0 + 3)}{i})>0.1)"], font=red))
    r = r + 2 + len(FEAT) + 1
    sec(dct, r, "b. Phân bố độ dài đoạn speech / khoảng lặng (số đoạn theo độ dài, toàn bộ file); nhãn theo bin nên độ dài là bội của 0.5 s")
    LCOL = [c for c in seg_hist.columns if c not in ("Category", "kind", "n")]; nL = len(LCOL)
    head(dct, r + 1, ["Category", "Loại", "Số đoạn"] + [f"{c} s" for c in LCOL] + [f"% {c} s" for c in LCOL])
    for i, (_, row) in enumerate(seg_hist.iterrows(), r + 2):
        w(dct, i, 1, row["Category"], left=True); w(dct, i, 2, row["kind"], left=True); w(dct, i, 3, row["n"], blue=True)
        for j, c in enumerate(LCOL):
            w(dct, i, 4 + j, row[c], blue=True); w(dct, i, 4 + nL + j, f'=IF($C{i}=0,"",{L(4 + j)}{i}/$C{i})', P)
    r = r + 2 + len(seg_hist) + 1
    sec(dct, r, "c. Cửa sổ 0.96 s theo tỉ lệ speech của nhãn trong cửa sổ (toàn bộ file)")
    FCOL = [c for c in frac_hist.columns if c not in ("Category", "n_win")]; nF = len(FCOL)
    head(dct, r + 1, ["Category", "", "Số cửa sổ"] + FCOL + [f"% {c}" for c in FCOL])
    for i, (_, row) in enumerate(frac_hist.iterrows(), r + 2):
        w(dct, i, 1, row["Category"], left=True); w(dct, i, 2, None); w(dct, i, 3, row["n_win"], blue=True)
        for j, c in enumerate(FCOL):
            w(dct, i, 4 + j, row[c], blue=True); w(dct, i, 4 + nF + j, f'=IF($C{i}=0,"",{L(4 + j)}{i}/$C{i})', P)
    r = r + 2 + len(frac_hist) + 1
    sec(dct, r, "d. Thang thời gian dùng ở trên")
    for i, t in enumerate([
            "0.5 s = bin nhãn (độ phân giải của nhãn); 0.08 s = hop model; 0.96 s = cửa sổ model; lưới chung 0.02 s nên mọi tỉ lệ đều chính xác.",
            "Đoạn / khoảng lặng < 0.96 s: không cửa sổ nào nằm trọn bên trong -> model không bao giờ thấy 'thuần' đoạn đó.",
            "0.96–1.46 s: chứa được 1 cửa sổ nhưng chưa đủ cửa sổ + 1 bin; ≥ 2 s: có bin ở giữa không chịu ảnh hưởng của biên.",
            f"Khoảng lặng < merge gap + pad trước + pad sau = {FG:g} s bị hậu xử lý lấp kể cả khi model đúng hoàn toàn.",
            "Khoảng cách tới biên: A0 gộp các cửa sổ có tâm trong bin (tầm ±0.73 s quanh tâm bin); A1–A4 gộp cửa sổ quanh từng hop "
            "rồi làm mượt (tầm tới ~1.4 s).",
            "Trần oracle: điểm mỗi cửa sổ = tỉ lệ speech thật trong cửa sổ, qua đúng pipeline A0 / A4 ở ngưỡng 0.5. "
            "Khoảng cách giữa kết quả model và trần này mới là phần model có thể cải thiện."], r + 1):
        c = dct.cell(i, 1, t); c.font = FN_
    dct.column_dimensions["A"].width = 40; dct.column_dimensions["B"].width = 52
    for c in range(3, 3 + max(4 * len(PC), 3 + 2 * max(nL, nF))): dct.column_dimensions[L(c)].width = 10
    if (R / "fig6_do_dai_doan.png").exists():
        r = r + 8; sec(dct, r, "e. Độ dài đoạn speech / khoảng lặng so với frame 0.5 s (hình Python, fig6_do_dai_doan.png)")
        img = XLImage(str(R / "fig6_do_dai_doan.png")); k = 1100 / img.width; img.width, img.height = 1100, int(img.height * k)
        dct.add_image(img, f"A{r + 1}")
    dct.freeze_panes = "C6"

# ======================= Hinh du lieu that (vẽ bằng Python) =======================
title(hd, "Hình – đánh giá (vẽ bằng Python)",
      f"{SRC} Hình cố định ở ngưỡng Python {THR} (A4); đổi tham số ở KQ du lieu that không đổi hình – chạy lại evaluate_real.py.")
r = 4
for f, t in [("fig1_kiem_tra_du_lieu.png", "Hình 1. Kiểm tra dữ liệu"), ("fig2_tang_A.png", "Hình 2. Tầng A (test)"),
             ("fig3_tang_B.png", "Hình 3. Ngưỡng và tầng B"), ("fig4_timeline.png", "Hình 4. Timeline mẫu (test)"),
             ("fig5_dac_trung_tap.png", "Hình 5. Đặc trưng tập theo độ phân giải model"),
             ("fig6_do_dai_doan.png", "Hình 6. Độ dài đoạn speech / khoảng lặng so với frame 0.5 s")]:
    if not (R / f).exists(): continue
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

# Bao cao category <- KQ mục C (template có 10 hàng: 5–14)
ws = wb["Bao cao category"]
for i, c in enumerate(CATS[:10]):
    r_, kr = 5 + i, ROW_C[c]
    put(ws, f"A{r_}", c, yellow=r_ == 5)
    for col, kc in zip("BCDEFGHIJKLM", ["B", "O", "P", "Q", "G", "H", "I", "J", "S", "T", "X", "Y"]):
        put(ws, f"{col}{r_}", f"={KQ}!{kc}{kr}", yellow=r_ == 5)
note(ws["A5"], f"Công thức lấy từ KQ du lieu that, mục C (test, cấu hình KQ!B7, ngưỡng KQ!B11). CI là số Python ở ngưỡng {THR}."
               + (f" Có {len(CATS)} category, template chỉ có 10 hàng." if len(CATS) > 10 else "") + f" {SRC}", 320, 160)

# Ablation <- KQ mục D
ws = wb["Ablation"]
for m in MODES:
    r_, kr = 5 + MODES.index(m), ROW_D[m]
    for col, kc in zip("GHIJ", "PQRS"): put(ws, f"{col}{r_}", f"={KQ}!{kc}{kr}", yellow=True)
    put(ws, f"M{r_}", f'="Ngưỡng "&{KQ}!$B$8&" trên dev = "&FIXED({KQ}!C{kr},2)')
ROW_OVERLAP = {"nearest": 16, "mean": 17, "max": 18, "median": 19, "tri": 20}[META.get("rescore_weight", "mean")]   # mục B: O0–O4
if ROW_OVERLAP != 16:
    for c in "FGHIJ": put(ws, f"{c}16", None, yellow=True)          # O0 chưa đo -> bỏ số ví dụ
    put(ws, "M16", "Chưa đo")
PRE, POST, GAP, DROP = (round(PR(k, d) * 1000) for k, d in (("pad_pre", .10), ("pad_post", .12), ("merge_gap", .50), ("drop", 0)))
for r_, vals in zip(range(5, 10), [(0, 0, 0), (0, 0, 0), (PRE, POST, 0), (0, 0, GAP), (PRE, POST, GAP)]):   # mục A: pad/merge (ms)
    for col, v in zip("CDE", vals): put(ws, f"{col}{r_}", v)
for col, v in zip("BCDE", (PRE, POST, GAP, DROP)): put(ws, f"{col}24", v)                                   # mục C: dòng C0 = cấu hình đang dùng
seg_ratio = S["n_pred_seg"] / S["n_gt_seg"] if S["n_gt_seg"] else None
for r_, txt in ((ROW_OVERLAP, f"Giống A4 ({RESCORE})"), (24, f"Giống A4 (drop {DROP} ms)" if DROP else "Giống A4 (không drop)"),
                 (32, "Giống A4: padding → drop → merge" if DROP else "Giống A4: padding → merge, drop = 0")):
    for col, kc in zip("GHIJ", "PQRS"): put(ws, f"{col}{r_}", f"={KQ}!{kc}{ROW_D['A4']}", yellow=True)
    put(ws, f"M{r_}", txt)
    if r_ != ROW_OVERLAP: put(ws, f"F{r_}", seg_ratio, yellow=True, text=f"Python: {S['n_pred_seg']} đoạn dự đoán / {S['n_gt_seg']} đoạn GT ở ngưỡng {THR}.")

# Bien va timestamp (mục C)
ws = wb["Bien va timestamp"]
syn_peak = S["best_shift_auc"].get("AUC_synth")
put(ws, "C24", round(S["best_shift_center"] * 1000), yellow=True,
    text=f"Python: quét F1 (thr 0.5) trên nhãn chính, đỉnh {S['best_shift_center']*1000:+.0f} ms. "
         + (f"Theo AUC trên nhãn chính xác (val tổng hợp): đỉnh {syn_peak*1000:+.0f} ms. " if syn_peak is not None else "")
         + f"AUC trên nhãn chính tăng {S['auc_gain_best_shift_real']:.4f} nếu dịch tới đỉnh.")
put(ws, "C25", int((~struct.win_count_ok).sum()), text="Python: số cửa sổ = floor((N_mẫu − 15600)/1280) + 1. Đo trên output model, chưa phải lib."
    + ("" if S["n_audio_checked"] else " Không có audio nên không kiểm tra được."))

# KPI va Gate
ws = wb["KPI va Gate"]
put(ws, "F10", f"={KQ}!$D${ROW_F['Test']}")
put(ws, "F11", f"={KQ}!$C${R_EVT0}", text=f"Python: event-F1 ở ngưỡng {THR} (KQ du lieu that, mục I).")
kpi_notes = {5: f'="F1 gộp toàn test = "&FIXED({KQ}!$S${R_TOT},3)&"; ô Baseline = TB {len(CATS)} category. Ngưỡng "&FIXED({KQ}!$B$11,2)&" ("&{KQ}!$B$8&" dev, "&{KQ}!$B$7&")"',
             6: f'="F1 ±1 bin gộp = "&FIXED({KQ}!$T${R_TOT},3)',
             7: f'="DCF gộp = "&FIXED({KQ}!$U${R_TOT},3)&" (trọng số miss "&FIXED({KQ}!$B$6,2)&")"',
             8: f'="MR gộp = "&FIXED({KQ}!$P${R_TOT}*100,1)&"%"', 9: f'="FAR gộp = "&FIXED({KQ}!$Q${R_TOT}*100,1)&"%"',
             10: f'="AUC hình thang (công thức); Python chính xác = "&FIXED({KQ}!$E${ROW_F["Test"]},3)&", CI95 "&FIXED({KQ}!$B${R_AUCCI},3)&"–"&FIXED({KQ}!$C${R_AUCCI},3)',
             11: f'="Python ở ngưỡng "&FIXED({KQ}!$B$12,2)&"; phụ thuộc cách nhãn cắt đoạn"',
             15: "Đo trên output Python; xem comment Bien va timestamp!C24."}
for r_, t in kpi_notes.items(): put(ws, f"S{r_}", t)

# Kiem tra du lieu (Bước 8): mục đo được bằng công thức hoặc số Python; còn lại là mô tả
ws = wb["Kiem tra du lieu"]
prob = S["problems"]; n_aud = S["n_audio_checked"]
gt_peak = S["best_shift_center"]; peak = syn_peak if syn_peak is not None else gt_peak
checks = {
 "5a": (f"{S['n_pairs']} file ghép đủ output model ↔ nhãn" + (" ↔ audio" if S["n_audio"] else " (không có audio)") + f"; {len(prob)} vấn đề",
        "Đạt" if not prob else "Không đạt", "; ".join(f"{k}: {v}" for k, v in prob[:5]) or None),
 "5b": (f"{S['n_struct_flag']} file lệch số bin / số cửa sổ / thời lượng", "Đạt" if not S["n_struct_flag"] else "Không đạt",
        "n_bins = floor(thời lượng / 0.5)" + ("" if n_aud else "; không có audio nên chỉ kiểm tra được hop / cửa sổ")),
 "5c": ((f"{S['n_audio_16k_mono']}/{n_aud} file 16 kHz mono", "Đạt" if S["n_audio_16k_mono"] == n_aud else "Không đạt", None)
        if n_aud else ("Không có audio / không đọc được", "Chưa làm", None)),
 "5d": (f"hop sai {S['n_hop_bad']}, cửa sổ sai {S['n_win_bad']}, score ngoài [0,1] {S['n_score_bad']}",
        "Đạt" if S["n_hop_bad"] + S["n_win_bad"] + S["n_score_bad"] == 0 else "Không đạt", None),
 "5e": (f"Đỉnh F1 (thr 0.5) {gt_peak*1000:+.0f} ms trên nhãn chính"
        + (f"; đỉnh AUC trên nhãn chính xác (val tổng hợp) {syn_peak*1000:+.0f} ms" if syn_peak is not None else ""),
        "Đạt" if abs(peak) <= 0.08 + 1e-9 else "Không đạt",
        (f"Dùng val tổng hợp (nhãn chính xác) làm mốc; " if syn_peak is not None else "") +
        f"AUC trên nhãn chính tăng {S['auc_gain_best_shift_real']:.4f} nếu dịch; {len(S['lag_flag_files'])} file |lag| ≥ 0.4 s (xem tương quan đỉnh)."),
 "1a": (f'={H("n_deg")}&"/"&{H("n_file")}&" file ("&FIXED({H("pct_deg")}*100,1)&"%) chỉ có 1 lớp"',
        f'=IF({H("pct_deg")}>0.1,"Không đạt","Đạt")', "Loại khỏi AUC theo file; giữ trong chỉ số gộp. Quy tắc: > 10% -> Không đạt."),
 "1b": (f'=FIXED({H("pct_ext")}*100,1)&"% file có tỉ lệ speech < 0.1 hoặc > 0.9"', f'=IF({H("pct_ext")}>0.3,"Không đạt","Đạt")', None),
 "1c": (f'="Trung vị "&FIXED({H("med_tr")},1)&" lần chuyển trạng thái / file"', f'=IF({H("med_tr")}<=1,"Không đạt","Đạt")', None),
 "1d": (f'="AUC gộp "&FIXED({H("auc_pool")},3)&" vs macro theo file "&FIXED({H("auc_mac")},3)&"; "&{H("n_lt09")}&" file < 0.9, tệ nhất "'
        f'&FIXED({H("auc_min")},3)&" ("&{H("worst")}&")"', f'=IF(OR(ABS({H("auc_gap")})>0.005,{H("n_lt09")}>0),"Không đạt","Đạt")',
        "Xem Hinh du lieu that, hình 1 và 4."),
 "2a": ("Chưa có metadata nguồn / người nói; dev/test chia theo file, phân tầng category", "Chưa làm", None),
 "2b": (None, "Chưa làm", None),
 "2c": ("Cần danh sách nguồn dữ liệu train để đối chiếu", "Chưa làm", None),
 "3a": ((f'="κ nhãn – {META["label2"]} = "&FIXED({KQ}!$K${R_G["all"]},2)&" (trong "&FIXED({KQ}!$K${R_G["trong"]},2)&", biên "&FIXED({KQ}!$K${R_G["bien"]},2)&")"',
         f'=IF({KQ}!$K${R_G["trong"]}<0.9,"Không đạt","Đạt")', "KQ du lieu that, mục G") if HAS_L2 else
        ("Không có người gán nhãn thứ 2", "Chưa làm", "Thêm --label2 (nhãn người gán 2) khi chạy evaluate_real.py")),
 "3b": (f"{S['n_suspicious_seg']} đoạn / {S['n_suspicious_bins']} bin điểm trái ngược mạnh với nhãn"
        + (f"; người gán 2 đồng ý với model ở {S['suspicious_l2_agrees_model']:.0%} số bin" if S.get("suspicious_l2_agrees_model") is not None else ""),
        "Đang làm", f"KQ du lieu that mục J; đủ danh sách: {R.as_posix()}/suspicious.csv"),
 "4a": (f"{S['pct_ambiguous']:.1%} bin có 0.1<p<0.9; F1 thay đổi {S['f1_range_0.2_0.9']:.3f} trong ngưỡng 0.2–0.9",
        "Không đạt" if S["pct_ambiguous"] < 0.02 and S["f1_range_0.2_0.9"] < 0.02 else "Đạt", None),
 "4b": (f"Category: {', '.join(CATS)}", "Đang làm", "Kiểm tra thủ công có đủ nền khó: nhạc có lời, đám đông, TV, tiếng cười"),
 "4c": ("Không có metadata SNR", "Chưa làm", None),
 "4d": ((f'="AUC val tổng hợp "&FIXED({KQ}!$B${R_AUCSYN},3)&" vs test "&FIXED({KQ}!$D${ROW_F["Test"]},3)', "Đạt", "Báo cả hai; ưu tiên số thật")
        if HAS_SYN else ("Không có val tổng hợp (syn_val_*) để so", "Chưa làm", None)),
}
for r_ in range(5, 23):
    code = ws[f"A{r_}"].value
    if code in checks:
        g, h, i_ = checks[code]; put(ws, f"G{r_}", g); put(ws, f"H{r_}", h); put(ws, f"I{r_}", i_)

# Quy trinh 8 buoc
ws = wb["Quy trinh 8 buoc"]
steps = {5: ("Đạt", f"Dev {S['n_dev']} / test {S['n_test']} file, chia theo file, phân tầng category (seed {SEED}). Không có metadata nguồn để chia theo nhóm."),
         6: ("Đạt", "Bin 0.5 s từ nhãn chính; cờ biên ±1 bin" + ("; bin 'nhãn chắc' = các nhãn và người gán 2 cùng đồng ý." if HAS_L2 else ".")),
         7: ("Đang làm", "Alignment 5a–5e xong. Rò rỉ 2a–2c chưa làm (thiếu metadata)."),
         8: ("Đạt", "KQ du lieu that mục F: AUC, PR-AUC, TPR@FPR bằng công thức; tất cả / không biên" + (" / nhãn chắc." if HAS_L2 else ".")
               + " Chưa vẽ reliability diagram."),
         9: ("Đang làm", "KQ du lieu that mục A, E: ngưỡng chọn bằng công thức theo tiêu chí ở B8. Cần chốt tiêu chí (DCF hay FAR cố định)."),
         10: ("Đạt", "KQ du lieu that mục B, C, I: MR/FAR/F1/DCF 3 chế độ, FEC/MSC/OVER/NDS, event-F1, Δbiên. Chưa tính DetER."),
         11: ("Đạt", "KQ du lieu that mục D + heatmap DCF (Hinh du lieu that, hình 3)."),
         12: ("Đang làm", "Có CI + per-category. Chưa nghe kiểm tra top-K (mục J).")}
for r_, (g, h) in steps.items(): put(ws, f"G{r_}", g); put(ws, f"H{r_}", h)

# Ke hoach baseline: bằng chứng nhóm Q
ws = wb["Ke hoach baseline"]
ev = {8: "DL theo file (cột Tập)", 9: "DL theo file, DL nhan va phan bo", 10: "KQ du lieu that mục H; DL nhan va phan bo mục b; Hinh du lieu that, hình 1",
      11: "KQ du lieu that mục F; DL ROC", 12: "KQ du lieu that mục A, E; DL quet nguong", 13: "KQ du lieu that mục B, C, I",
      14: "KQ du lieu that mục D; DL nhan va phan bo mục c; hình 3", 15: f"KQ du lieu that; Hinh du lieu that; {R.as_posix()}/"}
for r_, t in ev.items(): put(ws, f"O{r_}", t)

# Snapshot (cột Baseline)
ws = wb["Snapshot"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "(không tìm thấy file)"
snap = {"C23": RESCORE, "C24": f"={KQ}!$B$11", "C25": PRE, "C26": POST, "C27": DROP, "C28": GAP,
        "C31": f"{META['mo_ta_nhan']}; chia theo file, phân tầng category, seed {SEED}",
        "C32": f"{META['mo_ta_nhan']}; chia theo file, phân tầng category, seed {SEED}",
        "C33": f'=COUNTIFS({DFc("C")},"dev")&" / "&COUNTIFS({DFc("C")},"test")', "C34": f'=SUMIFS({DFc("D")},{DFc("C")},"test")/3600',
        "C37": "evaluate_real.py + make_excel_real.py (vad_demo)", "C38": f"0 / {NB}"}
if S["windows_drop_tail"]: snap["C22"] = "Bỏ phần dư (chỉ cửa sổ đủ 15 600 mẫu)"
if a.run:
    head_w = a.run / "best" / "head_weights.npz"
    snap.update({"C5": f"{a.run.name}/best", "C8": f"{a.yamnet.name} + {a.run.name}/best/head_weights.npz",
                 "C9": f"yamnet: {sha(a.yamnet)}\nhead: {sha(head_w)}", "C11": "float32"})
for ref, v in snap.items(): put(ws, ref, v, fmt="0.00" if ref == "C34" else None)
note(ws["C24"], "Công thức = KQ du lieu that!B11 (tiêu chí ở B8, trọng số DCF ở B6). Cần chốt tiêu chí trước khi đóng băng.", 320, 100)

# Bieu do can ve
ws = wb["Bieu do can ve"]
drawn = {5: "hình 2", 6: "hình 2", 8: "hình 3 – chỉ sau hậu xử lý A4" + (", dev vs val tổng hợp" if HAS_SYN else ""), 9: "hình 3", 10: "hình 3",
         11: "hình 3 – heatmap DCF (không phải F1)", 13: "hình 1 – chấm theo category thay histogram + boxplot",
         15: "hình 1 – chưa có F1 theo ngưỡng", 16: "hình 1 – ΔAUC theo shift + histogram lag", 17: "hình 4"}
for r_, t in drawn.items(): put(ws, f"E{r_}", "Rồi", text=f"Hinh du lieu that – {t}")

# Tong quan: nguồn số liệu
ws = wb["Tong quan"]
r0 = ws.max_row + 2
ws.cell(r0, 1, f"Số liệu đã điền ({TODAY})").font = Font(name=F, size=11, bold=True, color="FFFFFF"); ws.cell(r0, 1).fill = HDR
info = [("Model", META["mo_ta_model"] + (f"; head: {a.run.as_posix()}/best" if a.run else "")),
        ("Dữ liệu", f"{S['n_pairs']} file ({S['hours']:.2f} giờ), {len(CATS)} category: {', '.join(CATS)}; dev {S['n_dev']} / test {S['n_test']} file"
                    + (f"; thư mục {META['root']}" if META["root"] else "")),
        ("Nhãn", META["mo_ta_nhan"]),
        ("Người gán nhãn 2", META["label2"] or "Không có – bỏ qua phần đồng thuận nhãn"),
        ("Val tổng hợp", f"{S['n_synth_val']} file syn_val" if HAS_SYN else "Không có – chỉ chọn ngưỡng trên dev"),
        ("Loại số liệu", "Python reference – CHƯA phải output lib.so; Parity, hiệu năng, robustness chưa đo"),
        ("Ngưỡng dùng", f'=FIXED({KQ}!$B$11,2)&" ("&{KQ}!$B$8&" trên dev, "&{KQ}!$B$7&") – chỉnh ở KQ du lieu that, mục Tham số"'),
        ("Sheet thêm", "KQ du lieu that (chỉ số = công thức), Hinh du lieu that (hình Python), DL … (số đếm thô từ Python, chữ xanh)"
         + (", Dac trung tap danh gia (đặc trưng tập theo độ phân giải model)" if HAS_PROF else ""))]
for i, (k, v) in enumerate(info, 1):
    ws.cell(r0 + i, 1, k).font = Font(name=F, size=11, bold=True)
    c = ws.cell(r0 + i, 2, v); c.font = Font(name=F, size=11); c.alignment = Alignment(wrap_text=True, vertical="top")

wb.calculation.fullCalcOnLoad = True
wb.save(a.out); print("saved", a.out)
