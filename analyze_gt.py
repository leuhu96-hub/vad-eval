"""
analyze_gt.py — Phân tích tập ground truth (GT) cho bài toán face detection + tracking trong video.

Input: một hoặc nhiều file GT định dạng MOT (CVAT export "MOT 1.1"), mỗi file = 1 video:
    frame,id,x,y,w,h,not_ignored,class_id,visibility        (frame bắt đầu từ 1)
  - Cột 7 (not_ignored) = 0  -> box bị bỏ qua khi đánh giá (đếm riêng)
  - Cột 9 (visibility) trong [0,1] nếu có -> phân tích mức độ bị che

Thông tin video (độ phân giải, FPS) lấy theo thứ tự ưu tiên:
  --videos (đọc trực tiếp file video) > seqinfo.ini cạnh thư mục gt (cấu trúc MOTChallenge) > --width/--height/--fps

Cách dùng:
  python analyze_gt.py gt_video1.txt gt_video2.txt --videos v1.mp4 v2.mp4 --out gt_report
  python analyze_gt.py data/*/gt/gt.txt --width 1920 --height 1080 --fps 30 --out gt_report

Output (trong --out):
  report.html            báo cáo có biểu đồ, mở bằng trình duyệt
  suggested_params.json  tham số gợi ý cho eval_face_postproc.py
  issues.csv             danh sách nghi vấn lỗi nhãn (để mở lại trong CVAT kiểm tra)
  events.csv             các đoạn 2 người chồng nhau / đi ngang nhau (dễ đổi ID)
  split.csv              gợi ý chia video thành tập tune / test

Các phần phân tích:
  1. Tổng quan            5. Chuyển động (IoU giữa frame, tốc độ, dấu hiệu camera lia)
  2. Kích thước mặt       6. Tương tác giữa các ID (chồng nhau, đi ngang nhau)
  3. Hình dạng & vị trí   7. Chất lượng nhãn
  4. Thời gian & track    8. Gợi ý chia tập tune / test
"""
import argparse
import base64
import configparser
import csv
import io
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

YUNET_MIN, YUNET_MAX = 10, 300  # YuNet được huấn luyện cho mặt khoảng 10..300 px
SIZE_EDGES = [0, 10, 24, 64, 300, 1e9]
SIZE_NAMES = ["<10", "10-24", "24-64", "64-300", ">300"]
COLORS = ["#1D9E75", "#7F77DD", "#D85A30", "#BA7517", "#888780"]


# =========================================================== đọc dữ liệu
class Video:
    def __init__(self, name, path):
        self.name, self.path = name, path
        self.W = self.H = self.fps = None
        rows = []
        with open(path) as f:
            for r in csv.reader(f):
                if not r or r[0].strip().startswith("#"):
                    continue
                v = [float(x) for x in r[:9]] + [1.0] * max(0, 9 - len(r))
                if len(r) < 9:
                    v[8] = np.nan  # không có visibility
                if len(r) < 7:
                    v[6] = 1.0
                rows.append(v[:9])
        a = np.array(rows, float) if rows else np.zeros((0, 9))
        self.ign = a[a[:, 6] == 0]
        a = a[a[:, 6] != 0]
        self.fr, self.id = a[:, 0].astype(int), a[:, 1].astype(int)
        self.x, self.y, self.w, self.h = a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        self.vis = a[:, 8]
        self.n = len(a)
        self.has_vis = self.n > 0 and np.isfinite(self.vis).any()

    def tracks(self):
        """id -> chỉ số các box, đã sắp theo frame."""
        d = defaultdict(list)
        for i in np.argsort(self.fr, kind="stable"):
            d[self.id[i]].append(i)
        return {k: np.array(v) for k, v in d.items()}

    def by_frame(self):
        d = defaultdict(list)
        for i, f in enumerate(self.fr):
            d[f].append(i)
        return d


def load_meta(v, video_path, a):
    if video_path:
        import cv2

        cap = cv2.VideoCapture(video_path)
        v.W, v.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v.fps = cap.get(cv2.CAP_PROP_FPS) or None
        v.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
    else:
        ini = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(v.path))), "seqinfo.ini")
        if os.path.exists(ini):
            c = configparser.ConfigParser()
            c.read(ini)
            s = c["Sequence"]
            v.W, v.H = int(s.get("imWidth", 0)) or None, int(s.get("imHeight", 0)) or None
            v.fps = float(s.get("frameRate", 0)) or None
            v.nframes = int(s.get("seqLength", 0)) or None
        else:
            v.nframes = None
    v.W, v.H = v.W or a.width, v.H or a.height
    v.fps = v.fps or a.fps
    if not getattr(v, "nframes", None):
        v.nframes = int(v.fr.max()) if v.n else 0


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


# =========================================================== 1. tổng quan
def overview(vs):
    rows = []
    for v in vs:
        bf = v.by_frame()
        per = np.array([len(x) for x in bf.values()]) if bf else np.zeros(1)
        rows.append(dict(video=v.name, res=f"{v.W}x{v.H}" if v.W else "?", fps=v.fps or "?",
                         frames=v.nframes, labeled_frames=len(bf),
                         empty_frames=max(0, v.nframes - len(bf)), boxes=v.n, ids=len(set(v.id)),
                         faces_per_frame_mean=round(float(per.mean()), 2), faces_per_frame_max=int(per.max()),
                         ignored=len(v.ign)))
    return rows


# =========================================================== 2. kích thước
def sizes(vs):
    H = np.concatenate([v.h for v in vs])
    W = np.concatenate([v.w for v in vs])
    side = np.maximum(W, H)
    out = dict(h_p5=pct(H, 5), h_p50=pct(H, 50), h_p95=pct(H, 95))
    scale = {}
    for name, long_side in (("native", None), ("640", 640), ("960", 960), ("1280", 1280)):
        s = []
        for v in vs:
            k = 1.0 if (long_side is None or not v.W) else long_side / max(v.W, v.H)
            s.append(np.maximum(v.w, v.h) * k)
        s = np.concatenate(s)
        hist = np.histogram(s, SIZE_EDGES)[0] / max(len(s), 1)
        scale[name] = dict(zip(SIZE_NAMES, hist.round(4).tolist()))
        scale[name]["in_yunet_range"] = float(((s >= YUNET_MIN) & (s <= YUNET_MAX)).mean())
    out["by_input_size"] = scale
    best = max(scale, key=lambda k: scale[k]["in_yunet_range"])
    out["suggest_size"] = best
    out["side"] = side
    return out


# =========================================================== 3. hình dạng & vị trí
def shape_pos(vs):
    ar = np.concatenate([v.w / np.maximum(v.h, 1e-6) for v in vs])
    cx, cy, edge = [], [], 0
    nb = 0
    for v in vs:
        if not v.W:
            continue
        cx.append((v.x + v.w / 2) / v.W)
        cy.append((v.y + v.h / 2) / v.H)
        e = (v.x <= 2) | (v.y <= 2) | (v.x + v.w >= v.W - 2) | (v.y + v.h >= v.H - 2)
        edge += int(e.sum())
        nb += v.n
    lo, hi = pct(ar, 0.5), pct(ar, 99.5)
    return dict(ar=ar, ar_p=[pct(ar, q) for q in (0.5, 5, 50, 95, 99.5)],
                # nới 10% quanh phân vị 0,5..99,5, nhưng không hẹp hơn [0,6, 1,3]
                suggest_ar=[round(min(lo * 0.9, 0.6), 2), round(max(hi * 1.1, 1.3), 2)],
                cx=np.concatenate(cx) if cx else None, cy=np.concatenate(cy) if cy else None,
                edge_frac=edge / nb if nb else float("nan"))


# =========================================================== 4. thời gian
def temporal(vs):
    lens, gaps, segs, reapp = [], [], [], []
    per_id = []
    for v in vs:
        for tid, idx in v.tracks().items():
            f = v.fr[idx]
            span = f[-1] - f[0] + 1
            g = np.diff(f) - 1
            g = g[g > 0]
            lens.append(len(f))
            gaps += g.tolist()
            segs.append(len(g) + 1)
            fps = v.fps or 25
            long = g[g > fps]  # vắng mặt > 1 giây
            reapp += long.tolist()
            per_id.append(dict(video=v.name, id=int(tid), first=int(f[0]), last=int(f[-1]), boxes=len(f),
                               coverage=round(len(f) / span, 3), segments=len(g) + 1,
                               longest_gap=int(g.max()) if len(g) else 0))
    gaps = np.array(gaps)
    fps = np.median([v.fps for v in vs if v.fps]) if any(v.fps for v in vs) else 25
    short = gaps[gaps <= fps] if len(gaps) else gaps
    mid = gaps[gaps <= 2 * fps] if len(gaps) else gaps
    return dict(track_len=np.array(lens), gaps=gaps, segments=np.array(segs), per_id=per_id,
                n_reappear=len(reapp), reappear=np.array(reapp),
                # max_gap: phủ 95% khoảng trống <= 1 giây, tối thiểu 2 frame
                suggest_max_gap=int(np.clip(np.ceil(pct(short, 95)), 2, fps)) if len(short) else 8,
                # max_lost: phủ 99% khoảng trống <= 2 giây, trong khoảng [0,5 s, 2 s]
                suggest_max_lost=int(np.clip(np.ceil(pct(mid, 99)), fps / 2, 2 * fps)) if len(mid) else int(fps),
                fps=fps)


# =========================================================== 5. chuyển động
def motion(vs):
    ious, disp, scale = [], [], []
    cam_by = {}
    for v in vs:
        cam_pairs, cam_moving = 0, 0
        step = defaultdict(dict)  # frame -> id -> (dx, dy, h)
        for tid, idx in v.tracks().items():
            for a, b in zip(idx, idx[1:]):
                if v.fr[b] - v.fr[a] != 1:
                    continue
                A = (v.x[a], v.y[a], v.w[a], v.h[a])
                B = (v.x[b], v.y[b], v.w[b], v.h[b])
                ious.append(iou_xywh(A, B))
                dx = (B[0] + B[2] / 2) - (A[0] + A[2] / 2)
                dy = (B[1] + B[3] / 2) - (A[1] + A[3] / 2)
                h = max(A[3], 1)
                disp.append(np.hypot(dx, dy) / h)
                scale.append(B[3] / h)
                step[v.fr[b]][tid] = (dx / h, dy / h)
        for f, d in step.items():
            if len(d) < 2:
                continue
            m = np.array(list(d.values()))
            mag = np.linalg.norm(m, axis=1)
            if (mag > 0.08).all():
                u = m / mag[:, None]
                cos = u @ u.T
                cam_pairs += 1
                cam_moving += int((cos[np.triu_indices(len(m), 1)] > 0.9).all())
            else:
                cam_pairs += 1
        cam_by[v.name] = cam_moving / cam_pairs if cam_pairs else float("nan")
    ious, disp, scale = np.array(ious), np.array(disp), np.array(scale)
    return dict(iou=ious, disp=disp, scale=scale,
                iou_lt03=float((ious < 0.3).mean()) if len(ious) else float("nan"),
                iou_lt05=float((ious < 0.5).mean()) if len(ious) else float("nan"),
                disp_p95=pct(disp, 95),
                cam_by=cam_by,
                cam_frac=max([x for x in cam_by.values() if x == x], default=float("nan")),
                suggest_iou1=round(float(np.clip(pct(ious, 2), 0.1, 0.3)), 2) if len(ious) else 0.3)


# =========================================================== 6. tương tác
def interactions(vs, close=1.0):
    events = []
    for v in vs:
        bf = v.by_frame()
        state = {}  # (id1,id2) -> dict
        for f in sorted(bf):
            idx = bf[f]
            seen = set()
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    a, b = idx[i], idx[j]
                    ia, ib = sorted((v.id[a], v.id[b]))
                    if v.id[a] > v.id[b]:
                        a, b = b, a
                    A = (v.x[a], v.y[a], v.w[a], v.h[a])
                    B = (v.x[b], v.y[b], v.w[b], v.h[b])
                    o = iou_xywh(A, B)
                    d = np.hypot((A[0] + A[2] / 2) - (B[0] + B[2] / 2), (A[1] + A[3] / 2) - (B[1] + B[3] / 2))
                    if o > 0 or d < close * max(A[3], B[3]):
                        key = (ia, ib)
                        seen.add(key)
                        order = np.sign((A[0] + A[2] / 2) - (B[0] + B[2] / 2))
                        e = state.get(key)
                        if e is None or f - e["end"] > 1:
                            if e:
                                events.append(e)
                            e = dict(video=v.name, id1=ia, id2=ib, start=f, end=f, max_iou=0.0,
                                     first_order=order, crossed=False)
                            state[key] = e
                        e["end"] = f
                        e["max_iou"] = max(e["max_iou"], round(o, 3))
                        if order != 0 and e["first_order"] != 0 and order != e["first_order"]:
                            e["crossed"] = True
            for key in [k for k in state if k not in seen and state[k]["end"] < f - 1]:
                events.append(state.pop(key))
        events += list(state.values())
    for e in events:
        e["frames"] = e["end"] - e["start"] + 1
        e.pop("first_order", None)
    events.sort(key=lambda e: (-e["crossed"], -e["max_iou"]))
    return events


# =========================================================== 7. chất lượng nhãn
def label_qa(vs):
    issues = []

    def add(v, f, tid, kind, detail):
        issues.append(dict(video=v.name, frame=int(f), id=int(tid), issue=kind, detail=detail))

    for v in vs:
        for i in range(v.n):
            if v.w[i] <= 0 or v.h[i] <= 0:
                add(v, v.fr[i], v.id[i], "kích thước <= 0", f"w={v.w[i]:.1f} h={v.h[i]:.1f}")
            elif max(v.w[i], v.h[i]) < 4:
                add(v, v.fr[i], v.id[i], "box quá nhỏ (<4px)", f"{v.w[i]:.1f}x{v.h[i]:.1f}")
            if v.W and (v.x[i] + v.w[i] < 0 or v.y[i] + v.h[i] < 0 or v.x[i] > v.W or v.y[i] > v.H):
                add(v, v.fr[i], v.id[i], "box nằm ngoài khung", f"x={v.x[i]:.0f} y={v.y[i]:.0f}")
            if v.h[i] > 0 and not (0.3 <= v.w[i] / v.h[i] <= 2.5):
                add(v, v.fr[i], v.id[i], "tỉ lệ w/h bất thường", f"w/h={v.w[i] / v.h[i]:.2f}")
        seen = defaultdict(int)
        for f, t in zip(v.fr, v.id):
            seen[(f, t)] += 1
        for (f, t), c in seen.items():
            if c > 1:
                add(v, f, t, "ID trùng trong 1 frame", f"{c} box")
        for tid, idx in v.tracks().items():
            for a, b in zip(idx, idx[1:]):
                if v.fr[b] - v.fr[a] != 1 or min(v.w[a], v.h[a], v.w[b], v.h[b]) <= 0:
                    continue
                h = max(v.h[a], 1)
                d = np.hypot(v.x[b] + v.w[b] / 2 - v.x[a] - v.w[a] / 2, v.y[b] + v.h[b] / 2 - v.y[a] - v.h[a] / 2)
                r = v.h[b] / h
                if d > 1.0 * h:
                    add(v, v.fr[b], tid, "nhảy vị trí", f"dịch {d / h:.1f} lần chiều cao mặt (có thể nhầm ID)")
                if r > 1.6 or r < 1 / 1.6:
                    add(v, v.fr[b], tid, "nhảy kích thước", f"x{r:.2f}")
            f = v.fr[idx]
            g = np.where(np.diff(f) == 2)[0]
            if len(g) > 5:  # thiếu lặp lại nhiều lần -> gộp 1 dòng
                add(v, f[g[0]] + 1, tid, "thiếu 1 frame giữa track",
                    f"{len(g)} lần từ frame {f[g[0]] + 1} đến {f[g[-1]] + 1} (thiếu theo chu kỳ? kiểm tra cả ID này)")
            else:
                for k in g:
                    add(v, f[k] + 1, tid, "thiếu 1 frame giữa track", "có thể quên gán nhãn")
            # đoạn dài chuyển động thẳng đều tuyệt đối -> dấu hiệu nội suy keyframe của CVAT
            if len(idx) >= 12:
                cxs = v.x[idx] + v.w[idx] / 2
                d2 = np.abs(np.diff(cxs, 2)) + np.abs(np.diff(v.y[idx] + v.h[idx] / 2, 2)) + np.abs(np.diff(v.h[idx], 2))
                lin = d2 < 0.05
                run = 0
                for k, ok in enumerate(lin):
                    run = run + 1 if ok and f[k + 2] - f[k] == 2 else 0
                    if run == 10:
                        add(v, f[k + 2], tid, "đoạn nội suy (keyframe)",
                            "≥ 12 frame chuyển động thẳng đều tuyệt đối: kiểm tra box có còn ôm mặt")
    kinds = defaultdict(int)
    for i in issues:
        kinds[i["issue"]] += 1
    return issues, dict(kinds)


# =========================================================== 8. chia tập
def split(vs, sz, ev, test_frac=0.5):
    if len(vs) < 2:
        return [], "Chỉ có 1 video: nên chia theo đoạn thời gian dài (ví dụ nửa đầu / nửa sau) hoặc gán nhãn thêm video."
    ev_by = defaultdict(int)
    for e in ev:
        ev_by[e["video"]] += 1
    rows = []
    for v in vs:
        k = 640 / max(v.W, v.H) if v.W else 1
        small = float((np.maximum(v.w, v.h) * k < 24).mean()) if v.n else 0
        rows.append(dict(video=v.name, boxes=v.n, small_frac_at640=round(small, 3),
                         interactions=ev_by[v.name]))
    s = np.array([r["small_frac_at640"] for r in rows])
    it = np.array([r["interactions"] for r in rows], float)
    diff = s / (s.max() or 1) + it / (it.max() or 1)
    order = np.argsort(-diff)
    tot = sum(r["boxes"] for r in rows)
    acc = {"tune": 0, "test": 0}
    for rank, i in enumerate(order):
        want = "test" if acc["test"] / tot < test_frac * (acc["test"] + acc["tune"] + rows[i]["boxes"]) / tot else "tune"
        if rank < 2:
            want = ["test", "tune"][rank]
        rows[i]["split"] = want
        rows[i]["difficulty"] = round(float(diff[i]), 3)
        acc[want] += rows[i]["boxes"]
    return rows, f"tune: {acc['tune']} box, test: {acc['test']} box (cân bằng theo độ khó: tỉ lệ mặt nhỏ + số lần tương tác)"


# =========================================================== biểu đồ
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax, title, xl=None, yl=None):
    ax.set_title(title, fontsize=11, loc="left")
    if xl:
        ax.set_xlabel(xl, fontsize=9)
    if yl:
        ax.set_ylabel(yl, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)


def charts(vs, S, SP, T, M):
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    h = SP["side"]
    bins = np.logspace(np.log10(max(h.min(), 2)), np.log10(h.max() + 1), 40)
    ax[0].hist(h, bins=bins, color=COLORS[0])
    ax[0].set_xscale("log")
    for x in (YUNET_MIN, 24, 64, YUNET_MAX):
        ax[0].axvline(x, color="#888780", ls="--", lw=0.8)
    style(ax[0], "Cỡ mặt ở độ phân giải gốc (cạnh dài, px)", "px (log)", "số box")
    names = list(SP["by_input_size"])
    bottom = np.zeros(len(names))
    for k, b in enumerate(SIZE_NAMES):
        vals = np.array([SP["by_input_size"][n][b] for n in names]) * 100
        ax[1].bar(names, vals, bottom=bottom, color=["#D85A30", "#BA7517", "#7F77DD", "#1D9E75", "#888780"][k], label=b)
        bottom += vals
    ax[1].legend(fontsize=7, title="cỡ mặt (px)", title_fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    style(ax[1], "Cỡ mặt sau khi đưa vào model", "cạnh dài input", "% box")
    out["size"] = fig_b64(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(S["ar"], bins=40, color=COLORS[1])
    for x in S["suggest_ar"]:
        ax[0].axvline(x, color="#D85A30", ls="--", lw=1)
    style(ax[0], "Tỉ lệ w/h (vạch đỏ: ngưỡng gợi ý)", "w/h", "số box")
    if S["cx"] is not None:
        ax[1].hist2d(S["cx"], S["cy"], bins=24, range=[[0, 1], [0, 1]], cmap="Greens")
        ax[1].invert_yaxis()
        style(ax[1], "Vị trí tâm mặt trong khung hình", "x / width", "y / height")
    else:
        ax[1].text(0.5, 0.5, "Cần độ phân giải video", ha="center")
        ax[1].axis("off")
    out["shape"] = fig_b64(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].hist(T["track_len"], bins=30, color=COLORS[1])
    style(ax[0], "Độ dài track (số box / ID)", "box", "số ID")
    if len(T["gaps"]):
        ax[1].hist(T["gaps"], bins=np.arange(1, min(T["gaps"].max(), 200) + 2) if T["gaps"].max() < 200 else 50, color=COLORS[3])
        ax[1].axvline(T["suggest_max_gap"], color="#1D9E75", ls="--", lw=1, label="max_gap gợi ý")
        ax[1].axvline(T["suggest_max_lost"], color="#D85A30", ls="--", lw=1, label="max_lost gợi ý")
        ax[1].legend(fontsize=7)
    style(ax[1], "Khoảng trống bên trong 1 ID", "frame vắng mặt", "số lần")
    ax[2].hist(T["segments"], bins=np.arange(1, T["segments"].max() + 2) - 0.5, color=COLORS[0])
    style(ax[2], "Số đoạn của mỗi ID", "đoạn", "số ID")
    out["temporal"] = fig_b64(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    if len(M["iou"]):
        ax[0].hist(M["iou"], bins=40, range=(0, 1), color=COLORS[0])
        ax[0].axvline(0.3, color="#D85A30", ls="--", lw=1)
        ax[1].hist(np.clip(M["disp"], 0, 2), bins=40, color=COLORS[3])
    style(ax[0], "IoU cùng ID giữa 2 frame liên tiếp (vạch: 0,3)", "IoU", "số cặp")
    style(ax[1], "Độ dịch tâm / chiều cao mặt mỗi frame", "lần chiều cao mặt", "số cặp")
    out["motion"] = fig_b64(fig)

    if any(v.has_vis for v in vs):
        vis = np.concatenate([v.vis for v in vs])
        vis = vis[np.isfinite(vis)]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(vis, bins=20, range=(0, 1), color=COLORS[1])
        style(ax, "Visibility (1 = thấy rõ, 0 = bị che hết)", "visibility", "số box")
        out["vis"] = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.28 * min(len(T["per_id"]), 40)))
    rows = sorted(T["per_id"], key=lambda r: (r["video"], r["id"]))[:40]
    for k, r in enumerate(rows):
        v = next(x for x in vs if x.name == r["video"])
        f = np.sort(v.fr[v.id == r["id"]])
        br = np.split(f, np.where(np.diff(f) > 1)[0] + 1)
        ax.broken_barh([(b[0], b[-1] - b[0] + 1) for b in br], (k - 0.35, 0.7), color=COLORS[k % 4])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['video'][:12]}:{r['id']}" for r in rows], fontsize=7)
    style(ax, "Timeline từng ID (khoảng trắng = vắng mặt)" + (" — 40 ID đầu" if len(T["per_id"]) > 40 else ""), "frame")
    out["timeline"] = fig_b64(fig)
    return out


# =========================================================== báo cáo
def table(rows, cols=None, limit=None):
    if not rows:
        return "<p class='m'>Không có.</p>"
    cols = cols or list(rows[0])
    body = "".join("<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>" for r in rows[:limit])
    more = f"<p class='m'>… và {len(rows) - limit} dòng nữa (xem file csv).</p>" if limit and len(rows) > limit else ""
    return f"<div class='t'><table><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr>{body}</table></div>{more}"


def img(b):
    return f"<img src='data:image/png;base64,{b}'/>" if b else ""


def report(path, vs, OV, SP, S, T, M, EV, QA, QAK, SPL, SPLmsg, P, CH):
    f1 = lambda x: f"{x * 100:.1f}%" if x == x else "?"
    nb = sum(v.n for v in vs)
    sz = SP["by_input_size"]
    findings = []
    findings.append(f"Ở 640, <b>{f1(sz['640']['<10'])}</b> mặt nhỏ hơn 10 px (YuNet gần như không thấy); ở native là "
                    f"<b>{f1(sz['native']['<10'])}</b>. Cỡ input giữ nhiều mặt trong vùng 10–300 px nhất: <b>{SP['suggest_size']}</b>.")
    if sz["native"][">300"] > 0.01:
        findings.append(f"{f1(sz['native']['>300'])} mặt lớn hơn 300 px ở native: nên thu nhỏ ảnh hoặc chạy thêm 1 lần ở cỡ nhỏ.")
    findings.append(f"{f1(M['iou_lt03'])} cặp frame liên tiếp có IoU &lt; 0,3: "
                    + ("tracker chỉ dùng IoU sẽ hay mất dấu, cần Kalman tốt / CMC." if M["iou_lt03"] > 0.05 else "chuyển động chậm, tracker IoU + Kalman là đủ."))
    if M["cam_frac"] == M["cam_frac"]:
        lia = [k for k, x in M["cam_by"].items() if x == x and x > 0.2]
        findings.append("Camera lia / rung (mọi người cùng dịch một hướng): "
                        + (f"<b>{', '.join(lia)}</b> → nên bật CMC." if lia else "không thấy, camera khá ổn định."))
    findings.append(f"{T['n_reappear']} lần một ID vắng mặt hơn 1 giây rồi quay lại: "
                    + ("cần bước gộp tracklet dài hạn (HAC)." if T["n_reappear"] else "ít cần gộp dài hạn."))
    nc = sum(e["crossed"] for e in EV)
    findings.append(f"{len(EV)} đoạn 2 người ở sát nhau, trong đó {nc} đoạn đi ngang qua nhau: đây là nơi dễ ID switch nhất "
                    + ("→ nên có ReID." if nc else "."))
    findings.append(f"{len(QA)} nghi vấn lỗi nhãn ({', '.join(f'{k}: {n}' for k, n in QAK.items()) or 'không có'}).")

    html = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phân tích ground truth</title><style>
body{{font:15px/1.6 system-ui,sans-serif;max-width:1000px;margin:auto;padding:20px;color:#1f1e1d;background:#faf9f5}}
h1{{font-size:24px}}h2{{font-size:18px;margin-top:34px;border-bottom:1px solid #e3e1d8;padding-bottom:4px}}
img{{max-width:100%;background:#fff;border:1px solid #e3e1d8;border-radius:8px;margin:6px 0}}
.t{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;margin:8px 0}}td,th{{border:1px solid #e3e1d8;padding:4px 8px;text-align:left}}
th{{background:#f1efe8}}.m{{color:#6b6a65;font-size:13px}}.box{{background:#fff;border:1px solid #e3e1d8;border-radius:10px;padding:10px 16px}}
code{{background:#f1efe8;padding:1px 4px;border-radius:4px}}</style></head><body>
<h1>Phân tích ground truth</h1><p class="m">{len(vs)} video · {nb} box · {sum(len(set(v.id)) for v in vs)} ID</p>
<div class="box"><b>Kết luận nhanh</b><ul>{''.join(f'<li>{x}</li>' for x in findings)}</ul>
<b>Tham số gợi ý cho eval_face_postproc.py</b><br><code>{' '.join(f'--{k.replace("_", "-")} {v}' for k, v in P['cli'].items())}</code>
<p class="m">Tracker: max_lost={P['tracker']['max_lost']}, iou1={P['tracker']['iou1']} · Lọc hình học: ar={P['geometry']['ar']}</p></div>

<h2>1. Tổng quan</h2>{table(OV)}
<h2>2. Kích thước mặt</h2><p>Câu hỏi: chạy model ở độ phân giải nào? YuNet bắt tốt mặt khoảng 10–300 px.</p>{img(CH['size'])}
{table([dict(input=k, **{b: f1(v[b]) for b in SIZE_NAMES}, trong_vung_10_300=f1(v['in_yunet_range'])) for k, v in sz.items()])}
<h2>3. Hình dạng &amp; vị trí</h2><p>Tỉ lệ w/h (phân vị 0,5/5/50/95/99,5): {', '.join(f'{x:.2f}' for x in S['ar_p'])} → ngưỡng lọc gợi ý <b>{S['suggest_ar']}</b>.
Mặt chạm mép khung: <b>{f1(S['edge_frac'])}</b> (hay bị miss / box bị cắt).</p>{img(CH['shape'])}
{('<h3>Visibility</h3>' + img(CH['vis'])) if 'vis' in CH else '<p class="m">GT không có cột visibility. Nên gắn thuộc tính occlusion/blur trong CVAT để sau này tính recall theo từng loại.</p>'}
<h2>4. Thời gian &amp; track</h2><p>Độ dài track trung vị {pct(T['track_len'], 50):.0f} box. Khoảng trống bên trong ID: {len(T['gaps'])} lần,
trung vị {pct(T['gaps'], 50) if len(T['gaps']) else 0:.0f} frame. → <code>max_gap</code> ≈ {T['suggest_max_gap']} (phủ 95% khoảng trống ≤ 1 giây),
<code>max_lost</code> ≈ {T['suggest_max_lost']} (phủ 99% khoảng trống ≤ 2 giây). Vắng &gt; 1 giây rồi quay lại: {T['n_reappear']} lần.</p>
{img(CH['temporal'])}{img(CH['timeline'])}
<h2>5. Chuyển động</h2><p>IoU cùng ID giữa 2 frame: &lt;0,3: <b>{f1(M['iou_lt03'])}</b>, &lt;0,5: {f1(M['iou_lt05'])}. Độ dịch p95 = {M['disp_p95']:.2f} lần chiều cao mặt / frame.
Tỉ lệ frame có dấu hiệu camera lia (trong các frame có ≥ 2 người): {', '.join(f'{k}: {f1(x)}' for k, x in M['cam_by'].items())}.</p>{img(CH['motion'])}
<h2>6. Tương tác giữa các ID</h2><p>{len(EV)} đoạn có 2 người sát nhau (chồng box hoặc cách nhau &lt; 1 chiều cao mặt), {nc} đoạn đi ngang qua nhau.
Mở các đoạn này trong video để xem khi đánh giá tracking.</p>
{table(EV, ['video', 'id1', 'id2', 'start', 'end', 'frames', 'max_iou', 'crossed'], 15)}
<h2>7. Chất lượng nhãn</h2>{table([dict(loai=k, so_luong=n) for k, n in QAK.items()])}
{table(QA, ['video', 'frame', 'id', 'issue', 'detail'], 25)}
<h2>8. Chia tập tune / test</h2><p>{SPLmsg}</p>{table(SPL)}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt", nargs="+", help="file GT định dạng MOT (mỗi file = 1 video)")
    ap.add_argument("--videos", nargs="*", help="file video tương ứng (cùng thứ tự) để lấy độ phân giải/FPS")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--out", default="gt_report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vs = []
    for k, p in enumerate(a.gt):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "gt":  # cấu trúc MOTChallenge: <seq>/gt/gt.txt
            name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))
        v = Video(name, p)
        load_meta(v, a.videos[k] if a.videos and k < len(a.videos) else None, a)
        vs.append(v)
    if not any(v.W for v in vs):
        print("Cảnh báo: không biết độ phân giải video -> bỏ qua phần vị trí và cỡ mặt theo input. Dùng --videos hoặc --width/--height.")

    OV = overview(vs)
    SP = sizes(vs)
    S = shape_pos(vs)
    T = temporal(vs)
    M = motion(vs)
    EV = interactions(vs)
    QA, QAK = label_qa(vs)
    SPL, SPLmsg = split(vs, SP, EV)

    P = dict(cli=dict(size=SP["suggest_size"], max_gap=T["suggest_max_gap"]),
             tracker=dict(max_lost=T["suggest_max_lost"], iou1=M["suggest_iou1"]),
             geometry=dict(ar=S["suggest_ar"]),
             notes=dict(need_cmc=bool(M["cam_frac"] > 0.2) if M["cam_frac"] == M["cam_frac"] else None,
                        need_reid=bool(sum(e["crossed"] for e in EV) > 0),
                        need_hac=bool(T["n_reappear"] > 0)))
    with open(os.path.join(a.out, "suggested_params.json"), "w") as f:
        json.dump(P, f, indent=2, ensure_ascii=False)
    write_csv(os.path.join(a.out, "issues.csv"), QA)
    write_csv(os.path.join(a.out, "events.csv"), EV)
    write_csv(os.path.join(a.out, "split.csv"), SPL)
    write_csv(os.path.join(a.out, "ids.csv"), T["per_id"])
    CH = charts(vs, S, SP, T, M)
    report(os.path.join(a.out, "report.html"), vs, OV, SP, S, T, M, EV, QA, QAK, SPL, SPLmsg, P, CH)

    print(f"{len(vs)} video, {sum(v.n for v in vs)} box, {sum(len(set(v.id)) for v in vs)} ID")
    print(f"Cỡ input gợi ý: {SP['suggest_size']}  | % mặt <10px ở 640: {SP['by_input_size']['640']['<10'] * 100:.1f}%")
    print(f"max_gap={T['suggest_max_gap']}  max_lost={T['suggest_max_lost']}  iou1={M['suggest_iou1']}  ar={S['suggest_ar']}")
    print(f"Tương tác: {len(EV)} đoạn ({sum(e['crossed'] for e in EV)} đi ngang)  | ID quay lại sau >1s: {T['n_reappear']}")
    print(f"Nghi vấn lỗi nhãn: {len(QA)}  {QAK}")
    print(f"-> {os.path.join(a.out, 'report.html')}")


if __name__ == "__main__":
    main()
