# Demo đánh giá VAD với dữ liệu và model giả định

Chạy lại từ đầu (Python 3, numpy, pandas, scikit-learn, scipy, matplotlib, openpyxl, soundfile):

    python gen_data.py      # sinh Model/, Groundtruth/, Annotator2/, meta.csv, train_sources.csv
    python evaluate.py      # kiểm tra dữ liệu + tầng A + chọn ngưỡng + tầng B + ablation -> results/
    python make_excel.py    # xuất Excel -> results/Ket_qua_danh_gia_VAD_gia_dinh.xlsx

## Đánh giá một bộ dữ liệu có nhãn (dữ liệu thật / dữ liệu mới)

Cấu trúc thư mục cần có (tên thư mục con không phân biệt hoa thường):

    <root>/Model/<Cat>_<id>.txt                         output model, mỗi cửa sổ 0.96 s một dòng
                                                        ('idx, start, end, score' hoặc 'start end score')
    <root>/.../audio/<Cat>/<Cat>_<id>.wav               audio (wav/flac/mp3…; không bắt buộc)
    <root>/.../Ground_truth/<Cat>/groundtruth/Gt_<cat>_<id>.txt   (hoặc <Cat>/<Cat>_<id>.txt)
                                                        mỗi dòng 'start end speech|non-speech',
                                                        theo bin 0.5 s hoặc theo thời điểm

Chạy (ghi ra thư mục / file riêng cho từng bộ dữ liệu để không đè nhau):

    python evaluate_real.py --root "C:/Users/leuhu/OneDrive/Máy tính/test" --out results_test
    python make_excel_real.py --res results_test --out results/Tong_hop_danh_gia_VAD_test.xlsx

Tuỳ chọn thêm cho evaluate_real.py (có thì dùng, không có thì bỏ qua phần tương ứng):
- `--label2 <thư mục>`: nhãn của người gán thứ 2, cùng định dạng -> đồng thuận nhãn (κ), bin "nhãn chắc".
- `--silero-cache <data/auto_labels/cache>`: dùng Silero (cache của auto_label.py) làm người gán thứ 2.
- `--gt-alt <thư mục>`: nhãn theo quy tắc khác (vd hát = non-speech) -> bảng tầng A phụ.
- File `syn_val_*` trong Model/ + `--synth-gt`: chọn ngưỡng trên val tổng hợp, mốc căn chỉnh nhãn chính xác.
- `--mo-ta-nhan`, `--mo-ta-model`: mô tả ghi vào báo cáo; `--raw`, `--gt`, `--audio`: chỉ rõ thư mục thay cho --root.

Tinh chỉnh (mặc định = cấu hình hiện tại; giá trị đã dùng được lưu trong summary.json và ghi vào Excel):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--overlap tri\|mean\|max\|median\|nearest` | tri | Gộp điểm các cửa sổ chồng nhau thành điểm mỗi hop 0.08 s (Ablation mục B: O4, O1, O2, O3, O0) |
| `--smooth N` | 5 | Làm mượt trung bình trượt N hop; 1 = không |
| `--pad-pre`, `--pad-post` | 0.10, 0.12 | Nới đầu / cuối đoạn speech (giây) |
| `--merge-gap` | 0.50 | Gộp hai đoạn cách nhau ít hơn N giây |
| `--drop` | 0 | Bỏ đoạn ngắn hơn N giây (padding → drop → merge), chỉ ở A4 |
| `--criterion DCF-min\|F1-max` | DCF-min | Tiêu chí chọn ngưỡng trên dev (Excel lấy làm mặc định ô B8) |
| `--dcf-miss` | 0.75 | Trọng số miss trong DCF (Excel lấy làm mặc định ô B6) |
| `--thr-min`, `--thr-max`, `--thr-step` | 0.01, 0.95, 0.01 | Dải quét ngưỡng |
| `--grid-pre`, `--grid-post`, `--grid-gap` | 0,50,100,200 / 0,60,120,240 / 0,250,500,750,1000,1500 | Lưới quét pad × merge trên dev (ms) |
| `--seed`, `--n-boot` | 3, 1000 | Seed chia dev/test, số lần bootstrap |

Chạy lại bộ audio thật trước đây (data/ đang nằm trong git, lấy lại bằng `git restore data`):

    python evaluate_real.py --raw C:/vad_work/run1/raw_real --gt C:/vad_work/real_test/gt_policy --audio data/audio \
        --gt-alt data/auto_labels/Groundtruth --silero-cache data/auto_labels/cache --out results_real
    python make_excel_real.py --res results_real --run C:/vad_work/run1

Kết quả:
- `<out>/`: csv, summary.json, 4 hình fig1–fig4, log; `<out>/excel/` = số đếm thô cho Excel.
- Excel (từ template trống results/Tong_hop_danh_gia_VAD_template.xlsx, template không bị ghi):
  - KQ du lieu that: MR, FAR, F1, DCF, AUC, κ, chọn ngưỡng đều là công thức; ô vàng B6:B10 là tham số
    (trọng số DCF, cấu hình, tiêu chí chọn ngưỡng, mốc FPR), đổi là mọi bảng tính lại. Số bảng theo số category.
  - Hinh du lieu that: 4 hình vẽ bằng Python ở ngưỡng Python; đổi tham số trong Excel không đổi hình.
  - DL …: số đếm thô từ Python (chữ xanh), không sửa tay.
  - Các sheet template (Bao cao category, Ablation, KPI va Gate, Snapshot, Kiem tra du lieu…) liên kết bằng
    công thức tới KQ du lieu that; chỉ ghi ô vàng/xám, giữ nguyên công thức của template.
  - Số chỉ Python tính được (CI bootstrap, event-F1, Δbiên, AUC chính xác) giữ là số, chữ xanh, có ghi chú.

## Các file

- vadlib.py: đọc file, gom bin 0.5s, hậu xử lý (rescore hop, pad, merge, rebin), metric.
  Đọc được output model cả dạng 'idx, start, end, score' lẫn 'start end score'.
  rescore_hop (gộp điểm các cửa sổ chồng nhau thành điểm mỗi hop 0.08 s, dùng cho A1–A4): mặc định trọng số
  tam giác theo tâm cửa sổ + làm mượt trung bình 5 hop. Đổi ở RESCORE_WEIGHT / RESCORE_SMOOTH đầu file
  ("mean", 1 = cách cũ). evaluate.py (dữ liệu giả định) cũng dùng hàm này.
- auto_label.py: gán nhãn tự động cho data/audio bằng FireRedVAD AED (+ Silero đối chiếu) -> data/auto_labels/
  (Groundtruth/ đúng format, review.csv = đoạn cần nghe lại, summary.csv). Quy tắc nhãn ở docstring đầu file.
- evaluate_real.py, make_excel_real.py: xem mục trên.
- evaluate.py chỉ dùng cho dữ liệu giả định 10 s (cần meta.csv, Annotator2/); dữ liệu khác dùng evaluate_real.py.
- Lỗi cài sẵn trong dữ liệu giả định: file thiếu cặp, 3 file lệch +0.5s, file toàn speech/non-speech,
  ~50% file dùng nguồn âm có trong train, nhạc có lời và babble gây false alarm.
