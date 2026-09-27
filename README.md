# Demo đánh giá VAD với dữ liệu và model giả định

Chạy lại từ đầu (Python 3, numpy, pandas, scikit-learn, scipy, matplotlib, openpyxl):

    python gen_data.py      # sinh Model/, Groundtruth/, Annotator2/, meta.csv, train_sources.csv
    python evaluate.py      # kiểm tra dữ liệu + tầng A + chọn ngưỡng + tầng B + ablation -> results/
    python make_excel.py    # xuất Excel -> results/Ket_qua_danh_gia_VAD_gia_dinh.xlsx

Audio thật (data/audio, output model ở C:/vad_work/run1/raw_real):

    python evaluate_real.py     # -> results_real/ (csv, summary.json, 4 hình, excel/ = số đếm thô cho Excel)
    python make_excel_real.py   # template -> results/Tong_hop_danh_gia_VAD_baseline_real.xlsx

- vadlib.py: đọc file, gom bin 0.5s, hậu xử lý (rescore hop, pad, merge, rebin), metric.
  Đọc được output model cả dạng 'idx, start, end, score' lẫn 'start end score'.
- auto_label.py: gán nhãn tự động cho data/audio bằng FireRedVAD AED (+ Silero đối chiếu) -> data/auto_labels/
  (Groundtruth/ đúng format, review.csv = đoạn cần nghe lại, summary.csv). Quy tắc nhãn ở docstring đầu file.
- evaluate_real.py: cùng quy trình cho audio thật data/audio (file dài khác nhau, Silero thay Annotator2,
  ngưỡng chọn trên dev thật và trên val tổng hợp) -> results_real/. Mặc định đọc C:/vad_work/run1/raw_real
  và nhãn C:/vad_work/real_test/gt_policy (nói hoặc hát = speech).
- make_excel_real.py: đọc template trống results/Tong_hop_danh_gia_VAD_template.xlsx (không bị ghi) và
  results_real/excel/, ghi ra Tong_hop_danh_gia_VAD_baseline_real.xlsx:
  - sheet DL …: số đếm thô từ Python (chữ xanh) – TP/FN/FP/TN theo ngưỡng × cấu hình × category, ROC, nhãn 2×2…
  - KQ du lieu that: MR, FAR, F1, DCF, AUC, κ, chọn ngưỡng đều là công thức; ô vàng B6:B10 là tham số
    (trọng số DCF, cấu hình, tiêu chí chọn ngưỡng, mốc FPR), đổi là mọi bảng và biểu đồ tính lại
  - Bieu do du lieu that: 19 biểu đồ Excel đọc từ ô (không có ảnh)
  - các sheet template (Bao cao category, Ablation, KPI va Gate, Snapshot, Kiem tra du lieu…) liên kết
    bằng công thức tới KQ du lieu that; chỉ ghi ô vàng/xám, giữ nguyên công thức của template
  Số chỉ Python tính được (CI bootstrap, event-F1, Δbiên, AUC chính xác) giữ là số, chữ xanh, có ghi chú.
- Để dùng với dữ liệu thật: bỏ gen_data.py, trỏ evaluate.py vào thư mục có Model/ và Groundtruth/.
  Các bước dùng meta.csv (rò rỉ, SNR) và Annotator2/ (đồng thuận nhãn) cần metadata tương ứng.
- Lỗi cài sẵn trong dữ liệu: file thiếu cặp, 3 file lệch +0.5s, file toàn speech/non-speech,
  ~50% file dùng nguồn âm có trong train, nhạc có lời và babble gây false alarm.
