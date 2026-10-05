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
| `--preset hien-tai\|de-xuat` | hien-tai | Bộ tham số hậu xử lý. `de-xuat` = median 3 hop, `--hyst 0.15`, `--order mdp`, pad 0/0, merge 0.5 s, drop 0.32 s. Tham số ghi rõ vẫn ghi đè preset |
| `--overlap tri\|hann\|mean\|max\|median\|nearest` | tri | Gộp điểm các cửa sổ chồng nhau thành điểm mỗi hop 0.08 s (Ablation mục B: O4, O1, O2, O3, O0). `hann` giảm trọng số mép cửa sổ mạnh hơn tam giác |
| `--domain prob\|logit` | prob | Gộp tri / hann / mean trên xác suất hay trên log-odds |
| `--smooth N` | 5 | Làm mượt N hop; 1 = không |
| `--smooth-kind mean\|median\|gauss` | mean | Kiểu làm mượt. median giữ biên sắc, bỏ gai đơn lẻ; gauss: sigma = N/4 |
| `--hyst D` | 0 | Ngưỡng kép: bắt đầu speech khi điểm ≥ ngưỡng, chỉ kết thúc khi < ngưỡng − D (như Silero, pyannote). 0 = ngưỡng đơn |
| `--order pdm\|mdp` | pdm | Thứ tự ở A4. pdm = pad → drop → merge (hiện tại). mdp = merge → drop → pad: merge trên đoạn chưa pad, drop đo độ dài trước pad |
| `--pad-pre`, `--pad-post` | 0.10, 0.12 | Nới đầu / cuối đoạn speech (giây); âm = co đoạn |
| `--merge-gap` | 0.50 | Gộp hai đoạn cách nhau ít hơn N giây |
| `--drop` | 0 | Bỏ đoạn ngắn hơn N giây, chỉ ở A4 (pdm: đo sau pad; mdp: đo trước pad) |
| `--no-compare` | | Bỏ mục 6 (so sánh phương án hậu xử lý) |
| `--tune N`, `--tune-seed` | 0, 0 | Tìm cấu hình hậu xử lý tốt nhất trên dev với N lần thử (Optuna TPE nếu đã cài `optuna`, không thì tìm ngẫu nhiên) |
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

## So sánh phương án hậu xử lý (evaluate_real.py mục 6)

Mặc định evaluate_real.py chạy thêm mục này (tắt bằng `--no-compare`). Cơ sở của các phương án:
[vad_post_processing.md](vad_post_processing.md) và báo cáo tối ưu hậu xử lý. Bảng so sánh từng bước có minh hoạ: [SO_SANH_TOI_UU.md](SO_SANH_TOI_UU.md) (English: [POSTPROCESSING_COMPARISON.md](POSTPROCESSING_COMPARISON.md)).

    python evaluate_real.py --root <root> --out results_test --tune 300                 # so sánh + tìm cấu hình trên dev
    python evaluate_real.py --root <root> --out results_dexuat --preset de-xuat         # cả báo cáo (A0–A4, Excel) theo cấu hình đề xuất
    python make_excel_real.py --res results_test --out results/Tong_hop_danh_gia_VAD_test.xlsx   # thêm sheet "So sanh hau xu ly"

Cách làm:
- Mỗi cấu hình đi trọn pipeline: gộp cửa sổ → làm mượt → ngưỡng (đơn / kép) → đoạn → merge / drop / pad → bin 0.5 s.
- Ngưỡng được chọn riêng cho từng cấu hình trên dev (`--criterion`), rồi đánh giá một lần trên test.
- Δ so với B0 có CI 95% bootstrap ghép cặp theo file test: cùng một mẫu file dùng cho cả hai cấu hình, nên khác biệt nhỏ vẫn đo được.

| Mã | Cấu hình |
|---|---|
| B0 | Cấu hình hiện tại: tam giác + trung bình 5 hop, ngưỡng đơn, pad 100/120 → drop → merge < 500 ms. Luôn cố định (không theo `--preset`), để các lần chạy so được với nhau |
| CLI | Tham số dòng lệnh, chỉ hiện khi khác B0 và R |
| C1–C7 | B0 + đúng một thay đổi: ngưỡng kép Δ 0.15 / merge → drop 0.32 s → pad / pad 0/0 / median 3 hop / không làm mượt / Hann / miền logit |
| R | Đề xuất (`--preset de-xuat`) |
| R-h16, R-h24 | Như R nhưng chỉ giữ 1/2, 1/3 cửa sổ: mô phỏng hop 0.16 / 0.24 s, giảm 2–3× số lần chạy model (CPU, pin) |
| T | Có `--tune N`: cấu hình tốt nhất trên dev. N càng lớn càng dễ khớp riêng dev, nên chỉ tin T khi test và các category cũng tốt |

Đọc kết quả:
- AUC (cột `AUC_hop_bin`) chỉ đổi theo cách gộp, làm mượt và hop. Ngưỡng kép, pad, drop, merge không đổi AUC, nên các bước này phải so bằng F1, DCF, MR, FAR.
- CI của Δ chứa 0: chưa đủ bằng chứng để nói khác B0 với số file test hiện có.
- Chọn cấu hình theo dev và theo nhiều category. Không chọn theo hàng tốt nhất trên test.

Output:
- `pp_compare.csv`: mỗi cấu hình một dòng gồm ngưỡng dev, MR, FAR, Precision, Recall, F1, F1 bỏ bin biên, DCF, AUC, event-F1, số đoạn, ΔF1 / ΔDCF và CI.
- `pp_compare_category.csv`: như trên, theo category.
- `pp_tune_trials.csv`: mọi lần thử của `--tune`.
- `fig7_so_sanh_hau_xu_ly.png`.
- Khoá `pp_*` trong summary.json.

## Đặc trưng tập đánh giá theo độ phân giải model

Model cho điểm theo cửa sổ 0.96 s (hop 0.08 s), còn nhãn theo bin 0.5 s. evaluate_real.py (mục 2b) mô tả tập đánh giá
theo thang thời gian này. Các đặc trưng chỉ tính từ nhãn và vị trí cửa sổ, không dùng điểm model. Lưới chung 0.02 s
(0.5 = 25 ô, 0.08 = 4 ô, 0.96 = 48 ô) nên mọi tỉ lệ đều chính xác.

| Nhóm | Đặc trưng | Ý nghĩa |
|---|---|---|
| Độ dài đoạn / khoảng lặng | % đoạn speech và khoảng lặng < 0.96 s, 0.96–1.46 s, 1.46–2 s, ≥ 2 s | < 0.96 s: không cửa sổ nào nằm trọn trong đoạn |
| Hậu xử lý | % khoảng lặng < merge gap + pad (bị lấp), % đoạn bị drop | Lỗi vẫn bị tính kể cả khi model đúng hoàn toàn |
| Độ thuần cửa sổ | % cửa sổ thuần speech / thuần non-speech / lẫn | Cửa sổ lẫn: điểm tuỳ cách model hiểu nhãn cửa sổ |
| Khoảng cách tới biên | % bin cách chuyển trạng thái ≤ 0.48 / 0.48–0.96 / 0.96–1.44 / > 1.44 s | Vùng mà các cửa sổ góp vào điểm của bin có chứa biên |
| Trần oracle | F1 / MR / FAR / DCF của model hoàn hảo cùng cửa sổ (điểm = tỉ lệ speech trong cửa sổ, ngưỡng 0.5) qua A0 và A4 | Mức tốt nhất đạt được với độ phân giải + hậu xử lý hiện tại |
| Cỡ mẫu | Số bin, số cửa sổ không chồng, số đoạn, % bin < 6 tâm cửa sổ | Bin đầu/cuối file có điểm A0 dựa trên ít cửa sổ hoặc nội suy |
| Dev / test | Mọi đặc trưng tính riêng cho dev và test | Lệch > 10 điểm % ghi vào `profile_devtest_flags` và tô đỏ trong Excel |

Output: `dataset_profile.csv` (category × dev/test/tất cả), `seg_gap_hist.csv` (số đoạn theo độ dài),
`window_frac_hist.csv` (số cửa sổ theo tỉ lệ speech), `fig5_dac_trung_tap.png`,
`fig6_do_dai_doan.png` (độ dài đoạn speech / khoảng lặng theo giây, 1 cột = 1 frame 0.5 s; hai đường % tích luỹ
theo số đoạn và theo thời lượng), các cột mới trong `per_file.csv`,
khoá `profile*` trong summary.json. make_excel_real.py thêm sheet "Dac trung tap danh gia" (bỏ qua nếu kết quả cũ không có các file này).

### Cách đọc Hình 6 (`fig6_do_dai_doan.png`)

Các ví dụ số lấy từ bộ test (`results_test/`).

**Bố cục**
- Hàng trên: đoạn speech. Hàng dưới: khoảng lặng nằm giữa hai đoạn speech (không tính phần im lặng ở đầu và cuối file).
- Mỗi category một ô; ô cuối "Tất cả" gộp mọi file.
- Trục x dưới: độ dài (s), bước 0.5 s. Nhãn theo bin 0.5 s nên độ dài luôn là bội của 0.5 s.
  Trục x trên: cùng độ dài, tính bằng số frame 0.5 s.
- Trục y 0–100% dùng chung cho cột và hai đường.

**Cột màu: % số đoạn có đúng độ dài đó**
- Chiều cao cột = số đoạn dài đúng x ÷ tổng số đoạn n của category. Tổng các cột = 100%.
- Ví dụ Asm, hàng speech (n = 440): 55 đoạn dài 0.5 s → cột 12.5%; 98 đoạn dài 1 s → cột 22.3%.
- Cột cuối "≥ 5" gộp mọi đoạn dài từ 5 s trở lên. Cột này cao không có nghĩa là nhiều đoạn dài đúng 5 s
  (Clean, hàng speech: 47%, vì phần lớn đoạn rất dài).
- Cột chỉ đếm số đoạn: đoạn 0.5 s và đoạn 5 s đều tính là 1.

**Đường đen (chấm tròn): % tích luỹ theo số đoạn**
- Giá trị tại x = % số đoạn dài ≤ x, bằng tổng các cột từ trái tới x. Mỗi cột làm đường nhảy lên đúng bằng chiều cao
  của nó (Asm: 12.5% → 34.8% → 52.5% → … → 100%).
- Điểm đường cắt 50% là độ dài trung vị (Asm: 1.5 s).
- Dốc đứng ở bên trái = toàn đoạn ngắn (Music chạm 100% ngay tại 1.5 s). Đoạn nằm ngang = không có đoạn nào ở độ dài đó.
- Điểm cuối luôn là 100%.

**Đường tím nét đứt (chấm vuông): % tích luỹ theo thời lượng**
- Giá trị tại x = % tổng thời lượng nằm trong các đoạn dài ≤ x. Mẫu số là tổng thời lượng các đoạn cùng loại
  (tổng thời gian speech ở hàng trên, tổng thời gian khoảng lặng ở hàng dưới).
- Mỗi đoạn được tính theo độ dài của nó, nên đoạn 2 s nặng gấp 4 lần đoạn 0.5 s.
- Ví dụ Asm, hàng speech (tổng khoảng 1 129 s): tại 0.5 s là 2.4%; tại 1 s là 11%; tại 4.5 s là 64%.
  Như vậy 34 đoạn dài ≥ 5 s (7.7% số đoạn) chứa 36% thời lượng speech.

**So hai đường**
- Tím nằm xa dưới đen: đoạn ngắn nhiều nhưng chiếm ít thời gian, nên ít ảnh hưởng tới metric theo bin
  (Asm, Babble, Noise; Clean ở hàng speech).
- Tím bám sát đen: đoạn ngắn chiếm cả số lượng lẫn thời gian, nên ảnh hưởng thật lên metric
  (Music, hàng speech: 45% số đoạn và 26% thời lượng dưới 0.96 s; Clean, hàng khoảng lặng: 58% số khoảng và 23% thời lượng).
- Đánh giá theo sự kiện (event-F1, số biên): nhìn đường đen. Đánh giá theo bin (MR, FAR, F1, DCF): nhìn đường tím.

**Vạch đứng**
- Nét đứt xám, 0.5 s: 1 frame, độ dài ngắn nhất nhãn ghi được. Tiếng ngắn hơn (vd một từ 0.3 s) vẫn được ghi là 1 frame.
- Chấm đỏ, 0.96 s: cửa sổ model. Đoạn bên trái vạch không chứa trọn một cửa sổ nào. Cửa sổ nào chứa đoạn đó cũng lẫn
  ít nhất 0.46 s phần xung quanh, nên:
  - đoạn speech ngắn dễ bị bỏ sót (miss);
  - khoảng lặng ngắn dễ bị nhận nhầm là speech.

  Đoạn 1 s là trường hợp sát nút: tuỳ vị trí, có 1 cửa sổ nằm trọn trong đoạn hoặc không có cửa sổ nào
  (cửa sổ dịch từng bước 0.08 s).
- Gạch-chấm cam, 0.72 s (chỉ hàng khoảng lặng): pad nới mỗi đoạn speech 0.10 s ở đầu và 0.12 s ở cuối, rồi merge gộp
  hai đoạn nếu khoảng lặng còn lại < 0.5 s. Vì vậy khoảng lặng ban đầu < 0.5 + 0.22 = 0.72 s bị lấp thành speech,
  kể cả khi model đúng hoàn toàn, và các bin đó bị tính là FA ở A2–A4.
  Mốc này thay đổi theo `--pad-pre`, `--pad-post`, `--merge-gap` (vd `--merge-gap 1.0` → 1.22 s).
- Giá trị của hai đường tại vị trí một vạch = % số đoạn và % thời lượng nằm bên trái vạch đó.

**Tiêu đề mỗi ô**
- n: số đoạn, tức cỡ mẫu. n nhỏ thì % dao động mạnh (Music: 20 đoạn speech, nên 1 đoạn = 5%).
- "< 0.96 s: x% số đoạn · y% thời lượng": tỉ lệ đoạn ngắn hơn cửa sổ model, theo hai cách đếm.
- "bị lấp: …" (hàng khoảng lặng): tỉ lệ khoảng lặng bị pad + merge lấp. Với nhãn bin 0.5 s và tham số mặc định,
  con số này trùng với "< 0.96 s", vì chỉ khoảng lặng 0.5 s nằm dưới cả hai mốc.
- Mẫu số thời lượng ở đây là tổng thời lượng khoảng lặng giữa hai đoạn speech. Muốn ước lượng ảnh hưởng lên FAR thì
  chia cho toàn bộ thời gian non-speech (tính cả đầu và cuối file). Ví dụ với Clean: 23% theo hình, nhưng là 21.3% thời gian non-speech.

### Cách đọc Hình 5 (`fig5_dac_trung_tap.png`)

- (a) Đoạn speech gộp thành 4 nhóm theo cửa sổ model (< 0.96 s, 0.96–1.46 s, 1.46–2 s, ≥ 2 s). Mỗi màu là một category.
  Chiều cao cột = % số đoạn speech của category rơi vào nhóm. Đây là bản gộp của các cột trong Hình 6.
- (b) Giống (a) cho khoảng lặng. Chú thích ghi n và % khoảng lặng bị pad + merge lấp.
- (c) Mỗi cột chồng là toàn bộ cửa sổ 0.96 s của một category (100%), chia ba phần:
  - xanh dương: cửa sổ toàn non-speech;
  - cam: cửa sổ lẫn speech và non-speech, có ghi %;
  - xanh lá: cửa sổ toàn speech.

  Phần cam càng lớn thì kết quả càng phụ thuộc vào cách model chấm cửa sổ lẫn (Asm cao nhất: 21%).
- (d) Trần F1 của model hoàn hảo cùng cửa sổ 0.96 s (điểm = tỉ lệ speech trong cửa sổ, ngưỡng 0.5):
  - cột xanh: dùng điểm bin thô (A0). ≈ 1.00 ở mọi category, tức riêng độ phân giải không gây mất mát;
  - cột cam: đi qua hậu xử lý hiện tại (A4).

  Chênh lệch giữa hai cột là phần mất do hậu xử lý, lớn nhất ở Music (0.86).

## Các file

- vadlib.py: đọc file, gom bin 0.5s, hậu xử lý (rescore hop, pad, merge, rebin), metric, đặc trưng tập theo độ phân giải
  model (window_frac, oracle_model, run_lengths, dist_to_transition, centers_per_bin).
  Các phương án hậu xử lý tối ưu: `smooth_scores` (mean / median / gauss), `hysteresis` (ngưỡng kép),
  `postprocess_segs` (thứ tự pdm / mdp, pad âm), `segs_to_bins_fast`, `frames_to_bin_scores`;
  `rescore_hop` thêm `weight="hann"`, `smooth_kind`, `domain="logit"`.
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
