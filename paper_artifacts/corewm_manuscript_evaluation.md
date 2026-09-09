# CoRe-WM — block, prefix imagination và policy evaluation

## Bộ checkpoint và dữ liệu

Dùng bộ Full CoRe-WM tại `production_runs/corewm_atari100k_v1`, 26 game × 5 training seed, checkpoint 100,000 actions. Bộ này khác lượt constraint 26 game seed 0. Policy evaluation đã có 100 episode tại checkpoint cuối và 10 episode ở mỗi mốc 10k–90k. Hàm `load_full_policy_curves` kiểm tra đủ 130 run, hash provenance, số episode và lịch checkpoint.

Các số representation mới cần chạy evaluator; tài liệu này mô tả implementation và không thay thế bảng kết quả được tạo sau khi job hoàn tất.

## Block distinctness

`corewm_eval/manuscript_eval.py` lấy posterior ở từng thời điểm, gom chúng thành batch độc lập rồi gọi RSSM `imagine(..., single=True)` với action đi ra ở cùng thời điểm. Mỗi hàng CKA vì thế là một dự đoán đúng một bước, không phải posterior hoặc bước của rollout dài.

Projection tạo năm block với ranh giới lấy trực tiếp từ config checkpoint. Với cumulative reconstruction H_l, lấy D_l(z_l)=H_l−H_(l−1), riêng block đầu lấy H_1. Cách này giữ đúng residual head của block và không thêm bias của những head khác khi nhận vector zero. Observation decoder nhận riêng residual đó để tạo ảnh block-only.

Linear CKA được tính sau khi center từng cột, cho mọi cặp block; kết quả là ma trận 5×5 và heatmap. Không sử dụng cumulative prefix để tính CKA. CKA thấp và hình khác nhau chỉ là bằng chứng nội dung khác nhau trên các clip đã lấy mẫu; chưa chứng minh semantic disentanglement.

## Prefix imagination 64 bước

Từ posterior t=16, RSSM chạy một rollout 64 bước theo action thật. Tất cả prefix được reconstruct từ cùng imagined trajectory. Reconstruction không được đưa trở lại dynamics. Sai số là trung bình absolute pixel error trong miền [0,1] tại từng horizon k=1..64. Lưu đầy đủ ma trận 5×64 và đường MAE của observation decoder nhận trực tiếp imagined RSSM state.

Ảnh được xuất tại k∈{1,2,4,8,16,32,64}. Cả năm prefix được hiển thị. Các số 8,16,32,64 không phải block index trong checkpoint hiện có.

Mỗi game dùng bốn clip bắt đầu từ reset, do policy Full seed 0 tạo, dùng chung cho năm checkpoint. Mỗi clip tối đa 96 observations, một rollout start cố định t=16. Đây là phạm vi đầu episode; không đại diện toàn bộ phân phối gameplay. Không cho phép rollout xuyên terminal. Nếu clip không đủ dài, evaluator báo lỗi để chẩn đoán thay vì nối episode.

Đường `Base dynamics` trong figure là backbone của cùng CoRe-WM checkpoint. So sánh với mô hình baseline huấn luyện độc lập cần checkpoint baseline tương ứng; không dùng đường này để tuyên bố đã so sánh DreamerV3/STORM/TWISTER.

## Policy effectiveness

HNS_g=(R_g−R_random)/(R_human−R_random), không clip. Sử dụng bảng random/human trong `baselines.yaml` và báo cáo theo phần trăm. Return cuối được lấy trung bình trên 100 episode mỗi seed, sau đó lấy mean và sample SD trên năm seed. Mean/median HNS toàn suite tính trên 26 game sau khi trung bình seed.

AUC dùng tích phân hình thang của HNS tại các checkpoint 10k,20k,…,100k và chia cho 90k. Không giả tạo điểm evaluation tại 0 actions. Vì vậy tên chính xác là normalized AUC trên khoảng 10k–100k, không phải AUC từ 0.

## Chạy và output

`scripts/slurm_manuscript_suite.sh RUNROOT` chạy test trước, rồi `corewm_eval.manuscript_suite`. RUNROOT chứa source snapshot và `results/`. Job sử dụng hai H200, bốn worker, mỗi worker tám CPU riêng. Mỗi worker xử lý lần lượt năm seed của một game. Không training hoặc cập nhật checkpoint.

Mỗi game/seed xuất `result.json`, `one_step_blocks.png`, `cka.png`, `prefix_imagination.png`, `mae.png`. JSON chứa checkpoint hash, clip hash, full CKA, mọi MAE theo horizon và kiểm tra parameters/optimizer counter không đổi. `results/README.md` chứa bảng policy đủ 26 game ngay khi bắt đầu và được bổ sung representation sau khi đủ 130 checkpoint. `status.json` chỉ ghi COMPLETE khi toàn bộ evaluation kết thúc thành công.

## Kiểm chứng implementation

Test kiểm tra tách residual block với giá trị signed/bias và tính đối xứng, đường chéo, bất biến scale/translation của CKA trên block khác chiều. Runtime kiểm tra action alignment, hash checkpoint, parameter hash và optimizer counter trước/sau. Test và evaluation chạy trong Slurm; trạng thái thực nghiệm chưa được coi là hoàn thành chỉ vì đã tạo code hoặc submit job.
