# CoRe-WM — block, prefix imagination và policy evaluation

## Bộ checkpoint và dữ liệu

Dùng bộ Full CoRe-WM tại `production_runs/corewm_atari100k_v1`, 26 game × 5 training seed, checkpoint 100,000 actions. Bộ này khác lượt constraint 26 game seed 0. Policy evaluation đã có 100 episode tại checkpoint cuối và 10 episode ở mỗi mốc 10k–90k. Hàm `load_full_policy_curves` kiểm tra đủ 130 run, hash provenance, số episode và lịch checkpoint.

Các số representation mới cần chạy evaluator; tài liệu này mô tả implementation và không thay thế bảng kết quả được tạo sau khi job hoàn tất.

## Implementation map

| Thành phần | Công thức thực thi | File và hàm |
| --- | --- | --- |
| Posterior và dự đoán một bước | `posterior = dyn.observe(...)`; với mọi $(h_t,a_t)$ hợp lệ: $\widetilde h_{t+1}=f_{dyn}(h_t,a_t)$; $\widetilde z_{t+1}=S_\psi(\widetilde h_{t+1})$ | `corewm_eval/manuscript_eval.py:_evaluate` |
| Chia block | $\widetilde z^{(\ell)}=\widetilde z[d_{\ell-1}:d_\ell]$, với $d=(128,256,512,1024,2048)$ của checkpoint hiện tại | `corewm_eval/slicing.py:block_dims`; `manuscript_suite.py:worker` |
| Decode block riêng | $B_1=H_1$, $B_\ell=H_\ell-H_{\ell-1}=D_\ell(\widetilde z^{(\ell)})$; decode $O_\theta(B_\ell)$ | `corewm_eval/manuscript_eval.py:residual_blocks`, `_evaluate` |
| CKA | $C_{ij}=\|\bar Z_i^T\bar Z_j\|_F^2/(\|\bar Z_i^T\bar Z_i\|_F\|\bar Z_j^T\bar Z_j\|_F)$ | `corewm_eval/metrics.py:linear_cka`, `cka_matrix` |
| Rollout shared trajectory | $\widetilde h_{t+k+1}=f_{dyn}(\widetilde h_{t+k},a_{t+k})$, $k=0,\ldots,63$ | `corewm_eval/manuscript_eval.py:_evaluate` |
| Prefix reconstruction | $H_\ell=\sum_{j\leq\ell}D_j(\widetilde z^{(j)})$; $\widehat o_k^{(1:\ell)}=O_\theta(H_\ell)$ | `dreamerv3/htp.py:ProgressiveRecon.reconstruct`; `manuscript_eval.py:_evaluate` |
| Pixel error | $\operatorname{MAE}_\ell(k)=\frac1{CHW}\|o_{t+k}-\widehat o_{t+k}^{(1:\ell)}\|_1$ | `corewm_eval/manuscript_suite.py:worker` |
| Policy score | $\operatorname{HNS}_g=(R_g-R_g^{random})/(R_g^{human}-R_g^{random})$; $\operatorname{AUC}=\operatorname{trapz}(\operatorname{HNS},[10k,\ldots,100k])/90k$ | `corewm_eval/policy.py`; `manuscript_suite.py:report` |

All evaluation calls are read-only. `evaluate_readonly` saves the parameter tree and optimizer update counter before evaluation, then asserts both are unchanged afterward. It also SHA-256 checks `agent.pkl` before and after the run. The action tensor passed to the rollout is compared exactly with the action tensor the RSSM consumed.

## Block distinctness

`corewm_eval/manuscript_eval.py` lấy posterior ở từng thời điểm, gom chúng thành batch độc lập rồi gọi RSSM `imagine(..., single=True)` với action đi ra ở cùng thời điểm. Mỗi hàng CKA vì thế là một dự đoán đúng một bước, không phải posterior hoặc bước của rollout dài.

Projection tạo năm block với ranh giới lấy trực tiếp từ config checkpoint. Với cumulative reconstruction H_l, lấy D_l(z_l)=H_l−H_(l−1), riêng block đầu lấy H_1. Cách này giữ đúng residual head của block và không thêm bias của những head khác khi nhận vector zero. Observation decoder nhận riêng residual đó để tạo ảnh block-only.

Điểm này khác với việc gọi tất cả $D_j$ trên một masked vector: MLP head có bias nên $D_j(0)$ thường khác zero. Hiệu cumulative reconstruction loại đúng các head còn lại theo kiến trúc residual đã train. Vì vậy ảnh `one_step_blocks.png` là đóng góp của một block trong model thực tế, không bị trộn bởi zero-input bias của các block khác.

Linear CKA được tính sau khi center từng cột, cho mọi cặp block; kết quả là ma trận 5×5 và heatmap. Không sử dụng cumulative prefix để tính CKA. CKA thấp và hình khác nhau chỉ là bằng chứng nội dung khác nhau trên các clip đã lấy mẫu; chưa chứng minh semantic disentanglement.

## Prefix imagination 64 bước

Từ posterior t=16, RSSM chạy một rollout 64 bước theo action thật. Tất cả prefix được reconstruct từ cùng imagined trajectory. Reconstruction không được đưa trở lại dynamics. Sai số là trung bình absolute pixel error trong miền [0,1] tại từng horizon k=1..64. Lưu đầy đủ ma trận 5×64 và đường MAE của observation decoder nhận trực tiếp imagined RSSM state.

Trong output `result.json`, `prefix_mae[ell][k]` chứa MAE của prefix level `ell+1` tại horizon `k+1`; `backbone_mae[k]` là MAE của decoder nhận $\widetilde h_{t+k}$ trực tiếp. `cka[i][j]` là CKA giữa block `i+1` và `j+1`, từ tất cả one-step predicted vectors của bốn clips. `one_step_samples` ghi số vector thực sự đi vào ma trận này.

Ảnh được xuất tại k∈{1,2,4,8,16,32,64}. Cả năm prefix được hiển thị. Các số 8,16,32,64 không phải block index trong checkpoint hiện có.

Mỗi game dùng bốn clip bắt đầu từ reset, do policy Full seed 0 tạo, dùng chung cho năm checkpoint. Mỗi clip tối đa 96 observations, một rollout start cố định t=16. Đây là phạm vi đầu episode; không đại diện toàn bộ phân phối gameplay. Không cho phép rollout xuyên terminal. Nếu clip không đủ dài, evaluator báo lỗi để chẩn đoán thay vì nối episode.

Đường `Base dynamics` trong figure là backbone của cùng CoRe-WM checkpoint. So sánh với mô hình baseline huấn luyện độc lập cần checkpoint baseline tương ứng; không dùng đường này để tuyên bố đã so sánh DreamerV3/STORM/TWISTER.

## Policy effectiveness

HNS_g=(R_g−R_random)/(R_human−R_random), không clip. Sử dụng bảng random/human trong `baselines.yaml` và báo cáo theo phần trăm. Return cuối được lấy trung bình trên 100 episode mỗi seed, sau đó lấy mean và sample SD trên năm seed. Mean/median HNS toàn suite tính trên 26 game sau khi trung bình seed.

AUC dùng tích phân hình thang của HNS tại các checkpoint 10k,20k,…,100k và chia cho 90k. Không giả tạo điểm evaluation tại 0 actions. Vì vậy tên chính xác là normalized AUC trên khoảng 10k–100k, không phải AUC từ 0.

## Chạy và output

`scripts/slurm_manuscript_suite.sh RUNROOT` chạy test trước, rồi `corewm_eval.manuscript_suite`. RUNROOT chứa source snapshot và `results/`. Job sử dụng hai H200, bốn worker, mỗi worker tám CPU riêng. Mỗi worker xử lý lần lượt năm seed của một game. Không training hoặc cập nhật checkpoint.

Mỗi game/seed xuất `result.json`, `one_step_blocks.png`, `cka.png`, `prefix_imagination.png`, `mae.png`. JSON chứa checkpoint hash, clip hash, full CKA, mọi MAE theo horizon và kiểm tra parameters/optimizer counter không đổi. `results/README.md` chứa bảng policy đủ 26 game ngay khi bắt đầu và được bổ sung representation sau khi đủ 130 checkpoint. `status.json` chỉ ghi COMPLETE khi toàn bộ evaluation kết thúc thành công.

Artifact đã publish nằm ở `paper_artifacts/corewm_manuscript_evaluation_results/`: có 130 `result.json`, figure seed 0 cho cả 26 game, `README.md` tổng hợp 5 seed, nguồn policy và completion manifest. Mỗi figure trên README dùng đường dẫn tương đối trong cùng artifact nên mở trực tiếp trên GitHub được.

## Kiểm chứng implementation

Test kiểm tra tách residual block với giá trị signed/bias và tính đối xứng, đường chéo, bất biến scale/translation của CKA trên block khác chiều. Runtime kiểm tra action alignment, hash checkpoint, parameter hash và optimizer counter trước/sau. Test và evaluation chạy trong Slurm; trạng thái thực nghiệm chưa được coi là hoàn thành chỉ vì đã tạo code hoặc submit job.
