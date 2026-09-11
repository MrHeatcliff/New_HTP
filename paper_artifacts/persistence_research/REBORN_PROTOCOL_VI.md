# Reborn: conditional refinement và predictive outcomes

Ngày: 2026-09-12. Branch: `reborn`.

Đây là phương pháp mới theo đề xuất của người dùng, chưa có kết quả xác nhận
cải thiện Atari. Lần đầu là screening bốn game, một training seed. Báo cáo này
ghi giả thuyết trước khi có kết quả và phân biệt tính chất kiến trúc với kết quả
phải đo. Nó không thay thế mô tả constraint của đường 26 game cũ.

## 1. Giả thuyết và dấu hiệu bác bỏ

| Giả thuyết | Kỳ vọng quan sát | Dấu hiệu giả thuyết chưa được ủng hộ |
|---|---|---|
| H1: conditional context + cumulative rate giúp block sau thêm thông tin hữu ích | Held-out probe có thêm block làm giảm lỗi trên **cùng** detail target; cải thiện xuất hiện trên sampled code | Rate tăng nhưng incremental probe không tốt hơn; ảnh có coarse/fine chỉ do output basis |
| H2: successor targets giữ thông tin có ích ở horizon dài | Probe cùng target ở gamma lớn tốt hơn action-only; trained SF/Q tốt hơn zero baseline trên rollout thật | TD loss giảm nhưng rollout MSE/MAE không cải thiện; Q không hơn zero khi có reward |
| H3: fixed-noise code không collapse | Mu có phương sai/rank hữu ích và sampled-code probe tốt hơn baseline | Sample covariance lớn nhưng mu variance, probe utility gần zero |
| H4: representation mới vẫn đủ cho control | Return/learning curve không suy giảm rõ so với các run 100K cũ; reward prediction ở mẫu có reward vẫn hữu ích | Return giảm cùng sampled-code probe hoặc reward calibration, dù pixel loss đẹp |

Một run full không xác định đóng góp nhân quả của từng thay đổi. Đối chứng cũ
chỉ là tham chiếu screening. Kết luận H1/H2 cần các ablation ở mục 6 và nhiều seed.
Không dùng episode bootstrap để thay cho training-seed uncertainty.

## 2. Kiến trúc và objective thực thi

`dreamerv3/main_htp.py:make_agent` chọn `Agent_Reborn` khi
`agent.reborn.enabled=true`. Agent này kế thừa hạ tầng Dreamer và giữ RSSM,
observation decoder, actor/critic, reward/continuation head, các chuẩn hóa return
và value. Cấu hình bắt buộc `agent.htp.enabled=false` để không thực thi hai
objective HTP và các persistence/isotropy/rank/CKA penalty cũ.

### Projection và bottleneck

`dreamerv3/reborn.py:AffineProjection`, `sample_code`, `rate_terms`:

$$
\mu=Wh+b,\quad z=\mu+\epsilon,\quad \epsilon\sim\mathcal N(0,I).
$$

Projection là affine trực tiếp. Các prefix có kích thước
`128,256,512,1024,2048`; backbone size25m có
`d_h=3072+32*24=3840`, nên `D=2048 <= 3840`. Constructor kiểm tra điều kiện
này; không có shared trunk 512 trước output 2048. Backbone và các head vẫn phi tuyến.

$$
R_j=\tfrac12\|\mu^{(j)}\|_2^2,\qquad
R_{\leq\ell}=\sum_{j\leq\ell}R_j,\qquad
R=\tfrac15\sum_{\ell=1}^5R_{\leq\ell}.
$$

Trọng số theo block là `1,0.8,0.6,0.4,0.2`, trên tổng KL của block,
không phải mean theo chiều. Rate được log theo nats/state. Với fixed unit
Gaussian noise, expected rate là upper bound của mutual information của
stochastic code đối với phân phối state đang lấy kỳ vọng. KL lớn không đồng
nghĩa nhiều useful information: mean offset cũng tốn rate.

Rate loss được áp dụng trên posterior replay states. Không tuyên bố con số
rate này cũng là bound định lượng cho một imagined-state distribution khác.
Các head nhận code ở compute dtype của Dreamer (bfloat16); các thống kê và
distortion được tính float32. Rounding deterministic sau noise không tăng
mutual information.

### Conditional Haar refinement

`haar_bands` thực hiện Haar 2D trực chuẩn theo từng channel RGB. Tại mỗi
ô 2x2 có giá trị a,b,c,d, lowpass là `(a+b+c+d)/2`; ba detail là
`(a-b+c-d)/2`, `(a+b-c-d)/2`, `(a-b-c+d)/2`.

Lặp lowpass đến 4x4 rồi sắp thứ tự coarse-to-fine cho các lưới
`4,8,16,32,64`. Số hệ số từng band là `48,144,576,2304,9216`, tổng 12.288.
Observation đầu vào là RGB `image/255`, không có target encoder học được.

`Refinement` có head cho từng band, nhận **toàn bộ prefix** tương ứng.
`context=false` là ablation chỉ nhận block mới. `haar_image` tổng hợp các
band; suffix bị bỏ được thay bằng hệ số zero.

$$
\widehat b^{(\ell)}=G_\ell(z^{(1:\ell)}),\qquad
L_{\mathrm{rec}}=\frac1{12288}\sum_\ell
\|b^{(\ell)}-\widehat b^{(\ell)}\|_2^2.
$$

Các detail subspace trực giao nên trong một forward pass, thêm band không
đổi phần coarse đã dựng. Parseval cho distortion toàn ảnh bằng tổng
distortion hệ số. Đây không phải guarantee error giảm khi thêm band: predicted
detail sai có thể tệ hơn bỏ detail. Không detach context giữa các prefix.

### Multi-timescale predictive outcomes

`Outcomes` có năm trunk nhận prefix và **action hiện tại**. Mỗi trunk có hai
output: SF hệ số Haar tích lũy tới level đó và Q với distribution head
`symexp_twohot` 255 bins của repository. `q.pred()` dùng expectation trên
các symexp bins của Dreamer, không tự đặt một reward distribution mới.

$$
\gamma=(15/16,7/8,3/4,1/2,0),
$$
$$
y^\Psi_\ell=(1-\gamma_\ell)b_{t+1}^{(1:\ell)}
+\gamma_\ell c_{t+1}\bar\Psi_\ell(\bar z_{t+1},a'),
$$
$$
y^Q_\ell=r_{t+1}+\gamma_\ell c_{t+1}\bar Q_\ell(\bar z_{t+1},a').
$$

`bellman_targets` detach cả hai target. `a'` do **current actor** lấy mẫu
từ online sampled code tại next state, dùng chung action cho năm level.
Target projection và target outcome heads dùng EMA rate 0.02, cập nhật sau
optimizer; không thêm EMA RSSM. EMA bắt đầu bằng bản sao online parameters.
`gamma=0` học next observation và immediate reward.

$$
L_{\mathrm{outcome}}=\frac15\sum_\ell
\mathbb E_{\mathrm{valid}}
\left[\frac{\|\widehat\Psi_\ell-y^\Psi_\ell\|^2}{12288}
+\lambda_Q\ell_{\mathrm{twohot}}(p^Q_\ell,y^Q_\ell)\right].
$$

`transition_mask` dùng `~is_first[:,1:] & ~is_last[:,:-1]`. Action t lấy
từ `prevact[:,t+1]`; image/reward/terminal target lấy tại t+1. Transition
đi vào terminal vẫn được học, với continuation zero. Time limit không được
tự coi là terminal: đi vào time limit vẫn bootstrap nếu `is_terminal=false`,
nhưng không học transition từ state cuối episode sang reset. Loss lấy trung
bình trên **số valid transitions**, rồi chuẩn hóa/pad để phù hợp mean B*T
của agent. Batch không có valid transitions trả zero.

Objective tổng:

$$
L=L_{\mathrm{Dreamer}}+
\lambda_{\mathrm{rec}}(L_{\mathrm{rec}}+\beta R)+
\lambda_{\mathrm{outcome}}L_{\mathrm{outcome}}.
$$

Giá trị ban đầu: `lambda_rec=1`, `lambda_outcome=1`, `beta=1e-5`,
`q_weight=0.1`, `hidden=512`. Đây là hyperparameter screening chưa được tune.
Discount/horizon của actor–critic Dreamer không đổi.

### Dùng đúng stochastic code trên đường control

`Agent_Reborn:loss` lấy một noise draw trên posterior state và reuse cho
refinement/outcome/reward/continuation/replay-value. Auxiliary có thể detach
h theo `grad_to_backbone=false`; việc đó không đổi giá trị sampled code.

`imagine_codes` chạy RSSM một bước, lấy một sample z/state, sinh action và
trả về chính z đó. Policy log-probability, reward, continuation, critic và
slow critic cùng dùng code lưu từ lúc sinh action. Không lấy lại noise khi
tính policy loss. Online policy và evaluation cũng đi qua `feat2tensor`
có noise. Bản `stochastic=false` là ablation riêng, không được viện dẫn bound
Gaussian stochastic bottleneck cho đường deterministic đó.

## 3. Metric và cách đọc

Metric training được lưu trong `GAME/full/metrics.jsonl`, thông thường có
prefix logger `train/`. Tên dưới đây là tên nội bộ agent.

| Metric | Định nghĩa/ý nghĩa |
|---|---|
| `loss/reborn_rec`, `loss/reborn_rate`, `loss/reborn_outcome` | Raw objective từng nhánh |
| `weighted_loss/*` | Raw mean nhân đúng coefficient; đối chiếu với Dreamer loss |
| `reborn/l*/rate_nats`, `prefix_rate_nats` | KL theo block và prefix/state |
| `mu_variance`, `mu_dead_fraction` | Phương sai trên batch×time của mean code; dead khi <1e-6 |
| `noise_mse` | Sanity check noise thực sự hiện diện; không phải chống collapse |
| `band_mse_per_pixel`, `band_zero_mse` | Distortion band so với target; đối chiếu zero-band baseline |
| `prefix_image_mse` | Tổng lỗi các band đã dự đoán + energy của band bị bỏ; không clipping ảnh |
| `sf_td_mse`, `sf_target_energy`, `sf_zero_mse` | TD residual và độ lớn target; chưa phải rollout calibration |
| `q_twohot`, `q_td_mae`, `q_target_abs` | Distribution loss, TD MAE reward units và target scale |
| `q_td_mae_nonzero_reward` | TD MAE riêng transition có immediate reward khác zero |
| `valid_transition_count/fraction`, `terminal_transition_count`, `nonzero_reward_transition_count` | Denominator/coverage; metric conditional zero khi count zero không phải thành tích |
| `ema_mu_mse` | Chênh EMA/online projection trên cùng h |
| `opt/param_count`, `opt/grad_norm`, `opt/update_rms` | Tổng trainable params, global gradient norm và update scale |
| `episode/score`, action counter, throughput của runner | Control và chi phí thực tế; không gộp raw score giữa game |

Hiện chưa có per-objective gradient cosine trên shared parameters. Raw loss
scale không thay thế gradient scale; nếu rate hoặc Q có dấu hiệu chi phối,
cần audit gradient trước khi kết luận nguyên nhân.

## 4. Diagnostic tự chạy sau training

`corewm_eval/reborn_diagnostic.py` load checkpoint cuối, giữ policy cố định,
thu 30 clip tối đa 1024 bước/clip từ environment thật. Lưu image, action,
reward, reset/terminal để tái phân tích. RSSM posterior được trích theo chunk
với carry liên tục; anchor mỗi 16 bước và cần đủ 128 future steps trong clip.
Các clip ngắn không đủ horizon được báo count zero, không tạo target giả.

Các return thật dừng sau terminal (bao gồm image/reward của terminal một lần).
Không dùng target head bootstrap để chấm calibration. Tail tối đa của trọng
số geometric còn lại là `(15/16)^128 ≈ 0.000258`; với Q, error tail còn phụ
thuộc reward scale. Một sample rollout chứa stochastic outcome variance,
nên MSE không chỉ là lỗi conditional mean predictor.

Output `GAME/diagnostic/diagnostics.json` gồm:

- Full coefficient SF MSE và Q MAE so với finite real returns, kèm zero baseline.
- Band reconstruction MSE và lỗi khi bỏ mean của block mới nhưng reuse noise/context.
  Đây là head-reliance intervention; không coi nó tương đương retrain ablation.
- Reward MAE, nonzero-reward count/MAE, continuation Brier và actor entropy.
- CKA/participation rank trên **mu**, cùng variance từng block.
- Ridge probes cho action-only và năm prefix, báo riêng mu và sampled z.

Probe có cùng target cho mọi prefix: năm detail-band sketches, cùng full-image
feature sketch ở năm timescale và Q ở năm timescale. Sketch Gaussian cố định
32 chiều dùng để giảm chi phí probe; objective training vẫn dùng toàn bộ hệ số.
Các cột target của cache: detail `0:160`, SF `160:320`, Q `320:325`.

Split theo episode id modulo 3 thành train/validation/test; feature/target
normalization chỉ fit trên train. Ridge trong `{0.01,1,100}` chọn bằng validation
trên toàn bộ standardized targets. Với cùng target, so sánh lỗi test trước/sau
thêm block; R² dùng train-mean predictor làm baseline. Việc shared lambda được
ghi rõ vì có thể ưu tiên một nhóm target khi chọn model.

Các figure tự xuất: `prefix_target_probe_matrix.png`, `mu_block_cka.png`.
`probe_data.npz` chứa mu/z/action/target/episode id; `parameter_counts.json`
báo theo module (bao gồm EMA/optimizer nếu có, cần phân biệt với trainable count).

Giới hạn coverage: clip vẫn bắt đầu từ reset; 1024 bước rộng hơn các clip ngắn
cũ nhưng không đại diện đầy đủ replay distribution. Anchor cần 128 bước tiếp
theo có thể ít bao phủ sát terminal; luôn đọc event counts. Q từ current actor
trong training có policy drift, còn diagnostic dùng policy cuối cố định.

## 5. Thực nghiệm đầu và trạng thái xác minh

Job bốn game `3843` đã được hủy khi còn pending để phù hợp tài nguyên thực tế.
Screening thay thế dùng một allocation: 1 H200, 16 CPU, 128 GB RAM, wall limit
8 giờ. Hai game Boxing và Frostbite chạy đồng thời, mỗi game 8 CPU affinity
riêng. Chạy từ đầu 100.000 environment actions, seed 0, size25m,
train ratio 256, checkpoint mỗi 10K actions. Sau đó 20 episode eval và diagnostic.
100K actions là budget mới, không tự coi tương đương run 26 game 110K cũ.

Run root và job ID mới được ghi trong `production_runs/reborn_two_game_*/submission.json`.
Source snapshot và SHA-256 nằm trong run root. Kết quả theo game có
`status.json`, command JSON, train/eval/diagnostic logs và timing từng stage.

Trạng thái tại lúc viết: đang chuyển từ job bốn game sang job hai game để dùng
GPU còn trống trên node hiện tại. Chưa có kết quả training hoặc GPU test. Kiểm
tra syntax Python, shell và whitespace đã qua. `tests/test_reborn.py` và test
CPU affinity sẽ chạy **trong allocation**
trước training; fail test sẽ dừng pipeline. Không chạy JAX/train trên login.

Test bao gồm roundtrip/Parseval/no-coarse-overwrite, cumulative-rate gradient,
noise versus mu collapse, terminal/time-limit/reset và target detach,
dimension guard, real-rollout terminal indexing, ablation commands, cùng
forward + optimizer gradient step + EMA trên agent thực kích thước nhỏ.
Không ghi “tests passed” trước khi có log xác nhận.

Ước tính sơ bộ **2–4 giờ sau khi được cấp GPU**, gồm training/eval/diagnostic;
không gồm chờ scheduler. Đây chưa phải benchmark tốc độ Reborn. Script lưu
timing để cập nhật estimate bằng dữ liệu thật.

## 6. Ablation đã chuẩn bị

`corewm_eval/reborn_experiment.py:command(game, output, arm)` hỗ trợ:

| Arm | Yếu tố thay đổi so với full |
|---|---|
| `no_rec` | Bỏ distortion/refinement heads; giữ rate để tách reconstruction khỏi bottleneck |
| `no_outcome` | Bỏ SF/Q modules và EMA; giữ reconstruction/rate |
| `no_q` | q_weight=0; SF giữ nguyên, Q head không có auxiliary supervision |
| `no_rate` | beta=0, noise giữ nguyên |
| `flat_rate` | Tổng full KL một lần thay mean cumulative-prefix KL; khác cả effective weighting |
| `no_context` | Decoder band chỉ nhận block mới |
| `deterministic` | Noise off, các term còn lại giữ; không áp dụng stochastic MI claim |
| `affine_control` | Affine/noisy code + Dreamer, bỏ cả hai distortion và rate; là đối chứng package |

Chỉ **full** được submit ở vòng này theo yêu cầu bốn game. Chưa có factorial,
multiple seed hay parameter-matched baseline mới. Khi xem kết quả, ưu tiên
ablation làm rõ failure cụ thể: sampled probe mất tín hiệu → noise/rate;
SF tốt nhưng Q kém → reward grounding/gradient scale; band loss tốt nhưng
incremental probe không tốt → context/rate allocation; predictor tốt nhưng
control giảm → reward/value/imagination path và đối chứng affine/noisy code.

## 7. Nguồn nền tảng

Variational stochastic bottleneck dựa trên [Alemi et al., Deep Variational
Information Bottleneck](https://arxiv.org/abs/1612.00410). Cấu trúc discounted
feature prediction có nền tảng từ [Barreto et al., Successor Features for
Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312).
Việc ghép conditional Haar refinement, cumulative-prefix rate và các
action-conditioned outcomes là thiết kế cần thực nghiệm của Reborn; các
nguồn này không chứng minh hiệu quả Atari của cấu hình mới.
