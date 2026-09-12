# Reborn: ablation rate, Q và noise; đóng góp vào actor–critic

## Giả thuyết và thiết kế trước khi chạy

Haar đã đảm bảo coarse-to-fine ở output. Reconstruction tốt chưa chứng minh
incremental information hữu ích cho policy. Chúng ta giữ reconstruction và
target Haar nguyên trạng trong vòng này, kiểm tra ba failure cụ thể.

| Arm | Một thay đổi so với full | Kỳ vọng nếu giả thuyết đúng | Dấu hiệu phản bác |
|---|---|---|---|
| full | Không đổi; đối chứng chạy lại với cùng instrumentation | Tái hiện hành vi Reborn | Chênh lệch lớn với seed-0 cũ cần audit reproducibility |
| low_rate | beta 1e-5 → 1e-6, noise std vẫn 1 | Rate/SNR tăng, khoảng cách probe mu–z giảm, actor ít nhạy noise, control tốt hơn | Rate tăng nhưng usable information/control không tăng |
| no_q | q_weight 0.1 → 0 | Nếu Q gây hại, giảm xung đột với actor/critic, cải thiện reward/value/control | Control giảm hoặc không cải thiện dù bỏ Q |
| deterministic | stochastic true → false | Nếu noise gây khó control, actor/critic và score tốt hơn | Bỏ noise không giúp hoặc làm control kém |

Deterministic giữ beta và kiến trúc nhưng L2 không còn diễn giải information
bound như stochastic encoder. Đây là ablation chẩn đoán, không phải tuyên bố
phương pháp mới tốt hơn. No-Q giữ head/parameter count, bỏ supervision Q;
không dùng Q calibration của head không được huấn luyện để đánh giá arm này.

Bốn game: Boxing, Up N Down, Frostbite, Road Runner. Training seed 0, 100K
environment actions, cùng kiến trúc/optimizer/backbone/dims. Mỗi wave chạy
bốn game trên hai H200, hai game/GPU, tám CPU/game. Bốn wave tuần tự trong
một job; không thêm concurrent allocation. Các arm không kết hợp thay đổi.
Đây là screening một seed, không phải kết luận thống kê nhiều training seed.

## Evaluation và representation

Mỗi game lưu checkpoint 10K–100K. Evaluation riêng 10 episode tại 10K–90K,
100 episode tại 100K, eval seed 0, một environment. Kiểm tra đúng số episode,
return hữu hạn và checkpoint hash không đổi. Cấu hình ablation được dùng
nhất quán cả training và evaluation; không vô tình bật noise lại khi eval
deterministic. Dùng score theo checkpoint và AUC để phân biệt tốc độ học và
final performance. Lịch này là protocol evaluation đã áp dụng trong repo.

Diagnostic cuối: 30 frozen-policy clips tối đa 1024 actions, targets rollout
128 bước, probes split theo episode. Giữ rate/block, variance mu, rank/CKA mu,
same-target detail/SF/Q prefix probes và trained-head calibration SF/Q. Không
dùng covariance sampled z làm bằng chứng chống collapse. Chú ý dataset từ
policy yếu có thể dễ dự đoán; không suy từ fit tốt ra control tốt.

Critic được đổi từ normalized output sang reward scale bằng valnorm. Chỉ
episode clip kết thúc terminal thật được chấm against complete discounted
return, với discount actor–critic 1−1/horizon, reward từ t+1. Timeout/censored
clip không tạo giá trị calibration giả. Số episode đủ điều kiện được log;
không có mẫu đủ điều kiện nghĩa là chưa kết luận được, không phải lỗi bằng 0.

## Logging và audit actor–critic

Training giữ các metric Dreamer: entropy, normalized advantage mean/std/mag,
return/value/slowvalue, continuation weight, raw/weighted loss. Bổ sung
`reborn/weighted_sf`, `reborn/weighted_q` để tách SF và Q trong outcome,
`reborn/imag_return_raw_mean/std` để thấy return ở thang reward thực.
Không thêm random draw vào training cho việc logging.

Audit cuối checkpoint trên ba episode cố định 0,10,20, tối đa 64 state đầu,
hai seed cho noise/imagination. Đây là sample hẹp, không đại diện toàn replay.
Mỗi arm được audit trên clips riêng; ba ablation còn audit trên clips của
đối chứng full. Bản full own chính là shared-state reference. Chỉ audit
gradient/intervention trên dữ liệu shared; không diễn giải off-policy return
như on-policy value calibration.

### Gradient của loss thật

`corewm_eval/reborn_actor_audit.py::gradient_objective` gọi `model.loss` và
lấy riêng chín branch đã nhân coefficient: rec, rate, SF, Q, reward,
continuation, actor, imagined critic, replay critic. Mỗi phép so sánh dùng
cùng state batch và PRNG seed. Không cập nhật parameters/optimizer/EMA.

Với tập parameters theta của module, g_i = gradient_theta(w_i L_i), log:

$$
\|g_i\|_2,\qquad
\cos(g_i,g_j)=\frac{g_i^\top g_j}{\|g_i\|_2\|g_j\|_2}.
$$

Log riêng module projection, policy, value, reward, continuation. Actor–critic
aggregate là tổng gradient actor + imagined critic + replay critic (đã weighted).
Cosine undefined khi gradient zero được lưu null. Gradient là trước AGC/Adam,
không được gọi là update thực tế. Cosine âm là xung đột cục bộ, không tự chứng
minh auxiliary gây giảm score. Gradient auxiliary ở policy/value head thường
bằng zero vì chúng tác động gián tiếp qua projection; đây là cấu trúc graph.

### Can thiệp paired trên cùng posterior state

Giữ backbone state cố định, so z=mu+epsilon với mẫu epsilon khác, mu không noise,
và z trừ mu của một block (giữ nguyên noise, context các block khác). Log:

$$
TV(\pi,\pi')=\tfrac12\sum_a|\pi(a)-\pi'(a)|,
\qquad KL(\pi\|\pi')=\sum_a\pi(a)\log\frac{\pi(a)}{\pi'(a)}.
$$

Thêm tỷ lệ đổi argmax action, mean absolute change value/reward/continuation,
entropy gốc, reward MAE và zero baseline, continuation Brier. Noise sensitivity
cùng value scale giúp nhận ra actor/critic phản ứng với noise. Bỏ block có thể
out-of-distribution: chỉ cho thấy reliance, không thay thế retrained probes.
Mean-code intervention không thay đổi policy của run evaluation chính.

## Cách quyết định sau khi có kết quả

1. Kiểm tra protocol, completion, nonfinite và tái hiện full trước.
2. Với low_rate: đối chiếu rate → sampled utility → policy/value sensitivity → score.
3. Với no_q: xem gradient Q với actor–critic trên shared states và thay đổi score;
   không kết luận chỉ từ trị số cross-entropy lớn.
4. Với deterministic: phân biệt lợi ích bỏ noise với giả thuyết giảm rate.
5. Nếu Frostbite vẫn yếu dù gradient/usable information tốt, ưu tiên nghiên cứu
   exploration, temporal horizon và task-relevant target. Không tiếp tục tune
   reconstruction chỉ để cải thiện hình ảnh.
6. Hướng có lợi cần thêm training seed trước khi claim improvement.

## Implementation và artifacts

- `dreamerv3/reborn.py`: tách metric loss SF/Q, giữ nguyên tổng loss.
- `dreamerv3/agent_reborn.py`: weighted SF/Q và imagined return raw.
- `corewm_eval/reborn_experiment.py`: switch low_rate và các arm độc lập.
- `corewm_eval/reborn_diagnostic.py`: representation + complete-return critic calibration.
- `corewm_eval/reborn_actor_audit.py`: actual branch gradient + paired interventions.
- `corewm_eval/reborn_hypothesis_suite.py`: bốn wave, exact-budget train, evaluation, audit.
- `corewm_eval/reborn_checkpoint_eval.py`: cùng arm trong evaluation, hash validation.
- `tests/test_reborn.py`: kiểm tra component sum, gradient route, intervention bounds.

Run root chứa snapshot source và SHA256 manifest trước submit. Mỗi arm/game
có train log, config, checkpoint, evaluation returns/hash, diagnostics.json,
actor_audit_own/audit.json và actor_audit_shared/audit.json (ablation). Progress
và stage timing được lưu để phân biệt chờ, đang chạy, hoàn thành hoặc lỗi.

Ước tính ban đầu 8–12 giờ cho bốn wave gồm training, toàn bộ evaluation và
audit; time limit job 24 giờ. Thời gian có thể tăng với episode dài, JIT và
gradient diagnostic. Không cần monitor dài hạn sau khi xác nhận job hoạt động.
