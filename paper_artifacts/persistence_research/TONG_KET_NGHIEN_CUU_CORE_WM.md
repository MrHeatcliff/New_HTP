# Tổng kết nghiên cứu constraint cho CoRe-WM

Cập nhật lần cuối: ngày 12 tháng 9 năm 2026. Nhánh Git: `constraint`.

Tài liệu này tổng hợp toàn bộ quá trình nghiên cứu kể từ khi đặt ra yêu cầu làm cho compact prefix ổn định hơn theo thời gian, cho đến thí nghiệm factorial có constraint ở job 3669. Mỗi vòng được trình bày theo thứ tự: **giả thuyết → thay đổi → kết quả → chẩn đoán**.

Trạng thái của giả thuyết được chia thành bốn mức:

* **Được ủng hộ cục bộ:** có bằng chứng trong phạm vi thí nghiệm đã chạy, chưa phải kết luận tổng quát.
* **Bị bác bỏ:** kết quả không đạt tiêu chí đã đặt ra trước thí nghiệm.
* **Kết quả hỗn hợp:** tác động đổi dấu theo game, seed hoặc điều kiện đi kèm.
* **Còn mở:** dữ liệu hiện tại chưa đủ để kết luận.

Phần lớn thí nghiệm phát triển chỉ dùng một training seed, bốn game đã được lựa chọn để nghiên cứu, một đoạn continuation ngắn hoặc tập clip ở đầu episode. Vì vậy, kết quả tốt trong tài liệu không được hiểu là một định lý hay kết luận chung cho Atari 100K.

## 1. Điểm yếu ban đầu

Multi-Stride Prefix Dynamics ban đầu yêu cầu:

$$
z_t^{(1:\ell)} \longrightarrow z_{t+\Delta_\ell}^{(1:\ell)}
$$

phải dự đoán được từ prefix hiện tại và chuỗi action ở giữa. Tuy nhiên, khả năng dự đoán không đảm bảo cùng prefix giữ cùng yếu tố ổn định hoặc cùng vai trò ngữ nghĩa theo thời gian.

Ví dụ, một tín hiệu quay nhanh nhưng hoàn toàn xác định vẫn rất dễ dự đoán. Ngược lại, một representation hằng luôn ổn định nhưng không chứa thông tin hữu ích.

Mục tiêu thực nghiệm vì vậy phải đồng thời thỏa mãn bốn yêu cầu:

1. Compact prefix thay đổi chậm hơn ở horizon dài.
2. Prefix vẫn giữ thông tin dự đoán được và có ích cho task.
3. Prefix không collapse và các block không trở nên dư thừa.
4. Control performance không giảm.

Không loss nào đã triển khai có thể tự xác định semantic factor. Mọi kết luận đều bị giới hạn bởi target và trajectory thực tế được đo.

## 2. Method tham chiếu và constraint hiện tại

Representation cơ sở là:

$$
h_t=[h_t^{\mathrm{deter}},\mathrm{vec}(h_t^{\mathrm{stoch}})],
\qquad z_t=S_\psi(h_t).
$$

Các prefix tích lũy có kích thước `[128,256,512,1024,2048]`, tương ứng với stride `[16,8,4,2,1]`. Actor, critic, reward head và continuation head đều nhận toàn bộ $z$. Policy imagination vẫn dùng RSSM cơ sở, không dùng trực tiếp các predictor nhiều stride để rollout.

Constraint đang được giữ làm cấu hình tham chiếu là:

$$
\mathcal L_C=
0.1\mathcal L_{\mathrm{persist}}+
0.01\mathcal L_{\mathrm{iso}}.
$$

Persistence đo độ dịch chuyển đã chuẩn hóa theo covariance của 128 chiều đầu, tại mọi lag từ 1 đến 16 và loại các cặp vượt qua reset:

$$
\mathcal L_{\mathrm{persist}}=
\frac1K\sum_{k=1}^{K}\frac1{2d}
\mathrm{tr}\left[(C_k+\epsilon_kI)^{-1}Q_k\right],
\qquad d=128.
$$

Isotropy hạn chế việc toàn bộ variance tập trung vào quá ít hướng:

$$
\mathcal L_{\mathrm{iso}}=
\frac1d\left\|
\frac{C}{\max(\mathrm{tr}(C)/d,10^{-8})}-I
\right\|_F^2.
$$

Phần toán học và đường gradient đầy đủ nằm trong [CONSTRAINT_MATH_AND_INTUITION.md](CONSTRAINT_MATH_AND_INTUITION.md).

Lượt chạy 26 game sử dụng chính xác package sau:

* whitened persistence: `0.1`;
* mọi lag từ 1 đến 16;
* isotropy: `0.01`;
* progressive reconstruction về latent $h_t$;
* cửa sổ target reconstruction đầu tiên bằng 1;
* prediction reset mask tắt;
* auxiliary gradient không đi trực tiếp về RSSM qua nhánh CoRe-WM.

Toàn bộ 26 saved config đã được kiểm tra và không có game nào lệch cấu hình.

## 3. Cách đọc độ mạnh của bằng chứng

Các loại thí nghiệm trong quá trình nghiên cứu có độ mạnh khác nhau:

* Thí nghiệm synthetic kiểm tra cơ chế trong mô hình tuyến tính đơn giản.
* Continuation ngắn kiểm tra độ nhạy sau khi load một checkpoint và tạo replay mới; đây không phải một lần train độc lập từ đầu.
* Thí nghiệm from-scratch trên bốn game cho phép so sánh các arm trong cùng protocol, nhưng thường chỉ có một training seed.
* Bootstrap theo evaluation episode chỉ đo biến thiên episode khi policy đã cố định; nó không đo uncertainty giữa các training seed.
* Representation probe thường dùng 10 episode train, 10 validation và 10 test ở đầu episode của source policy. Target thường là random projection của RSSM feature, không phải nhãn semantic.
* Horizon 64 là phép ngoại suy: prefix đầu chỉ được train ở stride 16 và imagination length của actor là 15 trong config đã kiểm tra.
* Boxing, Up N Down, Frostbite và Road Runner đã được dùng nhiều lần để phát triển method, nên không còn là tập game held-out.

## 4. Toàn bộ các vòng giả thuyết và thực nghiệm

### 4.1. Từ pooled temporal penalty sang covariance whitening

**Giả thuyết.** Pooled ratio cũ có thể bị giảm bằng cách tăng biên độ của một chiều thay đổi chậm, qua đó che các chiều thay đổi nhanh trong tổng variance. Chuẩn hóa bằng full covariance sẽ giảm shortcut theo từng chiều này.

**Thay đổi.** Thay tỷ lệ tổng displacement/tổng variance bằng phép giải full covariance có relative ridge. Thêm isotropy cho prefix đầu vì whitening riêng lẻ không ngăn concentration hoặc exact collapse. Cơ chế được kiểm tra trên synthetic; Atari được kiểm tra bằng Alien continuation và checkpoint dài hơn.

**Kết quả.** Checkpoint Alien dài với whitened persistence 0.1 đạt `812.8`, so với `434.0` của checkpoint pooled-weight-1 trước đó trên 50 episode. Khoảng chênh lệch có điều kiện xấp xỉ `[270.6,502.6]`. Effective rank của prefix đầu tăng `7.87→40.49`, CKA giữa block giảm `.845→.356`, và held-out whitened displacement tốt hơn. Tuy nhiên, cải thiện future probe của prefix đầu chưa rõ; block 2 vẫn dự đoán tốt hơn block 1 tại horizon 16 và 32.

**Chẩn đoán: được ủng hộ làm constraint tham chiếu; temporal hierarchy còn mở.** So sánh control đồng thời thay đổi cả metric và weight, không có independent training seed. Package này xử lý một shortcut toán học thật và cho tín hiệu tốt về diversity/control, nhưng chưa chứng minh semantic hierarchy.

### 4.2. Target reconstruction trung bình nhân quả

**Giả thuyết.** Ép prefix nhỏ nhất tái tạo $h_t$ tức thời sẽ thưởng cho các chi tiết thay đổi nhanh, xung đột với stride 16. Target là trung bình nhân quả của 16 observation cho riêng reconstruction level đầu có thể giữ nội dung ổn định hơn.

**Thay đổi.** Job 3332 so sánh cửa sổ target 1 và 16, giữ prediction và constraint cố định. Mỗi arm tiếp tục cùng checkpoint Alien thêm 5k action.

**Kết quả.** Control tăng `499.4→742.8`, khoảng chênh lệch `[132.2,363.4]`; rank tăng `37.73→41.98`; CKA giảm `.399→.336`; displacement tốt hơn nhẹ. Nhưng future R2 của B1 giảm tại mọi horizon dài:

* horizon 16: `.15956→.15075`;
* horizon 32: `.09391→.08588`;
* horizon 64: `.03748→.03300`.

Cả hai arm đều thấp hơn source policy 812.8.

**Chẩn đoán: bị bác bỏ như một lời giải cho temporal hierarchy.** Một số proxy và control continuation tốt hơn, nhưng metric dự đoán cốt lõi xấu đi. Fresh replay và một continuation seed tiếp tục hạn chế kết luận. Window 16 không được chọn làm method mới.

### 4.3. Loại prediction pair vượt episode reset

**Giả thuyết.** Source-target pair vượt qua biên episode tạo supervision sai, đặc biệt đối với stride dài. Audit tìm thấy 2.37% pair lỗi ở stride 16.

**Thay đổi.** Job 3333 so sánh reset mask tắt/bật từ cùng Alien checkpoint, thêm 5k action, window 1 và 50 evaluation episode.

**Kết quả.** Masking cải thiện B1 R2:

* horizon 32: `.08847→.10837`, khoảng `[.00562,.03407]`;
* horizon 64: `.03444→.03887`, khoảng `[.00096,.00767]`.

Rank giảm `37.51→36.43`. Control giảm `575.6→475.4`, chênh lệch `-100.2`, khoảng `[-298,49]`. Khi loại pair lỗi, effective prediction-loss weight cũng giảm.

**Chẩn đoán: xác nhận lỗi supervision; lợi ích control chưa được chứng minh.** Target trong cùng episode là đúng về mặt khái niệm, nhưng thí nghiệm chưa weight-match và không chứng minh tăng control. Một bản sửa sạch trong tương lai phải chuẩn hóa theo valid-pair count rồi train lại từ đầu.

### 4.4. Constraint suite trên 26 game

**Giả thuyết.** Package whitened persistence 0.1 + isotropy 0.01 có thể train ổn định trên toàn bộ Atari 100K thay vì chỉ Alien.

**Thay đổi.** Job 3335 train 26 game, seed 0, size25m, 110k action/game, hai H200, hai game/GPU và tám CPU/game.

**Kết quả.** Cả 26 game hoàn thành với saved config đồng nhất và các metric prediction, persistence, isotropy khác 0. Run này tạo đường màu tím trong folder `full_vs_dreamerv3_constraint_suite_learning_curves`.

**Chẩn đoán: tính khả thi được ủng hộ; comparative performance chưa được giải quyết.** Đường constraint chỉ có một seed và dùng training-episode returns. Các curve lịch sử có năm seed, và Full CoRe-WM dùng isolated evaluation. Run cho thấy không có numerical collapse phổ quát, nhưng không phải so sánh năm-seed có kiểm soát với Full không constraint.

### 4.5. Dừng isotropy bằng rank floor 32

**Giả thuyết.** Full isotropy tiếp tục thưởng cho fast nuisance dimension ngay cả khi prefix đã đủ diversity. Dừng penalty khi participation rank đạt 32 có thể giảm xung đột.

**Thay đổi.** Job 3353 tiếp tục các source checkpoint 100k thêm 20k action, so isotropy với rank-floor 32.

| Game | Isotropy | Rank floor 32 | Rank B1 cũ→mới | CKA cũ→mới |
|---|---:|---:|---:|---:|
| Boxing | 63.55 | 49.6 | 65.31→34.94 | .429→.623 |
| Up N Down | 6716.5 | 5419 | 55.13→33.09 | .458→.602 |
| Frostbite | 2336.5 | 2622 | 46.68→25.21 | .449→.701 |
| Road Runner | 15545 | 15255 | 64.27→29.03 | .482→.677 |

**Chẩn đoán: bị bác bỏ.** Control giảm rõ ở Boxing và Up N Down; rank và redundancy cùng dịch chuyển ngược kỳ vọng. Rank floor trên training batch không đảm bảo held-out rank.

### 4.6. Giảm tương quan giữa block 1 và block 2

**Giả thuyết.** Giảm linear CKA trực tiếp sẽ khiến block 2 học nội dung bổ sung thay vì lặp lại block 1.

**Thay đổi.** Job 3368 so isotropy với isotropy + CKA weight 0.01. Block 1 bị detach trong CKA loss.

| Game | Isotropy | Thêm decorrelation | Rank B2 cũ→mới | CKA cũ→mới |
|---|---:|---:|---:|---:|
| Boxing | 63.55 | 30.35 | 7.96→5.56 | .429→.363 |
| Up N Down | 6716.5 | 4947 | 9.79→4.02 | .458→.341 |
| Frostbite | 2526 | 2540 | 4.54→4.42 | .407→.383 |
| Road Runner | 12215 | 15540 | 15.21→4.88 | .508→.308 |

**Chẩn đoán: bị bác bỏ.** CKA giảm ở cả bốn game, đúng metric được tối ưu, nhưng rank B2 giảm thêm và control giảm ở hai game. Loss đã loại shared signal thay vì đảm bảo block 2 giữ thông tin bổ sung hữu ích. B2 rank guard từng được đề xuất, nhưng không tìm thấy artifact của một comparison hoàn tất; do đó nó không được tính là bằng chứng thực nghiệm.

### 4.7. Predict trực tiếp từ h và bỏ projection/reconstruction

**Giả thuyết.** Multi-stride prediction trực tiếp trên $h_t$ có thể khiến projection và progressive reconstruction trở nên không cần thiết. Constraint vẫn phải được giữ trong ablation.

**Thay đổi.** Bỏ projection và reconstruction, cho auxiliary gradient đi vào backbone, rồi so h prediction không/có persistence-isotropy trên bốn game.

| Game | h prediction | h prediction + constraint |
|---|---:|---:|
| Boxing | 65.35 | 68.1 |
| Up N Down | 7241 | 6508 |
| Frostbite | 248.5 | 307.5 |
| Road Runner | 10445 | 8755 |

**Chẩn đoán: hỗn hợp, không được chọn.** Constraint giúp Boxing và Frostbite nhưng làm giảm Up N Down và Road Runner. Các lát cắt coordinate cố định của $h$ không có cơ chế learned ordering; việc gán horizon cho 128/256/... chiều đầu không tự tạo semantic hierarchy. Kết quả ủng hộ việc giữ learned projection/reconstruction làm tham chiếu, nhưng không chứng minh mọi reconstruction objective đều có ích.

### 4.8. Reconstruction ảnh theo spatial pyramid

**Giả thuyết.** Tái tạo ảnh theo độ phân giải có thể tạo vai trò rõ cho prefix: prefix nhỏ giữ cấu trúc thô, block sau thêm chi tiết. Giả thuyết cạnh tranh là coarse trong không gian không đồng nghĩa chậm theo thời gian; 4x4 còn có thể xóa vật thể nhỏ quan trọng.

**Thay đổi.** Job 3405 train from scratch seed 0 với ba arm: reconstruction latent $h$, ảnh 64x64 ở mọi level, và pyramid `[4,8,16,32,64]`. Prediction và constraint giữ nguyên.

| Game | Latent h | Ảnh uniform | Pyramid 4x4→64 |
|---|---:|---:|---:|
| Boxing | 80.05 | 81.85 | 91.25 |
| Up N Down | 27616.5 | 9286 | 35569 |
| Frostbite | 2083 | 2753.5 | 3238 |
| Road Runner | 9035 | 19000 | 9775 |

Pyramid thắng uniform ở ba game, nhưng B1 R2@64 lần lượt thay đổi `-.0561→-.0561`, `.2119→.1936`, `.3729→.3730`, `.1966→.1682`. Up N Down tăng control trong khi rank B1/B2 giảm và CKA tăng. Không có dead coordinate trong prefix đầu.

**Chẩn đoán: có tín hiệu control ở một seed; temporal hierarchy không được ủng hộ.** Pixel loss bị chi phối bởi background và độ khó target. Control và predictive probe không cùng chiều.

### 4.9. Nâng target đầu từ 4x4 lên 8x8

**Giả thuyết.** Target 4x4 quá thô, làm mất vật thể nhỏ. Lịch `[8,16,32,64,64]` sẽ phục hồi control và vẫn giữ thông tin dài hạn.

Ở seed 1, job 3502 cho kết quả ban đầu thuận lợi: 8x8 thắng 4x4 ở Boxing `72.55 vs 64.2`, Frostbite `2489.5 vs 371.5` và Road Runner `13525 vs 8450`, nhưng thua Up N Down `6417 vs 7164.5`. Nó chỉ thắng uniform ở Road Runner.

Replication seed 2 của job 3525 đã được định nghĩa trước khi xem kết quả:

| Game | Uniform | Pyramid 4x4 | Pyramid 8x8 |
|---|---:|---:|---:|
| Boxing | 72.05 | 74.15 | 63.55 |
| Up N Down | 176639.5 | 70242 | 66576.5 |
| Frostbite | 2312.5 | 3625.5 | 391 |
| Road Runner | 1855 | 13820 | 8420 |

8x8 thua 4x4 ở cả bốn game. Absolute-future-image B1 R2@64 ở seed 1 cũng không đi cùng control: Frostbite `.5101→.4907`; Road Runner `.3564→.3564`. Delta-image probe có shortcut từ ảnh hiện tại nên không được xem là bằng chứng về motion semantics.

**Chẩn đoán: bị bác bỏ.** Tiêu chí đặt trước là thắng lại ít nhất ba game ở seed 2, nhưng không đạt. Biến thiên rất lớn giữa các run, ví dụ Road Runner uniform `19000→1945→1855`, cũng cho thấy episode bootstrap không thể thay thế training-seed replication.

### 4.10. Audit tín hiệu mà từng component thực sự học

**Các giả thuyết.** Predictor có thể chỉ copy hoặc bỏ qua action; aggregate reward loss có thể che lỗi ở reward hiếm; actor có thể không dùng compact prefix; posterior có thể tốt nhưng imagination làm mất tín hiệu task.

**Thay đổi.** Job 3526 và 3668 đánh giá 12 checkpoint ảnh seed 1 trên cùng mười held-out episode/game. Audit gồm:

* predictor thật so với copy baseline;
* action đúng, action roll và action 0;
* reward error tách zero/nonzero event;
* posterior và RSSM rollout với cùng frame và cùng RNG;
* can thiệp actor theo từng nhóm 128 chiều.

**Kết quả prediction.** Tỷ lệ MSE predictor/copy của B1 nằm trong `.431–.690`. Rolling action làm MSE B1 tăng khoảng `2.9–15.8%`. Episode-bootstrap interval cho copy-minus-predictor và rolled-minus-true đều dương ở cả 12 checkpoint.

**Kết quả reward.** Khi so đúng cùng frame:

* Frostbite uniform: reward MAE `.401→1.125` từ posterior sang imagination, chênh lệch `.723 [.552,.879]`.
* Frostbite pyramid: `.232→.373`; soft: `.156→.461`.
* Up N Down pyramid: `6.471→8.538`, chênh lệch `2.067 [1.120,3.050]`.
* Road Runner đã gần zero-reward baseline ngay trên posterior: pyramid `10.416`, soft `10.485`, baseline `10.46875`; rollout chỉ thêm rất ít lỗi.
* Tất cả clip đều không có terminal event, nên chưa đánh giá được continuation calibration ở terminal.

Can thiệp cùng 128 chiều cho thấy B1 tạo actor KL thấp hơn trung bình các chunk thuộc B5 ở cả 12 checkpoint. Road Runner uniform là `.0036` so với `.0409`. Tuy nhiên, covariance và độ lớn perturbation vẫn khác; reliance không đồng nghĩa usefulness.

**Chẩn đoán: tìm thấy nhiều failure cục bộ, không có một nguyên nhân chung.** Predictor thật có học vượt copy và có dùng action trên các clip đã đo. Frostbite và một phần Up N Down có degradation qua imagination. Road Runner thất bại sớm hơn, ở reward prediction, information access hoặc sparse coverage ngay trên posterior. Actor ít nhạy với B1, phù hợp với việc full-$z$ head không bị buộc phải dùng compact prefix.

### 4.11. Factorial 2x2 có constraint trên formulation latent h

Đây là kết quả mới nhất và là phép cô lập component mạnh nhất trong chuỗi nghiên cứu.

**Giả thuyết H1.** Prediction chỉ giúp policy rõ khi reconstruction bảo toàn đủ thông tin hữu ích; kỳ vọng interaction dương.

**Giả thuyết H2.** Reconstruction cung cấp phần lớn lợi ích; prediction có thể tối ưu target nhưng không giúp control.

**Thay đổi.** Job 3669 thêm `prediction_loss_scale`, nhờ đó prediction MSE có thể bằng 0 trong khi persistence 0.1 và isotropy 0.01 vẫn chạy. Bốn arm seed 0 được train từ đầu 100k action:

| Tên trong report | Reconstruction h | Prediction | Constraint |
|---|---:|---:|---:|
| flat | Tắt | Tắt | Bật |
| rec_only | Bật | Tắt | Bật |
| pdyn_only | Tắt | Bật | Bật |
| full | Bật | Bật | Bật |

`flat` ở đây là tên nội bộ của arm constraint-only, không phải DreamerV3. Objective là:

$$
\mathcal L=\mathcal L_{\mathrm{other}}+
R\mathcal L_{\mathrm{rec}}+
\lambda_{\mathrm{pdyn}}
(P\mathcal L_{\mathrm{prediction}}+0.1\mathcal L_{\mathrm{persist}}+0.01\mathcal L_{\mathrm{iso}}).
$$

Cả 16 condition hoàn tất. Mean return trên 20 evaluation episode:

| Game | Chỉ constraint | Thêm reconstruction | Thêm prediction | Full |
|---|---:|---:|---:|---:|
| Boxing | 43.2 | 56.9 | 70.1 | **80.05** |
| Up N Down | **81872** | 6652 | 31592 | 13565 |
| Frostbite | 2387 | **3157** | 429.5 | 2038 |
| Road Runner | 14950 | 13025 | 6960 | **15790** |

Factorial effect và conditional episode-bootstrap 95% interval:

| Game | Hiệu ứng prediction khi không/có reconstruction | Hiệu ứng reconstruction khi không/có prediction | Interaction |
|---|---|---|---:|
| Boxing | `+26.9 [18.0,35.65]` / `+23.15 [16.05,30.30]` | `+13.7 [3.75,23.70]` / `+9.95 [4.60,15.45]` | `-3.75 [-14.95,7.60]` |
| Up N Down | `-50280 [-68799.5,-32042.8]` / `+6913 [648.0,14622.1]` | `-75220 [-90548.7,-60821.9]` / `-18027 [-31066.5,-5606.0]` | `+57193 [37755.5,77034.7]` |
| Frostbite | `-1957.5 [-2142,-1767]` / `-1119 [-1578.5,-700]` | `+770 [473.5,1131]` / `+1608.5 [1260,1948.5]` | `+838.5 [343,1296]` |
| Road Runner | `-7990 [-8795,-7125]` / `+2765 [1870,3690]` | `-1925 [-2820,-1065]` / `+8830 [7945,9655]` | `+10755 [9510,11985]` |

Các interval chỉ phản ánh variation giữa evaluation episode khi bốn policy seed 0 đã được train xong. Chúng không phải training-seed confidence interval.

Representation metrics cũng không đưa ra một lời giải thích chung:

* **Boxing:** Full có control tốt nhất, nhưng B1 R2@64 âm ở mọi arm và chỉ dao động `-.0542` đến `-.0583`.
* **Up N Down:** Full có B1 R2@16 tốt nhất `.4972` và R2@64 `.2233`, nhưng constraint-only có return cao gấp khoảng sáu lần Full.
* **Frostbite:** prediction-only có B1 R2@16 `.5707` và R2@64 `.3835`, nhưng return thấp nhất 429.5 và displacement cao nhất 1.332. Reconstruction-only có return tốt nhất và R2@64 `.3886`.
* **Road Runner:** Full có return và R2@16 tốt nhất; prediction-only có R2@64 tốt nhất `.1914` nhưng control thấp. B2 rank của Full chỉ 5.88 và CKA cao nhất `.685`.
* Mọi arm đều có 0 dead coordinate ở B1 theo threshold hiện tại. Điều này loại exact coordinate collapse, không loại low-rank concentration hoặc redundancy.

**Chẩn đoán: H1 hỗn hợp; H2 bị bác bỏ nếu xem là giải thích chung.** Interaction dương ở Up N Down, Frostbite và Road Runner, nhưng âm/không rõ ở Boxing. Prediction giúp mạnh ở Boxing, gây hại mạnh ở Frostbite, và đổi dấu theo việc có reconstruction ở Up N Down/Road Runner. Reconstruction cũng giúp Boxing/Frostbite nhưng có thể gây hại ở Up N Down/Road Runner. Full chỉ tốt nhất ở Boxing và nhỉnh hơn ở Road Runner.

Kết quả này chứng minh hai auxiliary component đều có khả năng tạo **negative transfer**, và tác động phụ thuộc mạnh vào context. Nó chưa xác định context đó là game dynamics, reward sparsity, gradient conflict, exploration hay variation từ một training seed.

## 5. Sổ trạng thái các giả thuyết

| Mã | Giả thuyết | Trạng thái bằng chứng | Quyết định hiện tại |
|---|---|---|---|
| H0 | Predictability tự đảm bảo semantic role ổn định | Bị bác bỏ về khái niệm và thực nghiệm | Không tuyên bố guarantee này |
| H1 | Covariance whitening giảm pooled scale shortcut | Được ủng hộ về cơ chế | Giữ làm constraint tham chiếu |
| H2 | Whitened persistence + isotropy tạo temporal hierarchy | Còn mở | Alien tốt hơn, hierarchy probe chưa thuyết phục |
| H3 | Causal average target giải quyết xung đột reconstruction | Bị bác bỏ | Không dùng window 16 |
| H4 | Pair vượt reset làm hại long-horizon prediction | Được ủng hộ cục bộ | R2 tăng; control và weight-matched effect chưa rõ |
| H5 | Rank-floor 32 tránh nuisance do isotropy | Bị bác bỏ | Rank, CKA và control thường xấu hơn |
| H6 | CKA decorrelation tạo B2 bổ sung | Bị bác bỏ | CKA giảm bằng cách làm nghèo B2 |
| H7 | Direct prediction từ h khiến projection/reconstruction không cần thiết | Không được ủng hộ | Control hỗn hợp; coordinate h không được learned ordering |
| H8 | Spatial pyramid trực tiếp tạo temporal hierarchy hữu ích | Không được ủng hộ | Control một seed không đi cùng probe |
| H9 | Target 4x4 quá thô; 8x8 cải thiện ổn định | Bị seed 2 bác bỏ | 8x8 thua 4x4 ở cả bốn game |
| H10 | PDyn chỉ copy hoặc bỏ qua action | Bị bác bỏ trên clip đã đo | Predictor thắng copy và nhạy với action |
| H11 | Reward failure chủ yếu sinh ra trong imagination | Phụ thuộc game | Đúng ở Frostbite/một phần Up N Down; Road Runner lỗi sớm hơn |
| H12 | Actor ít dùng compact prefix | Được ủng hộ như reliance pattern | B1 perturbation ảnh hưởng thấp hơn; usefulness còn mở |
| H13 | Prediction chủ yếu giúp khi có reconstruction | Hỗn hợp | Interaction dương 3/4 game nhưng main effect đổi dấu |
| H14 | Reconstruction luôn cần thiết | Bị bác bỏ | Constraint-only thắng lớn ở Up N Down và cạnh tranh ở game khác |
| H15 | Multi-stride prediction luôn tăng control | Bị bác bỏ | Giúp Boxing, hại Frostbite, phụ thuộc context ở game khác |

## 6. Thay đổi nào đã đủ bằng chứng để giữ lại?

Thay đổi cốt lõi duy nhất được giữ là **whitened persistence 0.1 + all-lag + isotropy 0.01**. Lý do là nó xử lý shortcut toán học có thật và cho tín hiệu tốt về control/diversity trên Alien. Giới hạn semantic vẫn giữ nguyên.

Các hướng sau chưa đạt điều kiện để đưa vào method:

* target trung bình nhân quả;
* rank floor 32;
* CKA decorrelation;
* direct-h prediction;
* spatial pyramid 8x8;
* bỏ reconstruction hoặc prediction trên mọi game.

Hạ tầng đánh giá đã được cải thiện đáng kể:

* phát hiện và tùy chọn loại reset-crossing pair;
* đo rank, CKA, dead coordinate, displacement và probe dài hạn trên split cố định;
* so actual PDyn với copy/action control;
* so posterior và same-RNG imagined reward trên cùng frame;
* can thiệp actor theo nhóm cùng 128 chiều;
* tắt prediction độc lập mà vẫn giữ constraint;
* lưu seed, action budget, source checkpoint và limitation trong manifest.

Các thay đổi hạ tầng này cải thiện khả năng quy nguyên nhân, không tự cải thiện policy score.

## 7. Mô hình failure hiện tại

Dữ liệu không còn ủng hộ một failure duy nhất cho mọi game.

1. **Auxiliary conflict phụ thuộc context.** Reconstruction và prediction đều có thể giúp hoặc hại. Interaction có thể cứu lẫn nhau ở một số game, rõ nhất là Road Runner, nhưng không phổ quát.
2. **Prediction học đúng supervised target nhưng target có thể lệch khỏi control.** Frostbite prediction-only là phản ví dụ rõ: future probe tốt nhưng return rất thấp.
3. **Policy không có ràng buộc cấu trúc phải dùng compact prefix.** Mọi control head nhận full $z$; các chunk phía sau thường làm actor thay đổi nhiều hơn B1.
4. **World-model failure xuất hiện ở vị trí khác nhau.** Frostbite suy giảm qua imagination; Road Runner reward error đã cao trên posterior.
5. **Probe hiện tại đo khả năng truy xuất, không đo semantic identity.** Random-projected $h$, image average, pixel MAE và CKA đều có thể tốt hơn trong khi control giảm.
6. **Training variance lớn.** Ranking giữa các arm đã đảo chiều qua seed. Một thay đổi chỉ được promote sau replication from scratch bằng independent seed.

## 8. Các giả thuyết nghiên cứu tiếp theo

### N1. Xung đột gradient giữa auxiliary và task objective

**Giả thuyết.** Các game bị negative transfer có gradient reconstruction/prediction đối nghịch hoặc lớn quá mức so với reward/value/policy gradient trong shared projection $S_\psi$.

**Phép đo cần chạy trước.** Trên cùng replay batch của latent factorial checkpoint, tính gradient norm và cosine theo từng parameter group: projection trunk và từng output block. Tách reconstruction, prediction, persistence, isotropy, reward, continuation, value, replay-value và policy.

**Kỳ vọng.** Frostbite prediction arm và Up N Down reconstruction arm có negative cosine hoặc auxiliary/task norm ratio cao hơn các effect có lợi ở Boxing. Nếu không có quan hệ này, bác bỏ gradient conflict là nguyên nhân chính.

**Cách sửa nếu được ủng hộ.** Thay một yếu tố: stop-gradient route, adaptive gradient projection hoặc loss weight của component gây xung đột. Luôn giữ một control arm không sửa.

### N2. Thông tin task bị mất ở h→z hay tồn tại nhưng không được dùng

**Giả thuyết.** Một số failure xảy ra vì thông tin có trong $h$ nhưng mất trong $z$; số khác xảy ra vì $z$ có thông tin nhưng actor/reward head không khai thác được.

**Phép đo.** Dùng cùng protocol để fit held-out probe từ $h$, B1, các cumulative prefix và full $z$ đến immediate reward, nonzero-reward class, discounted return và terminal. Cần thu thập trajectory có terminal vì tập clip hiện tại không có terminal. Sau đó freeze representation và train lại lightweight head có capacity bằng nhau.

**Kỳ vọng chẩn đoán.** Nếu h tốt nhưng z kém, sửa projection/reconstruction. Nếu z probe tốt nhưng existing head kém hơn retrained head, sửa head optimization hoặc policy access. Nếu h đã kém, tập trung vào encoder/RSSM hoặc data coverage.

### N3. Temporal prediction có chọn lọc theo task

**Giả thuyết.** Uniform latent MSE thưởng cho predictable nuisance dimension. Prediction nên ưu tiên future feature liên quan reward, value và continuation.

**Điều kiện trước khi sửa.** N2 phải xác nhận task information nằm ở đâu và N1 phải đánh giá optimization conflict. Nếu chưa, reward weighting có thể chỉ khuếch đại sparse/noisy head.

**Candidate.** Weight prediction residual theo task relevance hoặc dự đoán thêm compact task sufficient statistics. Tiếp tục giữ copy/action control và chỉ chấp nhận khi B1 task probe, control và rank cùng cải thiện.

### N4. Buộc policy truy cập hierarchy hợp lý

**Giả thuyết.** Full-$z$ head bypass compact-prefix specialization. Prefix dropout hoặc horizon-conditioned access có thể tăng khả năng dùng B1 mà vẫn giữ full capacity khi cần.

**Rủi ro và kiểm tra.** Ép dùng prefix có thể loại fast control signal. Trước tiên train lại frozen head trên từng prefix. Thí nghiệm train chỉ thay một access rule, theo dõi entropy/reward coverage, và bác bỏ nếu B1 reliance tăng nhưng control giảm.

### N5. Reset masking có weight matching

**Giả thuyết.** Episode-safe prediction cải thiện representation nếu valid-pair normalization giữ effective level weight không đổi.

**Cách sửa.** Chuẩn hóa từng level theo valid count và broadcast loss đã weight-match, thay vì đặt pair lỗi bằng 0 rồi lấy outer mean. Sau đó so from scratch với đúng một yếu tố thay đổi. Đây là sửa tính đúng của supervision, không phải semantic constraint.

## 9. Quyết định nghiên cứu hiện tại

Không promote spatial 8x8, rank floor, CKA decorrelation, causal averaging hoặc direct-h prediction. Chưa bỏ reconstruction hoặc prediction trên toàn bộ method chỉ từ bốn game và một seed. Family checkpoint 26-game với constraint tiếp tục là mốc tham chiếu.

Bước có giá trị chẩn đoán cao nhất tiếp theo là **gradient/task-information audit chỉ đọc trên các latent factorial checkpoint đã hoàn thành**. Sau audit, chọn đúng một intervention dựa trên failure quan sát được. Cách này phù hợp với yêu cầu mỗi vòng causal experiment chỉ thay đổi một yếu tố chính.

## 10. Chỉ mục artifact

| Giai đoạn | Artifact chính |
|---|---|
| Toán học constraint | `CONSTRAINT_MATH_AND_INTUITION.md` |
| Whitening, Alien, coarse target, reset mask | `RESEARCH_PROTOCOL.md`, `latest_3331/`, `coarse_target_3332/`, `episode_mask_3333/` |
| Suite 26 game | `production_runs/constraint_full26_seed0_OzirkNB8`, `full_vs_dreamerv3_constraint_suite_learning_curves/` |
| Rank floor | `FOUR_GAME_RANK_FLOOR.md`, `RANK32_DIAGNOSIS_AND_DECORRELATION.md` |
| Decorrelation | `DECORRELATION_DIAGNOSIS_AND_RANK_GUARD.md`, `production_runs/decorrelation_four_game_ro2vQg01/` |
| Direct h | `production_runs/direct_h_constraint_NpMOUZXh/` |
| Spatial seed 0/1/2 | `SPATIAL_RECONSTRUCTION.md`, `SPATIAL_SEED1_DIAGNOSIS.md`, `production_runs/spatial_seed2_UOW5Hr6o/` |
| Component audit | `COMPONENT_FAILURE_AUDIT.md`, `COMPONENT_AUDIT_RESULTS_3526.md`, `MATCHED_AUDIT_RESULTS_3668.md` |
| Constrained factorial | `CONSTRAINED_FACTORIAL_PROTOCOL.md`, `production_runs/constrained_factorial_NOBwTnNw/factorial_summary.json` |

Các đường dẫn `production_runs` chứa checkpoint và raw log trên research server, không mặc định có trên GitHub. Tài liệu này đã chép lại các số liệu cần thiết cho quyết định. Machine-readable factorial summary vẫn nằm tại `production_runs/constrained_factorial_NOBwTnNw/factorial_summary.json` trên server.
