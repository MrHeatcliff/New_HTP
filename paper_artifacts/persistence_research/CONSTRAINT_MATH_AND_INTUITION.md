# Constraint của CoRe-WM: công thức, trực giác và giới hạn

Tài liệu đối chiếu với code nhánh `constraint` và cấu hình thực nghiệm ngày 11/09/2026. Phần constraint được mô tả đã có trong commit `ae752ba`; các biến thể spatial reconstruction là thí nghiệm riêng, không phải thành phần mới của constraint.

**Constraint đang bật là tổng của hai regularizer mềm:** (1) giảm biến thiên theo thời gian của prefix đầu bằng khoảng cách có chuẩn hóa covariance; (2) chống tập trung phương sai vào quá ít hướng bằng isotropy. Nó bổ sung vào prediction loss, không thay thế prediction. Nó **không đảm bảo** prefix giữ cùng semantic role, không đảm bảo không collapse, và không đảm bảo cải thiện control.

## 1. Ký hiệu và đối tượng được ràng buộc

World model tạo trạng thái:

$$
h_t=[h_t^{\mathrm{deter}},\mathrm{vec}(h_t^{\mathrm{stoch}})],
\qquad z_t=S_\psi(h_t).
$$

Các chiều prefix tích lũy trong lượt đang nghiên cứu là:

$$
(d_1,d_2,d_3,d_4,d_5)=(128,256,512,1024,2048).
$$

Vì vậy các **block riêng lẻ** rộng $(128,128,256,512,1024)$ chiều. Không nhầm block $z^{(\ell)}$ với prefix $z^{(1:\ell)}$.

Hai regularizer đang bật chỉ áp dụng trực tiếp lên:

$$
x_t=z_t^{(1:1)}=z_t^{(1)}\in\mathbb R^{128}.
$$

Chúng không áp dụng trực tiếp lên toàn bộ năm prefix, không dùng nhãn object/spatial và không so ảnh. Shared projection vẫn có thể khiến việc cập nhật block đầu ảnh hưởng gián tiếp các block khác.

## 2. Prediction có gì chưa đủ?

Multi-stride prediction gán các horizon:

$$
(\Delta_1,\Delta_2,\Delta_3,\Delta_4,\Delta_5)=(16,8,4,2,1).
$$

Tại level $\ell$, predictor nhận prefix và chuỗi action, rồi khớp target từ slow projection:

$$
\widehat z_{t+\Delta_\ell}^{(1:\ell)}
=F_{\omega_\ell}(\widetilde z_t^{(1:\ell)},a_{t:t+\Delta_\ell-1}),
\qquad
\bar z_{t+\Delta_\ell}=\mathrm{sg}(S_{\bar\psi}(\mathrm{sg}(h_{t+\Delta_\ell}))).
$$

Ở level lớn hơn 1, input dùng $[\mathrm{sg}(z^{(1:\ell-1)}),z^{(\ell)}]$: các block trước là context bị chặn gradient trực tiếp trong loss của level này. Đây là isolation ở activation, không phải cách ly hoàn toàn các tham số của shared trunk.

Prediction tối thiểu hóa MSE trên prefix tương lai. Một tín hiệu đổi nhanh nhưng rất dễ dự đoán vẫn có thể đạt loss thấp. Ví dụ $x_t=(\cos(\omega t),\sin(\omega t))$ có thể được dự đoán bằng một phép quay dù thay đổi nhiều giữa hai thời điểm. Vì thế **predictable không đồng nghĩa persistent**.

Constraint bổ sung yêu cầu: trong số những biểu diễn vẫn hữu ích cho các loss khác, ưu tiên prefix đầu biến thiên ít hơn theo thời gian, sau khi tính đến độ biến thiên tự nhiên của từng hướng.

## 3. Temporal persistence: công thức đúng với implementation

### 3.1. Chỉ dùng cặp cùng episode

Giả sử minibatch có $B$ chuỗi, mỗi chuỗi dài $T$. Đặt:

$$
e_{b,t}=\sum_{u=0}^{t}\mathbf 1[\mathrm{is\_first}_{b,u}],
\qquad
m_{b,t}^{(k)}=\mathbf 1[e_{b,t}=e_{b,t+k}],
\quad 0\leq t<T-k.
$$

Với lag $k$, đặt $M_k=\sum_{b,t}m_{b,t}^{(k)}$. Các công thức tiếp theo giả sử $M_k>0$. Mỗi cặp hợp lệ có hai endpoint $a=x_{b,t}$ và $b'=x_{b,t+k}$. Trung bình được tính chung từ cả hai endpoint:

$$
\mu_k=\frac{1}{2M_k}\sum_{b,t}m_{b,t}^{(k)}(x_{b,t}+x_{b,t+k}).
$$

Các endpoint xuất hiện nhiều lần nếu tham gia nhiều cặp; đây là covariance của tập endpoint được ghép cặp trong minibatch, không phải covariance toàn dataset.

### 3.2. Covariance trạng thái và covariance sai khác

$$
C_k=\frac{1}{2M_k}\sum_{b,t}m_{b,t}^{(k)}
\left[(x_{b,t}-\mu_k)(x_{b,t}-\mu_k)^\top
+(x_{b,t+k}-\mu_k)(x_{b,t+k}-\mu_k)^\top\right],
$$

$$
Q_k=\frac{1}{M_k}\sum_{b,t}m_{b,t}^{(k)}
(x_{b,t+k}-x_{b,t})(x_{b,t+k}-x_{b,t})^\top.
$$

$Q_k$ là second moment của sai khác; code không trừ trung bình riêng của các sai khác. Tất cả thống kê này được tính ở float32.

### 3.3. Khoảng cách có chuẩn hóa covariance

Với $d=128$, implementation dùng:

$$
\epsilon_k=\max\left(10^{-4}\frac{\mathrm{tr}(C_k)}d,10^{-8}\right),
$$

$$
\boxed{\mathcal P_k=
\frac{1}{2d}\mathrm{tr}\left[(C_k+\epsilon_k I)^{-1}Q_k\right].}
$$

Code dùng `solve(C + ridge * I, difference)`, không tạo inverse tường minh. Dạng tương đương về mặt toán học là:

$$
\mathcal P_k=
\frac{1}{2dM_k}\sum_{b,t}m_{b,t}^{(k)}
\left\|(C_k+\epsilon_kI)^{-1/2}(x_{b,t+k}-x_{b,t})\right\|_2^2.
$$

Tên `whitened` chỉ phép đo này. **Không có thao tác thay $z$ bằng một representation đã whiten trước khi đưa vào actor.**

Nếu không có cặp hợp lệ, code dùng mẫu số được chặn dưới, $C_k=Q_k=0$, nên $\mathcal P_k=0$. Nếu $T\leq k$, lag đó trả về 0.

### 3.4. Vì sao không dùng MSE thông thường hoặc chỉ chia tổng variance?

MSE $\mathbb E\|x_{t+k}-x_t\|^2$ có thể giảm chỉ bằng cách nhân toàn bộ representation với một số nhỏ. Chuẩn hóa covariance giảm động cơ này: nếu $x\mapsto c x$, cả $C_k$ và $Q_k$ cùng nhân $c^2$. Khi relative ridge chi phối và chưa chạm floor, tỷ số giữ nguyên.

Bản `pooled` cũ gần tương ứng:

$$
\mathcal P_k^{\mathrm{pooled}}=
\frac{\mathrm{tr}(Q_k)}{2\mathrm{tr}(C_k)}.
$$

Nó có thể ưu tiên một hướng chậm có phương sai rất lớn, khiến các hướng nhanh bị che trong tổng. Nếu $C_k$ chéo với phương sai $v_i$, bản whitened có dạng:

$$
\mathcal P_k=\frac1{2d}\sum_i
\frac{\mathbb E[(x_{t+k,i}-x_{t,i})^2]}{v_i+\epsilon_k}.
$$

Mỗi hướng được đánh giá theo variance của chính nó. Khi có tương quan giữa các chiều, full covariance còn tính đến tương quan đó. Ridge làm phép đo chỉ xấp xỉ chuẩn hóa lý tưởng; không nên tuyên bố bất biến với mọi biến đổi tuyến tính hoặc mọi scale, đặc biệt gần suy biến.

Trong trường hợp lý tưởng full rank, không ridge và phân phối dừng: $Q_k=0$ cho persistence bằng 0; hai endpoint độc lập có cùng covariance cho $Q_k\approx2C_k$, nên giá trị khoảng 1. Đây là trực giác chuẩn hóa, **không phải khoảng giá trị bắt buộc [0,1]**.

### 3.5. Tại sao dùng mọi lag từ 1 đến 16?

Cấu hình `persistence_all_lags=True` dùng:

$$
K=\min(16,T-1),\qquad
\boxed{\mathcal L_{\mathrm{persist}}=\frac1K\sum_{k=1}^{K}\mathcal P_k.}
$$

Mỗi lag có cùng trọng số dù số cặp hợp lệ khác nhau; lag không có cặp hợp lệ vẫn đóng góp 0 vào trung bình đó. Với $T\leq1$, kết quả là 0.

Nếu chỉ đo ở lag 16, một representation dao động có chu kỳ 16 có thể giống nhau ở hai endpoint nhưng thay đổi mạnh ở giữa. Các lag 1..16 giảm lối tắt này. Tuy nhiên chúng cũng có thể phạt chuyển động nhanh hữu ích cho control. `all_lags` không phải ràng buộc ở mọi horizon: **không trực tiếp ràng buộc lag 32 hoặc 64**.

## 4. Isotropy: vì sao persistence cần thêm một thành phần khác?

Representation hằng có persistence bằng 0. Ngoài ra một representation có 128 chiều nhưng chỉ biến thiên theo vài hướng vẫn có thể quá nghèo thông tin.

Lấy tất cả $N=BT$ vector của prefix đầu, trừ mean và tính:

$$
C=\frac1N\sum_{n=1}^{N}(x_n-\mu)(x_n-\mu)^\top,
\qquad s=\max\left(\frac{\mathrm{tr}(C)}d,10^{-8}\right).
$$

Regularizer đang dùng là:

$$
\boxed{\mathcal L_{\mathrm{iso}}=\frac1d\left\|\frac Cs-I\right\|_F^2.}
$$

Covariance isotropy được tính riêng trên toàn bộ $BT$ activation, không phải covariance endpoint của từng lag. Nó không loại state ở reset; đây là thống kê phân bố trạng thái, không phải ghép transition qua reset.

Khi không chạm floor, isotropy khuyến khích các trị riêng covariance gần nhau: không để một vài hướng mang gần hết variance trong khi các hướng còn lại không hoạt động. Nó không đặt độ lớn variance tuyệt đối bằng 1.

Với participation rank:

$$
r_{\mathrm{PR}}(C)=\frac{\mathrm{tr}(C)^2}{\mathrm{tr}(C^2)},
$$

khi $C\neq0$ và floor không kích hoạt:

$$
\mathcal L_{\mathrm{iso}}=\frac d{r_{\mathrm{PR}}(C)}-1.
$$

Nếu variance phân bố đều trên $r$ hướng, isotropy bằng $d/r-1$. Với $d=128$, rank 128 cho 0, rank 32 cho 3. Đây là minh họa đại số của loss, không phải kết quả thí nghiệm.

**Giới hạn quan trọng:** isotropy có thể khuyến khích thêm nhiễu hoặc thông tin không hữu ích chỉ để tăng rank. Tại representation hằng chính xác, $C=0$, loss bằng 1 nhưng gradient qua covariance bằng 0; regularizer không đảm bảo tự thoát khỏi collapse. Gần các numerical floor, tính chất scale-free và quan hệ với rank cũng không còn đúng như trường hợp lý tưởng.

## 5. Ghép vào objective như thế nào?

Gọi $\mathcal L_{\mathrm{other}}$ là tổng có trọng số của các loss world model, reconstruction, actor, critic và replay value còn lại. Với các option khác đang tắt:

$$
\boxed{\mathcal L_{\mathrm{total}}=
\mathcal L_{\mathrm{other}}+
\lambda_{\mathrm{pdyn}}\left[
\mathcal L_{\mathrm{prediction}}
+0.1\mathcal L_{\mathrm{persist}}
+0.01\mathcal L_{\mathrm{iso}}\right].}
$$

Trong các lượt spatial đang bàn, $\lambda_{\mathrm{pdyn}}=1$. Do đó đóng góp thêm trực tiếp là $0.1\mathcal L_{\mathrm{persist}}+0.01\mathcal L_{\mathrm{iso}}$.

Implementation trả prediction loss dạng $(B,T)$, rồi cộng hai scalar regularizer bằng broadcasting. Khi lấy mean trên batch/time, mỗi scalar vẫn đóng góp đúng hệ số trên, **không bị nhân thêm $BT$**. Prediction có zero-padding các vị trí cuối không đủ horizon, còn regularizer đã tổng hợp riêng; vì vậy raw loss/weighted gradient của chúng không thể suy từ hệ số đơn thuần.

```yaml
agent:
  htp:
    use_proj: true
    use_pdyn: true
    grad_to_backbone: false
    persistence_scale: 0.1
    persistence_metric: whitened
    persistence_all_lags: true
    persistence_isotropy: 0.01
    persistence_min_rank: 0.0
    persistence_decorrelation: 0.0
    persistence_block2_rank_scale: 0.0
    use_vicreg: false
  loss_scales:
    htp_pdyn: 1.0
```

Đây là trích cấu hình **đã chạy**, không phải tuyên bố mọi default trong `configs.yaml` đều tự bật constraint. Các launcher nghiên cứu có override explicit.

## 6. Gradient đi đâu?

Trong nhánh auxiliary của các lượt projection hiện tại:

$$
z_t=S_\psi(\mathrm{sg}(h_t)).
$$

* Persistence lấy hai endpoint từ **online $z$**. Cả hai endpoint, covariance, difference và relative ridge đều nằm trên đường gradient; không detach covariance hoặc một endpoint. Slow projection chỉ phục vụ target prediction, không phục vụ persistence.
* Persistence/isotropy cập nhật projection $\psi$. Chúng không cập nhật trực tiếp các predictor head $\omega$ và không truyền gradient về RSSM qua nhánh auxiliary khi `grad_to_backbone=false`.
* Chỉ 128 coordinate đầu nhận gradient trực tiếp từ hai regularizer; shared trunk có thể làm output block khác thay đổi sau optimizer step.
* `grad_to_backbone=false` **không đóng băng toàn backbone**. Reward/continuation/replay-value có đường gradient khác; actor/critic cũng cập nhật projection. Các loss này có thể tương tác với constraint.

Nếu dùng direct-$h$ ablation với `use_proj=false`, đối tượng thay bằng 128 coordinate đầu của $h$ và `grad_to_backbone=true` mới cho constraint định hình backbone. Những coordinate đó không tự nhiên tương ứng với object hoặc vùng ảnh.

## 7. Constraint không phải spatial pyramid

| Thành phần | Tín hiệu tối ưu | Điều chưa suy ra được |
|---|---|---|
| Multi-stride prediction | Tương lai của prefix, có action conditioning | Semantic bất biến |
| Persistence | Ít biến thiên theo thời gian sau chuẩn hóa covariance | Thông tin đó có ích cho control |
| Isotropy | Phân bố variance trên nhiều hướng | Các hướng khác nhau về semantic |
| Progressive reconstruction | Giữ đủ thông tin để tái tạo target | Trật tự temporal tự động đúng |
| Spatial pyramid | Target ảnh từ thô tới chi tiết | Thô về không gian đồng nghĩa chậm về thời gian |

Reconstruction target $h$, ảnh uniform và ảnh pyramid đều đã được thử với cùng package persistence + isotropy. Đổi độ phân giải reconstruction **không đổi công thức constraint**. Predictor, reconstruction và control heads có thể tạo áp lực giữ thông tin; điều đó chỉ giảm nguy cơ nghiệm vô dụng, không tạo một định lý chống collapse.

## 8. Option có trong code nhưng không bật ở các lượt hiện tại

* `persistence_min_rank > 0`: thay isotropy loss bằng một penalty dừng khi participation rank đạt ngưỡng. Giá trị hiện tại là 0 nên vẫn dùng isotropy đầy đủ.
* `persistence_decorrelation > 0`: thêm CKA giữa block 1 và 2, chặn gradient trực tiếp vào block 1. Hiện bằng 0; **không có loss CKA hoạt động** trong package này.
* `persistence_block2_rank_scale > 0`: thêm rank guard riêng cho block 2. Hiện bằng 0.
* `use_vicreg`: regularizer variance khác, hiện tắt. Không gọi isotropy hiện tại là một bộ VICReg đầy đủ.

## 9. Hai chi tiết cần chú ý khi thiết kế ablation

**Constraint đang nằm trong `if self.htp_use_pdyn`.** Tắt `use_pdyn` sẽ tắt cả prediction, persistence và isotropy. Vì thế ablation prediction on/off hiện không tự cô lập prediction. Muốn giữ constraint trong nhánh prediction-off phải tách đường thực thi trước khi chạy factorial.

**Mask của persistence và mask của prediction là hai việc khác nhau.** Persistence luôn loại cặp vượt reset. Trong các lượt spatial, `pdyn.mask_episode_boundaries=false`, nên prediction vẫn chứa một số cặp vượt reset. Không được suy từ công thức persistence rằng toàn bộ supervision đã episode-safe.

## 10. Phải đo gì để biết constraint học đúng kỳ vọng?

Kỳ vọng có điều kiện là: giảm temporal displacement **đồng thời** giữ thông tin dự đoán và task information, không suy giảm rank quá mức, không tăng redundancy vô ích và không làm giảm control.

Các phép đo cần được dùng cùng nhau:

1. Whitened displacement trên episode held-out, qua nhiều horizon; thấp hơn riêng lẻ chưa đủ.
2. Predictor thật so với copy/mean baseline và action control; tốt hơn baseline chưa đủ chứng minh semantics.
3. Rank/covariance spectrum, dead coordinates, similarity giữa block; không metric nào tự chứng minh chống collapse hoàn toàn.
4. Probe task-relevant information, ví dụ reward/terminal/return; so $h$ và $z$ để phát hiện thông tin mất qua projection.
5. Actor/critic reliance và control trên nhiều training seed. Phải kiểm soát số chiều khi can thiệp các block khác kích thước.

Audit 3526 trên 12 checkpoint seed 1 cho thấy predictor B1 thắng copy và nhạy với action, nhưng đó là hệ thống đầy đủ **đã có constraint**. Nó không cô lập đóng góp riêng của persistence hay isotropy. Các kết quả spatial cũng chưa chứng minh compact prefix giữ cùng semantic role theo thời gian. Các báo cáo lợi ích constraint trước đây cần được hiểu trong phạm vi từng cấu hình/ablation, không phải guarantee tổng quát.

Muốn xác nhận đóng góp riêng: so cùng architecture, data budget và seed, thay một regularizer mỗi lần; báo cả control, representation và uncertainty giữa training seed. Đặc biệt không dùng việc đổi reconstruction target để kết luận riêng về hiệu quả constraint.

## 11. Bản đồ công thức sang code

| Công thức / hành vi | File và symbol |
|---|---|
| Episode mask, $C_k$, $Q_k$, ridge, trace solve, all-lags | [`dreamerv3/htp.py`](../../dreamerv3/htp.py), `prefix_persistence` |
| Covariance isotropy | [`dreamerv3/htp.py`](../../dreamerv3/htp.py), `prefix_isotropy` |
| Participation rank và rank floor tùy chọn | [`dreamerv3/htp.py`](../../dreamerv3/htp.py), `prefix_participation_rank`, `prefix_rank_floor` |
| CKA tùy chọn | [`dreamerv3/htp.py`](../../dreamerv3/htp.py), `prefix_block_redundancy` |
| Prediction, action windows, isolation và mask riêng | [`dreamerv3/htp.py`](../../dreamerv3/htp.py), `MultiStridePDyn` |
| Hệ số, broadcasting và gate `use_pdyn` | [`dreamerv3/agent_htp.py`](../../dreamerv3/agent_htp.py), `Agent_HTP.loss` |
| $h\to z$, gradient routing và head consumers | [`dreamerv3/agent_htp.py`](../../dreamerv3/agent_htp.py), `feat2h`, `feat2tensor`, `loss` |
| Cấu hình và default | [`dreamerv3/configs.yaml`](../../dreamerv3/configs.yaml), `agent.htp` |

Các symbol constraint cốt lõi nêu trên đã có trong commit `ae752ba`. Saved config dùng để kiểm tra hệ số trong tài liệu: `production_runs/spatial_seed2_UOW5Hr6o/boxing/image_uniform/config.yaml` (artifact cục bộ, không giả định được lưu trên GitHub). Tài liệu này độc lập với các file audit/spatial chưa commit trong workspace; không cần các artifact đó để hiểu hoặc kiểm tra công thức constraint.
