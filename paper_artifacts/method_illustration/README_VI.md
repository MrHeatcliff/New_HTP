# Các thành phần của CoRe-WM · Reborn

![Tổng quan phương pháp](reborn_two_stage_v2.png)

**Ý tưởng:** prefix nhỏ phải giữ thông tin hữu ích cho dự đoán dài hạn; block sau bổ sung chi tiết, đồng thời representation phải trả chi phí thông tin.

Hai panel là **hai objective học đồng thời**, không phải hai giai đoạn pretrain nối tiếp. Hình mô tả thiết kế Reborn; **không thể hiện phần cân bằng loss Harmony**. Các ảnh game là minh họa, không phải kết quả thực nghiệm.

## 1. RSSM và stochastic projection

![Tạo representation](block_encoder.svg)

RSSM tổng hợp lịch sử quan sát–hành động thành trạng thái $h_t$. Projection affine tạo mean, sau đó thêm Gaussian noise:

$$
\mu_t = Wh_t+b, \qquad z_t=\mu_t+\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
$$

$z_t$ được chia thành năm block. **Prefix level $\ell$ chứa toàn bộ block từ 1 đến $\ell$**, không chỉ block mới. Noise được dùng thật trên đường projected representation, kể cả khi đánh giá; không chỉ thêm vào auxiliary loss.

**Ý nghĩa các phép:**

- $Wh_t+b$: trộn và chọn các hướng thông tin từ trạng thái RSSM sang hệ tọa độ mới. Đây là phép affine học được, không phải phép chia ảnh thành vùng; một coordinate không mặc nhiên tương ứng với một vật thể.
- $+\epsilon_t$: tạo độ bất định cố định trên kênh truyền thông tin. Tín hiệu quá nhỏ so với noise sẽ khó được decoder/actor sử dụng. Vì thế model không thể chỉ thu nhỏ code rồi khuếch đại decoder để né chi phí mà vẫn giữ nguyên độ tin cậy.
- $z_t^{(1:\ell)}$: lấy phần đầu của **cùng một code**, không cộng hay lấy trung bình các block. Prefix dài có quyền truy cập thêm coordinate; thêm chiều chưa chắc thêm thông tin hữu ích.

Ví dụ trực giác: vị trí xe cần được mã hóa đủ rõ để vẫn đọc được khi có noise; chi tiết không giúp các mục tiêu học có thể không đáng giữ.

## 2. Cumulative rate — chi phí thông tin

![Chi phí tích lũy](block_rate.svg)

Với Gaussian variance cố định, KL của mỗi block bằng:

$$
\mathcal R_j=\tfrac12\|\mu_t^{(j)}\|_2^2,
\qquad
\mathcal L_{\mathrm{rate}}
=\frac{\beta}{L}\sum_{\ell=1}^{L}\sum_{j=1}^{\ell}\mathcal R_j.
$$

Block đầu xuất hiện trong nhiều prefix nên **cùng một đơn vị rate bị tính phí nhiều hơn**. Thông tin chỉ cần cho chi tiết phía sau có động lực được giữ muộn hơn.

Các cột giảm dần trong hình là **trọng số chi phí**, không phải lượng thông tin đo được. Rate là upper bound thông tin của code ngẫu nhiên; không bảo đảm chống collapse khi $\beta$ quá lớn.

**Ý nghĩa các phép:** $\|\mu^{(j)}\|_2^2$ là tổng bình phương các coordinate, không phải variance giữa các state. Công thức $\tfrac12\|\mu^{(j)}\|^2$ chính là KL từ $\mathcal N(\mu^{(j)},I)$ tới prior $\mathcal N(0,I)$: đo mức code rời khỏi một nguồn noise không mang tín hiệu về state.

Tổng trong cộng rate của mọi block thuộc một prefix; tổng ngoài tính phí cho mọi prefix. Với năm level:

$$
\mathcal L_{\mathrm{rate}}
=\beta(\mathcal R_1+0.8\mathcal R_2+0.6\mathcal R_3+0.4\mathcal R_4+0.2\mathcal R_5).
$$

Cùng một đơn vị rate ở block 1 đắt gấp 5 lần block 5, **không có nghĩa block 1 phải chứa ít thông tin gấp 5 lần**. Tăng $\beta$ làm việc giữ tín hiệu đắt hơn; giảm $\beta$ cho phép giữ nhiều tín hiệu hơn nhưng làm yếu áp lực tiết kiệm. Rate lớn cũng có thể do mean lệch cố định, nên không thể dùng nó một mình để kết luận representation hữu ích.

## 3. Spatial refinement — tái tạo coarse-to-fine

![Tái tạo theo dải chi tiết](block_spatial.svg)

Observation được phân rã bằng **Haar cố định** thành phần coarse và các dải chi tiết. Mỗi decoder dùng cả prefix để dự đoán đúng dải của mình:

$$
\widehat b_t^{(\ell)}=G_\ell(z_t^{(1:\ell)}),
\qquad
\widehat o_t^{(1:\ell)}
=\sum_{j=1}^{\ell}B_j\widehat b_t^{(j)}.
$$

Block mới được dùng lại context từ block trước. Nhờ các không gian Haar trực giao, **thêm dải chi tiết không ghi đè phần coarse đã tái tạo trong cùng forward pass**.

**Ý nghĩa các phép:**

- Haar lấy các tổ hợp **trung bình và chênh lệch đã chuẩn hóa** của pixel. Phần trung bình giữ bố cục coarse; các chênh lệch giữ biến thiên theo hướng ngang, dọc và chéo ở từng scale. Đây là phân tách theo tần số/scale, không phải tự động tách object hay semantics.
- $b^{(\ell)}=B_\ell^\top o$: đưa ảnh vào hệ số của dải Haar level $\ell$. $B_\ell$ cố định, không học theo latent.
- $G_\ell(z^{(1:\ell)})$: dự đoán hệ số dải đó bằng cả context trước và block mới. Ví dụ, biết vị trí xe từ context giúp head bổ sung biên xe mà không phải tự suy lại toàn bộ vị trí.
- $B_j\widehat b^{(j)}$: đưa hệ số dự đoán trở lại không gian ảnh; $\sum_j$ chồng các **dải ảnh cùng kích thước** lên nhau, không cộng trực tiếp các ma trận hệ số khác kích thước.

Distortion dùng tổng bình phương sai số hệ số chia cho $N$, là tổng số phần tử ảnh gồm cả channel:

$$
\mathcal D_{\mathrm{spatial}}
=\frac1N\sum_{\ell=1}^{L}\|b^{(\ell)}-\widehat b^{(\ell)}\|_2^2.
$$

Do Haar trực chuẩn, tổng này bằng pixel MSE của full reconstruction trước clipping. Normalization chung giữ đúng ý nghĩa sai số ảnh; không lấy mean riêng từng dải rồi vô tình cho dải ít hệ số trọng số quá lớn. Một dải dự đoán sai vẫn có thể làm ảnh cuối kém hơn dù không ghi đè phần coarse.

Các mức ảnh tích lũy là $4,8,16,32,64$. Kích thước dải detail trong hình là lưới hệ số Haar; không phải kích thước ảnh tái tạo tích lũy. Ảnh coarse-to-fine đẹp chưa chứng minh block mới mang thông tin hữu ích: vẫn cần probe cùng target khi có/không có block mới.

## 4. Multi-timescale prediction — dự đoán kết quả tích lũy

![Dự đoán nhiều thang thời gian](block_outcome.svg)

Mỗi predictor nhận **prefix và action hiện tại**, xuất hai loại dự đoán:

- $\Psi_\ell$: feature quan sát tương lai được cộng có trọng số, ở độ phân giải của level đó.
- $Q_\ell$: reward tương lai tích lũy theo action, dưới reference policy.

$$
\gamma_\ell=1-1/\Delta_\ell,
\qquad
\Delta=(16,8,4,2,1).
$$

Prefix nhỏ được giao thang thời gian dài; full prefix học next-observation và immediate reward khi $\gamma=0$. Các đường giảm dần là **trọng số discount**, không phải learning curve.

**Ý nghĩa của phép tích lũy**, với $\pi$ là policy dùng cho các action tiếp theo:

$$
\Psi_\ell^\pi(h_t,a_t)
=\mathbb E_\pi\!\left[(1-\gamma_\ell)\sum_{k=1}^{\infty}
\gamma_\ell^{k-1}b_{t+k}^{(1:\ell)}\mid h_t,a_t\right],
$$

$$
Q_\ell^\pi(h_t,a_t)
=\mathbb E_\pi\!\left[\sum_{k=1}^{\infty}
\gamma_\ell^{k-1}r_{t+k}\mid h_t,a_t\right].
$$

- $\gamma^{k-1}$: giảm trọng số của tín hiệu xa. $\gamma$ lớn làm tín hiệu xa còn ảnh hưởng lâu hơn; không có điểm cắt cứng ở $\Delta$.
- $(1-\gamma)$ ở $\Psi$: chuẩn hóa các trọng số thành tổng 1 trên chuỗi vô hạn trước xét terminal. Nhờ đó một feature hằng được giữ nguyên scale thay vì phình lên khi horizon tăng. Sau terminal, contribution bằng 0 nên tổng trọng số hữu hiệu có thể nhỏ hơn 1.
- $Q$ không nhân $(1-\gamma)$ vì mục tiêu là **tổng reward chiết khấu**, không phải reward trung bình; scale của Q vì thế có thể khác giữa các horizon.
- $\mathbb E_\pi$: dự đoán trung bình qua các tương lai có thể xảy ra khi tiếp tục theo policy, không đoán chắc một trajectory duy nhất. Điều kiện $a_t$ giúp phân biệt hệ quả của các lựa chọn hành động.

Với trọng số hình học đã chuẩn hóa, khoảng cách bước trung bình là $1/(1-\gamma)=\Delta$. Ví dụ $\Delta=16$ cho $\gamma=15/16$: feature ở bước 1 có trọng số $1/16$, bước 2 là $(1/16)(15/16)$, rồi giảm dần. Việc lấy tổng làm yếu dao động nhanh trong **target**, nhưng encoder vẫn có thể cần giữ velocity để dự đoán target đó.

Đây **không phải dự đoán một frame ở endpoint**, cũng không ép latent phải thay đổi chậm. Những frame tương lai trong hình chỉ minh họa nguồn của tổng tích lũy; predictor không xuất cả chuỗi frame. Ký hiệu reward bên cạnh $Q$ là minh họa reward grounding, không phải head reward độc lập thứ ba.

## 5. One-step TD targets — tín hiệu học từ transition thật

![Tạo target TD](block_td.svg)

Dùng next-observation, reward thật và bootstrap từ target networks:

$$
y^\Psi_\ell=(1-\gamma_\ell)b_{t+1}^{(1:\ell)}
+\gamma_\ell c_{t+1}\bar\Psi_\ell(\bar z_{t+1},a'),
$$

$$
y^Q_\ell=r_{t+1}
+\gamma_\ell c_{t+1}\bar Q_\ell(\bar z_{t+1},a').
$$

$a'$ lấy từ actor hiện tại; predictor dùng prefix tương ứng của $\bar z$. Target projection và predictor cập nhật bằng EMA; **toàn bộ target được detach**. EMA cập nhật tham số mạng, không làm trung bình ảnh hoặc reward.

Loại transition vượt reset; terminal thật có continuation bằng 0. Observation branch dùng squared error; Q branch dùng twohot loss. TD loss thấp chưa đủ: cần đối chiếu với rollout thật dưới policy cố định.

**Ý nghĩa các phép:**

- **Bootstrap:** cộng phần đã quan sát thật ở bước kế tiếp với phần tương lai còn lại do target network ước lượng. Cách đệ quy này học mục tiêu dài hạn từ transition một bước, nhưng vẫn có sai số bootstrap.
- $c_{t+1}$: bật/tắt phần tương lai sau transition. Khi terminal thật, $c=0$ bỏ bootstrap nhưng **vẫn giữ reward và observation của transition cuối**. Reset boundary phải mask riêng; không nối episode mới vào tương lai episode cũ, và không tự coi time limit là terminal thật.
- **EMA:** với quy ước $\tau$ là tốc độ cập nhật, $\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta$. $\tau$ nhỏ làm target thay đổi chậm hơn nhưng trễ hơn so với mạng online. Dấu gạch trên biểu thị mạng target, không phải temporal average của latent.
- **Stop-gradient/detach:** coi target là hằng trong lần backward này. Model học sửa prediction online, không sửa luôn target để làm loss giảm dễ dàng; target vẫn được cập nhật qua EMA ở các bước sau.
- **Squared error:** phạt độ lệch giữa successor features dự đoán và target, chia chung cho số phần tử ảnh $N$.
- **Twohot:** biểu diễn target Q bằng trọng số nội suy trên hai bin kề nhau, rồi dùng cross-entropy để học phân phối bin. Cách này tránh trực tiếp dùng squared error trên các giá trị reward có scale rất khác nhau; giá trị đọc ra theo cơ chế phân phối của Dreamer. Nó không có nghĩa reward chỉ được phép nhận hai giá trị.

Loss transition được lấy trung bình trên **số transition hợp lệ**, để batch có nhiều reset không tự làm loss nhỏ đi chỉ vì có ít mẫu học hơn.

## 6. Dreamer control — học actor và critic

![Đường actor–critic](block_control.svg)

RSSM vẫn là dynamics dùng cho imagination. Actor và critic nhận **cùng sampled representation của mỗi imagined state**, nhưng là hai nhánh song song:

- **Actor:** chọn action.
- **Critic:** ước lượng giá trị để hỗ trợ học policy.

Không đưa ảnh tái tạo hoặc output successor vào đường chọn action. Các head auxiliary định hình representation trong training; không phải chạy chúng để chọn từng action lúc evaluation. Backbone và observation decoder gốc của Dreamer được giữ lại.

**Ý nghĩa của imagination và reuse sample:** RSSM dự đoán trạng thái tiếp theo từ trạng thái hiện tại và action, cho phép actor–critic học từ trajectory latent mà không cần tương tác môi trường thật ở từng bước tưởng tượng. Mỗi imagined state lấy một mẫu $z$ rồi dùng lại mẫu đó khi chọn action và tính policy log-probability; nếu lấy noise khác, ta đang chấm action dưới một input khác với input đã sinh ra nó.

Auxiliary $Q_\ell$ không thay critic Dreamer: nó giám sát representation theo các discount riêng. Critic phục vụ policy objective gốc, với discount/control setup gốc. Vì vậy auxiliary Q tốt hơn chưa tự động bảo đảm policy tốt hơn.

## Ghép các thành phần

$$
\mathcal L
=\mathcal L_{\mathrm{WM}}
+\lambda_1\mathcal L_{\mathrm{spatial}}
+\lambda_2\mathcal L_{\mathrm{outcome}}.
$$

Spatial loss gồm distortion Haar và cumulative rate; outcome loss gồm successor-feature và action-value prediction. Công thức là ký hiệu gộp cho hai nhánh auxiliary, không thay thế objective actor–critic gốc.

**Ý nghĩa các hệ số:** $\lambda_1,\lambda_2$ điều chỉnh mức ảnh hưởng của hai auxiliary objective lên các tham số nhận gradient; $\beta$ cân bằng giữ thông tin với tiết kiệm rate bên trong spatial objective; $\lambda_Q$ cân bằng reward/action-value với successor observation bên trong outcome objective. Đây là các vai trò khác nhau, không nên coi giảm $\beta$ tương đương giảm toàn bộ reconstruction loss.

Phép cộng loss kết hợp các hướng gradient, không có nghĩa mọi nhánh hỗ trợ nhau. Loss số học lớn hơn cũng chưa chắc tạo gradient lớn hơn trên representation. Khi phân tích hyperparameter hoặc Harmony, cần quan sát cả gradient theo nhánh, rate, chất lượng prediction và control score; bản hình này không mô tả cơ chế Harmony.

**Điều cần kiểm chứng:** block mới giảm lỗi dự đoán cùng target, prefix nhỏ dự đoán tốt kết quả dài hạn, và control tốt hơn ở cùng budget. Thiết kế không tự bảo đảm ba điều này.

---

Hình tổng quan được tạo bằng image generation; các sơ đồ từng khối là SVG để dễ đọc trên GitHub. [Prompt và ghi chú hình tổng quan](reborn_two_stage_v1_prompt.md).
