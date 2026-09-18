# Run tracker đã điền

[Tải Excel](ablation_run_tracker_filled.xlsx) · [CSV để import vào Google Sheets](filled_run_tracker.csv) · [Chỉ 212 run đã hoàn tất](completed_runs.csv)

Nguồn template: Google Sheet `1-q0NLVsGY9qyVmhYLmWRzvc68DjxOqxvfp0l8AtCN1s`, tab **gid1403460040**. Không sửa trực tiếp Google Sheet; đây là bản xuất đã điền từ artifact thực.

## Vì sao không điền vào 42 dòng kế hoạch gốc?

42 dòng có sẵn đều là **Breakout, size25m, seeds0–2**, trong khi sweep hoàn tất là **Boxing / Up N Down / Frostbite / Road Runner, size12m, seed0**. Một số lệnh warmup của template còn chỉ định110000 thay vì budget100000 của sweep. Không có dòng nào khớp đủ arm/task/seed/size/warmup; không chuyển score giữa các setting.

- `Template original`: giữ57 dòng gốc (42 dòng kế hoạch và15 chỗ trống).
- `Template audit`: giữ42 kế hoạch, ghi `Not matched`, để trống score/HNS/thời gian. Trạng thái `Queued` gốc được ghi lại trong Notes; không khẳng định chúng đang nằm trong Slurm queue.
- `Completed runs`: điền đủ212 run thật với size12m và đúng task/seed.
- `Filled tracker`:42 kế hoạch +212 completed =254 dòng; dùng run_043…run_254 cho kết quả mới, tận dụng các ID trống gốc trước.
- `Run details`: tên arm nội bộ, comparator artifact link, ngưỡng bật/tắt auxiliary, reference scores, episode SD, timing và checkpoint hash.

Tên arm gốc được map về cách đặt tên Excel, ví dụ `full`→`htp_full`, `flat`→`htp_empty_projection`, `slim_variance`→`htp_slim_vicreg`. Các đối chứng bổ sung giữ tên sweep. Warmup full dùng `Arm=htp_full` cùng cột `Warmup` tương ứng. `sweep_arm` trong Run details phân biệt tuyệt đối từng cấu hình; ngưỡng early-stop ở `aux_stop_actions`.

## Định nghĩa các cột đã điền

- **Final score:** mean raw return của100 final evaluation episodes tại checkpoint100K executed agent actions; không phải peak score.
- **HNS:** `(Final score − random)/(human − random)`, dạng ratio; human=1, không clip.
- **Start / End / Hours:** bắt đầu/kết thúc/lượng giờ của invocation worker thành công. Bao gồm những phase thực sự chạy trong invocation đó; không tính queue và các attempt lỗi trước. Không phải training-only hoặc GPU-hours.
- **Train hours / Eval hours** trong Run details: thời gian phase training thành công / tổng thời gian mười phase checkpoint evaluation, tính từ timing logs. Hai game dùng chung GPU nên không cộng process-hours rồi gọi là allocated GPU-hours.
- **Timezone:** Asia/Jakarta (UTC+7). Timestamp hiển thị tới phút; Hours tính từ timestamp đầy đủ.
- **W&B URL:** trống vì sweep này dùng logging local, không có link W&B được ghi lại. Không chế URL.
- **Command:** lệnh train thực đã đổi executable thành `python` và output thành `./reruns/...`. Phải dùng code gốc commit `5ee4f27a0ba7fd7bdbdcbd4823fb2d539f8b4b47` với [source.patch](../source.patch), không mặc định lệnh chạy được trên main. Chọn logdir mới trước khi chạy. Lệnh kế hoạch chưa khớp vẫn là nguyên văn template, không coi là đã thực thi.

Không có score cho các setting chưa khớp; single-seed không được diễn giải thành five-seed result. Episode SD là sample SD (`ddof=1`) của episodes, không phải SD qua training seeds.

## Import / tạo lại Excel

Có thể upload `.xlsx` vào Google Drive để mở bằng Sheets, hoặc import `filled_run_tracker.csv` vào **tab mới** để giữ tab gốc. File Excel gồm sáu sheet, freeze header và filters; numeric scores là số, không phải chuỗi hay công thức.

```bash
python -m pip install 'openpyxl>=3.1,<4'
python render_workbook.py
```

Script chỉ đọc CSV cạnh nó, không đọc checkpoint/log bên ngoài. `validation.json` ghi số dòng và hash nguồn; `checksums.csv` ghi SHA256 artifact.
