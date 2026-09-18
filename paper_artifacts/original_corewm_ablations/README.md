# Original CoRe-WM — ablation results

[Run tracker Excel/CSV đã điền từ tab gid1403460040](run_tracker/): giữ template gốc, phân biệt42 kế hoạch chưa khớp và212 run đã hoàn tất.

53 cấu hình ×4 game ×seed0 = **212 run hoàn tất**; mỗi run100 final evaluation episodes tại100K actions. Không phải Harmony hoặc constraint suite. Không có nhận xét performance trong bộ tài liệu này.

Mỗi folder độc lập chứa CSV của ablation và comparator, episode ledger, config, provenance, README, figure và script vẽ chỉ đọc CSV tại chỗ. Role `ablation` là cấu hình của folder; role `comparator` là đối chứng đã đăng ký, không phải baseline được chọn sau khi xem score. Việc lặp lại dữ liệu đối chứng giúp copy một folder vẫn vẽ được.

Protocol: size12m, seed0, action repeat4, replay ratio256,100K executed actions; eval10 episodes ở mỗi checkpoint10K–90K và100 ở100K. Không smoothing, không chọn best checkpoint/seed, không dựng confidence interval từ một training seed. HNS dùng human/random decimal references trong CSV, SD là giữa episodes. Các smoke test và lỗi hạ tầng không được nhập làm performance.

Nguồn: original commit `5ee4f27a0ba7fd7bdbdcbd4823fb2d539f8b4b47` cùng `source.patch`; source experiment `production_runs/original_ablation_20260918`. Bản full không có các constraint bổ sung; chỉ variant được ghi rõ mới bật variance/mask/normalization. Environment seed=True trong sweep này; không coi scores của bộ5seed cũ là paired controls.

`requested_ablation.csv` là input từ tab Google Sheet gid1530063492; `ablation_design.csv` là registry53 cấu hình; `index.csv` ánh xạ tên và comparator. Nguồn Google Sheet: https://docs.google.com/spreadsheets/d/1-q0NLVsGY9qyVmhYLmWRzvc68DjxOqxvfp0l8AtCN1s/edit?gid=1530063492

## Chạy lại figure

```bash
python -m pip install -r requirements.txt
python full/plot.py
```

Không cần training data ngoài folder con. `checksums.csv` chứa SHA256 của artifact; `validation.json` ghi các kiểm tra episode/checkpoint/aggregate và việc chạy lại toàn bộ53 figure từ các CSV đã copy sang thư mục tạm độc lập.

| Ablation | Nhóm | Đối chứng |
|---|---|---|
| [baseline](baseline/) | backbone | `full` |
| [full](full/) | factorial | `flat` |
| [flat](flat/) | factorial | `baseline` |
| [rec_only](rec_only/) | factorial | `flat` |
| [pdyn_only](pdyn_only/) | factorial | `flat` |
| [flat_pruned](flat_pruned/) | pruning | `flat` |
| [rec_pruned](rec_pruned/) | pruning | `rec_only` |
| [pdyn_pruned](pdyn_pruned/) | pruning | `pdyn_only` |
| [joint_full](joint_full/) | joint_factorial | `full` |
| [joint_flat](joint_flat/) | joint_factorial | `flat` |
| [joint_rec_only](joint_rec_only/) | joint_factorial | `rec_only` |
| [joint_pdyn_only](joint_pdyn_only/) | joint_factorial | `pdyn_only` |
| [separable_full](separable_full/) | projection | `full` |
| [separable_flat](separable_flat/) | projection | `flat` |
| [reverse](reverse/) | timescale | `full` |
| [all1](all1/) | timescale | `full` |
| [all16](all16/) | timescale | `full` |
| [permuted](permuted/) | timescale | `full` |
| [full_validnorm](full_validnorm/) | normalization | `full` |
| [reverse_validnorm](reverse_validnorm/) | normalization | `reverse` |
| [all1_validnorm](all1_validnorm/) | normalization | `all1` |
| [all16_validnorm](all16_validnorm/) | normalization | `all16` |
| [reset_mask](reset_mask/) | boundaries | `full` |
| [reset_mask_validnorm](reset_mask_validnorm/) | boundaries | `full_validnorm` |
| [recon_no_sg](recon_no_sg/) | reconstruction | `full` |
| [recon_final_only](recon_final_only/) | reconstruction | `recon_no_sg` |
| [pdyn_no_sg](pdyn_no_sg/) | prediction | `full` |
| [online_target](online_target/) | target | `full` |
| [no_actions](no_actions/) | conditioning | `full` |
| [ema_slow](ema_slow/) | target | `full` |
| [ema_fast](ema_fast/) | target | `full` |
| [rec_weak](rec_weak/) | loss_scale | `full` |
| [rec_strong](rec_strong/) | loss_scale | `full` |
| [pdyn_weak](pdyn_weak/) | loss_scale | `full` |
| [pdyn_strong](pdyn_strong/) | loss_scale | `full` |
| [wide_trunk](wide_trunk/) | capacity | `full` |
| [compact_code](compact_code/) | capacity | `full` |
| [level1_short](level1_short/) | levels | `all1` |
| [level1_long](level1_long/) | levels | `all16` |
| [level3](level3/) | levels | `full` |
| [slim](slim/) | slim | `slim_zero` |
| [slim_zero](slim_zero/) | slim | `baseline` |
| [slim_posthoc](slim_posthoc/) | slim | `slim_zero` |
| [slim_variance](slim_variance/) | slim | `slim` |
| [slim_variance_only](slim_variance_only/) | slim | `slim_zero` |
| [warmup_40000](warmup_40000/) | timing | `full` |
| [early_60000](early_60000/) | timing | `warmup_40000` |
| [warmup_60000](warmup_60000/) | timing | `full` |
| [early_40000](early_40000/) | timing | `warmup_60000` |
| [warmup_80000](warmup_80000/) | timing | `full` |
| [early_20000](early_20000/) | timing | `warmup_80000` |
| [raw_readout_full](raw_readout_full/) | readout | `full` |
| [raw_readout_flat](raw_readout_flat/) | readout | `flat` |
