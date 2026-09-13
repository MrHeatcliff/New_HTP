# Tái chạy Reborn + Harmony (bản job 4080)

Đây là gói source **đúng bản Harmony đang chạy 26 game × training seeds 1–4**,
để bổ sung seed 0 đã report. Không phải source nghiên cứu mới nhất trong workspace.
Không chứa checkpoint, replay, ROM Atari, token hoặc kết quả train mới.

## Vì sao có frozen_source.tar.gz?

Model/config của job 4080 được lấy từ source seed 0 của job 3997, không lấy từ
working tree có nhiều thay đổi chưa commit. Archive đóng gói nguyên source của job
4080 (bỏ bytecode/cache); launcher kiểm tra SHA256 trước khi giải nén.
Vì vậy clone repo mới không cần có thư mục `production_runs` của tác giả.
Source trong archive là Python/YAML thông thường, có thể đọc/chỉnh sửa sau giải nén.
Không thay đổi source của job đang chạy khi sử dụng gói này.

## Môi trường

- Linux, Python **3.12**, Slurm, `taskset`, `nvidia-smi`, storage dùng chung giữa login/compute.
- 2 NVIDIA H200 cùng node; 32 CPU, 192 GiB RAM, walltime 48 giờ.
- 4 processes đồng thời, 2 game/GPU, pin riêng 8 CPU/game. Runner kiểm tra đúng H200.
- CUDA driver tương thích JAX CUDA12; kiểm tra thiết bị **trong allocation**, không trên login.
- Dự trù khoảng **1 TB storage trống** cho 104 runs (checkpoint/replay/log); thực tế tùy dữ liệu.

```bash
git clone https://github.com/MrHeatcliff/New_HTP.git
cd New_HTP
git switch reborn
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduction/harmony/environment.freeze.txt
```

File freeze ghi package versions của môi trường gốc tại lúc đóng gói; đã bỏ entry
editable của chính repo để không kéo một revision khác đè lên frozen source.
Đây là dependency snapshot, không phải container bảo đảm chạy trên mọi driver.
Model chạy trực tiếp từ extracted source, không cần `pip install -e .`.
ROM Atari cần được cài hợp lệ và ALE nhìn thấy. Nếu đồng ý giấy phép ROM,
có thể dùng `AutoROM --accept-license` trong virtualenv; bước này tải ROM riêng,
repo không phân phối ROM. Tuân thủ quy định cluster khi cài/build dependency.

## Chuẩn bị trước, không submit

```bash
.venv/bin/python reproduction/harmony/launch.py \
  --run-root /shared/your_user/harmony_preview \
  --python /absolute/path/New_HTP/.venv/bin/python
```

Lệnh này chỉ giải nén/kiểm tra hash và lưu `submission.json`; **không train, không
submit**. `run-root` phải mới và trên shared storage. Có thể đọc các file trong
`harmony_preview/source` để review toàn bộ model, config và scheduler.

## Submit một job cho 104 runs

```bash
.venv/bin/python reproduction/harmony/launch.py \
  --run-root /shared/your_user/harmony_seeds1to4 \
  --python /absolute/path/New_HTP/.venv/bin/python \
  --partition gpu_general --qos gpu_general_qos \
  --gres gpu:nvidia_h200:2 --submit
```

Thay đường dẫn, partition/QoS/GRES theo cluster. Dùng `--qos ''` nếu không có QoS,
`--account ACCOUNT` nếu cần, `--dependency afterok:JOBID` để đợi một job được chỉ định.
Không chạy lại launcher để xem trạng thái: mỗi lần `--submit` là một job mới.
Launcher không cancel hay sửa job khác. Không chạy runner trực tiếp trên login.
Historical scripts trong archive có đường dẫn của tác giả; **dùng launch.py ở trên**.

## Method và protocol cố định

| Thành phần | Cấu hình |
|---|---|
| Backbone | `atari100k size25m`; RSSM/imagination/observation decoder gốc |
| Representation | affine projection, stochastic code với unit Gaussian noise |
| Prefix dimensions | 128, 256, 512, 1024, 2048 |
| Spatial objective | Haar detail bands, prefix context |
| Rate | cumulative rate; beta = 1e-5 |
| Temporal objective | successor observation + action-value; horizons 16,8,4,2,1 |
| Q weight / EMA | 0.1 / 0.02 |
| Harmony | `--agent.reborn.harmony True` (config mặc định false được override bởi command) |
| Auxiliary backbone gradients | `grad_to_backbone: false` |
| Training | seeds 1,2,3,4; 26 Atari100K games; 100K exact agent actions, repeat 4 |
| Checkpoints | 10K,20K,…,100K |
| Isolated eval | 10 episodes tại 10K–90K, 100 episodes tại 100K; eval RNG seed 0 |

Full mathematical context: [technical report](../../paper_artifacts/harmony_technical_report/TECHNICAL_REPORT_VI.md).
Seed-0 results: [single-seed report](../../paper_artifacts/harmony_single_seed_report/README.md).
Không diễn giải 100 eval episodes thành 100 training seeds. Không chọn best checkpoint.
Seed-0 eval RNG được giữ nguyên cho mọi policy; training seed thực tế là 1–4 và
được ghi riêng trong evaluation summary. Hạn chế terminal/time-limit của bản frozen
được giữ nguyên, không âm thầm sửa giữa các seeds.

## File nào làm gì? (bên trong extracted source)

- `dreamerv3/reborn.py`: representation, Haar reconstruction, rate, successor/Q và Harmony.
- `dreamerv3/agent_reborn.py`: tích hợp model, actor/critic và sampling.
- `dreamerv3/configs.yaml`: hyperparameters; `dreamerv3/main_htp.py`: entrypoint.
- `corewm_eval/reborn_experiment.py`: command cấu hình Harmony.
- `corewm_eval/harmony_multiseed.py`: đổi training seed, queue 104 runs, train/eval.
- `corewm_eval/h200_suite.py`: 4 CPU-pinned slots trên 2 GPUs và atomic JSON.
- `corewm_eval/reborn_checkpoint_eval.py`: checkpoint digest dùng kiểm tra bất biến.
- `tests/test_reborn.py`, `tests/test_h200_suite.py`: chạy trước training trong allocation.

## Log, trạng thái và failure handling

```text
RUNROOT/
  source/                         frozen source
  source_sha256.json              hashes từng file
  submission.json                 lệnh Slurm, job ID
  manifest.json                   slots, seeds, games, protocol
  status.json                     TESTING / RUNNING / COMPLETE / FAILED
  tests.log                       preflight tests
  slurm-JOBID.out, slurm-JOBID.err
  seed_1/alien/
    train.log, train_request.json, train_timing.json
    stage.json                    trạng thái lượt này
    full/                         training logs + checkpoints
    evaluation/summary.json        returns/mean/hash/training_seed/evaluation_seed
    evaluation/progress.json       checkpoint đang eval
    evaluation/100000/scores.jsonl
  seed_2/... seed_3/... seed_4/...
```

Kiểm tra `squeue -j JOBID`, `tail` Slurm log, các `stage.json` và evaluation summaries.
Root `status.json` không có live aggregate count; đọc trạng thái từng run khi cần.
Frozen legacy logger có thể ghi nhãn `Backbone/official` và Git HEAD của repo cha:
**không dùng các nhãn này để xác định method**; dùng command/config và source SHA256.

Mỗi evaluation kiểm tra đúng episode count, return hữu hạn và checkpoint hash không
đổi. Một lượt lỗi được ghi FAILED, các lượt còn lại tiếp tục. Scheduler không tự
resume khi hết 48 giờ, không tự submit thêm job và không tự skip runs đã hoàn tất.
Nếu timeout, `status.json` có thể còn RUNNING: đối chiếu Slurm trước khi kết luận.
Cần kiểm tra/sắp lịch riêng phần còn thiếu, không chạy lại mù quáng cả suite.

Tham chiếu thời gian: 36–46 giờ trên setup gốc; game/seed có episode dài có thể vượt
ước lượng. Gói này không cam kết bitwise reproducibility qua hardware/driver khác.
Runner cố định seeds 1–4; đây là đợt replication bổ sung seed 0, không phải API tùy
ý cho số seeds/game. Nếu sửa scheduler/protocol, ghi lại cấu hình mới và không gộp
nhầm vào cùng thí nghiệm.

## Kiểm chứng gói phát hành

Ngày 13/09/2026, CPU Slurm job 4131 chạy `test_launch.py`: PASS chuẩn bị source,
kiểm tra manifest hashes, import frozen scheduler, cấu hình Harmony/seeds/budget,
4 slots và từ chối overwrite. Test không submit GPU job, không train lại method.
Frozen model tests của job training 4080 trước đó: 11 passed.
Chưa xác nhận end-to-end trên một cluster độc lập; package versions/driver/ROM
vẫn cần được người chạy kiểm tra trong allocation của họ.
