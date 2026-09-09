# CoRe-WM: đánh giá 26 game × 5 seed

Policy: checkpoint 100k actions, 100 evaluation episodes/seed; AUC hình thang trên 10k–100k, chia 90k. HNS không clip; mean/median tính sau khi lấy trung bình 5 seed mỗi game.

Representation: 4 clip từ policy seed 0 mỗi game dùng chung cho 5 checkpoint; rollout bắt đầu ở posterior t=16, chạy 64 actions thật. Đây là diagnostic đầu episode. CKA dùng các dự đoán một bước độc lập từ từng posterior. Mọi prefix dùng chung một rollout; reconstruction không đưa lại vào dynamics. Block-only được tính bằng hiệu hai cumulative reconstruction, tương đương đúng D_l(z_l), tránh bias từ head bị mask.

Checkpoint có 5 block: dùng tất cả 5 block. Các mốc 1,2,4,8,16,32,64 là horizon. Base dynamics trong hình là RSSM của cùng checkpoint; chưa phải mô hình baseline huấn luyện độc lập.

| Game | Return mean ± SD (5 seed) | HNS % | AUC HNS % | CKA off-diagonal | MAE prefix 1 @64 | MAE full prefix @64 | MAE base @64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| alien | 518.24 ± 80.45 | 4.21 | 2.21 | 0.9850 | 0.0072 | 0.0075 | 0.0080 |
| amidar | 112.59 ± 34.29 | 6.23 | 3.88 | 0.9798 | 0.0051 | 0.0054 | 0.0050 |
| assault | 626.49 ± 30.43 | 77.77 | 59.27 | 0.9651 | 0.1148 | 0.1148 | 0.1149 |
| asterix | 544.90 ± 19.14 | 4.04 | 1.69 | 0.9820 | 0.0072 | 0.0076 | 0.0076 |
| bank_heist | 27.08 ± 5.67 | 1.74 | 0.52 | 0.9912 | 0.1112 | 0.1121 | 0.1073 |
| battle_zone | 11022.00 ± 4528.62 | 24.87 | 4.56 | 0.9940 | 0.0133 | 0.0134 | 0.0135 |
| boxing | 76.70 ± 5.76 | 638.33 | 368.08 | 0.9409 | 0.0142 | 0.0146 | 0.0148 |
| breakout | 6.61 ± 1.46 | 17.05 | 7.92 | 0.9706 | 0.0039 | 0.0039 | 0.0040 |
| chopper_command | 1141.60 ± 168.60 | 5.03 | 0.27 | 0.9914 | 0.0104 | 0.0106 | 0.0107 |
| crazy_climber | 58110.80 ± 6741.97 | 188.95 | 122.72 | 0.9974 | 0.0041 | 0.0042 | 0.0038 |
| demon_attack | 284.51 ± 113.13 | 7.28 | 5.57 | 0.9553 | 0.0986 | 0.0985 | 0.0986 |
| freeway | 0.00 ± 0.00 | 0.00 | 0.08 | 0.9370 | 0.0050 | 0.0050 | 0.0050 |
| frostbite | 1570.60 ± 1161.05 | 35.26 | 15.42 | 0.9666 | 0.0276 | 0.0278 | 0.0270 |
| gopher | 1405.84 ± 245.21 | 53.29 | 21.08 | 0.9930 | 0.0105 | 0.0102 | 0.0099 |
| hero | 7502.05 ± 73.15 | 21.73 | 11.00 | 0.9827 | 0.0044 | 0.0051 | 0.0042 |
| jamesbond | 351.00 ± 250.79 | 117.60 | 48.86 | 0.9697 | 0.0050 | 0.0048 | 0.0046 |
| kangaroo | 1306.80 ± 695.59 | 42.07 | 14.90 | 0.9779 | 0.0049 | 0.0050 | 0.0049 |
| krull | 7876.36 ± 269.71 | 588.14 | 450.82 | 0.9903 | 0.0070 | 0.0072 | 0.0069 |
| kung_fu_master | 18675.00 ± 6356.22 | 81.93 | 58.06 | 0.9896 | 0.0095 | 0.0096 | 0.0096 |
| ms_pacman | 896.28 ± 206.88 | 8.86 | 5.74 | 0.9849 | 0.0100 | 0.0106 | 0.0100 |
| pong | -6.41 ± 1.72 | 40.48 | 12.94 | 0.9669 | 0.0046 | 0.0047 | 0.0044 |
| private_eye | 440.26 ± 1494.93 | 0.60 | 0.68 | 0.9723 | 0.0095 | 0.0097 | 0.0089 |
| qbert | 809.70 ± 138.49 | 4.86 | 2.53 | 0.9809 | 0.0083 | 0.0084 | 0.0080 |
| road_runner | 12558.00 ± 4268.78 | 160.16 | 118.52 | 0.9501 | 0.0056 | 0.0060 | 0.0049 |
| seaquest | 515.52 ± 113.99 | 1.06 | 0.79 | 0.9812 | 0.0101 | 0.0101 | 0.0101 |
| up_n_down | 57996.54 ± 49975.54 | 514.91 | 364.67 | 0.9672 | 0.0386 | 0.0392 | 0.0401 |

Mean HNS: 101.79%; median HNS: 23.30%; games above human: 6/26.

## Figures

### alien

![alien one_step_blocks](alien/seed_0/one_step_blocks.png)
![alien cka](alien/seed_0/cka.png)
![alien prefix_imagination](alien/seed_0/prefix_imagination.png)
![alien mae](alien/seed_0/mae.png)

### amidar

![amidar one_step_blocks](amidar/seed_0/one_step_blocks.png)
![amidar cka](amidar/seed_0/cka.png)
![amidar prefix_imagination](amidar/seed_0/prefix_imagination.png)
![amidar mae](amidar/seed_0/mae.png)

### assault

![assault one_step_blocks](assault/seed_0/one_step_blocks.png)
![assault cka](assault/seed_0/cka.png)
![assault prefix_imagination](assault/seed_0/prefix_imagination.png)
![assault mae](assault/seed_0/mae.png)

### asterix

![asterix one_step_blocks](asterix/seed_0/one_step_blocks.png)
![asterix cka](asterix/seed_0/cka.png)
![asterix prefix_imagination](asterix/seed_0/prefix_imagination.png)
![asterix mae](asterix/seed_0/mae.png)

### bank_heist

![bank_heist one_step_blocks](bank_heist/seed_0/one_step_blocks.png)
![bank_heist cka](bank_heist/seed_0/cka.png)
![bank_heist prefix_imagination](bank_heist/seed_0/prefix_imagination.png)
![bank_heist mae](bank_heist/seed_0/mae.png)

### battle_zone

![battle_zone one_step_blocks](battle_zone/seed_0/one_step_blocks.png)
![battle_zone cka](battle_zone/seed_0/cka.png)
![battle_zone prefix_imagination](battle_zone/seed_0/prefix_imagination.png)
![battle_zone mae](battle_zone/seed_0/mae.png)

### boxing

![boxing one_step_blocks](boxing/seed_0/one_step_blocks.png)
![boxing cka](boxing/seed_0/cka.png)
![boxing prefix_imagination](boxing/seed_0/prefix_imagination.png)
![boxing mae](boxing/seed_0/mae.png)

### breakout

![breakout one_step_blocks](breakout/seed_0/one_step_blocks.png)
![breakout cka](breakout/seed_0/cka.png)
![breakout prefix_imagination](breakout/seed_0/prefix_imagination.png)
![breakout mae](breakout/seed_0/mae.png)

### chopper_command

![chopper_command one_step_blocks](chopper_command/seed_0/one_step_blocks.png)
![chopper_command cka](chopper_command/seed_0/cka.png)
![chopper_command prefix_imagination](chopper_command/seed_0/prefix_imagination.png)
![chopper_command mae](chopper_command/seed_0/mae.png)

### crazy_climber

![crazy_climber one_step_blocks](crazy_climber/seed_0/one_step_blocks.png)
![crazy_climber cka](crazy_climber/seed_0/cka.png)
![crazy_climber prefix_imagination](crazy_climber/seed_0/prefix_imagination.png)
![crazy_climber mae](crazy_climber/seed_0/mae.png)

### demon_attack

![demon_attack one_step_blocks](demon_attack/seed_0/one_step_blocks.png)
![demon_attack cka](demon_attack/seed_0/cka.png)
![demon_attack prefix_imagination](demon_attack/seed_0/prefix_imagination.png)
![demon_attack mae](demon_attack/seed_0/mae.png)

### freeway

![freeway one_step_blocks](freeway/seed_0/one_step_blocks.png)
![freeway cka](freeway/seed_0/cka.png)
![freeway prefix_imagination](freeway/seed_0/prefix_imagination.png)
![freeway mae](freeway/seed_0/mae.png)

### frostbite

![frostbite one_step_blocks](frostbite/seed_0/one_step_blocks.png)
![frostbite cka](frostbite/seed_0/cka.png)
![frostbite prefix_imagination](frostbite/seed_0/prefix_imagination.png)
![frostbite mae](frostbite/seed_0/mae.png)

### gopher

![gopher one_step_blocks](gopher/seed_0/one_step_blocks.png)
![gopher cka](gopher/seed_0/cka.png)
![gopher prefix_imagination](gopher/seed_0/prefix_imagination.png)
![gopher mae](gopher/seed_0/mae.png)

### hero

![hero one_step_blocks](hero/seed_0/one_step_blocks.png)
![hero cka](hero/seed_0/cka.png)
![hero prefix_imagination](hero/seed_0/prefix_imagination.png)
![hero mae](hero/seed_0/mae.png)

### jamesbond

![jamesbond one_step_blocks](jamesbond/seed_0/one_step_blocks.png)
![jamesbond cka](jamesbond/seed_0/cka.png)
![jamesbond prefix_imagination](jamesbond/seed_0/prefix_imagination.png)
![jamesbond mae](jamesbond/seed_0/mae.png)

### kangaroo

![kangaroo one_step_blocks](kangaroo/seed_0/one_step_blocks.png)
![kangaroo cka](kangaroo/seed_0/cka.png)
![kangaroo prefix_imagination](kangaroo/seed_0/prefix_imagination.png)
![kangaroo mae](kangaroo/seed_0/mae.png)

### krull

![krull one_step_blocks](krull/seed_0/one_step_blocks.png)
![krull cka](krull/seed_0/cka.png)
![krull prefix_imagination](krull/seed_0/prefix_imagination.png)
![krull mae](krull/seed_0/mae.png)

### kung_fu_master

![kung_fu_master one_step_blocks](kung_fu_master/seed_0/one_step_blocks.png)
![kung_fu_master cka](kung_fu_master/seed_0/cka.png)
![kung_fu_master prefix_imagination](kung_fu_master/seed_0/prefix_imagination.png)
![kung_fu_master mae](kung_fu_master/seed_0/mae.png)

### ms_pacman

![ms_pacman one_step_blocks](ms_pacman/seed_0/one_step_blocks.png)
![ms_pacman cka](ms_pacman/seed_0/cka.png)
![ms_pacman prefix_imagination](ms_pacman/seed_0/prefix_imagination.png)
![ms_pacman mae](ms_pacman/seed_0/mae.png)

### pong

![pong one_step_blocks](pong/seed_0/one_step_blocks.png)
![pong cka](pong/seed_0/cka.png)
![pong prefix_imagination](pong/seed_0/prefix_imagination.png)
![pong mae](pong/seed_0/mae.png)

### private_eye

![private_eye one_step_blocks](private_eye/seed_0/one_step_blocks.png)
![private_eye cka](private_eye/seed_0/cka.png)
![private_eye prefix_imagination](private_eye/seed_0/prefix_imagination.png)
![private_eye mae](private_eye/seed_0/mae.png)

### qbert

![qbert one_step_blocks](qbert/seed_0/one_step_blocks.png)
![qbert cka](qbert/seed_0/cka.png)
![qbert prefix_imagination](qbert/seed_0/prefix_imagination.png)
![qbert mae](qbert/seed_0/mae.png)

### road_runner

![road_runner one_step_blocks](road_runner/seed_0/one_step_blocks.png)
![road_runner cka](road_runner/seed_0/cka.png)
![road_runner prefix_imagination](road_runner/seed_0/prefix_imagination.png)
![road_runner mae](road_runner/seed_0/mae.png)

### seaquest

![seaquest one_step_blocks](seaquest/seed_0/one_step_blocks.png)
![seaquest cka](seaquest/seed_0/cka.png)
![seaquest prefix_imagination](seaquest/seed_0/prefix_imagination.png)
![seaquest mae](seaquest/seed_0/mae.png)

### up_n_down

![up_n_down one_step_blocks](up_n_down/seed_0/one_step_blocks.png)
![up_n_down cka](up_n_down/seed_0/cka.png)
![up_n_down prefix_imagination](up_n_down/seed_0/prefix_imagination.png)
![up_n_down mae](up_n_down/seed_0/mae.png)

