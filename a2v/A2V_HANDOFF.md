# A2V 项目交接文档（新会话入口）

> 新开 Claude 会话时**先读本文**。它汇总了"把 Wan 基模改造成 action-conditioned (A2V) 世界模型"
> 这件事的目标、已核实事实、已产出文档、当前代码状态、以及待办与阻塞点。
> 详细内容散在 `.cache/analysis/` 的各专题文档里，本文给出索引与摘要。

最后更新：2026-06-21。

> ## ⭐ 接手清单（新会话先读这一段）
> **Multi-episode I2V-14B 已训练/PASS（2026-06-21，8 卡 DDP）**：train=ep0-39，held-out=ep40-49，105 帧，240x320。
> final ckpt（全参 vace）：`models/train/a2v_robotwin_train40_vace_i2v/step-800.safetensors`。
> **held-out：REAL MAE-GT=4.25、NONE=30.09、real<none=10/10、motion ratio=1.06、real-vs-none=30.21，PASS**（比 1.3B 更强、与 train 几乎无 gap）。
> 配方 `NPROC=8 FRAMES=105 LR=1e-5 REPEAT=4 EPOCHS=40 bash a2v/run_overfit.sh wan2.1-i2v-14b-480p`（800 步~2.2h）。评测：`.cache/a2v_robotwin/eval_heldout_i2v/`。详见 `A2V_HISTORY.md §16`。
>
> **Multi-episode VACE-1.3B PASS（2026-06-18）**：同数据 LoRA `models/train/a2v_robotwin_train40_vace1p3b_lora/step-4800.safetensors`；
> held-out REAL=6.25/NONE=26.83/10-10/PASS。详见 `A2V_HISTORY.md §15`。
>
> **工具**：`a2v.eval_multiep --base_spec <s> --lora <ckpt> --dataset <heldout> --controls real,none` 多行泛化评测（模型只加载一次、per-row 容错、聚合判定 + metrics.json）。`run_overfit.sh` 支持 `NPROC=`（多卡 DDP）+ `REPEAT/EPOCHS/FRAMES`（多 ep 配方），默认仍是单卡单 ep 过拟合。
>
> **T5 (Wan2.2-TI2V-5B) 已打通/PASS（2026-06-16）**：SEAM-2 新 VAE 路径可用。
> `wan2.2-ti2v-5b` 已接入：`z_dim=48`、`vae_spatial_factor=16` →
> `vace_in_dim=352`/`mask_pq=16`，`first_frame_mode="ti2v_fused"`，15 层 VACE。
> 训练侧 dataset operator 已按 spec 使用 32 倍空间整除；推理/provision 对 `ti2v_fused` 走 `input_image`。
>
> **环境状态**：deepspeed 已卸载，`gradient_checkpoint._HAS_DEEPSPEED=False`，跑 a2v 命令**不需要** stub-nvcc（diffsynth 直接 import 即可）。
> 若日后重装 deepspeed 又遇 nvcc 探测崩溃，最简办法是再次卸载它（A2V 单卡 8-bit + 低 lr 不需要 deepspeed）。8 张 A100 可用。
>
> **🧹 代码整理（2026-06-17，中度统一）**：
> - **训练脚本合并**：`run_overfit_{t2v,i2v,ti2v}.sh` → 统一 **`bash a2v/run_overfit.sh <base_spec>`**（按 spec 自带 lr/数据集/H×W/优化器/首帧/LoRA-vs-全参预设；env `LR=/OUT=/HEIGHT=/...` 可覆盖）。下文历史段里出现的 `run_overfit_<x>.sh` 一律等价于 `run_overfit.sh <对应 spec>`。
> - **加载冒烟合并**：`check_t4_load.py`/`check_t5_load.py` → 统一 **`a2v.check_load --base_spec <spec>`**。
> - **删除废弃产物**：`a2v/.fakecuda/`、`run_overfit_i2v_fp32.sh`、`accelerate_2gpu_zero2.yaml`（fp32 多卡弯路已废弃，历史见 §13.1）。
> - `causal_metrics` 的 `--height/--width` 改为可选，默认从 GT PNG 自动派生（避免 T5 256 vs 默认 240 的形状错）。
>
> **T5 产物（单样本过拟合，8-bit Adam，step-1000）**：
> - 数据集：`.cache/a2v_robotwin/ep0_dataset_phys_256x320`（T5 高度需 32 倍整除，故从 240×320 重采到 256×320）。
> - ckpt：`models/train/a2v_robotwin_ep0_vace_ti2v/step-1000.safetensors`（全参 vace，5.22GB）。
> - 视频：`.cache/a2v_robotwin/gen_ti2v_s1000_{real,none,shuffle}.mp4`。
> - 因果门：REAL MAE-GT=**5.27**、motion=5.38（GT=4.74）；NONE MAE-GT=15.62、real-vs-none=16.02；
>   SHUFFLE MAE-GT=14.91、real-vs-shuffle=13.44。T5/I2V 类判据同 T4：REAL 更近 GT 且与负对照明确不同，负对照不要求静止。
>
> **T5 复现命令**：
> ```bash
> cd /vepfs/wangshilong/code/DiffSynth-Studio
> PY=.venv/bin/python
> $PY -m a2v.check_load --base_spec wan2.2-ti2v-5b
> $PY -m a2v.check_provision --base_spec wan2.2-ti2v-5b \
>   --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 --height 256 --width 320 --num_frames 13 --steps 4
> CUDA_VISIBLE_DEVICES=<g> bash a2v/run_overfit.sh wan2.2-ti2v-5b
> for c in real none shuffle; do $PY -m a2v.infer_a2v --base_spec wan2.2-ti2v-5b \
>   --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
>   --lora models/train/a2v_robotwin_ep0_vace_ti2v/step-1000.safetensors \
>   --num_frames 121 --height 256 --width 320 --control $c \
>   --output .cache/a2v_robotwin/gen_ti2v_s1000_$c.mp4; done
> # causal_metrics auto-derives H,W from the GT PNGs (256x320 here) — no --height needed:
> $PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 --num_frames 121 \
>   --real .cache/a2v_robotwin/gen_ti2v_s1000_real.mp4 --none .cache/a2v_robotwin/gen_ti2v_s1000_none.mp4
> ```
>
> **T4 产物（单样本过拟合，8-bit Adam，step-1000）**：
> | run | lr | 输出目录 | REAL MAE-GT | REAL motion | NONE MAE-GT | real-vs-none |
> | --- | --- | --- | --- | --- | --- | --- |
> | lr5e6 | 5e-6 | `models/train/a2v_robotwin_ep0_vace_i2v_lr5e6` | 5.35 | 5.04 | 28.59 | 29.83 |
> | **lr1e5** | **1e-5** | `models/train/a2v_robotwin_ep0_vace_i2v_lr1e5` | **4.83** | **4.94** | **36.04** | **36.79** |
> | lr2e5 | 2e-5 | `models/train/a2v_robotwin_ep0_vace_i2v_lr2e5` | 5.19 | 4.97 | 37.80 | 38.62 |
> 视频：`.cache/a2v_robotwin/gen_i2v_lr{5e6,1e5,2e5}_s1000_{real,none}.mp4`。
> 复现/覆盖：`LR=<lr> OUT=<dir> CUDA_VISIBLE_DEVICES=<g> bash a2v/run_overfit.sh wan2.1-i2v-14b-480p`（默认 `LR=1e-5`）。
>
> **下一步**：推进 **I2V-14B 多 episode**（复用 `.cache/a2v_robotwin/ep_train40_phys` / `ep_heldout10_phys` 与 `a2v.eval_multiep`，从 `LR=1e-5` 起步）；之后再做 **T6 I2V-A14B**（SEAM-5 双专家：Fun-A14B 的 `vace`/`vace2`、boundary=0.875、分带训练/验证）。见 §15/§4.2/§5。
> **勿用的坏产物**：`models/train/a2v_robotwin_ep0_vace_i2v/step-*`（8-bit lr=1e-4 噪声 ckpt）。
> 其余背景见 `A2V_HISTORY.md` §12（T4 调试史）、§13（06-16 比对 ABot + lr 修复全过程）、§14（T5 终态）。

---

> 📜 **各 Track 详细调试史 + 进展时间线（原进展 banner 与 §7–§14）已拆分到 [`A2V_HISTORY.md`](./A2V_HISTORY.md)**，章节编号不变、便于交叉引用。
> 本文只保留**当前状态 + 接手清单 + 速查**；下文与接手清单中凡“见下方 §N / 详见 §N”（N≥7）一律指 `A2V_HISTORY.md` 对应章节。

---

## 0. 一句话目标

在 **DiffSynth-Studio** 里，用一套**可扩展、分阶段**的框架，把多种 Wan 基模改造成
**动作→视频（A2V）**模型：把机器人末端轨迹渲染成 **3 通道 RGB 轨迹图**当作 `vace_video`，
走官方 **VACE** 路径，冻结 DiT 只训 VACE 分支。目标基模（按接入顺序）：

```
Wan2.1-VACE-1.3B → Wan2.1-T2V-1.3B → Wan2.1-I2V-14B-480P → Wan2.2-TI2V-5B → Wan2.2-I2V-A14B
```

两个代码库：
- **DiffSynth-Studio**：`/vepfs/wangshilong/code/DiffSynth-Studio`（改造在这里，分支 `a2v`）
- **ABot-PhysWorld**：`/vepfs/wangshilong/code/ABot-PhysWorld`（参考实现 + 分析文档在 `.cache/analysis/`）

---

## 1. 已核实的关键事实（都查过实际代码，可直接信任）

### 1.1 五基模配置（`diffsynth/configs/model_configs.py`）
| | VACE-1.3B | T2V-1.3B | I2V-14B-480P | TI2V-5B | I2V-A14B |
| --- | --- | --- | --- | --- | --- |
| dim / layers / heads / ffn | 1536/30/12/8960 | 同左 | 5120/40/40/13824 | 3072/30/24/14336 | 5120/40/40/13824 ×2 |
| in_dim | 16 | 16 | 36 | 48 | 36 ×2 |
| VAE | Wan2.1 | Wan2.1 | Wan2.1 | **Wan2.2** | Wan2.1 |
| z_dim / 空间压缩 | 16/8× | 16/8× | 16/8× | **48/16×** | 16/8× |
| **vace_in_dim** = 2·z+s² | **96** | **96** | **96** | **352** | **96** |
| **mask_pq** = s | **8** | **8** | **8** | **16** | **8** |
| **vace_layers** | (0,2,…,28)=15 | (0,2,…,28) | **(0,5,…,35)=8** | (0,2,…,28) | (0,5,…,35) ×2 |
| 自带 VACE 权重 | **有** | 无 | 无（官方 VACE-14B 结构同构可 warm-start） | 无 | **有(Fun-A14B,双)** |
| 首帧机制 | vace_reference(弱) | none | i2v_concat(强) | ti2v_fused(强) | i2v 式 |
| 专家数 | 1 | 1 | 1 | 1 | **2 (boundary 0.875, vace+vace2)** |

**三个最易错、务必记牢的数字**：
1. `vace_in_dim = 2·z_dim + spatial²`：Wan2.1-VAE 系全是 **96**；**TI2V-5B = 352**（mask 通道 64→256，不是 160）。
2. `mask_pq`（`WanVideoUnit_VACE` 里写死的 `P=Q=8`）= VAE 空间压缩因子：Wan2.1=8，**Wan2.2=16**。
3. `vace_layers` 按层数显式给：30 层→15 层(step2)，40 层→8 层(step5)。**不要 naive 派生。**

### 1.2 VACE 通道构成（`diffsynth/pipelines/wan_video.py:649-706`）
`vace_video`(3ch RGB) 过 VAE → `vace_context` = inactive(z) + reactive(z) + mask(P·Q)。
Wan2.1：16+16+64=96。mask 来自 `rearrange("T (H P) (W Q) -> 1 (P Q) T H W", P=8,Q=8)`，
P=Q **必须等于 VAE 空间压缩因子**（否则崩）——这是 master plan 唯一需要改 DiffSynth 数据流的点。

### 1.3 两个工程约束
- 官方 `train.py` **不能给无 VACE 权重的基模凭空建 VACE**（`pipe.vace=None` 时 `--trainable_models vace`
  静默不训，`training_module.py:290`）。→ T2V/I2V-14B/TI2V-5B/A14B 都需移植 ABot 的 `create_vace_from_dit`。
- 数据集对 `video` 和 `vace_video` **独立采样**（`unified_dataset.py:89-101`）→ 必须离线把两路固定成
  **同一批 N 帧、各存 PNG 列表**（这是 I2 对齐的落地）。

### 1.4 ABot 的 3 通道 vs 9 通道（已查源码）
- **3 通道** = 轨迹图（`get_traj_maps`），走官方 VACE，**生产推理/训练用的就是它**。
- **9 通道** = 3 轨迹 + 6 相机射线（Plücker），不过 VAE、直接 latent 注入，需自定义 vace_in_dim 从零训。
- ABot 14B 生产 A2V = VACE + 3 通道轨迹图，与本方案同构（参考实现：
  `ABot-PhysWorld/inference/inference_a2v.py`、`training/train_a2v.py`、`inference/diffsynth/utils/action_utils.py`）。

---

## 2. 已产出文档索引（`.cache/analysis/`，按阅读顺序）

| 文档 | 作用 |
| --- | --- |
| `Wan2.1_VACE_1.3B_A2V_plan.md` | 初版计划（被审计对象，已过时） |
| `Wan2.1_VACE_1.3B_A2V_plan_revised.md` | 修订版：厘清 3/9 通道、复用 ABot 编码、无损存储、内参缩放、帧记账 |
| `Wan2.1_VACE_1.3B_A2V_impl_plan.md` | 1.3B 单基模分阶段落地（含退出门/故障树） |
| `A2V_multibase_framework.md` | 多基模接缝抽象（WanBaseSpec + 5 seam） |
| **`A2V_master_plan.md`** | **★ 当前主计划**：5 基模、Tracks T0–T6、一次一个新变量、低 debug 难度 |
| `A2V_action_encoding_redesign.md` | 性能优先的编码重设计（Tier1 splat / Tier2 解耦通道），**不绑 ABot ckpt** |
| `A2V_radius_normalization_explained.md` | 半径归一化的原理详解（min-max bug + 正确的投影/逆深度归一化） |
| **`A2V_HANDOFF.md`** | **本文，新会话入口**：当前状态 + 接手清单 + 速查（§0–§6） |
| **`A2V_HISTORY.md`** | **历史附录**：进展时间线 + 各 Track 调试史与教训（§7–§14，编号沿用本文） |

代码侧（`DiffSynth-Studio/a2v/`）：
| 文件 | 作用 |
| --- | --- |
| `REVIEW_FIXES.md` | 第一阶段代码的 review 任务清单（T1–T7） |
| `render/traj_map.py` | 轨迹图渲染（移植 ABot；含那条**有意保留**的经验半径公式） |
| `render/action_io.py` | 动作/相机 IO |
| `render/check_projection.py` | I1 内参缩放重投影回归测试（T3 新增） |
| `data/prepare.py` | 离线数据生成（采样→渲染→PNG 列表→metadata.jsonl）;有 `--gripper_z_offset`(RoboTwin=0) |
| `data/operators.py` | `frame_list_video_operator`（PNG 列表的 list 路由，绕开框架独立采样） |
| `data/validate.py` | 用 UnifiedDataset 校验产出；支持 `--base_spec` 派生 T5 的 32×空间整除 |
| `data/smoke.py` | 端到端 smoke（合成数据 prepare+validate） |
| `data/robotwin_adapter.py` | **[T1]** RoboTwin hdf5 → actions/intrinsic/extrinsic.npy + manifest（§7） |
| `base_spec.py` | **[T2/T3/T4/T5]** `WanBaseSpec`+`REGISTRY`+`get_spec`；含 `wan2.2-ti2v-5b`(352/16,§14) |
| `vace_unit.py` | **[T2]** `ParamWanVideoUnit_VACE`(参数化 mask_pq)+`install_vace_unit`(§8) |
| `provision.py` | **[T2/T3]** `ensure_vace`(SEAM-1,幂等)+`provision_a2v`+`create_vace_from_dit`(从 DiT 造 VACE,只 zero-init after_proj,§11) |
| `check_provision.py` | **[T3/T4/T5]** SEAM-1 退出门:形状/零副作用/结构 parity(§11);I2V/TI2V 传 `input_image`(§12/§14) |
| `run_overfit.sh` | **[统一,06-17]** `bash a2v/run_overfit.sh <base_spec>` 单样本过拟合,按 spec 自带预设(lr/数据集/H×W/优化器/首帧/LoRA-vs-全参)。取代 run_overfit_{t2v,i2v,ti2v}.sh |
| `check_load.py` | **[统一,06-17]** `--base_spec` 加载冒烟(取代 check_t4_load/check_t5_load):DiT层数/VAE z·s/造 vace(in_dim·层数·after_proj=0)/mask_pq;i2v 验 CLIP+in_dim36,ti2v 验 in_dim48+fused |
| `causal_metrics.py` | **[可复用]** 因果门度量 MAE-GT/motion/real-vs-none;H,W 默认从 GT PNG 自动派生(§12) |
| `train_a2v.py` | **[T1/T2/T5]** 薄训练封装(swap operator)+`--base_spec`+`provision_a2v`；按 spec 派生数据整除因子 |
| `infer_a2v.py` | **[T1/T2/T4/T5]** 推理 harness;`--base_spec`/`--control real\|none\|shuffle`；I2V/TI2V 首帧走 `input_image`；现抽出可 import 的 build/generate 函数供多行评测复用 |
| `eval_multiep.py` | **[M1]** 多 episode 泛化评测；模型只加载一次，逐行生成 real/none，输出 mp4 + metrics.json + SUMMARY |
| `run_overfit.sh` | **[T1/T2]** 单样本 LoRA 过拟合(spec 驱动) |
| `check_parity.py` | **[T2]** 零行为变更对拍闸门(§8) |
| `README_A2V.md` | 用法 + 训练接线前向依赖 + reference 帧语义 |

---

## 3. 当前进度与环境（新会话先读这一节）

**走的是 RoboTwin 路线**（不是 AgiBot）。进度沿 master plan：

| Track | 状态 | 说明 |
| --- | --- | --- |
| T1 VACE-1.3B 垂直切片 | ✅ PASS | RoboTwin ep0 LoRA 过拟合,实控制复现/空白控制不复现。详见 §7 |
| T2 抽象插入(`WanBaseSpec`) | ✅ PASS | 零行为变更已对拍(MD5 一致)。详见 §8 |
| §9 半径修复端到端重验 | ✅ PASS (06-13) | physical 重渲染+重训+因果门(REAL MAE 9.2/NONE 静止);并修复 T2 LoRA 前缀 bug。详见 §10 |
| T3 接入 T2V-1.3B | ✅ PASS (06-14) | SEAM-1「从 DiT 造 VACE」。3 道 provision 门 + 全参过拟合因果门(REAL MAE-GT 20.9/NONE 静止 0.68/real-none 41.6)。**关键纠错:LoRA 不可训 from-DiT vace,且 patch_embedding 不可与 before_proj 同时零初始化(死锁)**。详见 §11 |
| **T4 接入 I2V-14B** | ✅ **PASS/定稿** (06-16) | 真因=**lr 1e-4 过高**(非 8-bit Adam/非缺 vace_reference)。三组 step-1000 lr 扫描完成，**lr1e5 胜出**:REAL MAE-GT **4.83**、motion 4.94≈GT 4.78、NONE MAE-GT 36.04、real-none 36.79。`run_overfit.sh` 默认已改 `1e-5`。详见 §12/§13 |
| **T5 接入 TI2V-5B** | ✅ **PASS** (06-16) | SEAM-2 新 VAE 打通：`vace_in_dim=352`/`mask_pq=16`，256×320 数据、load/provision/1000步过拟合/real-none-shuffle 因果门全过。REAL MAE-GT **5.27**，NONE 15.62，SHUFFLE 14.91。详见 §14 |
| **M1 VACE-1.3B 多 episode** | ✅ **PASS** (06-18) | train40/heldout10，105 帧，240x320，LoRA step-4800。held-out REAL MAE-GT **6.25** vs NONE **26.83**，real<none **10/10**，motion ratio **1.03**。详见 §15 |

**环境（已解决,不要再找）**：venv 在 `/vepfs/wangshilong/code/DiffSynth-Studio/.venv`。
`.venv/bin/python` 已含 torch 2.5.1+cu121 / numpy / einops / PIL / imageio / **h5py / tensorboard**。
（matplotlib/decord 缺失,但代码有 fallback,不影响。）GPU: A100-80GB。deepspeed 已卸载，`_HAS_DEEPSPEED=False`，无需 stub-nvcc。

**冒烟自检（任何改动后先跑,应全绿）**：
```bash
cd /vepfs/wangshilong/code/DiffSynth-Studio
.venv/bin/python -m a2v.render.check_projection   # check_projection: OK
.venv/bin/python -m a2v.data.smoke                # same_size/scaled_crop 通过 + bad_original_size 期望失败
.venv/bin/python -m a2v.check_parity           # T2 parity: OK (需 GPU,加载 VACE-1.3B)
```

**新半径可用产物（06-13 验证，直接可用，无需任何手工后处理）：**
- 数据集 `.cache/a2v_robotwin/ep0_dataset_phys`；LoRA `models/train/a2v_robotwin_ep0_lora_phys/step-1000.fixedkeys.safetensors`。
- `run_overfit.sh` 默认已指向 `ep0_dataset_phys`；`train_a2v` 前缀 bug 已修，**重训原生产出可加载 key**（`vace_blocks.*`）。
- ⚠ 同目录的原始 `step-1000.safetensors`（及 step-100..900）是**坏前缀**(`pipe.vace.*`)产物，**勿用**；只用 `*.fixedkeys.safetensors` 或重训。

> 第一阶段 REVIEW_FIXES.md 的 T1–T7 早已逐文件核实落实。
> **半径已修(2026-06-12,见 §9;端到端重验 06-13 PASS,见 §10)**:原 ABot `simple_radius_gen_func` 的"恒为常数"bug 已修;
> `traj_map.py` 现支持 `radius_mode = physical(默认) | normalized | constant`。`correct_matrix`/`gripper_z_offset` 仍不要顺手改
> (RoboTwin 用 `gripper_z_offset=0` 已参数化)。
> **LoRA 保存前缀 bug 已修(06-13,见 §10)**:T2 spec 驱动训练曾因 stock `--remove_prefix_in_ckpt` 默认 `pipe.dit.` 而存出加载不上的 LoRA;现已修。

---

## 4. 框架设计速查（来自 master plan）

### 4.1 五个接缝（新增基模理想情况只加一条 `WanBaseSpec`）
1. **SEAM-1 VACE 供给**：有权重→直接用；无→`create_vace_from_dit`（zero-init / 或官方权重 warm-start）。
2. **SEAM-2 通道**：`vace_in_dim=2·z+s²`、`mask_pq=s`（把官方 `WanVideoUnit_VACE` 的 `P=Q=8` 参数化，
   复制成 `a2v/vace_unit.py`，**不改原文件**）。
3. **SEAM-3 结构**：dim/heads/ffn/patch 派生自 DiT；`vace_layers` 显式给。
4. **SEAM-4 首帧**：`vace_reference | i2v_concat | ti2v_fused | none`。
5. **SEAM-5 专家集合**：单 / 双专家 MoE（high/low + boundary，vace+vace2，**分带双训练作业**）。

### 4.2 渐进路线（每步只引入一个新变量 → debug 半径最小）
```
T0 官方基线 smoke          → 隔离环境问题
T1 VACE-1.3B 垂直切片       → ground truth + 共享设施   ✅ PASS (§7)
T2 抽象插入(走 spec, 对拍T1) → 抽象=零行为变更           ✅ PASS (§8)
T3 T2V-1.3B   → SEAM-1 (从DiT造VACE)                  ✅ PASS (§11)
T4 I2V-14B    → SEAM-4(i2v)+scale (有 ABot 参考可 warm-start)  ✅ PASS (§13)
T5 TI2V-5B    → SEAM-2 (新VAE 352/16)                    ✅ PASS (§14)
T6 A14B       → SEAM-5 (双专家MoE)                       ← 下一步
```

### 4.3 对齐不变量（正确性核心）
- I1 空间：内参随 resize/crop 同步缩放（已在 `traj_map.py` 实现，`check_projection.py` 回归）。
- I2 时间：两路同一批帧的 PNG 列表（已在 `prepare.py`+`operators.py` 落地）。
- I3 帧数：4n+1。 I4：reference=首帧、vace_video 不重复计首帧。 I5：禁有损 mp4，PNG 无损。

---

## 5. 下一步待办（建议顺序）

T1、T2、T3、T4、T5 已 PASS（见 §7/§8/§11/§13/§14）。接下来：

1. **T6 接入 Wan2.2-I2V-A14B**（新变量 SEAM-5 双专家 MoE）：
   - 目标是 Fun-A14B 的双 VACE 专家（`vace`/`vace2`、boundary=0.875）。先只做加载/provision 门，再做单样本过拟合因果门。
   - 保持 T3/T4/T5 的原则：新增基模先 spec 化；from-DiT/新分支全参训；只 zero-init 回主流的 `after_proj`；I2V 首帧走 `input_image`。
   - 退出门建议沿用 T5：load smoke、provision shape/零副作用、1000 步单样本 overfit、real/none/shuffle 因果门。
2. **多 episode 泛化**（可先在 T4/T5 做）：adapter 已支持 `--episodes_range`/默认全 episode；prepare 支持多行 manifest。14B 多 episode 前优先考虑 encoded-cache，避免重复 T5/CLIP/VAE 编码。
3. **编码升级**（不依赖 ABot ckpt）：按 `A2V_action_encoding_redesign.md` 实现 Tier1 splat 编码，作为 `prepare.py` 的 `--encoding splat` 选项，与现编码过拟合对拍。

### 已完成但常被误判的分支
- T4 I2V-14B 已 PASS；坏的是旧 `models/train/a2v_robotwin_ep0_vace_i2v/step-*`（lr=1e-4 发散），可用的是 `..._lr1e5/step-1000.safetensors`。
- T5 TI2V-5B 已 PASS；`vace_unit.py` 的 `mask_pq=16` 路径已实测，不再是待办。

---

## 6. 必记的"坑"与硬约束
- `vace_in_dim`/`mask_pq` 必须派生，**TI2V-5B=352/16**，不是 96/8。
- `vace_layers` 按层数显式给（40 层 step5）。
- 不改 `DiffSynth-Studio/diffsynth/` 任何原文件；所有 A2V 改动限定在 `a2v/`。
- 半径公式已修（§9）：默认 `radius_mode='physical'`（`f_eff·R/z_cam` 透视标定）。原 ABot 的 buggy 经验公式
  按用户指示**不再保留**；`simple_radius_gen_func` 现是修正后的 min-max 归一化（`normalized` 模式用）。
- 走 ABot warm-start 的 T4 若需复现 ABot 控制图，须自行确认半径口径（旧 buggy 公式已不在仓库；用 `normalized` 近似或重训）。
- 真实数据务必确认：`original_size` 与视频真实分辨率一致、外参是 c2w、夹爪量纲、H/W 为 16 倍数。
