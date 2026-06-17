# A2V 项目交接文档（新会话入口）

> 新开 Claude 会话时**先读本文**。它汇总了"把 Wan 基模改造成 action-conditioned (A2V) 世界模型"
> 这件事的目标、已核实事实、已产出文档、当前代码状态、以及待办与阻塞点。
> 详细内容散在 `.cache/analysis/` 的各专题文档里，本文给出索引与摘要。

最后更新：2026-06-16。

> ## ⭐ CODEX 接手清单（2026-06-16，先读这一段）
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
> $PY -m a2v.check_t3_provision --base_spec wan2.2-ti2v-5b \
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
> **下一步**：推进 **T6 I2V-A14B**（SEAM-5 双专家：Fun-A14B 的 `vace`/`vace2`、boundary=0.875、分带训练/验证），或先做 T1/T3/T4/T5 多 episode 泛化。见 §4.2/§5。
> **勿用的坏产物**：`models/train/a2v_robotwin_ep0_vace_i2v/step-*`（8-bit lr=1e-4 噪声 ckpt）。
> 其余背景见下方 §12（T4 调试史）、§13（06-16 比对 ABot + lr 修复全过程）、§14（T5 终态）。

---

（历史最后更新：2026-06-15。）

> **2026-06-11 进展（RoboTwin 路线）**：用户改为直接在 **RoboTwin2.0**(`/data/RoboTwin2.0_unpacked`,见其
> `DATA_STRUCTURE.md`)上训练。已完成 master plan **T1 去留闸门（PASS）**：单 episode
> (`beat_block_hammer/aloha-agilex_clean_50/ep0`) LoRA-on-VACE 过拟合，原生 240×320、121 帧、1000 步。
> 实控制→复现 GT 机械臂轨迹；空白控制→不复现（臂不动）⇒ 控制信号因果驱动生成。数据→训练→推理全链路打通。
> 关键约定与新文件见下方 §7。半径/correct_matrix 不要"顺手修"的告诫仍适用于 AgiBot 路线。
>
> **2026-06-12 进展（T2 抽象插入 PASS）**：把 T1 重构进 `WanBaseSpec` 抽象，**零行为变更**已对拍证明 —— 见下方 §8。
>
> **2026-06-13 进展（§9 半径修复端到端重验 PASS）**：physical 半径重渲染+重训+因果对照通过（REAL MAE-GT=9.2 复现 /
> NONE 运动 0.44 不动）。顺带挖出并修复一个 T2 引入的 LoRA 保存前缀 bug（`remove_prefix_in_ckpt` 死代码，
> 导致 spec 驱动训练存出的 LoRA 加载不上）—— 详见下方 §10。**新半径可用产物在 `ep0_dataset_phys` + `*_lora_phys/step-1000.fixedkeys.safetensors`。**
>
> **2026-06-14 进展（T3 接入 T2V-1.3B PASS）**：SEAM-1「从 DiT 造 VACE」打通——无 VACE 权重的基模也能凭空造分支并训到
> 因果驱动。3 道 provision 门 + 全参过拟合因果门全过（REAL MAE-GT 20.9 / NONE 静止 0.68 / real-none 41.6）。
> **两个 from-DiT 致命纠错：①LoRA 训不动 from-DiT vace（控制进出口 patch_embedding/after_proj 非 LoRA target），必须全参；
> ②patch_embedding 与 before_proj 不能同时 zero-init（梯度死锁），只 zero-init after_proj。** 详见下方 §11。
>
> **2026-06-15 进展（T4 I2V-14B：基础设施 PASS，因果门 FAIL，调试中）**：权重下载齐（I2V-14B + VACE-14B 各 7 分片 + CLIP）。
> base_spec 加 `dit_glob`(7分片→list 合并) + `image_encoder_path`，注册 `wan2.1-i2v-14b-480p`。**CLIP `.pth` 直接加载**(免转换)；
> SEAM-4 首帧走 `input_image`(非 vace_reference)；**provision 两门全绿**(形状 96/8层/5120/13824 + 零副作用 max_abs=0)。
> 8-bit Adam 单卡跑完 1000 步全参 vace。**但因果门 FAIL：real/none 输出全是噪声**。逐层隔离锁定：base i2v(无vace)连贯、
> vace 初始态(after_proj=0)连贯、**训练后 vace 在 scale≥0.3 即崩成噪声且 step-100 早期就坏** ⇒ vace 训练学坏(结构/上下文/权重幅度
> 都正常、非结构 bug)，loss 非单调飙到 2.1。**嫌疑：①8-bit Adam 不稳(T3 用 fp32 AdamW)②去掉了 vace_reference_image③lr 偏高。**
> **下一步=用 T3 验证过的 fp32 AdamW 重训(单卡需 offload 或多卡 ZeRO)，最小变量回退。详见下方 §12。**
>
> **2026-06-16 进展（T4 定稿：lr 根因 + 扫描胜出 lr=1e-5）**：尝试 2×A100 ZeRO-2 跑 fp32 AdamW 两次均失败——
> ①no-offload→优化器步 OOM；②optimizer-offload→DeepSpeedCPUAdam 需 ninja+CUDA toolkit 编译 cpu_adam，本机只有 runtime 无 toolkit→死路。
> 随后比对 ABot(同构 I2V-14B+VACE 路线)实际配方，锁定主要差异是 lr：ABot A2V 用 5e-6，而失败 run 用 1e-4(高 20×)。
> 单卡 8-bit 低 lr 重训验证真因，三组 step-1000 扫描完成：**lr1e5 胜出**（REAL MAE-GT=4.83 / motion=4.94 / NONE=36.04 / real-none=36.79），
> 优于 lr5e6(REAL MAE-GT=5.35) 与 lr2e5(5.19)。`run_overfit_i2v.sh` 默认改为 `1e-5`，**T4 PASS**。详见 §13。
>
> **2026-06-16 进展（T5 TI2V-5B PASS）**：接入 Wan2.2-TI2V-5B（SEAM-2 新 VAE），`vace_in_dim=352`/`mask_pq=16`
> 由 spec 自动派生；训练/推理首帧走 `ti2v_fused → input_image`。256×320 数据、T5 load smoke、provision 零副作用门、1000 步全参 vace 过拟合、
> real/none/shuffle 因果门均完成：REAL MAE-GT=5.27，NONE=15.62，SHUFFLE=14.91，T5 PASS。用户同意后已卸载 deepspeed，a2v run 不再需要 stub-nvcc。详见 §14。

---

## 0. 一句话目标

在 **DiffSynth-Studio** 里，用一套**可扩展、分阶段**的框架，把多种 Wan 基模改造成
**动作→视频（A2V）**模型：把机器人末端轨迹渲染成 **3 通道 RGB 轨迹图**当作 `vace_video`，
走官方 **VACE** 路径，冻结 DiT 只训 VACE 分支。目标基模（按接入顺序）：

```
Wan2.1-VACE-1.3B → Wan2.1-T2V-1.3B → Wan2.1-I2V-14B-480P → Wan2.2-TI2V-5B → Wan2.2-I2V-A14B
```

两个代码库：
- **DiffSynth-Studio**：`/vepfs/wangshilong/code/DiffSynth-Studio`（改造在这里，commit `e5f88f0`）
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
| `Wan2.1_VACE_1.3B_A2V_plan.md` | codex 初版计划（被审计对象，已过时） |
| `Wan2.1_VACE_1.3B_A2V_plan_revised.md` | 修订版：厘清 3/9 通道、复用 ABot 编码、无损存储、内参缩放、帧记账 |
| `Wan2.1_VACE_1.3B_A2V_impl_plan.md` | 1.3B 单基模分阶段落地（含退出门/故障树） |
| `A2V_multibase_framework.md` | 多基模接缝抽象（WanBaseSpec + 5 seam） |
| **`A2V_master_plan.md`** | **★ 当前主计划**：5 基模、Tracks T0–T6、一次一个新变量、低 debug 难度 |
| `A2V_action_encoding_redesign.md` | 性能优先的编码重设计（Tier1 splat / Tier2 解耦通道），**不绑 ABot ckpt** |
| `A2V_radius_normalization_explained.md` | 半径归一化的原理详解（min-max bug + 正确的投影/逆深度归一化） |
| **`A2V_HANDOFF.md`** | **本文，新会话入口** |

代码侧（`DiffSynth-Studio/a2v/`）：
| 文件 | 作用 |
| --- | --- |
| `REVIEW_FIXES.md` | 对 codex 第一阶段代码的 review 任务清单（T1–T7） |
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
| `check_t3_provision.py` | **[T3/T4/T5]** SEAM-1 退出门:形状/零副作用/结构 parity(§11);I2V/TI2V 传 `input_image`(§12/§14) |
| `run_overfit.sh` | **[统一,06-17]** `bash a2v/run_overfit.sh <base_spec>` 单样本过拟合,按 spec 自带预设(lr/数据集/H×W/优化器/首帧/LoRA-vs-全参)。取代 run_overfit_{t2v,i2v,ti2v}.sh |
| `check_load.py` | **[统一,06-17]** `--base_spec` 加载冒烟(取代 check_t4_load/check_t5_load):DiT层数/VAE z·s/造 vace(in_dim·层数·after_proj=0)/mask_pq;i2v 验 CLIP+in_dim36,ti2v 验 in_dim48+fused |
| `causal_metrics.py` | **[可复用]** 因果门度量 MAE-GT/motion/real-vs-none;H,W 默认从 GT PNG 自动派生(§12) |
| `train_a2v.py` | **[T1/T2/T5]** 薄训练封装(swap operator)+`--base_spec`+`provision_a2v`；按 spec 派生数据整除因子 |
| `infer_a2v.py` | **[T1/T2/T4/T5]** 推理 harness;`--base_spec`/`--control real\|none\|shuffle`；I2V/TI2V 首帧走 `input_image` |
| `run_overfit.sh` | **[T1/T2]** 单样本 LoRA 过拟合(spec 驱动) |
| `check_t2_parity.py` | **[T2]** 零行为变更对拍闸门(§8) |
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
| **T4 接入 I2V-14B** | ✅ **PASS/定稿** (06-16) | 真因=**lr 1e-4 过高**(非 8-bit Adam/非缺 vace_reference)。三组 step-1000 lr 扫描完成，**lr1e5 胜出**:REAL MAE-GT **4.83**、motion 4.94≈GT 4.78、NONE MAE-GT 36.04、real-none 36.79。`run_overfit_i2v.sh` 默认已改 `1e-5`。详见 §12/§13 |
| **T5 接入 TI2V-5B** | ✅ **PASS** (06-16) | SEAM-2 新 VAE 打通：`vace_in_dim=352`/`mask_pq=16`，256×320 数据、load/provision/1000步过拟合/real-none-shuffle 因果门全过。REAL MAE-GT **5.27**，NONE 15.62，SHUFFLE 14.91。详见 §14 |

**环境（已解决,不要再找）**：venv 在 `/vepfs/wangshilong/code/DiffSynth-Studio/.venv`。
`.venv/bin/python` 已含 torch 2.5.1+cu121 / numpy / einops / PIL / imageio / **h5py / tensorboard**。
（matplotlib/decord 缺失,但代码有 fallback,不影响。）GPU: A100-80GB。deepspeed 已卸载，`_HAS_DEEPSPEED=False`，无需 stub-nvcc。

**冒烟自检（任何改动后先跑,应全绿）**：
```bash
cd /vepfs/wangshilong/code/DiffSynth-Studio
.venv/bin/python -m a2v.render.check_projection   # check_projection: OK
.venv/bin/python -m a2v.data.smoke                # same_size/scaled_crop 通过 + bad_original_size 期望失败
.venv/bin/python -m a2v.check_t2_parity           # T2 parity: OK (需 GPU,加载 VACE-1.3B)
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

---

## 7. RoboTwin2.0 路线（2026-06-11 新增，T1 已 PASS）

数据：`/data/RoboTwin2.0_unpacked/{task}/{robot}_{mode}_{n}/`，结构见 `DATA_STRUCTURE.md`。

**已核实的 RoboTwin→traj_map 约定（勿重推）：**
- `endpose/{left,right}_endpose (T,7)=[x,y,z,qx,qy,qz,qw]` **直接**对应 traj_map 16 维（xyzw 顺序一致）；
  夹爪 ∈[0,1]，`×120` 以铺满 `/120` 色表（纯视觉）。
- `observation/head_camera/intrinsic_cv`=OpenCV K；`extrinsic_cv (T,3,4)`=**w2c [R|t]** OpenCV。
  adapter 导出 **c2w=inv([R|t;0001])** 给 prepare.py（其内部再求逆得 w2c）。OpenGL `cam2world_gl` 故意不用。
- **关键：RoboTwin endpose 在 TCP，`gripper_z_offset=0.0`**（不是 AgiBot 的 0.23）。overlay 实测 0.23 偏低 ~165px。
  `traj_map.py` 已参数化 `gripper_z_offset`（默认 0.23 不变，AgiBot 路线零行为变更）；`prepare.py` 加了 `--gripper_z_offset`。
- head mp4(320×240,30fps) 与 hdf5 逐帧对齐（I2 平凡）；原生 240×320 均 ÷16 → scale=1，无 I1 风险。

**新文件：**
- `data/robotwin_adapter.py`：读 hdf5 → `actions.npy[T,16]`/`intrinsic.npy`/`extrinsic.npy(c2w)` + `manifest.jsonl`（video 指向原 mp4）。
- `train_a2v.py`：薄封装 stock `WanTrainingModule`+`wan_parser`，仅把 dataset 的 operator 换成 `frame_list_video_operator`（PNG 列表路由）。
- `infer_a2v.py`：加载 base+VACE-LoRA，从 `vace_video`+`vace_reference_image` 生成；`--control real|none|shuffle`（none/shuffle 为负对照）。
- `run_overfit.sh`：单样本 LoRA-on-vace 过拟合配方；本地 `--model_paths` JSON（DiT+VACE / Wan2.1_VAE / umt5-xxl t5，均在 `models/`）。

**复现命令（仓库根目录，venv=`.venv`，已装 h5py/tensorboard）：**
```bash
PY=.venv/bin/python
$PY -m a2v.data.robotwin_adapter --task beat_block_hammer --robot_mode aloha-agilex_clean_50 \
    --episodes 0 --work_dir .cache/a2v_robotwin/ep0_work
$PY -m a2v.data.prepare --manifest .cache/a2v_robotwin/ep0_work/manifest.jsonl \
    --output_dir .cache/a2v_robotwin/ep0_dataset --height 240 --width 320 --num_frames 121 \
    --resize_mode stretch --gripper_z_offset 0 --write_overlay
bash a2v/run_overfit.sh        # → models/train/a2v_robotwin_ep0_lora/step-*.safetensors
$PY -m a2v.infer_a2v --dataset .cache/a2v_robotwin/ep0_dataset \
    --lora models/train/a2v_robotwin_ep0_lora/step-1000.safetensors \
    --num_frames 121 --height 240 --width 320 --control real --output .cache/a2v_robotwin/gen.mp4
```

**下一步建议**：①多 episode 泛化（adapter 已支持 `--episodes_range`/默认全 episode；prepare 支持多行 manifest）；
②全参 VACE 分支（去掉 LoRA，`--trainable_models vace`）。③T2 抽象插入 **已完成**（见 §8）。

---

## 8. T2 抽象插入（2026-06-12，PASS）

把 T1 硬编码路径重构进声明式 `WanBaseSpec`，**零行为变更**已数值对拍证明。5 个 seam 收敛进 spec，
新增基模理想情况=加一条 REGISTRY。

**新文件：**
- `base_spec.py`：`WanBaseSpec`(frozen dataclass) + `REGISTRY`(目前仅 `wan2.1-vace-1.3b`) + `get_spec`。
  派生量 `vace_in_dim=2*z+s²`、`mask_pq=s`（**勿写死 96/8**）；`model_paths()` 返回本地 DiT/T5/VAE 绝对路径。
- `vace_unit.py`：`ParamWanVideoUnit_VACE`(把官方 `WanVideoUnit_VACE` 写死的 `P=Q=8` 改成 `mask_pq`，其余逐字复制) +
  `install_vace_unit(pipe, mask_pq)`(在 `pipe.units` 里就地替换那个 VACE 单元)。**不改 `diffsynth/`**。
- `provision.py`：`ensure_vace(pipe, spec)`(SEAM-1；有权重→返回 `pipe.vace`) + `provision_a2v`。
  （此处为 T2 当时状态；**无权重分支已在 T3 实现为 `create_vace_from_dit`,且 ensure_vace 已幂等化 —— 见 §11**。）
- `check_t2_parity.py`：**T2 对拍闸门**。

**对拍闸门结果（`python -m a2v.check_t2_parity`）：**
- SEAM-1：`ensure_vace(pipe, spec) is pipe.vace` ✓
- SEAM-2：固定同 seed 下 stock 与 param(mask_pq=8) 的 `vace_context`（shape `(1,96,32,30,40)`）**bit-identical, max_abs_diff=0** ✓
- 端到端：spec 驱动 infer(`--base_spec`) 复用 T1 的 step-1000 LoRA，输出与 T1 硬编码路径 **MD5 完全一致**（逐帧 Δ=0）。

**接线方式（保持 T1 命令可用）：** train/infer 新增 `--base_spec`(默认 `wan2.1-vace-1.3b`)；spec 填充
`model_paths/tokenizer/lora_base_model/remove_prefix` 的默认值，**显式 CLI 仍覆盖**。`run_overfit.sh` 已简化为
`--base_spec` 驱动。

**关键工程注意：** 默认训练 task=`sft`(非 `:train`/`:data_process`)，`split_pipeline_units` 不动 `pipe.units`，
故 `install_vace_unit` 能在 `pipe.units` 找到 VACE 单元（`training_module.py:332-341`）。

**（T2 当时的下一步=T3,现已 PASS —— 见 §11。后续=T4 I2V-14B,见 §5。）**

---

## 9. 半径归一化 bug 修复（2026-06-12，按 `A2V_action_encoding_redesign.md` §3.1）

**诊断（原 ABot `simple_radius_gen_func`）：** `clamp(1.0 - dist - 0.07/(0.8-0.07), 0,1)*100`。
本意是 min-max 斜坡 `(dist-0.07)/(0.8-0.07)`，但**少了括号** → `dist` 跑到归一化外、常数 `0.07` 被单独除，
塌成 `0.904 - dist`；再叠加 near/far=0.07/0.8m **魔法常数不匹配真实相机距离**。两者合力使半径在 ~1m
（AgiBot 距离）恒为负 → clamp 0 → **1px 死值，不携带深度**。
（注：RoboTwin 相机近 ~0.5m，旧公式恰好给 27–50px 不死——但靠距离巧合而非物理，量纲仍错。）

**修复（`a2v/render/traj_map.py`）：** 加 `radius_mode`，三选一：
- **`physical`（默认）** `physical_projection_radius`：`r_px = clamp(f_eff·R_phys/z_cam, r_min, H·r_max_frac)`，
  `z_cam`=EE 原点在相机系深度(`pts_*[:,2,0]`)、`f_eff=(fx+fy)/2` 用**同一套缩放内参** → 透视标定的近大远小。
- **`normalized`** 重写后的 `simple_radius_gen_func`：对轨迹**实际距离范围**做 min-max（最近→`r_max`、最远→`r_min`），
  不论绝对距离都随深度变化、永不恒定。修掉了括号 + 魔法常数两个病根。
- **`constant`** 固定 50px。

原 buggy 公式**按用户指示不再保留**。`get_vace_traj_maps_with_scaled_intrinsic` 默认 `radius_mode='physical'`；
`prepare.py` 加 `--radius_mode/--phys_radius_m/--radius_min_px/--radius_max_frac`，并写进 `metadata.jsonl`。

**已验证：** ① 合成深度变化序列下 physical 严格 `=f·R/z`、normalized 严格 min-max、constant=50，三者非恒定；
② 坏 mode 报错；③ `check_projection`/`smoke` 全绿（smoke 默认走 physical）；
④ 真实 ep0：`z_cam∈[0.345,0.618]m`，physical→23–40px、normalized→3–40px(std 9.46)，均随深度变化。

**注意：** 现有 `.cache/a2v_robotwin/ep0_dataset` 与 step-1000 LoRA 是用**旧半径**渲染/训练的。要用新半径需
**重跑 prepare**（默认即 physical）再重训；旧 LoRA 不要直接配新数据。

---

## 10. 半径修复的端到端重验（2026-06-13，PASS）+ 一个潜伏 LoRA 前缀 bug

**结论：§9 physical 半径 = 零质量回归，因果门干净通过。** 顺带挖出并修了一个 T2 引入的 LoRA 保存前缀 bug。

**重验流程：** physical 重渲染 `ep0_dataset_phys`（metadata 带 `radius_mode=physical/phys_radius_m=0.04/gripper_z_offset=0`，validate 通过）→ 1000 步 LoRA 过拟合 → real/none 因果对照。

**因果门（physical-训练 LoRA，逐帧度量 vs GT）：**
- REAL 控制：MAE-GT=**9.2**（T1 基线 10.8，更好）、运动能量 4.90≈GT 4.78、均值 213.7≈GT 214.9 → **复现 GT 轨迹**。
- NONE 控制：MAE-GT=30.1（远）、运动能量 **0.44**（≈静止）→ **不复现，臂不动**。
- real vs none 逐帧 MAE=31.7 → 控制信号因果驱动。蒙太奇目视确认（`.cache/a2v_robotwin/diag_fixed.png`）。
- 关键隔离对照：**旧 LoRA + 新 physical 数据 → MAE-GT=14.5**（复现良好）⇒ 证明 physical 数据本身无回归；
  新旧 `vace_video` 逐像素 MAE 仅 1.9/255（半径 23–40px vs 旧巧合 27–50px，差异极小），`reference`/`GT` 逐字节相同。

**潜伏 bug（已修，`a2v/train_a2v.py`）：** stock `diffsynth/diffusion/parsers.py:46` 把 `--remove_prefix_in_ckpt`
默认成 `"pipe.dit."`（**非 None**），导致 T2 的 `if args.remove_prefix_in_ckpt is None: = spec.vace_remove_prefix`
**永远不执行**（死代码）。后果：VACE LoRA 用 `pipe.dit.` 去 strip 对 `pipe.vace.*` key 无匹配 → 存成带
`pipe.vace.` 前缀的 key → `infer_a2v` 的 `pipe.load_lora(vace, …)` 静默匹配不到 → **LoRA 完全没生效 → 输出纯基模垃圾**
（不同 step 输出逐字节相同、real 比 none 还差，就是这个症状）。
- **为何 T1 没踩**：T1 时 run_overfit.sh 显式传了 `--remove_prefix_in_ckpt pipe.vace.`；T2 简化成 spec 驱动后暴露。
- **为何 T2 parity 漏检**：parity 复用 T1 旧 LoRA、**没重训**，从未走保存路径。
- **修法**：`parser.set_defaults(remove_prefix_in_ckpt=None)` 让"显式传入 vs 未传"可区分，spec 分支照常生效，
  无 spec 时回落 `"pipe.dit."`。三场景已模拟验证：spec无flag→`pipe.vace.`、spec+显式→`pipe.dit.`、无spec→`pipe.dit.`。

**当前可用产物（新半径）：**
- 数据集：`.cache/a2v_robotwin/ep0_dataset_phys`
- LoRA：`models/train/a2v_robotwin_ep0_lora_phys/step-1000.fixedkeys.safetensors`（手工剥前缀的可用版；
  原始 `step-1000.safetensors` 带坏前缀，勿用）。
- 因果对照视频：`gen_phys_fixed_real.mp4` / `gen_phys_fixed_none.mp4`。

**train_a2v 修复已实测（06-13）：** 用修好的代码跑 4 步重训，落盘 LoRA 的 300 个 key **零 `pipe.` 前缀、全为 `vace_blocks.*`**，
key-set 与已知可用旧 LoRA 命名空间逐一致 ⇒ **重训原生产出可加载 key，`*.fixedkeys` 这种手工剥前缀的临时产物以后不再需要**。
直接 `bash a2v/run_overfit.sh`（已默认指向 `ep0_dataset_phys`）即可。

> **教训给 T3+**：任何"抽象/重构"后，**对拍必须包含一次真·重训 + load_lora 往返**，不能只复用旧 ckpt 对拍推理，
> 否则保存路径的 seam（remove_prefix/key 命名）测不到。T3 的 `create_vace_from_dit` 退出门应显式加 LoRA 存取往返。

---

## 11. T3 接入 T2V-1.3B（2026-06-14，PASS）+ 两个 from-DiT 纠错

**结论：SEAM-1「从 DiT 造 VACE」成功。** 一个**不自带 VACE 分支**的基模（T2V-1.3B），从 DiT 克隆 + zero-init
造出 VACE 分支，**全参**过拟合后能因果驱动生成。过程中纠正了原计划/ABot 参考里的两个会让 from-DiT 彻底失效的坑。

**权重：** 官方 `Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors`（5.29GB，modelscope 下载；825 key，纯 DiT 无 `vace_*`）。
（注：T2V DiT == VACE-1.3B 的 DiT 基模；下载是为来源无歧义。下载/编辑期间踩过 `/vepfs` 100% 满 → 已腾挪。）

**新增/改动文件（均限 `a2v/`，不碰 `diffsynth/`）：**
- `base_spec.py`：加 `wan2.1-t2v-1.3b` 条（`has_pretrained_vace=False`、`first_frame_mode="none"`，dit_path 指官方 T2V）。
- `provision.py`：`create_vace_from_dit(pipe, spec)`（按 spec 形状建 `VaceWanModel`，逐层拷 DiT block 子模块，
  **只 zero-init `after_proj`**）；`ensure_vace` 幂等化（无 vace→造一次并打标记，避免二次覆盖）。
- `train_a2v.py`：`A2VWanTrainingModule` 子类，在 `switch_pipe_to_training_mode` 前 `ensure_vace`，使 from-DiT 分支
  在 freeze/挂训前就存在（否则 stock 在 `pipe.vace is None` 时静默跳过）。T2 路径幂等无变化。
- `check_t3_provision.py`：3 道门 —— ①形状(96/15层/拷贝核对) ②**零副作用**(真 `pipe()` 带控制 vs 不带控制逐位相同) ③(可选`--parity`)
  用 VACE-1.3B 自己的 DiT 造壳后 `load_state_dict(官方vace,strict)` 全等。**全绿。**
- `run_overfit_t2v.sh`：T3 **全参** vace 过拟合（`--trainable_models vace`，非 LoRA，见下纠错）；`run_overfit.sh` 参数化(SPEC/DATASET/OUT)。
- `infer_a2v.py`：`_load_vace_weights` 自动辨识 LoRA(含`lora_`键)→`load_lora` / 全参 vace→`load_state_dict`。

**★ 纠错 1（致命）：from-DiT vace 不能用 LoRA 训，必须全参。** 控制信号**只**经 `vace_patch_embedding`(Conv3d) 进、
经 `after_proj` 出；二者为 zero-init 且**都不是 LoRA target**(q,k,v,o,ffn)。LoRA 跑完它们仍为 0 ⇒ 控制零影响 ⇒
`gen(real)==gen(none)` **逐字节相同**(实测 MAE=0.00、均值同为 132.8)。ABot 也因此默认 `trainable="vace"`(全参)。

**★ 纠错 2（隐蔽）：`patch_embedding` 与 `before_proj` 不能同时 zero-init —— 会梯度死锁。** 原移植(同 ABot init_from_dit)
把 `patch_embedding`/`before_proj`/`after_proj` 全 zero。block0 做 `c = before_proj(c)+x`：若 `patch_embedding`=0 且
`before_proj.weight`=0，则①控制被切断②`patch_embedding` 拿不到梯度(因 `before_proj.weight`=0)、`before_proj.weight`
拿不到梯度(因其输入=`patch_embedding` 输出=0)，**互锁恒为 0**。实测全参训 100 步后 `patch_embedding` 仍全 0、
`after_proj` 已在动 ⇒ 控制永不进网络。**修法：只 zero-init `after_proj`**（它是唯一回主流的耦合，足够给"零副作用"），
`patch_embedding`/`before_proj` 留默认非零初始化保持控制通路可微。修后 100 步 `patch_embedding` 即非零(absmax~0.05)。

**退出门结果（physical 数据 `ep0_dataset_phys`，全参 1000 步）：**
- provision 三门：①形状 96/15层✓ ②零副作用 `pipe()` 带/不带控制 max_abs=0 逐位相同✓ ③结构 parity strict 全等✓。
- 因果门(逐帧 vs GT)：REAL MAE-GT=**20.9**、运动能量 5.23≈GT 4.78；NONE MAE-GT=35.4、运动能量 **0.68**(≈静止)；
  real vs none MAE=**41.6** ⇒ 控制因果驱动。(REAL 比 T1 的 9–11 略糙：from-scratch 全参 vace vs T1 预训练 vace+LoRA，预期。)
- 存取往返：全参 vace ckpt 439 key(`vace_blocks.*`+`vace_patch_embedding.*`，零 `pipe.`/lora 前缀)，
  infer `load_state_dict` missing=0/unexpected=0。

**当前可用产物：** ckpt `models/train/a2v_robotwin_ep0_vace_t2v/step-1000.safetensors`(全参 vace,1.47GB)；
视频 `.cache/a2v_robotwin/gen_t2v_{real,none}.mp4`。复现：`bash a2v/run_overfit_t2v.sh` 后用上方 infer 命令。
（注：同目录早期 `..._lora_t2v/` 是纠错 1 之前的 LoRA 死产物，**勿用**。）

**复现命令：**
```bash
PY=.venv/bin/python
$PY -m a2v.check_t3_provision --dataset .cache/a2v_robotwin/ep0_dataset_phys --parity   # 三门全绿
bash a2v/run_overfit_t2v.sh                                                              # 全参 1000 步
for c in real none; do $PY -m a2v.infer_a2v --base_spec wan2.1-t2v-1.3b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --lora models/train/a2v_robotwin_ep0_vace_t2v/step-1000.safetensors \
  --num_frames 121 --height 240 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_t2v_$c.mp4; done
```

> **教训给 T4+**：从 DiT 造分支时，"zero-init 求零副作用"只该作用在**回主流的那一层**(after_proj)，
> 任何位于**输入/控制通路上的层(patch_embedding、before_proj)若同时清零会梯度死锁**；且 from-DiT 新分支
> （输入/输出投影从零起）**必须全参训，LoRA 训不动**。退出门务必同时含「零副作用」与「过拟合后 real≠none」两面。

---

## 12. T4 接入 I2V-14B（2026-06-15，历史失败记录；已由 §13 修正）

**历史结论（已 superseded）**：本节记录旧 `lr=1e-4` run 的失败隔离。后来 §13.4 已证明真因是 lr 过高，
`run_overfit_i2v.sh` 默认改为 `lr=1e-5` 后 T4 PASS。保留本节用于解释坏 ckpt 和排除结构 bug；不要按本节的“换 fp32”作为当前下一步。

### 12.1 已下载权重（modelscope，均在 `models/Wan-AI/`）
- `Wan2.1-I2V-14B-480P/`：7 个 `diffusion_pytorch_model-0000N-of-00007.safetensors`（DiT，bf16 ~65GB on disk）
  + `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`（4.77GB，CLIP image encoder）+ `xlm-roberta-large/` tokenizer + index/config。
- `Wan2.1-VACE-14B/`：7 个分片（**warm-start 备用，暂未用**，见 §12.5）。
- **坑**：modelscope `--include` 对分片 glob 不稳（曾只下了 4 个 tokenizer 小文件就误报成功）；**改用 `--exclude`** 排除冗余的
  `models_t5_*.pth`/`Wan2.1_VAE.pth`（已有 converted safetensors）才正常拉分片。

### 12.2 代码改动（均限 `a2v/`，不碰 `diffsynth/`）
- `base_spec.py`：加 `dit_glob`（展开成 7 分片 list；`ModelConfig.path` 接受 `list[str]` → loader 自动合并成一个 DiT）+
  `image_encoder_path`（追加为额外 model_paths 项，自动识别为 `wan_video_image_encoder`）。注册 `wan2.1-i2v-14b-480p`
  （dim 5120/40层/40头/ffn 13824、`vace_layers=(0,5,…,35)=8`、`has_pretrained_vace=False`、`first_frame_mode="i2v_concat"`）。
- `infer_a2v.py`：按 `spec.first_frame_mode` 路由首帧 —— **i2v_concat → `input_image`（不传 vace_reference_image，二者是替代非叠加）**；
  其余基模仍走 vace_reference。
- `check_t3_provision.py`：gate_shape 去掉硬编码的 1.3B 数字（改 spec 驱动）；**gate_zero_side_effect 对 i2v 必须传 `input_image`**
  （I2V DiT in_dim=36，无首帧则 16ch latent 进不了 36ch patch_embedding 会崩）。
- `run_overfit_i2v.sh`（**新**）：T4 全参 vace 过拟合配方。
- `check_t4_load.py`（**新**）：14B 本地加载冒烟（7分片合并 / CLIP .pth 直载 / in_dim=36 / create_vace_from_dit 造 8 层）。
- `causal_metrics.py`（**新，可复用**）：因果门度量（MAE-GT / motion / real-vs-none，用 imageio ffmpeg 读 mp4；**pyav 没装，勿用 v3 pyav**）。
- `create_vace_from_dit` / `provision.py` **未改**（已通用，14B shape 全走 spec）。

### 12.3 已 PASS 的基础设施（可信）
- **加载**：`check_t4_load` 全绿 —— 7 分片自动合并成 40-block DiT；**CLIP `.pth` 直接加载成 `WanImageEncoder`（无需转 safetensors）**；
  in_dim=36 / has_image_input / require_clip / require_vae 全 True；create_vace_from_dit 造出 8 层 vace（vace_in_dim=96）、after_proj=0、
  patch_embedding 存活(0.051)。
- **provision 两门**（`check_t3_provision --base_spec wan2.1-i2v-14b-480p`）：①形状 96/8层/5120/13824、权重拷贝、after_proj=0 ✓
  ②**零副作用**：真 `pipe()` 带 `input_image`、切换 vace 控制 → 逐位相同 max_abs=0 ✓。
- **SEAM-4 训练自动取首帧**：stock `parse_extra_inputs` 把 `input_image → data["video"][0]`（GT 首帧），故训练只需
  `--extra_inputs vace_video,input_image` + `--data_file_keys video,vace_video`，**不用改 train_a2v.py**。
- **14B 显存**：纯推理/加载 ~44GB；8-bit Adam 全参训练峰值 ~75GB（单卡 A100-80GB 放得下）。

### 12.4 因果门为什么 FAIL（逐层隔离，关键）
8-bit Adam 单卡跑完 1000 步（11.4s/it，~3.2h，`models/train/a2v_robotwin_ep0_vace_i2v/step-{100..1000}.safetensors`，每个 6.9GB）。
infer real/none **输出全是彩色噪声**。隔离实验（脚本临时建已删，结论记此）：

| 测试 | motion | 目视 |
| --- | --- | --- |
| base i2v（**不挂 vace**） | 53.7 | **连贯**（认得出 hammer/block，i2v 自由幻想大幅运动） |
| 训练后 vace，**vace_scale=0.0** | 6.7 | **连贯**（=base） |
| 训练后 vace，vace_scale=0.3 | 14.2 | **噪声** |
| 训练后 vace，vace_scale=1.0 | 8.2 | **噪声** |
| step-100 / step-300 早期 ckpt | 22 / 14 | **也噪声**（早期就坏） |

- vace 的**结构、上下文（确认收到 i2v 的 `context=[clip(257)⊕text]`，`wan_video.py:1525-1527` 调 `vace(x,vace_context,context,…)`）、
  序列对齐、权重幅度（absmax 3.2、无 NaN/inf）全部正常** → **不是结构/加载 bug**。
- 但训练出的 vace **方向性破坏**输出：scale 一旦非零（连 0.3）就崩，且 **step-100 早期就坏**；CFG 不是因（cfg=1 也噪声）。
- 训练 loss **非单调、中途飙到 2.1**（单样本过拟合应单调下降）= **训练不稳定**。

**嫌疑（按可能性）**：①**8-bit Adam 不稳**（T3 用的是 fp32 AdamW + 1.3B，干净收敛；为省显存换了 8-bit 是 T4 唯一的优化器变量）；
②**去掉了 `vace_reference_image`**（T3 训练是带的，T4 i2v 我换成 input_image 且没保留 reference）；③lr=1e-4 对 14B from-scratch 偏高。

### 12.5 当时的下一步（历史；§13.4 已证明不必 fp32）
**当时的判断**是用 T3 验证过的 fp32 AdamW 重训，一次只回退一两个变量。后来 lr 扫描已修正该判断：低 lr + 8-bit Adam 可过门。以下仅保留为 14B 多卡/显存参考。fp32 优化器（~33.6GB）单卡放不下 →
- **路 A（最快，需用户扩多卡）**：多卡 DeepSpeed ZeRO-2 分片优化器到 CPU/多卡（官方 `examples/wanvideo/model_training/full/accelerate_config_14B.yaml`
  是 zero2+optimizer-offload 模板，改 `num_processes`），去掉 `--customized_optimizer`（回 fp32 AdamW）。需验证 `A2VWanTrainingModule` 子类
  在 DeepSpeed 下 from-DiT vace 构建是否兼容（vace 在 `__init__`→`switch_pipe_to_training_mode` 里建，先于 accelerate.prepare，应 OK）。
- **路 B（不扩卡，慢）**：单卡 `--enable_model_cpu_offload`（**不要** `--enable_optimizer_cpu_offload`，否则可训 vace 的 Conv3d 被搬 CPU →
  backward "two devices" 错）+ fp32 AdamW。~21.7s/it ≈ 6h。先验证是不是 8-bit Adam 的锅。
- **正交变量回退**：把 `vace_reference_image` 加回 `--extra_inputs`/`--data_file_keys`（与 input_image 并存，或二选一各试一版），隔离 §12.4 嫌疑②。
- 可考虑 lr 降到 5e-5（官方 VACE-14B 配方用 5e-5）。

**warm-start（暂缓）**：VACE-14B 的 vace 分支是 **T2V-14B 基础（无图像 cross-attn k_img/v_img）**，而我们从 I2V DiT 克隆的 shell **带**图像
cross-attn → VACE-14B **不能 strict-load** 进 I2V shell，只能非严格部分加载。所以 warm-start 是"又一个新变量"，留到 from-scratch 跑通后再单独验证
（或用于多 episode 阶段）。

### 12.6 产物与复现
- **数据集**（复用 T1/T3）：`.cache/a2v_robotwin/ep0_dataset_phys`（physical 半径）。
- **坏 ckpt（勿用）**：`models/train/a2v_robotwin_ep0_vace_i2v/step-*.safetensors`（8-bit Adam，破坏性 vace）。
- **诊断视频**：`.cache/a2v_robotwin/gen_i2v_{real,none}.mp4`（噪声）、`gen_i2v_BASE.mp4`（无 vace 基础，连贯）、`diag_*.png` 蒙太奇。
- **三道闸门（基础设施）复现**：
```bash
PY=.venv/bin/python
$PY -m a2v.check_t4_load                                                         # 加载冒烟全绿
$PY -m a2v.check_t3_provision --base_spec wan2.1-i2v-14b-480p \
   --dataset .cache/a2v_robotwin/ep0_dataset_phys                               # provision 两门绿
# 历史失败复现（旧 lr=1e-4 目录勿用）；当前 run_overfit_i2v.sh 默认 lr=1e-5，会复现 §13.4 的 PASS：
bash a2v/run_overfit_i2v.sh
# 因果门度量：
for c in real none; do $PY -m a2v.infer_a2v --base_spec wan2.1-i2v-14b-480p \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --lora models/train/a2v_robotwin_ep0_vace_i2v/step-1000.safetensors \
  --num_frames 121 --height 240 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_i2v_$c.mp4; done
$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --real .cache/a2v_robotwin/gen_i2v_real.mp4 --none .cache/a2v_robotwin/gen_i2v_none.mp4 --num_frames 121
```

> **教训给重训（已按 §13 修正）**：①from-scratch 14B vace 对 lr 很敏感，`1e-4` 会发散，`1e-5` 已实测过门；②单样本过拟合若 loss 非单调上行
> 就是训练有问题，别等跑完再看；③因果门对 i2v/ti2v 而言"NONE 静止"判据**不成立**，判据应是"REAL 复现 GT 轨迹 且 real≠none/shuffle 且 REAL 更近 GT"。
> bitsandbytes 0.49.2 已装在 `.venv`，8-bit Adam + 低 lr 单卡可用。

---

## 13. T4 调试（2026-06-16）：2 卡 ZeRO 环境墙 + lr 修复真因，T4 PASS

**结论：本日先撞到 2 卡 fp32 ZeRO 环境墙，随后比对 ABot 并做低 lr 单卡 8-bit 重训，证明 §12.4 因果门 FAIL 的真因是 `lr=1e-4` 过高，
不是 8-bit Adam 精度、也不是缺 vace_reference。`lr=1e-5` 胜出，T4 定稿。**

### 13.1 2 卡 ZeRO-2 跑 fp32 AdamW 的两次失败（环境，非方法）
环境：`/vepfs/.../DiffSynth-Studio` 现有 **2×A100-80GB**（不是 8 卡）。已 `pip install deepspeed==0.19.1` 到 `.venv`，gcc/g++ 在，但 **无 CUDA toolkit（只有 runtime，无 `/usr/local/cuda*/bin/nvcc`、无 cuda 头文件）**。
- **坑 0（deepspeed 导入即崩）**：`import deepspeed` 在 `nvcc -V` 版本探测处 FileNotFoundError。**解法：造 stub nvcc**（`a2v/.fakecuda/bin/nvcc` 只回应 `-V` 版本串），设 `CUDA_HOME=a2v/.fakecuda`。ZeRO-2 配 client torch 优化器时不真编译 op，stub 够用。
- **失败①（no-offload OOM）**：`offload_optimizer_device: none`。前向/反向过了（DiffSynth 强制开梯度检查点），**卡在优化器步 OOM**（每卡 77.9GB 占用、还差 6.46GB）。根因：**ZeRO-2 不分片参数**，~44GB 冻结 DiT/T5/CLIP/VAE 每卡都复制一份，留给 fp32 优化器分片(33.6GB/2=16.8GB/卡)的空间不够。
- **失败②（optimizer CPU offload → cpu_adam）**：`offload_optimizer_device: cpu`。accelerate 的 `map_pytorch_optim_to_deepspeed` 在 offload 时**强制**把 client AdamW 转成 `DeepSpeedCPUAdam` → JIT 编译 `cpu_adam` op → 先报 `Ninja is required`，即便装 ninja 后仍需 CUDA 头文件编译 → **本机无 toolkit，死路**。（no-offload 不会触发此转换，已由失败①证明。）
- **判定**：本机想跑 fp32 多卡，要么 (a) 上 **ZeRO-3 no-offload** 分片参数（不触发 cpu_adam，但 DiffSynth 存盘需验证 zero3 gather）；要么 (b) **encoded-cache 砍掉 T5/CLIP/VAE**（见 §13.3 第 4 点）把冻结栈降到 ~28GB → 2 卡 ZeRO-2 no-offload 就放得下 fp32；要么 (c) 单卡 `--enable_model_cpu_offload`+fp32（§12.5 路 B，最稳但 ~6h）。
- **新增文件（今天）**：`a2v/accelerate_2gpu_zero2.yaml`（2 进程 ZeRO-2 模板）、`a2v/run_overfit_i2v_fp32.sh`（fp32 版配方，含 stub-nvcc 接线）、`a2v/.fakecuda/bin/nvcc`（stub）。**坏 8-bit ckpt 仍在 `models/train/a2v_robotwin_ep0_vace_i2v/`（勿用）；fp32 输出目录设为 `..._vace_i2v_fp32` 不覆盖。**

### 13.2 比对 ABot（`/vepfs/wangshilong/code/ABot-PhysWorld`）——同构路线的实际配方
ABot 就是「I2V-14B + 动作轨迹图经 VACE 注入、full-param vace、from-DiT 初始化」，与我们 T4 **完全同构**，其训练脚本是最直接参考。核实到的关键数字（非 README，是真文件）：

| 项 | ABot 实际值 | 我们 T4(8-bit run) | 出处 |
| --- | --- | --- | --- |
| **A2V 训练 lr** | **5e-6** | **1e-4（高 20×）** | `training/run_train_a2v.sh:53` |
| 文本条件 | **关**（`DISABLE_TEXT_CONDITION=true`） | 开 | `run_train_a2v.sh:69` |
| 优化器 | 普通 AdamW（无 8-bit/无 customized） | 8-bit Adam | `run_train_a2v.sh`/zero2 yaml |
| ZeRO | ZeRO-2 **no-offload，8 卡** | 单卡 8-bit | `training/accelerate_config_zero2.yaml`（`num_processes:8`） |
| trainable | `vace`（全参） | `vace`（全参）✓一致 | `run_train_a2v.sh:74` |
| vace_layers/in_dim | (0,5,…,35)/96 ✓一致 | 同 | `wan_video_vace.py` converter |
| 时间采样 | chunk=121, stride 6-6 | 121 | `run_train_a2v.sh:60-64` |

**两条最可能修好因果门的零成本变量**：①**lr 1e-4→5e-6**（直击 §12.4 loss 非单调飙 2.1 的嫌疑③；ABot 全参 vace 就用 5e-6）；②**关文本条件**（让 action 主导，对齐 ABot 纯动作世界模型，近 §12.4 嫌疑②）。**这两个甚至可先在原单卡 8-bit 路线上快速重试**——若 loss 转单调、因果门过，说明根本不是优化器精度，省掉整个 fp32 多卡折腾。

### 13.3 ABot 中对改造其余有用点（落地优先级）
1. **Encoded cache 两阶段**（`training/train.py:75,206,347`）：存 `input_latents/context/y/clip_feature`，二阶段 `skip_vae/skip_text_encoder/skip_image_encoder`。**既加速多 epoch，又是 14B 省显存的钥匙**——跳过 T5/CLIP/VAE 加载后冻结栈 44→~28GB，**2 卡 ZeRO-2 no-offload 即可跑 fp32**，绕开 cpu_adam。
2. **⚠ 我们已领先 ABot 一点，别回退**：ABot `init_from_dit`（`models/wan_video_vace.py:108`，`zero_init_extra`）把 `vace_patch_embedding`+`before_proj`+`after_proj` **全部**零初始化——正是我们 §11 实测的**梯度死锁**陷阱。我们 `provision.py` 已修为**只零 after_proj**，正确，勿因"对齐参考"改回去。
3. **Resume+RNG 恢复 / 均匀·chunk·stride 采样 / cache 命中跳过读视频 / realtime_text_encode**（`trainers/utils.py`、`unified_dataset.py`）：多 episode 泛化阶段直接移植。
4. **9 通道动作编码**（3 轨迹 RGB + 6 Plücker 射线，`utils/action_utils.py:99`）：`A2V_action_encoding_redesign.md` 编码升级的现成参考（当前用 3 通道）。
5. **EZS-Bench**（8 视频质量指标 + 机器人 VQA 域分）：量化评测，替代目视蒙太奇。
6. **DPO LoRA 物理偏好**（`train_dpo.py`，reference 用 `disable_adapter()` 省一份模型）：T6 之后精修。
7. **避开 ABot 的坑**：encoded-cache 文件名仅由视频路径 MD5（跨数据集同相对路径会错误复用）；`sed -i` 改 accelerate yaml（并发实验互覆盖——我们已用独立 config 文件，勿学）。

### 13.4 ★ 结果（2026-06-16）：lr 是真因，lr1e5 胜出，T4 PASS
**只把 `--learning_rate 1e-4 → 5e-6`（其余与原失败 run 完全一致：单卡、8-bit Adam、带文本），因果门立刻从"纯噪声"变正常。**
step-100 已非噪声，step-400 REAL MAE-GT=24.95 / motion=5.88，证明 §12.4 的 FAIL 根因是 lr 过高导致发散，**不是 8-bit Adam 优化器精度，也不是 vace_reference**。fp32 多卡折腾（§13.1）不再必要。

随后完成三组单样本过拟合 1000 步 lr 扫描（8-bit Adam，全参 vace，带文本，seed=0）。i2v 判据按 §12.6：REAL MAE-GT 最低、REAL motion 接近 GT、real<none(MAE-GT)、real≠none；NONE 不要求静止。

| run | lr | REAL MAE-GT | REAL motion (GT=4.78) | NONE MAE-GT | NONE motion | real-vs-none | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr5e6 | 5e-6 | 5.35 | 5.04 (×1.05) | 28.59 | 2.99 | 29.83 | PASS |
| **lr1e5** | **1e-5** | **4.83** | **4.94 (×1.03)** | **36.04** | 4.35 | **36.79** | **WINNER** |
| lr2e5 | 2e-5 | 5.19 | 4.97 (×1.04) | 37.80 | 1.16 | 38.62 | PASS |

**定稿**：`run_overfit_i2v.sh` 默认学习率已改为 **1e-5**（仍可用 `LR=...` 覆盖）。T4 的最终可用 ckpt 是
`models/train/a2v_robotwin_ep0_vace_i2v_lr1e5/step-1000.safetensors`；视频是
`.cache/a2v_robotwin/gen_i2v_lr1e5_s1000_{real,none}.mp4`。

**仍勿用**：`models/train/a2v_robotwin_ep0_vace_i2v/step-*`（旧 1e-4 发散噪声 ckpt）。用户同意后已卸载 deepspeed，
`diffsynth.core.gradient.gradient_checkpoint._HAS_DEEPSPEED=False`，现在 a2v run/infer 不再需要 stub-nvcc；`a2v/.fakecuda/bin/nvcc` 仅作为日后重装 deepspeed 的备用。

### 13.5 下一步（按性价比排序，T5 后更新）
1. **T5 已完成**（见 §14）。下一步推进 **T6 Wan2.2-I2V-A14B**：SEAM-5 双专家 MoE（`vace`/`vace2`、boundary=0.875）。
2. （可选）T1/T3/T4/T5 多 episode 泛化：adapter 已支持 `--episodes_range`/默认全 episode；prepare 支持多行 manifest。
3. （可选）若日后做 14B 多 episode 全量训练，8 卡数据并行才真正提速；单样本过拟合 8 卡不加速（同梯度冗余算 8 次）。encoded-cache（§13.3 第 1 点）届时既省显存又免重复编码。
4. 因果门判据沿用 §12.6 教训③（i2v/ti2v：REAL 复现 GT 且 real≠none/shuffle 且 REAL 更近 GT；负对照不要求静止）。


---

## 14. T5 接入 TI2V-5B（2026-06-16，PASS）

**结论：SEAM-2 新 VAE 路径打通。** Wan2.2-TI2V-5B 使用 Wan2.2 VAE，`z_dim=48`、空间压缩 16×，
所以 `vace_in_dim = 2*48 + 16*16 = 352`、`mask_pq=16`。load smoke、provision 零副作用门、1000 步全参 vace 过拟合和
real/none/shuffle 因果门均通过。

### 14.1 代码改动
- `base_spec.py`：新增 `wan2.2-ti2v-5b`。DiT 3 分片，Wan2.2 converted VAE，复用 umt5 encoder/tokenizer；30 层、dim 3072、heads 24、ffn 14336；`vace_layers=(0,2,...,28)`；`first_frame_mode="ti2v_fused"`。
- `train_a2v.py`：dataset operator 的空间/时间整除因子改为按 spec 派生。T5 因 `vae_spatial_factor=16` 且 patch size=2，训练数据需 H/W 可被 32 整除。
- `infer_a2v.py` / `check_t3_provision.py`：`ti2v_fused` 与 `i2v_concat` 一样走 `input_image`，不走 `vace_reference_image`。
- `data/validate.py`：新增 `--base_spec`，按 spec 校验空间/时间整除。
- `check_t5_load.py`：TI2V-5B 加载冒烟与 VACE 壳检查。
- `run_overfit_ti2v.sh`：T5 单样本全参 vace 过拟合配方，默认 `LR=1e-5`、`HEIGHT=256`、`WIDTH=320`、`FRAMES=121`。

### 14.2 权重与数据
- DiT：`models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors`。
- VAE：`models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`。
- T5 encoder：`models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors`。
- 数据集：`.cache/a2v_robotwin/ep0_dataset_phys_256x320`。原 240×320 高度不能被 32 整除，T5 用 256×320 重渲染；validate 通过。

### 14.3 退出门结果
- `check_t5_load`：DiT 30 blocks、`in_dim=48`、fused VAE latent=True、separated timestep=True；WanVideoVAE38 `z_dim=48`、upsampling=16；`create_vace_from_dit` 造 15 层 VACE，`vace_in_dim=352`，`after_proj` 全零，`patch_embedding` 存活；`ParamWanVideoUnit_VACE(mask_pq=16)` 安装成功。
- `check_t3_provision --base_spec wan2.2-ti2v-5b`：形状门 352/15 层/3072/14336 通过；权重拷贝通过；零副作用门 `max_abs_pixel_diff=0`、`bit_identical=True`。
- 训练：`CUDA_VISIBLE_DEVICES=0 bash a2v/run_overfit_ti2v.sh` 完成 1000 steps，约 3.1s/it，峰值约 39GB，ckpt 在 `models/train/a2v_robotwin_ep0_vace_ti2v/step-{100..1000}.safetensors`。
- 推理：`step-1000.safetensors` 全参 vace 439 keys，加载 `missing=0 unexpected=0`。

### 14.4 因果门指标
| video | MAE-GT | motion | mean | frames |
| --- | --- | --- | --- | --- |
| GT | -- | 4.74 | 214.93 | 121 |
| REAL | **5.27** | 5.38 | 210.83 | 121 |
| NONE | 15.62 | 4.07 | 212.38 | 121 |
| SHUFFLE | 14.91 | 4.69 | -- | 121 |

- `real-vs-none MAE = 16.02`。
- `real-vs-shuffle MAE = 13.44`。
- 判定：T5 PASS。和 I2V 一样，强首帧模型在 NONE/SHUFFLE 下仍可能自由运动；判据不是“负对照静止”，而是 REAL 明显更贴近 GT 且与负对照不同。

### 14.5 复现命令
```bash
cd /vepfs/wangshilong/code/DiffSynth-Studio
PY=.venv/bin/python

$PY -m a2v.data.prepare --manifest .cache/a2v_robotwin/ep0_work/manifest.jsonl \
  --output_dir .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --height 256 --width 320 --num_frames 121 --resize_mode stretch \
  --gripper_z_offset 0 --radius_mode physical --write_overlay

$PY -m a2v.data.validate --base_spec wan2.2-ti2v-5b \
  --dataset_base_path .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --dataset_metadata_path .cache/a2v_robotwin/ep0_dataset_phys_256x320/metadata.jsonl \
  --height 256 --width 320 --num_frames 121

$PY -m a2v.check_t5_load
$PY -m a2v.check_t3_provision --base_spec wan2.2-ti2v-5b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --height 256 --width 320 --num_frames 13 --steps 4

CUDA_VISIBLE_DEVICES=<g> bash a2v/run_overfit_ti2v.sh

for c in real none shuffle; do $PY -m a2v.infer_a2v --base_spec wan2.2-ti2v-5b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --lora models/train/a2v_robotwin_ep0_vace_ti2v/step-1000.safetensors \
  --num_frames 121 --height 256 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_ti2v_s1000_$c.mp4; done

$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --real .cache/a2v_robotwin/gen_ti2v_s1000_real.mp4 \
  --none .cache/a2v_robotwin/gen_ti2v_s1000_none.mp4 \
  --num_frames 121 --height 256 --width 320
```

### 14.6 环境备注
本会话曾为 DeepSpeed ZeRO 试验安装 `deepspeed==0.19.1`，它会在导入时探测 `nvcc`，本机无 CUDA toolkit 时需 stub-nvcc。
用户已同意恢复旧环境，现已执行 `.venv/bin/python -m pip uninstall -y deepspeed`，并验证 `_HAS_DEEPSPEED=False`。
因此常规 a2v 训练/推理不再需要 `CUDA_HOME=a2v/.fakecuda`；stub 文件保留只为未来重新安装 deepspeed 时备用。

### 14.7 下一步
推进 **T6 Wan2.2-I2V-A14B**（SEAM-5 双专家 MoE），或先做 T4/T5 多 episode 泛化。T6 建议先只做加载/provision/零副作用门，确认双专家 `vace`/`vace2`、boundary 和首帧路由，再跑过拟合因果门。

### 14.8 本轮最终复查上下文
- 环境：已卸载 deepspeed；`.venv/bin/python -m pip show deepspeed` 返回 package not found；`from diffsynth.core.gradient.gradient_checkpoint import _HAS_DEEPSPEED` 输出 `False`。常规 a2v 命令不再需要 `CUDA_HOME=a2v/.fakecuda`。
- 静态验证：`.venv/bin/python -m py_compile a2v/base_spec.py a2v/train_a2v.py a2v/infer_a2v.py a2v/check_t3_provision.py a2v/check_t5_load.py a2v/data/validate.py` 通过；`bash -n a2v/run_overfit_i2v.sh` 和 `bash -n a2v/run_overfit_ti2v.sh` 通过。
- 资源：`nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader` 显示 8 张 A100-SXM4-80GB 均约 4 MiB 占用，训练/推理进程已结束。
- 自动记忆：`find /vepfs/wangshilong/code -maxdepth 5 -type f \( -name 'MEMORY.md' -o -name 'a2v-robotwin-convention.md' \)` 未找到文件；只找到本文 `a2v/A2V_HANDOFF.md`。若后续会话需要“记忆同步”，先确认这些文件是否在别的路径或由外部系统托管。
- 工作区：本轮代码和文档仍未提交；`a2v/A2V_HANDOFF.md`、`check_t5_load.py`、`run_overfit_ti2v.sh`、`causal_metrics.py` 等为当前工作区产物。继续前先看 `git status --short`，不要误删未跟踪文件。
- 工具注意：当前 Codex 默认 sandbox 的 `apply_patch`/普通读写曾因 `bwrap: No permissions to create a new namespace` 失败；实际文件位于可写根下，必要时使用已授权的外部 shell/Python 做定点替换，并在替换脚本里检查锚点。
