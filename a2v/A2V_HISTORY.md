# A2V 历史与调试记录（附录）

> 本文是 `A2V_HANDOFF.md` 的历史附录：按时间线 + 分 Track 记录各阶段的调试过程、纠错与教训。
> **当前状态 / 接手清单 / 速查在 `A2V_HANDOFF.md`。** 下面的 §7–§14 沿用原 handoff 的章节编号，便于交叉引用。

最后更新：2026-06-17（从 `A2V_HANDOFF.md` 拆出，历史内容逐字保留）。

## 进展时间线（banner）

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
>
> **2026-06-22 进展（T6 Wan2.2-I2V-A14B PASS，五基模收官）**：接入最后一个基模——双专家 MoE（SEAM-5）。
> DiffSynth 原生支持双专家（`dit/dit2`+`vace/vace2`，推理 `switch_DiT_boundary=0.875` 自动切换）；沿用 from-DiT，
> **两条独立单专家作业**分带训练（high `[0,0.358]`、low `[0.358,1]`，官方配方）。新 SEAM-4 变体 `i2v_vae`（input_image
> 走 VAE-concat，无 CLIP）。只下载两套 DiT 专家分片（107GB，VAE/T5/tokenizer 复用）。load/provision/双专家过拟合/合并因果门全过：
> **REAL MAE-GT=3.14（五基模最佳）**、motion 4.97≈GT 4.78、NONE 21.48/SHUFFLE 20.48，real≪负对照、切换不撕裂。详见 §18。

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
- `check_parity.py`：**T2 对拍闸门**。

**对拍闸门结果（`python -m a2v.check_parity`）：**
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
- `check_provision.py`：3 道门 —— ①形状(96/15层/拷贝核对) ②**零副作用**(真 `pipe()` 带控制 vs 不带控制逐位相同) ③(可选`--parity`)
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
$PY -m a2v.check_provision --dataset .cache/a2v_robotwin/ep0_dataset_phys --parity   # 三门全绿
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
- `check_provision.py`：gate_shape 去掉硬编码的 1.3B 数字（改 spec 驱动）；**gate_zero_side_effect 对 i2v 必须传 `input_image`**
  （I2V DiT in_dim=36，无首帧则 16ch latent 进不了 36ch patch_embedding 会崩）。
- `run_overfit_i2v.sh`（**新**）：T4 全参 vace 过拟合配方。
- `check_t4_load.py`（**新**）：14B 本地加载冒烟（7分片合并 / CLIP .pth 直载 / in_dim=36 / create_vace_from_dit 造 8 层）。
- `causal_metrics.py`（**新，可复用**）：因果门度量（MAE-GT / motion / real-vs-none，用 imageio ffmpeg 读 mp4；**pyav 没装，勿用 v3 pyav**）。
- `create_vace_from_dit` / `provision.py` **未改**（已通用，14B shape 全走 spec）。

### 12.3 已 PASS 的基础设施（可信）
- **加载**：`check_t4_load` 全绿 —— 7 分片自动合并成 40-block DiT；**CLIP `.pth` 直接加载成 `WanImageEncoder`（无需转 safetensors）**；
  in_dim=36 / has_image_input / require_clip / require_vae 全 True；create_vace_from_dit 造出 8 层 vace（vace_in_dim=96）、after_proj=0、
  patch_embedding 存活(0.051)。
- **provision 两门**（`check_provision --base_spec wan2.1-i2v-14b-480p`）：①形状 96/8层/5120/13824、权重拷贝、after_proj=0 ✓
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
$PY -m a2v.check_provision --base_spec wan2.1-i2v-14b-480p \
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
- `infer_a2v.py` / `check_provision.py`：`ti2v_fused` 与 `i2v_concat` 一样走 `input_image`，不走 `vace_reference_image`。
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
- `check_provision --base_spec wan2.2-ti2v-5b`：形状门 352/15 层/3072/14336 通过；权重拷贝通过；零副作用门 `max_abs_pixel_diff=0`、`bit_identical=True`。
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
$PY -m a2v.check_provision --base_spec wan2.2-ti2v-5b \
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
- 静态验证：`.venv/bin/python -m py_compile a2v/base_spec.py a2v/train_a2v.py a2v/infer_a2v.py a2v/check_provision.py a2v/check_t5_load.py a2v/data/validate.py` 通过；`bash -n a2v/run_overfit_i2v.sh` 和 `bash -n a2v/run_overfit_ti2v.sh` 通过。
- 资源：`nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader` 显示 8 张 A100-SXM4-80GB 均约 4 MiB 占用，训练/推理进程已结束。
- 自动记忆：`find /vepfs/wangshilong/code -maxdepth 5 -type f \( -name 'MEMORY.md' -o -name 'a2v-robotwin-convention.md' \)` 未找到文件；只找到本文 `a2v/A2V_HANDOFF.md`。若后续会话需要“记忆同步”，先确认这些文件是否在别的路径或由外部系统托管。
- 工作区：本轮代码和文档仍未提交；`a2v/A2V_HANDOFF.md`、`check_t5_load.py`、`run_overfit_ti2v.sh`、`causal_metrics.py` 等为当前工作区产物。继续前先看 `git status --short`，不要误删未跟踪文件。
- 工具注意：某些 agent sandbox 下 `apply_patch`/普通读写曾因 `bwrap: No permissions to create a new namespace` 失败；实际文件位于可写根下，必要时使用已授权的外部 shell/Python 做定点替换，并在替换脚本里检查锚点。

---

## 15. Multi-episode generalization (2026-06-18, VACE-1.3B LoRA, PASS)

**结论：VACE-1.3B LoRA 的正式多 episode 泛化门通过。** 使用 RoboTwin `beat_block_hammer/aloha-agilex_clean_50`，train=ep0-39，held-out=ep40-49，105 帧，240x320。同一个 final LoRA 在 held-out 10 个未见 episode 上满足主判据：REAL 明显比 NONE 更接近 GT，且 10/10 episode 均 real<none；REAL motion 与 GT 同量级。

### 15.1 代码与数据
- `run_overfit.sh` 增加 `REPEAT` / `EPOCHS` / `DRY_RUN` env override；默认仍是 `REPEAT=100`、`EPOCHS=10`、`FRAMES=121`，不破坏单样本过拟合复现。
- `infer_a2v.py` 抽出 `build_pipe_with_vace`、`first_frame_kwargs`、`generate_one`，供单行 CLI 与多行评测复用；原 CLI 参数保持不变。
- 新增 `a2v/eval_multiep.py`：一次加载模型，按 metadata 行循环生成 `real,none`，保存 mp4，写 `metrics.json`，打印逐 episode 表和 SUMMARY。
- 数据集已 validate：`.cache/a2v_robotwin/ep_train40_phys` 40 行、`.cache/a2v_robotwin/ep_heldout10_phys` 10 行；每行 105 帧、240x320。

### 15.2 训练配方
```bash
FRAMES=105 REPEAT=1 EPOCHS=120 \
OUT=models/train/a2v_robotwin_train40_vace1p3b_lora \
DATASET=.cache/a2v_robotwin/ep_train40_phys \
CUDA_VISIBLE_DEVICES=0 bash a2v/run_overfit.sh wan2.1-vace-1.3b
```
- 步数：40 rows * 120 epochs = 4800 steps；最终 ckpt `models/train/a2v_robotwin_train40_vace1p3b_lora/step-4800.safetensors`，大小 43,781,016 bytes。
- TensorBoard loss：4800 条；step1=0.01057，step1000=0.00974，step2400=0.02093，step4800=0.01559；全程有波动但未发散，最终以 causal eval gate 为准。

### 15.3 Held-out 10 ep 结果
```text
rows=10 mean_real_mae=6.25 mean_none_mae=26.83 real_lt_none=10/10
mean_real_motion=5.18 mean_none_motion=0.57 mean_gt_motion=5.01
mean_motion_ratio=1.03 mean_real_vs_none=26.68 verdict=PASS
```
产物：`.cache/a2v_robotwin/eval_heldout/row*_ep*_real.mp4`、`row*_ep*_none.mp4`、`metrics.json`。

### 15.4 Train subset 对照
```text
rows=5 mean_real_mae=6.35 mean_none_mae=29.89 real_lt_none=5/5
mean_real_motion=5.18 mean_none_motion=0.74 mean_gt_motion=5.06
mean_motion_ratio=1.02 mean_real_vs_none=30.30 verdict=PASS
```
产物：`.cache/a2v_robotwin/eval_train/row*_ep*_real.mp4`、`row*_ep*_none.mp4`、`metrics.json`。

### 15.5 复现命令
```bash
PY=.venv/bin/python
for d in ep_train40_phys ep_heldout10_phys; do
  $PY -m a2v.data.validate --base_spec wan2.1-vace-1.3b \
    --dataset_base_path .cache/a2v_robotwin/$d \
    --dataset_metadata_path .cache/a2v_robotwin/$d/metadata.jsonl \
    --height 240 --width 320 --num_frames 105 --max_items 50
done

CK=models/train/a2v_robotwin_train40_vace1p3b_lora/step-4800.safetensors
$PY -m a2v.eval_multiep --base_spec wan2.1-vace-1.3b --lora $CK \
  --dataset .cache/a2v_robotwin/ep_heldout10_phys --num_frames 105 --height 240 --width 320 \
  --controls real,none --output_dir .cache/a2v_robotwin/eval_heldout
$PY -m a2v.eval_multiep --base_spec wan2.1-vace-1.3b --lora $CK \
  --dataset .cache/a2v_robotwin/ep_train40_phys --rows 0,1,2,3,4 \
  --num_frames 105 --height 240 --width 320 --controls real,none \
  --output_dir .cache/a2v_robotwin/eval_train
```

### 15.6 下一步
同份 240x320 / 105-frame train40+heldout10 数据直接推进 **I2V-14B 多 episode**。沿用 `eval_multiep.py`；训练侧注意 14B 用 8-bit Adam、`LR=1e-5` 起步，必要时再引入 encoded-cache 提速和降显存。

---

## 16. Multi-episode generalization (2026-06-21, I2V-14B, 8-GPU DDP, PASS)

**结论：I2V-14B（生产基模）多 episode 泛化门通过，且强于 1.3B。** 复用同一份 train40/heldout10（240×320, 105 帧）数据，
8 卡 DDP 训练 from-DiT 全参 vace，held-out 10 个未见 episode **10/10 real<none**、REAL MAE-GT 4.25 ≪ NONE 30.09、
motion ratio 1.06；held-out(4.25) 与 train(4.76) 几乎无 gap ⇒ 真泛化非记忆。

### 16.1 代码硬化（eval_multiep）+ 多卡（run_overfit.sh）
- `eval_multiep.py`：**A1 每行 try/except**（单行失败记 `failures`、从聚合剔除、不中断整轮，metrics.json 加 `failures`/`failed`）；
  **A2 motion_ratio nan 防御**（聚合仅用有限值；全静止导致 nan 时跳过 ratio 子条件，主判据仍判）。1.3B held-out 复跑结果逐位一致（real=6.25/10-10/failed=0）= 零回归。
- `run_overfit.sh`：加 `NPROC` env（普通 accelerate DDP，默认 1 不变）。8 卡 20 步 smoke 干净退出，from-DiT vace 在 DDP 下每 rank 确定性克隆、训练正常、无 OOM。

### 16.2 训练配方（8 卡 DDP）
```bash
NPROC=8 FRAMES=105 LR=1e-5 REPEAT=4 EPOCHS=40 \
OUT=models/train/a2v_robotwin_train40_vace_i2v \
DATASET=.cache/a2v_robotwin/ep_train40_phys \
bash a2v/run_overfit.sh wan2.1-i2v-14b-480p
```
- 步数：`EPOCHS×(40×REPEAT)/NPROC = 40×160/8 = 800` 步（≈6400 sample-exposures）；~10s/it、每卡 ~80GB（8-bit + grad-ckpt，紧但不 OOM）、约 2.2h。
- loss：800 步从 ~0.06 降到 0.001–0.01，平稳无发散（lr=1e-5，对比单 ep lr=1e-4 发散）。
- final ckpt `models/train/a2v_robotwin_train40_vace_i2v/step-800.safetensors`（6.5GB，276 keys 全参 vace `vace_blocks.*`/`vace_patch_embedding.*`，无 lora/pipe 前缀）。

### 16.3 评测结果（step-800）
| 集合 | REAL MAE-GT | NONE MAE-GT | real<none | motion ratio | real-vs-none |
| --- | --- | --- | --- | --- | --- |
| **held-out ep40-49** | **4.25** | 30.09 | **10/10** | 1.06 | 30.21 |
| train ep0-4 | 4.76 | 32.10 | 5/5 | 1.06 | 32.19 |
- 每个 held-out episode 均 real<none，REAL MAE 稳定 3.8–4.8；REAL motion 5.31≈GT 5.01。**PASS**。
- 与 1.3B 多 ep 对照（held-out REAL 6.25）：14B 更准，泛化更强。
- 产物：`.cache/a2v_robotwin/eval_heldout_i2v/`、`eval_train_i2v/`（含 mp4 + metrics.json）。

### 16.4 复现命令
```bash
PY=.venv/bin/python
# 训练见 16.2；评测：
CK=models/train/a2v_robotwin_train40_vace_i2v/step-800.safetensors
CUDA_VISIBLE_DEVICES=0 $PY -m a2v.eval_multiep --base_spec wan2.1-i2v-14b-480p --lora $CK \
  --dataset .cache/a2v_robotwin/ep_heldout10_phys --num_frames 105 --height 240 --width 320 \
  --controls real,none --output_dir .cache/a2v_robotwin/eval_heldout_i2v
```

### 16.5 下一步
- （可选提速/降显存）encoded-cache（ABot 思路，`A2V_HISTORY.md §13.3`）：跳过 VAE/T5/CLIP 重编码，对 14B 多 ep/多 epoch 收益大。
- 更大规模：更多 episode / 多 task 混训、或扩 held-out。
- **T6 Wan2.2-I2V-A14B**（SEAM-5 双专家 MoE）：master plan 最后一个基模。
- eval_multiep 当前每行重编码（无 latent 缓存），14B 上 10 行 held-out ~40min；若评测规模变大可加缓存（计划书 Part A3，本轮未做）。

---

## 17. Encoded-cache 两阶段训练（2026-06-22，提速+降显存，PASS）

**结论：encoded-cache 打通，I2V-14B 多 ep 训练每步快 ~31%、每卡省 ~9GB，泛化不退化。** 关键发现：**stock DiffSynth 已内置该机制**（`:data_process` 预编码 → `load_from_cache` 训练），无需移植 ABot 的自定义实现；只需给 `train_a2v` 加一个 `--cache_train` 开关。

### 17.1 机制（stock，复用）
- **阶段A 预编码** `--task sft:data_process`：`split_pipeline_units` 只保留编码器单元，跑一遍把每个 episode 的编码器输出存成 `.pth`（`launch_data_process_task` → `{output}/{rank}/{data_id}.pth`）。实测缓存内容：`input_latents`、`context`(T5 posi/nega)、`vace_context`(96ch)、i2v 另含 `clip_feature`(1,257,1280) 与 `y`(1,20,T,H,W)。
- **阶段B cache-train** `metadata_path=None` → `UnifiedDataset.load_from_cache`（递归找 `*.pth`）；训练 `forward` 跳过 `get_pipeline_inputs`，编码器单元不跑。配 `--cache_train` 后**只加载 DiT**（不载 T5/VAE/CLIP）。每步仍随机采 noise/timestep（保持训练随机性）。

### 17.2 代码（仅 `a2v/`）
- `train_a2v.py`：加 `--cache_train`。① `model_paths` 砍到只剩 DiT 条目（`spec.model_paths()[0]`，含 dit_glob 分片 list；VACE-1.3B 的 DiT ckpt 自带 vace 权重，from-DiT 基模由 `ensure_vace` 从 DiT 造）；② `dataset_metadata_path=None` 触发 load_from_cache。
- **修了一个集成 bug**：`--task sft:train` 会把编码器侧 `WanVideoUnit_VACE`（产 `vace_context`）从 `pipe.units` split 掉，原 `provision_a2v` 的 `install_vace_unit` 因此报 "No WanVideoUnit_VACE found"。cache 模式下 `vace_context` 已缓存、不需该编码器单元，故 cache_train 只调 `ensure_vace`（建 vace 模型）、**跳过 install_vace_unit**。
- `run_overfit.sh` 的 `NPROC`（§16）用于多卡 cache-train。

### 17.3 实测（I2V-14B，train40，8 卡 DDP，800 步）
| | 非缓存（§16） | cache-train |
| --- | --- | --- |
| 每步 | ~10 s/it | **6.96 s/it（−31%）** |
| 每卡显存 | ~80 GB（贴 OOM） | **~71 GB（多 ~9GB 余量）** |
| 加载 | dit+t5+vae+clip | **仅 dit** |
| 800 步耗时 | ~2.2h | **~1.5h** |
- 缓存：`.cache/a2v_robotwin/cache_train40_i2v`（40 个 `.pth`，4.1GB；vace-1.3b 版在 `cache_train40_vace1p3b`）。
- final ckpt `models/train/a2v_robotwin_train40_vace_i2v_cached/step-800.safetensors`（276 keys 全参 vace）。

### 17.4 泛化（cache-trained，held-out ep40-49）
| 集合 | REAL MAE-GT | NONE MAE-GT | real<none | ratio | 对照非缓存 |
| --- | --- | --- | --- | --- | --- |
| held-out | 5.30 | 42.71 | **10/10** | 1.07 | 非缓存 4.25/10-10 |
| train ep0-4 | 4.96 | 38.88 | 5/5 | 1.06 | 非缓存 4.76/5-5 |
- **PASS**，10/10 real<none、real≪none（~8×）。REAL MAE 5.30 vs 非缓存 4.25 的差异属**训练随机性**（cache 模式 noise/timestep 仍每步随机；DDP 下 load_from_cache 按文件 glob 顺序分片 ≠ metadata 顺序 → 不同 SGD 轨迹），**非缓存导致的退化**——cache 喂的就是非缓存会算出的同一批 latent。

### 17.5 复现命令
```bash
PY=.venv/bin/python; D=.cache/a2v_robotwin/ep_train40_phys
# 阶段A 预编码（单卡，一次性；i2v 缓存 clip_feature/y/vace_context/input_latents/context）
CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec wan2.1-i2v-14b-480p --task sft:data_process \
  --dataset_base_path "$D" --dataset_metadata_path "$D/metadata.jsonl" \
  --data_file_keys video,vace_video --extra_inputs vace_video,input_image \
  --height 240 --width 320 --num_frames 105 --dataset_repeat 1 \
  --output_path .cache/a2v_robotwin/cache_train40_i2v
# 阶段B cache-train（8 卡 DDP，跳过编码器）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/accelerate launch --num_processes 8 --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec wan2.1-i2v-14b-480p --cache_train --task sft:train \
  --dataset_base_path .cache/a2v_robotwin/cache_train40_i2v --data_file_keys video,vace_video \
  --trainable_models vace --learning_rate 1e-5 --dataset_repeat 4 --num_epochs 40 --save_steps 100 \
  --output_path models/train/a2v_robotwin_train40_vace_i2v_cached \
  --customized_optimizer bitsandbytes.optim.Adam8bit --enable_tensorboard_log
# 评测同 §16 的 eval_multiep（--lora .../step-800.safetensors）
```

### 17.6 注意/下一步
- 缓存 key 是按 `data_id`（行号），**与分辨率/帧数/prompt/动作图无内容哈希**——换分辨率或重渲数据后**必须用新的 `--output_path` 缓存目录**，否则会复用过期 latent（沿用 ABot 的已知坑，stock 同样如此）。
- 数据并行顺序：cache 的 `.pth` 按文件 glob 顺序被发现（非按 episode 号），DDP 分片与 metadata 顺序不同——结果等价但非逐位复现。
- 显存进一步：编码器已不占（省 ~9GB/卡），DiT 本体 + 激活仍是大头；要再降需模型分片（ZeRO-3/FSDP）或更激进的 grad-ckpt。
- 适用面：多 ep / 多 epoch 训练收益最大（一次编码、多轮复用）；单 ep 单轮过拟合收益小（编码只省一遍）。

---

## 18. T6 接入 Wan2.2-I2V-A14B（2026-06-22，SEAM-5 双专家 MoE，PASS）

**结论：master plan 最后一个基模打通，五基模全部 PASS。** Wan2.2-I2V-A14B 是双专家 MoE：
高噪专家 `dit`+`vace`、低噪专家 `dit2`+`vace2`，推理按 timestep 用 DiffSynth **原生切换**
（`switch_DiT_boundary=0.875`）。沿用 from-DiT 原则（与 T3/T4/T5 同构，唯一新变量=SEAM-5），
**两条独立单专家作业**分带训练，推理合并。单样本因果门 **REAL MAE-GT=3.14**（五基模最佳）。

### 18.1 关键事实（都查过实际代码）
- **DiffSynth 原生支持双专家**：`wan_video.py:151-163` 用 `fetch_model(...,index=2)` 把两套权重装进
  `pipe.dit`/`pipe.dit2`、`pipe.vace`/`pipe.vace2`；`wan_video.py:314-317` 在 `timestep < switch_DiT_boundary*1000`
  时把 `models["dit"]=dit2`/`models["vace"]=vace2`。**不改 `diffsynth/`。**
- **官方配方=两条独立单专家作业**（`examples/wanvideo/model_training/full/Wan2.2-I2V-A14B.sh`、
  `…VACE-Fun-A14B.sh`）：high 训 `--max_timestep_boundary 0.358 --min 0`（timesteps[900,1000]），
  low 训 `--max 1 --min 0.358`（timesteps[0,900)）。stock loss 已按这两个 flag 截断采样带
  （`diffsynth/diffusion/loss.py:11-14,37-40`），`train_a2v` 早已透传。每条作业只载**一个** 14B 专家
  → 训练显存=T4。
- **A14B I2V 无 CLIP**：config（`model_configs.py:281`）`has_image_input=False, in_dim=36,
  require_clip_embedding=False`。首帧走 VAE-concat `y`（`WanVideoUnit_ImageEmbedderVAE`，gated on
  `require_vae_embedding`），CLIP 单元自动 no-op（`require_clip_embedding=False`）→ 新 SEAM-4 变体
  `i2v_vae`（input_image，但不载 image encoder）。
- **VAE/T5/tokenizer 与已有资产同**：Wan2.1 VAE（z16/s8 → vace_in_dim=96、mask_pq=8）、umt5-xxl T5、
  umt5 tokenizer，全部复用；DiT 结构==I2V-14B（5120/40/40/13824）、`vace_layers=(0,5,…,35)`。
  **只下载两套 DiT 专家分片**（各 6 片，共 107GB，`models/Wan-AI/Wan2.2-I2V-A14B/{high,low}_noise_model/`）。

### 18.2 代码改动（均限 `a2v/`）
- `base_spec.py`：新 `first_frame_mode="i2v_vae"`；新双专家字段 `experts/expert_dit_globs/switch_boundary/train_bands`；
  `model_paths(expert=None)`（`expert="high|low"`→单专家 DiT；`None`→两专家[high,low]→`dit,dit2`）；
  注册 `wan2.2-i2v-a14b`。
- `provision.py`：`create_vace_from_dit(pipe, spec, dit=None)`（可指定源 DiT，从 `dit2` 造 `vace2`）；
  新 `ensure_vace_experts`（两 DiT 都在时建两分支）；`provision_a2v` 双专家时建 vace+vace2。
- `train_a2v.py`：加 `--expert {high,low}`，选该专家 DiT。其余=T4（from-DiT 全参 vace、8-bit Adam、lr1e-5）。
- `infer_a2v.py`：`build_pipe_with_vace(..., ckpt_path_low=)` 双专家路径（两 DiT、vace+vace2、high/low ckpt）；
  `first_frame_kwargs` 加 `i2v_vae→input_image`；CLI 加 `--lora_low`。**双专家推理开 CPU offload**（两 14B 同载，
  非活跃专家 offload 到 CPU；后建的 vace/vace2 不受 vram 管理 → 显式 `.to(device)` 钉在 GPU）。
- `eval_multiep.py`：加 `--lora_low`。
- `check_load.py`/`check_provision.py`：处理 `i2v_vae`（in_dim36/无 CLIP）与双专家加载/零副作用。
- `run_overfit.sh`：`wan2.2-i2v-a14b` 预设 + `EXPERT=high|low`（必填；设带边界 flag + 输出后缀 `…_a14b_{high,low}`）。

### 18.3 退出门结果（单样本过拟合 `ep0_dataset_phys`，240×320，121 帧）
- **load smoke**：两 DiT（40 块/in_dim36/无 CLIP）、两 from-DiT VACE（8 层/vace_in_dim96/after_proj=0/patch_embedding 存活）、mask_pq=8、switch_boundary=0.875。✓
- **provision 门**：①形状 96/8层/5120/13824 + 权重拷贝 + after_proj=0 ✓；②**零副作用**：两专家都 provision、
  `pipe()` 4 步带/不带控制 `max_abs_pixel_diff=0 bit_identical=True`（原生切换跨步把两分支都跑到，一次覆盖两专家零初始化）✓。
- **训练**：high(GPU0)+low(GPU1) 并行各 1000 步（8-bit Adam，lr1e-5，~11.7s/it，~66GB/卡）。loss 平稳
  （high last20≈0.006、low≈0.023，无发散）。ckpt `models/train/a2v_robotwin_ep0_vace_a14b_{high,low}/step-1000.safetensors`
  （各 236 keys 全参 vace `vace_blocks.*`/`vace_patch_embedding.*`，无 lora/pipe 前缀）。
- **因果门**（推理合并两专家 + 原生切换，real/none/shuffle）：

| video | MAE-GT | motion | mean |
| --- | --- | --- | --- |
| GT | -- | 4.78 | 214.93 |
| **REAL** | **3.14** | 4.97（×1.04） | 214.28 |
| NONE | 21.48 | 2.72 | 221.16 |
| SHUFFLE | 20.48 | 4.32 | 217.43 |

  `real-vs-none MAE=21.56`、`real-vs-shuffle MAE=20.24`。判据（§12.6 i2v）：REAL MAE-GT 最低（3.14，
  五基模最佳，优于 T4 4.83/T5 5.27/多ep-I2V 4.25）、REAL motion≈GT（×1.04）、real≪none/shuffle（~6.8×）。
  **切换点不撕裂**：REAL motion 4.97≈GT 4.78（整段时间能量与 GT 同量级，无边界处运动突变），双专家在
  switch 两侧产出连贯视频。**T6 PASS。**

### 18.4 复现命令
```bash
cd /vepfs/wangshilong/code/DiffSynth-Studio
PY=.venv/bin/python
# 下载（只两套 DiT 专家分片；VAE/T5/tokenizer 复用 converted 资产）
.venv/bin/modelscope download --model Wan-AI/Wan2.2-I2V-A14B \
  --exclude 'models_t5_*.pth' 'Wan2.1_VAE.pth' 'assets/*' 'examples/*' 'google/*' 'nohup.out' \
  --local_dir models/Wan-AI/Wan2.2-I2V-A14B
CUDA_VISIBLE_DEVICES=0 $PY -m a2v.check_load --base_spec wan2.2-i2v-a14b
CUDA_VISIBLE_DEVICES=0 $PY -m a2v.check_provision --base_spec wan2.2-i2v-a14b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys --num_frames 13 --steps 4
# 两专家分带过拟合（可并行不同卡）
EXPERT=high CUDA_VISIBLE_DEVICES=0 bash a2v/run_overfit.sh wan2.2-i2v-a14b
EXPERT=low  CUDA_VISIBLE_DEVICES=1 bash a2v/run_overfit.sh wan2.2-i2v-a14b
# 合并因果门
HI=models/train/a2v_robotwin_ep0_vace_a14b_high/step-1000.safetensors
LO=models/train/a2v_robotwin_ep0_vace_a14b_low/step-1000.safetensors
for c in real none shuffle; do CUDA_VISIBLE_DEVICES=0 $PY -m a2v.infer_a2v --base_spec wan2.2-i2v-a14b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys --lora $HI --lora_low $LO \
  --num_frames 121 --height 240 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_a14b_s1000_$c.mp4; done
$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys --num_frames 121 \
  --real .cache/a2v_robotwin/gen_a14b_s1000_real.mp4 --none .cache/a2v_robotwin/gen_a14b_s1000_none.mp4
```

### 18.5 注意/下一步
- 双专家推理两 14B 同载，**必须开 CPU offload**（`infer_a2v` 已对 dual spec 自动加 `vram_config`+`vram_limit`）；
  非活跃专家 offload 时 from-DiT 的 vace/vace2 可能落 CPU，已 `.to(device)` 钉回 GPU。约 4s/it、~45GB GPU。
- 坏产物提醒：无（A14B 首训即过；坏 ckpt 仅历史 I2V `…_vace_i2v/step-*` 那批 lr1e-4）。
- 下一步（择一）：① **A14B 多 episode 泛化已完成（见 §18.6）**；② `Wan2.2-VACE-Fun-A14B` 自带双 VACE warm-start 对照；
  ③ 更大规模/多 task 混训；④ 编码升级（splat）。

### 18.6 A14B 多 episode 泛化（2026-06-22，encoded-cache + 双专家，PASS）

**结论：A14B 双专家在 10 个未见 episode 上泛化通过，且 held-out 与 train 几乎无 gap。** 复用 §16 同份
train40/heldout10（240×320，105 帧），用 §17 encoded-cache（双专家**共享同一份 cache**，一次预编码两专家复用），
两专家各 8 卡 DDP cache-train 800 步，推理合并 + 原生切换评测。

- **阶段A 预编码**（一次）：`--task sft:data_process --expert high` 把 train40 编码进 `.cache/a2v_robotwin/cache_train40_a14b`
  （40 个 `.pth`，4.1GB）。实测缓存内容：`input_latents`(1,16,27,30,40)、`y`(1,20,…)、`vace_context`(1,96,…)、`context`(T5 posi/nega)，
  **无 `clip_feature`**（A14B 无 CLIP，符合预期）。cache 是编码器输出、与专家无关 → 两专家共用。
- **阶段B cache-train**（高/低各一条 8 卡 DDP 作业，串行）：`--cache_train --task sft:train --expert {high,low}` +
  各自带边界 `--min/max_timestep_boundary`（high `[0,0.358]`、low `[0.358,1]`），`REPEAT=4 EPOCHS=40 → 800 步`，
  lr1e-5 8-bit Adam。**仅载单专家 DiT、跳过 T5/VAE/CLIP → ~65GB/卡**（比全编码 ~80GB 省 ~15GB），~7.2s/it、每专家~1.6h。
  loss：high first=1.49→last20≈0.074（高噪带本就难，初值高合理）、low 0.060→0.036，均收敛无发散。
  ckpt `models/train/a2v_robotwin_train40_vace_a14b_{high,low}_cached/step-800.safetensors`（各 236 keys 全参 vace）。
- **评测**（`eval_multiep --lora_low`，双专家合并 + 原生切换 + CPU offload，~4.5s/视频帧组、held-out 10 行约 1.5h）：

| 集合 | REAL MAE-GT | NONE MAE-GT | real<none | motion ratio | real-vs-none |
| --- | --- | --- | --- | --- | --- |
| **held-out ep40-49** | **5.36** | 56.72 | **10/10** | 1.05 | 56.50 |
| train ep0-4 | 5.60 | 102.24 | 5/5 | 1.06 | 101.56 |

  held-out(5.36) ≤ train(5.60) ⇒ **真泛化非记忆**；REAL motion 5.24≈GT 5.01（×1.05）；real≪none **~10.6×**
  （A14B 强 I2V 先验在无控制时自由幻想，离 GT 极远 → 因果足迹巨大）；failed=0/10。**PASS。**
  产物：`.cache/a2v_robotwin/eval_heldout_a14b/`、`eval_train_a14b/`（mp4 + metrics.json）。
- 对照：单 ep A14B 因果门 REAL 3.14（§18.3）；多 ep held-out 5.36 略高（cache 训练随机性 + 多 ep 摊薄容量，
  同 §17 观察）。与 §16 I2V-14B 多 ep（held-out 4.25/NONE 30.09）相比，A14B MAE 略高但 real-vs-none 分离更大。

**复现**：
```bash
PY=.venv/bin/python; D=.cache/a2v_robotwin/ep_train40_phys; CACHE=.cache/a2v_robotwin/cache_train40_a14b
# A: 预编码（单卡一次）
CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch --num_processes 1 --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec wan2.2-i2v-a14b --task sft:data_process --expert high \
  --dataset_base_path "$D" --dataset_metadata_path "$D/metadata.jsonl" \
  --data_file_keys video,vace_video --extra_inputs vace_video,input_image \
  --height 240 --width 320 --num_frames 105 --dataset_repeat 1 --output_path "$CACHE"
# B: 两专家 cache-train（8 卡 DDP，各自带边界）
for E in high low; do case $E in high) MN=0; MX=0.358;; low) MN=0.358; MX=1;; esac
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/accelerate launch --num_processes 8 --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec wan2.2-i2v-a14b --cache_train --task sft:train --expert $E \
  --dataset_base_path "$CACHE" --data_file_keys video,vace_video --trainable_models vace \
  --learning_rate 1e-5 --dataset_repeat 4 --num_epochs 40 --save_steps 100 \
  --min_timestep_boundary $MN --max_timestep_boundary $MX \
  --output_path models/train/a2v_robotwin_train40_vace_a14b_${E}_cached \
  --customized_optimizer bitsandbytes.optim.Adam8bit --enable_tensorboard_log; done
# C: 评测
HI=models/train/a2v_robotwin_train40_vace_a14b_high_cached/step-800.safetensors
LO=models/train/a2v_robotwin_train40_vace_a14b_low_cached/step-800.safetensors
CUDA_VISIBLE_DEVICES=0 $PY -m a2v.eval_multiep --base_spec wan2.2-i2v-a14b --lora $HI --lora_low $LO \
  --dataset .cache/a2v_robotwin/ep_heldout10_phys --num_frames 105 --height 240 --width 320 \
  --controls real,none --output_dir .cache/a2v_robotwin/eval_heldout_a14b
```

---

## 19. Warm-start comparison: Wan2.2-VACE-Fun-A14B (pretrained dual VACE) vs T6 from-DiT (2026-06-22~24, PASS)

**结论：两条路线都通过因果门，但 T6 的 from-DiT I2V-A14B（REAL MAE-GT 3.14）显著优于 Fun-A14B warm-start（6.45）。
根因是首帧机制（i2v_vae 强锚定 vs vace_reference 弱），不是 VACE 初始化质量——warm-start 起始 loss 更低，但
最终像素保真度被弱首帧拖累。** 这是一个"两条 A14B A2V 路线"的实用对照，不是单变量消融（见下"混淆"）。

### 19.1 Fun-A14B 的关键事实（查 config.json + 实测）
- `PAI/Wan2.2-VACE-Fun-A14B` 自带**预训练双 VACE**（high/low 各一份，与各自 noise_model 的 DiT 打包在同一
  safetensors，34.7GB/份 bf16）。`high_noise_model/config.json`：`VaceWanModel, dim5120/40L, in_dim=16,
  vace_layers(0,5..35), vace_in_dim96`——DiT 是 **in_dim=16 T2V 式**，控制完全经 VACE 分支 + `vace_reference_image`
  （**非 i2v concat**）。
- 结构与已注册的 `Wan2.1-VACE-14B` 完全一致（hash `7a513e1f257a861512b1afd387a8ecd9`，实测高噪 noise_model
  文件 hash 精确匹配）→ 每份 noise_model 被探测为 **DiT + VACE 打包**（`has_pretrained_vace=True`），**无需改 diffsynth**。
- 只下载两份 noise_model（~69GB），VAE(Wan2.1)/T5/tokenizer 复用 converted 资产。

### 19.2 代码（仅 `a2v/`）
- `base_spec.py`：注册 `wan2.2-vace-fun-a14b`（`has_pretrained_vace=True`、`experts=(high,low)`、
  `first_frame_mode="vace_reference"`、in_dim16 DiT、`expert_dit_globs` 指两份单文件 noise_model、switch 0.875、train_bands 同 I2V-A14B）。
- `provision.py`：`provision_a2v` 加"预训练+双专家"分支（不 from-DiT 造，只校验 pipe.vace/vace2 都在 + 装 unit）；
  新 `build_bare_vace(spec,...)`（按 spec 形状建裸 VaceWanModel）。
- `infer_a2v.py`：**关键修复**——from_pretrained 载入的预训练 vace/vace2 受 vram 管理（wrapped → state_dict 键带 `.module.`），
  标准 vace ckpt 无法 `load_state_dict`（报 unexpected keys）。推理时对预训练分支改为 **build_bare_vace 建裸分支 + 载训练 ckpt + 钉 GPU**
  （与 from-DiT 路同样的 un-managed 常驻 GPU 终态；DiT 仍 offload）。
- `run_overfit.sh`：加 `wan2.2-vace-fun-a14b` case（`FIRST_FRAME=reference`、full-param vace、adam8bit、lr1e-5、`EXPERT=high|low` 带边界）。
- `check_load.py`：双预训练 smoke（校验 vace + vace2 都预训练存在）。

### 19.3 训练（同 T6 配方：单 ep ep0_dataset_phys 240×320，full-param vace，8-bit Adam，lr1e-5，1000 步/专家）
- high/low 并行各 ~2.8h（vace_reference 加 ref 帧 → 每步略慢，~11s/it），~65GB/卡。ckpt
  `models/train/a2v_robotwin_ep0_vace_funa14b_{high,low}/step-1000.safetensors`（各 236 keys 全参 vace）。
- **warm-start 收敛**：high 专家 loss 起步 0.024（T6 from-DiT 起步 1.49）——预训练 VACE 已会控制，初始 loss 低得多。

### 19.4 对照结果（单样本因果门，REAL/NONE/SHUFFLE，merged + 原生切换）
| 指标 | **I2V-A14B from-DiT (T6)** | **Fun-A14B warm-start** |
| --- | --- | --- |
| VACE 初始化 | from-DiT 零初始化 | 预训练双 VACE |
| DiT / 首帧 | in_dim36 I2V / **i2v_vae** | in_dim16 T2V 式 / **vace_reference** |
| step-200 REAL MAE-GT | **12.99** | 40.95 |
| **step-1000 REAL MAE-GT** | **3.14** | 6.45 |
| step-1000 NONE MAE-GT | 21.48 | 31.94 |
| step-1000 SHUFFLE MAE-GT | 20.48 | 25.70 |
| REAL motion (GT 4.78) | 4.97 (×1.04) | 5.08 (×1.06) |
| real-vs-none | 21.56 | 32.43 |
| 因果门 | **PASS** | **PASS** |
产物：`.cache/a2v_robotwin/gen_funa14b_s1000_{real,none,shuffle}.mp4`、`conv_{funwarm,i2vfromdit}_s200_*.mp4`。

### 19.5 解读（混淆 + 结论）
- **不是干净的单变量消融**：Fun-A14B 与 T6 在三个轴上同时不同——VACE 初始化（预训练 vs from-DiT）、DiT（in_dim16 T2V 式 vs in_dim36 I2V）、
  首帧（vace_reference vs i2v_vae）。无法干净隔离，因为 Fun 的预训练 VACE 与其 in_dim16 DiT 绑定，硬塞进 I2V-A14B 壳并不"warm"。
- **主导因素是首帧机制**：i2v_vae 把 GT 首帧经 VAE-concat 直接喂入 → REAL 强锚定 GT → 像素 MAE 低（step-200 就 12.99）；
  vace_reference 锚定弱 → MAE 高（step-200 40.95）。warm-start 的预训练 VACE 让起始 loss 低，但赢不回弱首帧的像素差距。
- **两者都是有效 A2V 模型**（都 PASS：REAL≪NONE/SHUFFLE、REAL motion≈GT、控制因果驱动）。Fun-A14B 的 NONE 更静（vace_reference 无轨迹→少动，motion 2.30），
  符合 T1/T3 vace_reference 行为；I2V 类 NONE 自由运动。
- **生产取舍**：RoboTwin A2V 像素保真，**首帧机制（i2v_vae）比 warm-start 更关键 → T6 from-DiT I2V-A14B 是更好的生产选择**；
  Fun-A14B warm-start 可用但此处不占优。warm-start 的价值在收敛速度/少步数与"纯动作"(vace_reference 无强首帧泄漏) 场景。
- 未做（可选）：Fun-A14B 多 episode 泛化（同 §18.6 cache 流程，复用 `eval_multiep --lora_low`）；预计同样被首帧机制主导，模式不变。

### 19.6 复现命令
```bash
cd /vepfs/wangshilong/code/DiffSynth-Studio; PY=.venv/bin/python
# 下载（仅两份 noise_model；T5/VAE/tokenizer 复用）
.venv/bin/modelscope download --model PAI/Wan2.2-VACE-Fun-A14B \
  --exclude 'models_t5_umt5-xxl-enc-bf16.pth' 'Wan2.1_VAE.pth' 'google/*' \
  --local_dir models/PAI/Wan2.2-VACE-Fun-A14B
CUDA_VISIBLE_DEVICES=0 $PY -m a2v.check_load --base_spec wan2.2-vace-fun-a14b
EXPERT=high CUDA_VISIBLE_DEVICES=0 bash a2v/run_overfit.sh wan2.2-vace-fun-a14b
EXPERT=low  CUDA_VISIBLE_DEVICES=1 bash a2v/run_overfit.sh wan2.2-vace-fun-a14b
HI=models/train/a2v_robotwin_ep0_vace_funa14b_high/step-1000.safetensors
LO=models/train/a2v_robotwin_ep0_vace_funa14b_low/step-1000.safetensors
for c in real none shuffle; do CUDA_VISIBLE_DEVICES=0 $PY -m a2v.infer_a2v --base_spec wan2.2-vace-fun-a14b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys --lora $HI --lora_low $LO \
  --num_frames 121 --height 240 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_funa14b_s1000_$c.mp4; done
$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys --num_frames 121 \
  --real .cache/a2v_robotwin/gen_funa14b_s1000_real.mp4 --none .cache/a2v_robotwin/gen_funa14b_s1000_none.mp4
```
