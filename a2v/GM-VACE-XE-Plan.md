# GM-VACE：面向 Cross-Embodiment 的 Canonical Desired World-State-Conditioned Multi-View World Model

## 正式实施计划

**文档状态**：执行基线  
**适用范围**：DiffSynth-Studio 中基于 Wan/VACE 的多视角、跨本体视频世界模型研发  
**核心原则**：先证明条件可用与几何正确，再增加跨视角通信；任何阶段失败，不继续堆叠复杂度。

---

# 0. 摘要、目标与非目标

## 0.1 一句话目标

构建一个共享的、面向不同机器人本体的条件视频世界模型：

$$
p\!\left(
X^{future},
S^{future}_{scene}
\mid
X^{history},
S^{current}_{world},
S^{desired}_{robot,1:H},
G_{embodiment},
C_{1:H}
\right),
$$

其中本体原生状态与命令先由 `EmbodimentAdapter` 转换为统一的 canonical 表示；Shared World Model 不接触原生关节状态或原生 action space，只学习 canonical robot desired state、场景演化与可变多视角观测之间的关系。

## 0.2 核心贡献假设

本项目要验证的不是“一个模型可以同时生成多个视频”这一弱结论，而是以下可检验假设：

1. canonical desired world-state trajectory 能作为跨本体共享的因果条件；
2. embodiment-specific adapter 能隔离 native state/action 差异；
3. 共享世界模型在冻结后，能够通过新本体 adapter 接入 seen 或 unseen embodiment；
4. joint multi-view 模块能改善同一潜在世界在多视角下的状态、事件与几何一致性；
5. 上述收益在等数据、等参数、等算力及严格 episode split 下仍成立。

所有论文级主张必须由 G0–G5 中对应实验与统计结果支持。工程完成不自动等于论文主张成立。

## 0.3 非目标

第一版明确不做：

- 不让 Shared World Model 学习或统一不同机器人的 native joint/action space；
- 不把 future realized robot/object/camera/contact/image 当作正式推理条件；
- 不把对象未来状态混入 desired robot trajectory；
- 不把 inverse controller 纳入 Shared World Model；
- 不强制建模 wrench、force 或 contact；它们可作为后续增强、监督或未观测随机因素；
- 不为 Head、Wrist-L、Wrist-R 建三套独立模型；
- 不在基线阶段声称存在 cross-view communication；
- 不在 VACE 条件路径尚未证明不足前修改 stock Wan DiT；
- 不以像素完全相同作为多视角一致性的目标；
- 不以 Oracle 条件结果冒充可部署结果。

## 0.4 总体研发原则

每个阶段严格执行：

```text
冻结上一阶段
→ 定义唯一变量
→ 实现
→ 单元测试
→ data/projection smoke
→ 5-sample overfit
→ 受控消融
→ Gate
→ 通过后进入下一阶段
```

若 Gate 失败，回退并定位当前阶段，不追加新的网络模块掩盖失败。

---

# 1. 形式化任务与信息流

## 1.1 时间定义

- $t_0$：预测起点；
- history：所有时间戳 $\le t_0$ 的可用信息；
- $H$：未来预测时域；
- $t_h$，$h=1,\ldots,H$：统一同步网格上的未来时刻。

## 1.2 随机变量

- $X^{history}=\{X_{v,t}\mid t\le t_0\}$：历史多视角观测及其 view mask；
- $X^{future}=\{X_{v,t_h}\}$：待预测未来多视角视频；
- $S^{current}_{world}$：由 history 与当前测量得到的 canonical robot current state 和历史推断场景表示；
- $S^{desired}_{robot,1:H}$：执行前已知的 canonical desired robot world-state trajectory；
- $S^{future}_{scene}$：待预测场景未来状态，包括对象、可交互部件及事件；
- $G_{embodiment}$：本体几何、语义、外观与 adapter 元数据；
- $C_{1:H}$：计划或已知的多相机未来轨迹、标定和有效性信息；
- $Q^{pred}_{robot,1:H}$：可选 adapter-predicted trajectory、uncertainty 与 source；
- $Y_{goal}$：可选任务目标，独立于 desired robot trajectory。

正式模型可扩展为：

$$
p_\theta\!\left(
X^{future},S^{future}_{scene}
\mid
X^{history},S^{current}_{world},
S^{desired}_{robot,1:H},
Q^{pred}_{robot,1:H},
G_{embodiment},C_{1:H},Y_{goal}
\right).
$$

$Q^{pred}_{robot,1:H}$ 与 $Y_{goal}$ 均为可选输入；缺失时必须有显式 mask，不能用零值暗示“真实零轨迹”。

## 1.3 生成语义

模型预测的是：

- canonical desired robot motion 条件下的场景演化分布；
- 每个有效 camera 对同一潜在世界未来的观测；
- 必要时辅助预测 scene state、robot realized state 或事件，用作训练监督和诊断。

模型不承诺 desired trajectory 一定会被物理系统精确实现。desired 与 realized 的偏差由执行器、动力学、接触和 adapter 误差共同决定，因此训练与评估必须覆盖合理轨迹噪声。

## 1.4 部署路径

```text
native current state + native command
    ↓
EmbodimentAdapter
    ├── current canonical state
    ├── canonical desired trajectory
    ├── optional predicted trajectory + uncertainty/source
    ├── geometry metadata
    └── appearance metadata
    ↓
Shared World Model
    ↓
future scene + variable-view observations
```

若系统需要执行模型输出或 desired trajectory：

```text
canonical desired trajectory
    ↓
inverse controller / motion controller
    ↓
native actuator commands
```

inverse controller 属于执行系统，不属于 Shared World Model。

---

# 2. 术语、坐标与表示约定

## 2.1 三类 robot trajectory

### desired_state

执行前已知、正式输入模型的期望 canonical robot trajectory：

$$
S^{desired}_{robot,1:H}.
$$

它可以来自规划器、遥操作目标、策略意图或用户指定目标。正常 desired state 不是 Oracle。

### predicted_state

adapter 根据 native command 和 current state 预测的未来 canonical trajectory：

$$
Q^{pred}_{robot,1:H}
=
\left(
\hat S^{pred}_{robot,1:H},
\Sigma_{1:H},
source_{1:H}
\right).
$$

其中 uncertainty 可为对角协方差、分位数或离散置信等级，但格式必须版本化。`source` 至少区分解析运动学、学习动力学、规划器 rollout 与其他来源。

### realized_state

执行后测得或重建的真实轨迹：

$$
S^{realized}_{robot,1:H}.
$$

它只能用于：

- 训练监督；
- adapter 误差建模；
- 条件遵循诊断；
- Oracle 上界。

在线正式推理不得读取 $t>t_0$ 的 realized state。

## 2.2 坐标变换

统一定义：

$$
{}^{A}T_{B}
$$

将 B frame 中的列向量映射到 A frame：

$$
{}^{A}p = {}^{A}T_B\,{}^{B}p.
$$

全项目约定：

- 左乘；
- 列向量；
- 右手坐标系；
- 长度单位米；
- 时间单位秒；
- 角度单位弧度；
- 原始 quaternion 顺序为 `xyzw`；
- 模型旋转统一使用 continuous rotation-6D。

常用 frame：

- $W$：world；
- $B$：robot base；
- $E_k$：第 $k$ 个 effector；
- $C_v$：第 $v$ 个 camera；
- $L_j$：第 $j$ 个 body/link。

## 2.3 绝对 pose 与相邻增量

effector 的 world pose 为：

$$
{}^{W}T_{E_k}(t).
$$

旧式公式

$$
\left({}^{W}T_{E_k}(t)\right)^{-1}
{}^{W}T_{E_k}(t+\Delta t)
=
{}^{E_k(t)}T_{E_k(t+\Delta t)}
$$

是以时刻 $t$ body frame 表达的 right/body increment，不是 world-space action。仅允许在 adapter 诊断、速度估计或局部运动编码中使用，并命名为 `body_increment`。

若需要 world/left increment，定义：

$$
{}^{W}T_{E_k}(t+\Delta t)
\left({}^{W}T_{E_k}(t)\right)^{-1},
$$

并命名为 `spatial_increment`。二者均不是本计划的正式 action 条件；正式条件是绝对或相对起点表达的 canonical desired state trajectory。

camera 同理：

$$
\left({}^{W}T_{C_v}(t)\right)^{-1}
{}^{W}T_{C_v}(t+\Delta t)
$$

必须称为 camera body increment，不能称为 world camera motion。

## 2.4 SE(3) 统计

禁止直接对 $4\times4$ 齐次矩阵元素做 `Var`。相对变换误差统一通过 Lie algebra：

$$
\xi=\log\!\left(T_{ref}^{-1}T\right)
=
\begin{bmatrix}
\rho\\
\phi
\end{bmatrix}
\in\mathbb R^6.
$$

分别报告：

- 平移误差 $\|\rho\|_2$，单位米；
- 旋转误差 $\|\phi\|_2$，单位弧度；
- 二者各自的均值、分位数和方差；
- 不将米与弧度直接相加，除非预先声明归一化尺度。

## 2.5 旋转表示

### 原始 quaternion

- 磁盘顺序固定为 `xyzw`；
- 归一化后使用 canonical sign：优先令 $w\ge0$；
- 当 $|w|$ 接近 0 时，以第一个绝对值超过容差的非零分量为正，保证确定性；
- 相邻时间序列额外执行 hemisphere continuity：若 $q_t^\top q_{t-1}<0$，则令 $q_t\leftarrow-q_t$。

### 模型 rotation-6D

rotation-6D 取旋转矩阵前两列并通过 Gram–Schmidt 恢复 $SO(3)$。因此：

- pose 的“6D”不得与 twist 的 6D 混淆；
- `pose_r6d` 实际为 position 3D + rotation-6D，共 9 个标量；
- `pose_quat` 为 position 3D + quaternion 4D，共 7 个标量；
- schema、shape 与变量名必须明确表示是哪一种。

必须测试：

- quaternion 双覆盖不改变旋转；
- canonical sign 确定性；
- 179°→181° 附近的序列连续性；
- rotation-6D 编解码正交性、行列式与 round-trip；
- 非法零 quaternion 和退化 rotation-6D 被拒绝或显式 mask。

## 2.6 Camera-relative geometry

第 $v$ 个 camera 与第 $k$ 个 effector 的几何关系：

$$
G_{v,k}(t)
=
{}^{C_v}T_{E_k}(t)
=
\left({}^{W}T_{C_v}(t)\right)^{-1}
{}^{W}T_{E_k}(t).
$$

对 rigid wrist mount：

$$
{}^{W}T_{C_v}(t)
=
{}^{W}T_{E_k}(t)\,{}^{E_k}T_{C_v},
$$

于是：

$$
\left({}^{W}T_{C_v}(t)\right)^{-1}
{}^{W}T_{E_k}(t)
=
\left({}^{E_k}T_{C_v}\right)^{-1}.
$$

这是严格的 rigid transform closure。Wrist 自身对应 EEF 的近常量关系仅用于 calibration sanity，不作为主要动态 feature；其他 effector、body/link 与 scene 的 camera-relative geometry 仍可动态变化。

## 2.7 Camera ray 与 Plücker 表示

对像素齐次坐标 $\tilde u=(u,v,1)^\top$：

$$
r^{C_v}=K_v^{-1}\tilde u,
\qquad
d=\operatorname{normalize}\!\left({}^{W}R_{C_v}r^{C_v}\right),
$$

$$
o={}^{W}p_{C_v},
\qquad
m=o\times d.
$$

完整 Plücker ray 为：

$$
\ell=(d,m).
$$

必须验证 $\|d\|_2=1$ 与 $d^\top m\approx0$。crop/resize 后使用更新后的 $K_v$，不能使用原始 intrinsic。

---

# 3. 信息边界

## 3.1 正式推理允许输入

仅允许：

1. 所有时间戳 $\le t_0$ 的图像、深度、状态、事件或接触 history；
2. $t_0$ 的 current canonical state；
3. 执行前已知的 canonical desired trajectory；
4. 可选 adapter-predicted trajectory、uncertainty 与 source；
5. planned/known camera trajectory；
6. 由 history 推断的 scene representation；
7. static calibration；
8. embodiment geometry 与 appearance；
9. 可选 task goal。

## 3.2 正式推理禁止输入

禁止任何 $t>t_0$ 的：

- realized robot state；
- realized camera pose，除非该轨迹在执行前已被计划且冻结；
- object state；
- contact、force 或 wrench；
- image、depth 或 event；
- 从 future observation 反推得到的特征；
- 由 future realized trajectory 计算出的派生量。

## 3.3 监督、诊断与 Oracle

禁止项可用于训练标签或独立评估。所有 Oracle 结果必须：

- 单独命名为 `oracle_*`；
- 与 deployable setting 分开报告；
- 只作为上界或误差归因；
- 不参与核心贡献结论。

## 3.4 防泄漏实现要求

- dataset API 将 `condition`、`target`、`oracle` 分离；
- condition builder 只接收截断到 $t_0$ 的 history 和预先保存的 plan；
- 禁止 condition builder 持有 future record 的通用句柄；
- 单测在 future target 随机替换后验证 condition byte-equivalent；
- 数据缓存 key 必须包含 split、schema version、$t_0$ 与 horizon；
- adapter 在线接口不接受 `realized_future` 参数；
- 每次实验记录 feature provenance。

---

# 4. Canonical State Schema

## 4.1 设计目标

第一版支持：

- 可变数量 effectors $K$；
- 可选 mobile/manipulation base；
- 单臂、双臂及其他多 effector 本体；
- 统一时间、单位、frame、shape、dtype 和 mask；
- batch 内 padding，但语义上不把 padding 当真实 effector。

## 4.2 CanonicalRobotState

逻辑 schema：

```python
CanonicalRobotState:
    schema_version: str
    timestamp_s: float64
    world_frame_id: str
    effectors: list[CanonicalEffectorState]  # variable K
    base: CanonicalBaseState | None
    valid_mask: bool
```

`CanonicalEffectorState` 至少包含：

```python
CanonicalEffectorState:
    effector_id: str
    semantic_role: str
    pose_world:
        position_m: float32[3]
        rotation_6d: float32[6]
    gripper_openness: float32[1]
    valid_mask:
        pose: bool
        gripper: bool
```

约束：

- `semantic_role` 描述功能，如 `left_gripper`、`right_gripper`、`suction_tool`，不能依赖数组位置；
- `effector_id` 是 sample 内稳定身份；
- `gripper_openness` 归一化到 $[0,1]$，0/1 方向由 schema 固定；
- 无 gripper 的 effector 使用 `valid_mask=False`，不能用常数 0 冒充闭合；
- 双臂通常 $K=2$，单臂通过 effector mask 兼容；
- loader 可 padding 到 batch 内 $K_{max}$，并始终携带 effector mask。

## 4.3 可选 base

```python
CanonicalBaseState:
    pose_world:
        position_m: float32[3]
        rotation_6d: float32[6]
    twist_world_or_body: float32[6] | None
    twist_frame: Literal["world", "body"] | None
    valid_mask: dict[str, bool]
```

twist 若存在必须显式声明 frame。第一版可以只提供 base pose。

## 4.4 Desired trajectory

```python
CanonicalDesiredTrajectory:
    schema_version: str
    timestamps_s: float64[H]
    effectors: CanonicalEffectorTrajectory[K, H]
    base: CanonicalBaseTrajectory | None
    trajectory_valid_mask: bool[K, H]
    source: str
```

desired trajectory 只含机器人期望状态。对象目标不得塞入此结构。

## 4.5 Task goal

对象目标作为独立条件：

```python
GoalCondition:
    goal_type: str
    object_id_or_role: str | None
    target_pose_or_region: object | None
    language_or_symbolic_goal: object | None
    valid_mask: dict[str, bool]
```

goal 可描述“杯子到目标区”，但对象 future state 仍是预测目标，不是 desired robot condition。

## 4.6 Predicted trajectory

```python
AdapterPredictedTrajectory:
    canonical_trajectory: CanonicalDesiredTrajectory
    uncertainty:
        representation: str
        values: float32[...]
    source: str
    model_or_solver_version: str
```

训练时对 predicted trajectory 注入由真实 adapter 误差分布拟合的噪声，并保留无噪声、经验噪声和超分布噪声三档。

## 4.7 第一版暂不强制字段

以下字段只能以可选扩展出现：

- contact；
- wrench；
- force/torque；
- tactile；
- joint torque。

缺失应视为未观测随机因素，不得默认为零物理量。

---

# 5. Camera 与 Embodiment Schema

## 5.1 可变 camera 数量

每个 sample 支持可变 $V$：

```python
MultiViewObservation:
    video: float_or_uint8[V, T, H, W, C]
    camera: list[CameraStream]
    view_valid_mask: bool[V, T]
```

模型内部逻辑布局统一为：

```text
[B, V, C, T, H, W]
```

允许局部算子展开为 $B\times V$，但接口、mask、日志和 sample 级 diffusion timestep 必须保留 view 语义。

## 5.2 CameraStream

每个 camera 至少包含：

```python
CameraStream:
    camera_id: str
    role: str
    parent_link_or_world: str
    intrinsic_original: float64[3, 3]
    distortion_model: str
    distortion_coefficients: float64[D]
    static_mount_transform: float64[4, 4] | None
    pose_world_trajectory: float64[T, 4, 4]
    pose_valid_mask: bool[T]
    frame_valid_mask: bool[T]
    original_resolution_hw: int32[2]
    exposure_metadata: object | None
    readout_metadata: object | None
    crop_resize_transform: float64[3, 3]
    intrinsic_after_transform: float64[3, 3]
    calibration_version: str
```

静态 mount：

$$
{}^{parent}T_{C_v}
$$

与逐帧 world pose：

$$
{}^{W}T_{C_v}(t)
$$

必须分开保存，不能互相覆盖。

## 5.3 ID 与 role

`camera_id` 是具体设备/位置身份，`role` 是抽象观测功能。首轮可能存在：

```text
head ↔ world-centric
wrist-left/right ↔ egocentric
```

此时二者高度共线，不得声称模型已学到 role 泛化。必须做：

- no-ID/no-role；
- ID-only；
- role-only；
- ID+role。

只有在 role 跨不同 ID、安装位置和 embodiment 重复出现，且 held-out ID 上仍有效时，才可主张 role-level generalization。

## 5.4 Embodiment metadata

```python
EmbodimentCondition:
    embodiment_id: str
    geometry: GeometryCondition
    appearance: AppearanceCondition
    adapter_version: str
    schema_version: str
```

`embodiment_id` 仅用于审计、切分或受控消融；zero-shot 主设置中不得依靠新本体训练视频学习其 ID embedding。

---

# 6. EmbodimentAdapter Contract

## 6.1 统一接口

```python
class EmbodimentAdapter:
    def canonicalize_current_state(
        self,
        native_state_history,
        timestamps_s,
        calibration,
    ) -> CanonicalRobotState: ...

    def build_desired_trajectory(
        self,
        native_command,
        current_canonical_state,
        timestamps_s,
    ) -> CanonicalDesiredTrajectory: ...

    def predict_trajectory(
        self,
        native_command,
        current_canonical_state,
        timestamps_s,
    ) -> AdapterPredictedTrajectory | None: ...

    def build_geometry_condition(
        self,
        native_state_history,
        native_plan,
        current_canonical_state,
        desired_trajectory,
        timestamps_s,
        calibration,
    ) -> GeometryCondition: ...

    def appearance_condition(
        self,
        history_observation,
    ) -> AppearanceCondition: ...
```

所有输出遵循同一 schema、时间网格、单位和 frame 约定。

## 6.2 责任拆分

### StateCanonicalizer

负责：

- native state → current canonical state；
- effector role 与 ID 映射；
- unit/frame/rotation 转换；
- mask 与时间戳传播；
- 解析 FK 或必要的 learned mapping。

### Intent/Dynamics Adapter

负责：

- native command → desired canonical trajectory；
- 可选 current state + command → predicted canonical trajectory；
- 输出 uncertainty 和 source；
- 训练与验证 adapter prediction error。

### Geometry Adapter

负责：

- body/link/effector 拓扑；
- 静态尺寸、mesh proxy、关键点或 capsule；
- camera parent 与 mount；
- 根据推理前可用的 native current state、native plan、canonical desired trajectory 与时间网格构造 planned/predicted body 和 camera trajectory；
- 输出 trajectory source、uncertainty 与 valid mask；无法从 desired effector state 唯一恢复的 body/camera 状态不得伪造为确定值；
- camera–effector/body 的可计算关系；
- 不把 rigid wrist 自身 EEF 常量当主要动态特征。

### Appearance Adapter

负责：

- robot mask、reference image、纹理或外观 token；
- 防止外观条件携带 future 信息；
- unseen embodiment 下只使用预先提供或 history 可见的信息。

## 6.3 明确边界

Shared World Model：

- 不读取 native joints；
- 不读取 native action；
- 不调用 native controller；
- 不在线读取 future realized state；
- 只读取 adapter 规范化输出。

adapter 可以是解析、学习或混合实现。学习型 adapter 的训练数据与 Shared WM 训练数据必须分别记录。

## 6.4 Adapter Validation

Native action 相关测试只属于 Adapter Validation：

- native command → desired trajectory 的语义一致性；
- native FK 与 canonical pose round-trip；
- predicted 与 realized 的 endpoint/trajectory error；
- uncertainty calibration；
- frame、单位和时间对齐；
- 新本体无视频时 adapter 是否可由解析/预给参数运行。

这些测试不能替代 Shared WM 的 G2。

---

# 7. Shared World Model 与多视角拓扑

## 7.1 条件分解

对每个 sample：

```text
history observations
+ current canonical world state
+ canonical desired robot trajectory
+ optional predicted trajectory/uncertainty
+ planned camera trajectories
+ embodiment geometry/appearance
+ optional task goal
    ↓
Shared World Model
    ↓
shared future world evolution
    ↓
masked variable-view observations
```

## 7.2 MV-B0：shared-weight independent baseline

逻辑输入保持 $[B,V,C,T,H,W]$。实现可展开为：

$$
[B,V,C,T,H,W]\rightarrow[BV,C,T,H,W],
$$

并共享 VACE/Wan 权重。

限制：

- 各 view denoise 过程不通信；
- 同一 sample 的所有 view 使用相同 diffusion timestep；
- loss 先按 view mask 聚合，再按 sample 平均，避免 view 多的 sample 权重更高；
- 可以使用 per-view camera condition；
- 必须明确称为 shared-weight independent multi-view baseline；
- 不得声称 joint world modeling 或 cross-view consistency mechanism。

## 7.3 MV-J1：VACE 内 joint cross-view world bottleneck

在每个选定联合 denoise step：

1. 保持 view 维；
2. 各 view token 读取到 shared world bottleneck；
3. bottleneck 聚合同 sample 的有效 view；
4. bottleneck 回写各 view；
5. view mask 阻止缺失视角参与读写。

形式上：

$$
W'=\operatorname{Read}\left(
W,\{X_v\}_{v:m_v=1}
\right),
$$

$$
X'_v=X_v+\operatorname{Write}(X_v,W'),\qquad m_v=1.
$$

通信按 sample 隔离，并在每个配置的联合 denoise step 发生，不允许仅在预处理时一次性混合后宣称 joint denoising。

优先在 VACE 内实现，以保持 stock Wan DiT。仅当实验证明 VACE 条件容量不足，才 opt-in 修改 DiT attention；此时必须新建独立配置并重跑 Path A parity。

## 7.4 Geometry-aware cross-view module

该模块根据 camera rays、calibration、body/effector geometry 或可见性先验调制跨视角通信。

它与 world bottleneck 是 DAG 中相互独立的因素：

```text
E4 baseline
 ├── E5-A world bottleneck
 ├── E5-B geometry-aware cross-view
 └── E5-A + E5-B combined
```

必须分别比较：

- neither；
- bottleneck only；
- geometry-aware only；
- combined。

不能把二者捆绑后只报告一个结果。

## 7.5 Diffusion noise 消融

同一 sample 始终共享 diffusion timestep，但噪声不预设必须完全相同。比较：

1. independent per-view noise；
2. shared global noise；
3. shared-global + view-local noise：

$$
\epsilon_v
=
\sqrt{\alpha}\,\epsilon_{global}
+
\sqrt{1-\alpha}\,\epsilon_{local,v}.
$$

扫描少量预注册 $\alpha$，以多视角一致性和单视角质量共同选择，不在 test set 调参。

## 7.6 Camera 与 robot geometry condition

基础几何条件至少包括：

$$
G_{v,k}(t)
=
\left({}^{W}T_{C_v}(t)\right)^{-1}
{}^{W}T_{E_k}(t).
$$

body/link geometry 由 Geometry Adapter 提供。未来 desired effector trajectory 与 planned camera trajectory 可生成未来 $G^{desired}_{v,k}(t_h)$，但不得使用 future realized pose 计算正式条件。

---

# 8. 数据协议与 Loader Hard Gate

## 8.1 原始数据为 source of truth

所有 frame/state/command/pose 保存原始单调时间戳。派生量通过统一库在线或缓存计算，不作为唯一 source of truth。

每条记录至少包含：

- schema version；
- units；
- frame convention；
- shape；
- dtype；
- calibration version；
- provenance；
- raw timestamp；
- valid mask。

## 8.2 同步

原始多流数据同步到统一网格 $t_0,\ldots,t_H$，并保存：

- source index；
- 左右插值 index；
- interpolation method；
- signed/absolute time delta；
- valid mask；
- dropped-frame reason。

默认最大允许 skew：

$$
\Delta t_{max}\le \frac{1}{4}\Delta t_{target},
$$

最终阈值根据训练集同步统计预注册；不能根据 test 表现调整。

禁止：

- 跨 episode 插值；
- 对离散 command 使用连续插值而不声明；
- 对 pose quaternion 直接线性插值；
- 将超阈值样本静默视为有效。

pose 插值使用平移线性插值与旋转 SLERP，输出再转换为 rotation-6D。

## 8.3 图像与相机元数据

必须保存：

- 原始分辨率；
- 原始 $K$；
- distortion model 与 coefficients；
- exposure；
- rolling/global shutter 与 readout；
- crop/resize transform；
- 更新后的 $K$；
- static mount；
- per-frame world pose；
- calibration version。

图像变换矩阵记为 $A_{img}$，更新 intrinsic：

$$
K'=A_{img}K.
$$

## 8.4 切分与样本身份

- 按 episode 切分；
- 固定 test set；
- 同一连续轨迹的相邻窗口不得跨 train/val/test；
- 同一 scene reset 的近重复片段不得泄漏；
- split manifest 内容寻址并冻结；
- embodiment、task、scene、camera rig 维度均可审计。

## 8.5 Loader Gate

任何训练前必须通过：

### Round-trip

- pose matrix ↔ quaternion ↔ rotation-6D；
- native → canonical → 可验证 native/FK；
- crop/resize pixel ↔ transformed ray；
- serialization ↔ deserialization。

### Rigid closure

构造任意合法 $A(t)\in SE(3)$：

$$
{}^{W}T_E(t)=A(t),\quad
{}^{W}T_C(t)=A(t)\,{}^{E}T_C.
$$

验证：

$$
\left({}^{W}T_C(t)\right)^{-1}
{}^{W}T_E(t)
=
\left({}^{E}T_C\right)^{-1}
$$

在数值容差内对所有 $t$ 成立。该测试替代“同向运动”自然语言测试。

### Reprojection

- 已知 3D 点经 extrinsic/intrinsic 投影到预期像素；
- distortion/undistortion round-trip；
- crop/resize 前后投影一致；
- 完整 Plücker ray 满足约束。

### Synchronization

- index、插值、时间差与 mask 一致；
- 超 skew 正确拒绝；
- 边界时刻不读取未来不可用数据；
- command/state/image 时间方向一致。

### Permutation

- left/right effector 交换后，ID、role、pose、mask 和对应监督一起交换；
- view 顺序置换后，输出逆置换一致；
- padding view/effector 不改变有效项输出；
- 不允许模型依赖固定数组槽位表达语义。

---

# 9. 验证框架 G0–G5

## 9.1 G0：Experimental Protocol

G0 贯穿全部阶段，不是最终补做项。

### 固定协议

- episode split；
- no adjacent leakage；
- fixed test；
- 至少 3 个随机种子；
- 同 seed、同 sample 的 paired comparison；
- paired bootstrap 置信区间；
- data/parameters/compute control；
- 预注册主指标、次指标和最小效应；
- 记录 code/config/data/calibration/adapter/checkpoint hash；
- 报告失败运行与排除原因。

### 统计规则

- 论文主张必须有统计支持；
- 主结果报告点估计、95% paired bootstrap CI、seed 间离散度；
- 工程阶段可采用“达到预注册最小效应且所有 seed 方向稳定”作为推进条件；
- CI 跨零仍必须如实报告；
- 多重主假设采用预注册校正或限制主假设数量；
- 不以单 seed 最优 checkpoint 代表方法。

### 公平对照

分别建立：

- 等数据：相同训练 episode 与采样次数；
- 等参数：可训练参数差异控制在预注册容差，或增加 capacity-matched baseline；
- 等算力：相同训练 FLOPs/GPU-hours 或报告 compute-normalized 曲线；
- 相同调参预算；
- 相同 checkpoint selection rule。

## 9.2 G1：Engineering Hard Gate

必须全部通过：

- Path A parity；
- import/shape/dtype/device；
- mask 与 permutation；
- loader gate；
- data smoke；
- projection smoke；
- 5-sample overfit；
- 单视角与多视角 inference；
- deterministic synthetic geometry；
- 无 NaN/Inf；
- artifact 可复现。

任一失败，方法正确性不成立。

## 9.3 G2：Conditional Causality Hard Gate

G2 验证模型是否真正使用 current state 与 desired canonical trajectory。

### 配对设计

1. same $S_0$，different desired trajectory；
2. same desired trajectory，different $S_0$；
3. correct condition；
4. stay trajectory：所有有效 effector 保持 $S_0$，而不是把绝对 pose tensor 置零；
5. matched-shuffle condition；
6. time-reverse condition。

matched shuffle 必须匹配 trajectory 长度、速度幅度、effector role、task/scene 类别，避免仅凭边缘分布识别。

### 物理可实现性

分别测试：

- robot-trajectory-only；
- camera-trajectory-only；
- both。

每个反事实都必须满足关节/速度/工作空间/相机 mount 的物理约束。rigid wrist camera 不得独立于 parent link 做破坏运动学的 shuffle。若需要 camera-only 变化，应使用可动 head、外部 camera 或整体 rig 的合法轨迹。

### 条件遵循指标

至少报告：

- directional cosine；
- endpoint position error；
- endpoint rotation error；
- full trajectory position error；
- full trajectory rotation error；
- gripper/event timing error；
- trajectory-relevant region 的变化幅度；
- 非 trajectory-relevant region 的非预期变化。

输出差异应集中在 trajectory-relevant region。G2 不要求 FVD、视觉质量等所有指标在错误条件下都下降；错误条件仍可能生成高质量但条件不符的视频。

### Gate

在预注册主条件遵循指标上：

- correct 优于 stay、matched-shuffle 和 reverse；
- same $S_0$/different desired 能产生方向与终点一致的差异；
- same desired/different $S_0$ 能保留不同起点的物理后果；
- 至少 3 seeds 方向稳定；
- paired CI 与最小效应按 G0 规则报告。

Native action tests 不属于此 Gate。

## 9.4 G3：Multi-view World Consistency Hard Gate

分别报告 per-view quality 与 joint consistency。

### 指标族

- per-view visual/temporal quality；
- camera/view identity accuracy；
- robot canonical state consistency；
- object identity与属性一致性；
- event occurrence/timing consistency；
- 2D keypoint/segmentation reprojection；
- 有深度或多视角重建时的 3D point/pose consistency；
- epipolar、ray 或 calibrated correspondence consistency；
- occlusion-aware consistency。

几何一致不等于 pixel equality。不同视角的遮挡、曝光、分辨率和投影不同，禁止用逐像素相等作为 Gate。

### Gate

- 不降低预注册 per-view floor；
- 相对 MV-B0，在至少一个核心 world-consistency 指标达到最小效应；
- identity/state/event 不因 joint module 出现系统性冲突；
- 2D reprojection 必须通过；
- 数据允许时尽可能报告 3D consistency；
- 所有 seed 方向与 CI 按 G0 报告。

## 9.5 G4：Cross-Embodiment Core Contribution

评估时冻结 Shared World Model，仅改变 adapter 或本体条件。

### 三档设置

1. **Seen embodiment**：本体及其视频参与 Shared WM 训练；
2. **Adapter-only adaptation**：冻结 Shared WM，仅训练/标定新本体 adapter；
3. **Zero-shot unseen embodiment**：Robot-C 的视频完全不用于训练 Shared WM 或 adapter；只使用解析或预给 adapter、geometry、appearance/calibration。

每档必须明确允许的数据。不能把 Robot-C 视频用于 normalization、early stopping、appearance encoder 训练或阈值选择后仍称 zero-shot。

### Capacity control

对比：

- shared canonical WM + adapters；
- capacity-matched per-embodiment 或 mixture baseline；
- native-action conditioned baseline（仅作为本体内参考）；
- adapter-only parameter count 与 compute。

### Gate

核心贡献需同时满足：

- frozen Shared WM；
- adapter-only setting 有可测收益；
- zero-shot setting 可运行并显著优于无效/错误 adapter；
- 结果不由额外参数、数据或算力解释；
- G2、G3 在跨本体设置仍成立。

## 9.6 G5：Robustness 与 OOD

### Tier 1：最终主结果硬要求

- unseen scene；
- unseen task；
- camera variation；
- moderate calibration noise。

### Tier 2：系统鲁棒性

- missing view；
- time jitter；
- desired/predicted trajectory noise；
- moderate frame drop。

### Tier 3：压力测试

- severe calibration corruption；
- long missing-view burst；
- severe time misalignment；
- large trajectory error；
- appearance/geometry metadata corruption。

方法正确性定义为：

$$
G1\land G2\land G3.
$$

核心贡献由 G4 决定；G5 衡量泛化和鲁棒性。Tier 1 是最终主结果硬要求，Tier 2/3 不得替代 G1–G4。

---

# 10. 路线总览

```text
E0 Baseline Freeze + G0
 ↓
E1 Versioned Data Schema + EmbodimentAdapter Contract
 ↓
E2 MV-B0 shared-weight independent baseline
 ↓
E3 Canonical Desired-State Conditioning + G2
 ↓
E4 Embodiment Geometry + Camera Conditioning
 ├───────────────┐
 ↓               ↓
E5-A VACE        E5-B Geometry-aware
World Bottleneck Cross-view Module
 └───────┬───────┘
         ↓
E6 Cross-Embodiment Transfer（frozen Shared WM）
         ↓
E7 Robustness / OOD
```

E5-A 与 E5-B 可独立、可组合；任一失败可回退 E4。E6 不要求 E5 两支都成功，但采用的模型必须先通过 G1–G3。

---

# 11. E0：Baseline Freeze + G0

## 唯一变量

无模型变量；仅冻结当前可运行 baseline、数据切分、评估与实验协议。

## 实现

- 核验并记录 Path A 入口、配置与 checkpoint；
- 冻结 train/val/test episode manifests；
- 建立 G0 配对统计脚本；
- 记录现有数据、参数量、训练算力和指标；
- 建立 baseline artifact；
- 核验仓库内真实路径后，再把命令写入执行记录，不依据本计划猜测文件路径。

## 单元测试

- split 无 episode/adjacent overlap；
- metric 对相同输入可重复；
- bootstrap pairing key 唯一；
- config 与 checkpoint hash 可复现。

## Smoke

- Path A dry-run；
- data smoke；
- projection smoke；
- inference；
- metric pipeline；
- 5-sample overfit。

## 消融

无新增模块；仅复现至少 3 seeds 或先完成预注册 seed 子集并说明。

## Gate

G0 生效，Path A parity 与 G1 基础项通过。

## Failure action

停止 E1；修复 baseline、split 或评估协议，不修改新模型。

## 产物

```text
experiments/E0_baseline/
  protocol.yaml
  split_manifest.json
  baseline_config.yaml
  metrics_by_seed.json
  parity.log
  smoke.log
  environment.lock
  README.md
```

---

# 12. E1：Versioned Data Schema + EmbodimentAdapter Contract

## 唯一变量

将现有数据读入版本化 canonical schema；网络和训练目标保持 E0 等价。

## 实现

- 实现第 4–6 节 schema 与 adapter contract；
- 原始时间戳和 calibration 保留为 source of truth；
- 同步到统一网格并保存 provenance；
- 分离 `condition/target/oracle`；
- 实现统一 SE(3)、rotation-6D、projection 和 Plücker 库；
- 支持可变 $K$、可变 $V$ 与 mask；
- 建立 schema migration/version checker。

## 单元测试

- quaternion/rotation-6D/SE(3) round-trip；
- canonical sign 与 180° 连续性；
- rigid closure；
- reprojection 与 crop/resize $K$；
- synchronization/skew；
- left/right 与 view permutation；
- future target 替换不改变 condition；
- single-arm padding 与 dual-arm $K=2$。

## Smoke

- 小 episode 全链路读取；
- 可视化投影 overlay；
- 5 samples serialization→loader→model；
- 原 loader 与新 loader 在等价字段上的 parity。

## 消融

- old loader；
- canonical loader；
- canonical loader + permutation；
- 不改变网络输入语义。

## Gate

Loader Hard Gate 全部通过；Path A parity 通过；无 future leakage。

## Failure action

停留 E1；优先修 frame、timestamp、calibration、mask 或 schema，不进入多视角训练。

## 产物

```text
experiments/E1_schema_adapter/
  schema_version.json
  adapter_contract.md
  loader_gate.json
  sync_statistics.json
  projection_overlays/
  leakage_test.log
  parity.log
  README.md
```

---

# 13. E2：MV-B0 Shared-Weight Independent Baseline

## 唯一变量

从单视角扩展到可变多视角 shared-weight independent 训练；不引入 cross-view communication。

## 实现

- 逻辑张量 $[B,V,C,T,H,W]$；
- 局部展开 $BV$ 运行 shared VACE/Wan；
- sample 内共享 diffusion timestep；
- view mask 与 sample-balanced loss；
- 输出恢复 $[B,V,\ldots]$；
- camera ID/role 只做最小可配置条件，默认分别消融；
- 保持 stock Wan DiT。

## 单元测试

- $V=1$ 与 Path A parity；
- $V=2,3,\text{variable}$ shape；
- view permutation equivariance；
- missing/padded view 不影响有效 view；
- 同 sample timestep 一致；
- 不同 sample 不串扰；
- loss 不随有效 view 数量不当地放大。

## Smoke

- 5-sample multi-view overfit；
- 每个 view 独立生成；
- identity/view order 可视检查；
- 单视角与多视角 inference。

## 消融

- single-view；
- MV-B0 no ID/no role；
- ID-only；
- role-only；
- ID+role；
- independent noise 与 shared-global+view-local noise 初筛。

## Gate

G1 通过；5-sample overfit；per-view 指标不低于预注册 floor；文档中不出现 cross-view claim。

## Failure action

修复 batch/view/mask/loss；若 role 无增益或与 ID 共线，只保留为消融，不追加 joint 模块。

## 产物

```text
experiments/E2_MV-B0/
  configs/
  shape_and_mask_tests.json
  five_sample_overfit/
  per_view_metrics.json
  id_role_ablation.json
  noise_ablation.json
  qualitative/
  README.md
```

---

# 14. E3：Canonical Desired-State Conditioning + G2

## 唯一变量

在 MV-B0 上加入 canonical desired robot trajectory；不加入新 cross-view module。

## 实现

- trajectory encoder 支持可变 $K,H$ 与 mask；
- 编码 desired pose、gripper、可选 base；
- current canonical state 独立编码；
- 可选 predicted trajectory 使用独立 type/source/uncertainty embedding；
- planned adapter output 做经验误差与噪声鲁棒训练；
- condition 注入优先发生在 VACE；
- future realized state 只进入监督分支。

## 单元测试

- desired/predicted/realized 类型不能互换；
- desired 不含 object future state；
- current state 与 desired trajectory 分支可独立置换；
- masked effector/time 不贡献 token/loss；
- online adapter 不读取 future realized；
- trajectory noise 单位与 frame 正确。

## Smoke

- 5-sample overfit；
- same $S_0$/two desired trajectories 生成可辨差异；
- same desired/two $S_0$ 保留不同初始状态；
- trajectory-relevant region overlay；
- stay/reverse/shuffle inference 可运行。

## 消融

- no desired condition；
- desired only；
- current + desired；
- current + desired + predicted；
- predicted without uncertainty；
- clean trajectory；
- empirical adapter noise；
- matched shuffle、zero、reverse。

## Gate

完整通过 G2。视觉质量本身不作为错误条件必须下降的要求。

## Failure action

- 若 correct 与 shuffle 无差异：检查条件注入、数据配对与目标可观测性；
- 若差异遍布背景：加强空间归因诊断，检查 camera/scene 混杂；
- 若只在 train 有效：检查 leakage、split 与 desired 边缘分布；
- 不通过时不得进入 E4。

## 产物

```text
experiments/E3_desired_state/
  configs/
  causal_pairs_manifest.json
  adapter_noise_model.json
  G2_metrics_by_seed.json
  paired_bootstrap.json
  action_region_visualization/
  oracle_diagnostics/
  README.md
```

---

# 15. E4：Embodiment Geometry + Camera Conditioning

## 唯一变量

在已通过 G2 的模型上加入 geometry、planned camera trajectory、ID/role 与 appearance 条件；仍不做 joint cross-view communication。

## 实现

- Geometry Adapter 输出 effector/body/link geometry；
- 计算 desired $G_{v,k}(t_h)$；
- planned camera trajectory 与 pose uncertainty 可选输入；
- camera intrinsic/distortion/crop/readout 元数据编码；
- appearance 条件仅来自 static asset 或 history；
- wrist rigid closure 用于 sanity；
- Plücker ray 先用于 projection smoke，作为模型输入需独立开关；
- 保持 stock Wan DiT。

## 单元测试

- $G_{v,k}$ 变换方向；
- rigid closure；
- camera body increment 与 world pose 命名；
- Lie-log 平移/旋转统计；
- body/link 与 effector mask；
- planned camera 合法性；
- future realized camera pose 不进入正式 condition；
- Plücker $(d,m)$ 完整性。

## Smoke

- projection overlay；
- stationary external camera + moving effector；
- rigid wrist camera + parent effector；
- physically valid moving camera；
- variable camera rig；
- 5-sample overfit。

## 消融

- no geometry/camera condition；
- camera pose only；
- effector geometry only；
- body/link geometry；
- Plücker off/on；
- ID-only/role-only/combined；
- robot-trajectory-only/camera-trajectory-only/both；
- planned camera clean/noisy。

## Gate

G1、G2 继续通过；projection smoke 通过；基础 geometry/camera 条件在预注册相关指标上有收益或至少不回归。尚不要求 G3 joint gain。

## Failure action

- closure 失败先修坐标与 calibration；
- role 无独立证据则不做 role 泛化主张；
- Plücker 无收益则关闭；
- geometry 条件无收益时保留 E3，禁止直接增加 attention 掩盖问题。

## 产物

```text
experiments/E4_geometry_camera/
  geometry_config.yaml
  rigid_closure.json
  lie_statistics.json
  projection_overlays/
  camera_counterfactuals/
  id_role_ablation.json
  metrics_by_seed.json
  README.md
```

---

# 16. E5-A：VACE World Bottleneck

## 唯一变量

在 E4 上加入 VACE 内 shared world bottleneck；geometry-aware routing 关闭。

## 实现

- 每个 sample 独立的少量 world tokens；
- masked view read/write；
- 在预注册 VACE block/denoise step 通信；
- 同 timestep、可配置噪声相关性；
- stock Wan DiT 不变；
- 记录额外参数、FLOPs、显存与吞吐。

## 单元测试

- batch 间无 token 串扰；
- view mask；
- view permutation；
- $V=1$ 退化合理；
- 每个联合 denoise step 确实通信；
- 关闭模块时与 E4 parity；
- checkpoint 向后兼容。

## Smoke

- 5-sample joint overfit；
- 一个 view 改变时其他 view 的 world token 响应；
- missing view；
- mixed $V$ batch；
- 长短 horizon。

## 消融

- E4；
- bottleneck token 数；
- read/write 层位置；
- 通信频率；
- independent/shared/mixed noise；
- 等参数 E4 baseline。

## Gate

G1、G2、G3 通过；相对 E4/MV-B0 获得统计支持的 world-consistency gain，且 per-view 不低于 floor。

## Failure action

无收益或回归则删除 E5-A，回退 E4；不因“更像 world model”保留。

## 产物

```text
experiments/E5-A_world_bottleneck/
  configs/
  communication_tests.json
  G3_metrics_by_seed.json
  capacity_compute_control.json
  memory_profile.json
  qualitative/
  README.md
```

---

# 17. E5-B：Geometry-Aware Cross-View Module

## 唯一变量

在 E4 上加入 geometry-aware cross-view module；world bottleneck 默认关闭。

## 实现

- 以 rays、calibration、visibility、effector/body geometry 调制 correspondence；
- 优先稀疏或局部通信；
- invalid view/ray/geometry 完全 mask；
- 首版在 VACE 内实现；
- 只有 VACE 表达受限的实验证据成立后，才 opt-in DiT attention。

## 单元测试

- 相机置换等变；
- 几何对应 synthetic scene；
- 遮挡 mask；
- 错 calibration 降低对应质量；
- rigid wrist closure；
- 关闭模块与 E4 parity；
- 稀疏索引不跨 sample。

## Smoke

- 已知 3D 点的双视角 correspondence；
- external↔wrist view；
- missing view；
- camera variation；
- 5-sample overfit。

## 消融

- E4；
- geometry-aware only；
- unstructured capacity-matched cross-view；
- rays only；
- effector/body geometry only；
- correct/perturbed calibration；
- 与 E5-A combined。

## Gate

G1、G2、G3 通过；收益不能仅由额外参数解释；calibrated geometry 在 reprojection/3D consistency 上优于错误 geometry。

## Failure action

回退 E4；若 only geometry routing 失败，不自动修改 Wan DiT。DiT opt-in 必须形成新实验分支、独立 parity 与预算。

## 产物

```text
experiments/E5-B_geometry_crossview/
  configs/
  correspondence_tests.json
  calibration_ablation.json
  G3_metrics_by_seed.json
  capacity_compute_control.json
  qualitative/
  README.md
```

---

# 18. E6：Cross-Embodiment Transfer

## 唯一变量

冻结已通过 G1–G3 的 Shared World Model，只改变 embodiment adapter 和本体条件。

## 实现

- 定义 Robot-A/B/C 数据使用白名单；
- seen、adapter-only、zero-shot 三档独立配置；
- adapter 参数、训练数据、标定数据单独计数；
- Robot-C zero-shot 只加载解析/预给 adapter；
- 统一 canonical schema 和 evaluation scenes/tasks；
- capacity-matched baselines；
- 防止 embodiment ID 泄漏。

## 单元测试

- frozen WM 参数 hash 前后一致；
- optimizer 不含 WM 参数；
- Robot-C 视频未出现在任何训练、normalization、selection cache；
- adapter 输出 schema 一致；
- 新本体可变 $K$、base 和 camera rig；
- zero-shot 无需学习新 ID embedding。

## Smoke

- 每档各少量 episode 推理；
- adapter-only 短训练能收敛；
- zero-shot 全链路；
- wrong adapter 对照；
- appearance/geometry 缺失时明确报错或 mask。

## 消融

- seen；
- adapter-only；
- zero-shot；
- no/wrong adapter；
- analytic vs learned adapter；
- frozen vs 非冻结 WM（仅诊断）；
- capacity-matched per-embodiment baseline；
- geometry/appearance 分别移除。

## Gate

完整通过 G4，并确认跨本体 G2/G3 仍成立。

## Failure action

- adapter-only 失败：定位 state/intent/geometry/appearance 哪一部分失配；
- zero-shot 失败但 adapter-only 成功：限制论文主张，不把其包装为 zero-shot；
- WM 必须解冻才成功：核心 frozen-transfer 假设未成立，回到 adapter/schema 分析。

## 产物

```text
experiments/E6_cross_embodiment/
  data_usage_manifest/
  frozen_parameter_hashes.json
  adapter_configs/
  seen/
  adapter_only/
  zero_shot/
  capacity_control.json
  G4_metrics_by_seed.json
  README.md
```

---

# 19. E7：Robustness / OOD

## 唯一变量

冻结 E6 选定方法，系统扫描预注册扰动与 OOD 切分。

## 实现

- Tier 1/2/3 独立 manifest；
- calibration noise 在 $SE(3)$ Lie algebra 与 intrinsic 参数空间施加；
- time jitter 重跑同步与 mask；
- missing view 区分随机缺失和连续缺失；
- trajectory noise 来自 adapter error model；
- 记录 corruption seed 和 severity。

## 单元测试

- zero severity 与 clean parity；
- 噪声单位、frame 和协方差；
- corruption 只影响目标字段；
- missing view 不变成黑图有效帧；
- severe corruption 不产生未处理 NaN。

## Smoke

- 每种 corruption 最低/中/高各一批；
- OOD scene/task/camera；
- variable $V$；
- moderate calibration noise；
- trajectory noise。

## 消融

- clean；
- 各 corruption 单独；
- 组合 corruption；
- uncertainty-aware vs unaware；
- robust training vs no robust training；
- MV-B0/E4 与 joint 模型对照。

## Gate

Tier 1 必须达到预注册主结果要求；Tier 2 报告稳定退化曲线；Tier 3 作为压力测试，不以单点成败定义方法正确性。

## Failure action

- Tier 1 失败：不得提交强泛化主张；
- Tier 2/3 失败：明确 operating envelope，按收益决定是否增加 robust training；
- 不允许在 test corruption 上反复调阈值。

## 产物

```text
experiments/E7_robustness_ood/
  tier_manifests/
  corruption_configs/
  robustness_curves.json
  OOD_metrics_by_seed.json
  paired_bootstrap.json
  failure_gallery/
  README.md
```

---

# 20. 测试矩阵

## 20.1 表示与几何

- R1：quaternion xyzw 解析；
- R2：canonical sign；
- R3：hemisphere continuity；
- R4：180° 邻域连续性；
- R5：rotation-6D round-trip；
- R6：SE(3) inverse/composition；
- R7：body/right 与 spatial/left increment 区分；
- R8：Lie-log 平移/旋转统计；
- R9：rigid wrist closure；
- R10：完整 Plücker $(d,m)$；
- R11：crop/resize 后 $K'$；
- R12：2D/3D reprojection。

## 20.2 数据与信息边界

- D1：episode split；
- D2：no adjacent leakage；
- D3：fixed test manifest；
- D4：timestamp monotonicity；
- D5：max skew；
- D6：interpolation provenance；
- D7：condition/target/oracle 分离；
- D8：future target 替换不改变 condition；
- D9：schema/version/unit/shape/dtype；
- D10：calibration version；
- D11：left/right permutation；
- D12：view permutation。

## 20.3 可变本体与视角

- V1：$K=1$；
- V2：$K=2$；
- V3：mixed $K$ batch；
- V4：optional base；
- V5：$V=1$；
- V6：$V=2,3$；
- V7：mixed $V$ batch；
- V8：missing view；
- V9：camera ID-only/role-only/combined；
- V10：body/link geometry mask。

## 20.4 条件因果

- C1：same $S_0$, different desired；
- C2：same desired, different $S_0$；
- C3：correct；
- C4：zero；
- C5：matched shuffle；
- C6：reverse；
- C7：robot-only physically valid；
- C8：camera-only physically valid；
- C9：both physically valid；
- C10：desired noise；
- C11：predicted uncertainty；
- C12：realized Oracle 隔离。

## 20.5 多视角一致性

- M1：MV-B0 无通信；
- M2：sample 内同 timestep；
- M3：joint step 通信；
- M4：world bottleneck only；
- M5：geometry-aware only；
- M6：combined；
- M7：independent noise；
- M8：shared-global+view-local noise；
- M9：identity/state/event；
- M10：2D reprojection；
- M11：可用时 3D consistency；
- M12：pixel inequality 合理性。

## 20.6 Cross-embodiment 与鲁棒性

- X1：seen；
- X2：adapter-only；
- X3：zero-shot Robot-C；
- X4：wrong adapter；
- X5：frozen WM hash；
- X6：capacity control；
- X7：Tier 1；
- X8：Tier 2；
- X9：Tier 3；
- X10：等数据/参数/算力。

---

# 21. 实验产物与可复现性

## 21.1 统一目录

所有阶段采用：

```text
experiments/
  E{N}_{name}/
    run_{seed}_{config}/
      config.yaml
      code_version.json
      data_manifest.json
      adapter_manifest.json
      calibration_manifest.json
      checkpoint/
      train.log
      smoke.log
      metrics.json
      predictions_manifest.json
      qualitative/
    aggregate/
      metrics_by_seed.json
      paired_bootstrap.json
      compute.json
      README.md
```

该目录是计划要求的实验产物规范，不表示当前仓库已存在。实际落地位置在 E0 核验后写入 `protocol.yaml`。

## 21.2 每阶段 README 必答

1. 本阶段唯一变量是什么；
2. 相对冻结父节点改变了哪些条件、参数和算力；
3. 单测、smoke、5-sample overfit、Path A parity 是否通过；
4. 采用了哪些 seeds、split 和 checkpoint rule；
5. 消融与 capacity control 是否完整；
6. 对应 Gate 的点估计、CI 和方向稳定性；
7. 失败样本与已知限制；
8. 是否推进、回退或终止该分支；
9. Oracle 是否严格分离；
10. 复现所需版本与命令。

## 21.3 依赖文档清单

实施前只引用已在仓库核验存在的文档和入口。至少核验：

- Path A baseline 的真实执行说明；
- A2V setup/环境说明；
- VACE unit 与 stock Wan/VACE 实现；
- data loader 与 projection smoke；
- parity、train、infer、metric 入口。

本计划不保留不可访问的外部行号引用标记，也不伪造未核实的仓库相对路径。核验后的路径写入 E0 `protocol.yaml` 和阶段 README。

---

# 22. 显存、参数与算力预算

## 22.1 预算原则

每次实验必须报告：

- trainable/total parameters；
- peak allocated/reserved memory；
- tokens per view/sample；
- samples/sec 与 frames/sec；
- GPU model/count/hours；
- estimated FLOPs 或可靠 proxy；
- activation checkpointing、precision、offload；
- inference latency。

## 22.2 分阶段策略

### E0–E4

- shared per-view VACE；
- 轻量 trajectory/geometry/camera encoders；
- 局部 $BV$ 展开；
- 不使用 dense all-to-all cross-view attention。

### E5-A

- 小规模 world bottleneck；
- 扫描少量 token 数；
- 限制通信层和频率；
- 与 capacity-matched E4 对照。

### E5-B

- 优先 sparse/local geometry routing；
- 限制每 token correspondence 数；
- 记录 geometry index 构建成本；
- dense attention 只作受控上界。

## 22.3 OOM 处理顺序

1. 降低 micro-batch，保持 global batch；
2. gradient accumulation；
3. activation checkpointing；
4. mixed precision；
5. 降低训练分辨率或 clip length，并重做公平对照；
6. 减少 world tokens/通信频率；
7. 最后才改变 backbone。

OOM 后改变数据量、分辨率或训练步数必须同步更新 control；不得把算力减少后的结果直接与原设置比较。

---

# 23. 禁止的 Shortcuts

以下行为禁止：

1. Shared WM 直接读取 native joints/action；
2. 将 body/right increment 命名为 world action；
3. 用 future realized robot/camera/object/contact/image 作正式条件；
4. 将 future object state 塞入 desired robot trajectory；
5. 用 future realized pose 生成正式 geometry condition；
6. 将 desired state 误称 Oracle；
7. 将 realized Oracle 结果混入 deployable 主表；
8. 对 SE(3) 矩阵元素直接做方差；
9. 只保存派生 delta，不保存绝对 pose 与原时间戳；
10. 把 rigid wrist camera 做不满足 mount closure 的独立 shuffle；
11. 用“同向运动”替代严格 rigid transform closure；
12. 只保存 Plücker direction，不保存 moment；
13. 混淆 pose 9D、pose quaternion 7D 与 twist 6D；
14. 固定依赖数组槽位区分 left/right 或 camera role；
15. 将 ID/role 共线结果解释为 role 泛化；
16. 缺失 view 用黑图但标为有效；
17. MV-B0 声称 cross-view communication；
18. E2/E3 尚未通过就加入 world bottleneck；
19. E4 失败后直接加入 geometry-aware attention；
20. VACE 尚可用时直接修改 stock Wan DiT；
21. E5-A 与 E5-B 捆绑，无法归因；
22. 不做等数据、等参数、等算力对照；
23. 使用 frame/window random split 造成 adjacent leakage；
24. 根据 fixed test 反复调阈值；
25. 只报告单 seed 或最优 seed；
26. 只报告 FVD 而不报告条件遵循和多视角 world consistency；
27. 以 pixel equality 定义几何一致；
28. 失败后继续堆复杂度；
29. 跳过 Path A parity、5-sample overfit 或 projection smoke；
30. 使用未核实路径或不可访问引用。

---

# 24. 第一轮执行 Checklist

第一轮只推进 E0–E1，不开始 E5：

```text
[ ] 核验 Path A baseline 真实入口与配置
[ ] 核验环境、训练、推理、parity、metric 入口
[ ] 冻结 episode split 与 fixed test
[ ] 建立 >=3 seeds 与 paired bootstrap 协议
[ ] 记录 baseline 数据/参数/算力
[ ] 跑 Path A dry-run
[ ] 跑 data smoke
[ ] 跑 projection smoke
[ ] 跑 5-sample overfit
[ ] 跑 inference 与 metric pipeline
[ ] 定义 schema version、unit/frame/shape/dtype
[ ] 实现 canonical state schema
[ ] 实现 variable K/V 与 masks
[ ] 实现 EmbodimentAdapter contract
[ ] 分离 condition/target/oracle
[ ] 保存所有原始单调时间戳
[ ] 实现同步 provenance 与 skew gate
[ ] 实现 rotation/quaternion/SE(3) tests
[ ] 实现 rigid transform closure
[ ] 实现 reprojection 与完整 Plücker tests
[ ] 实现 left/right 与 view permutation tests
[ ] 实现 future leakage test
[ ] 新 loader 与 baseline parity
[ ] 生成 E0/E1 完整实验产物
[ ] Gate 通过后才进入 E2
```

---

# 25. 成功标准

## 25.1 工程正确

G1 全部通过，包含 Path A parity、loader gate、5-sample overfit、projection smoke、variable $K/V$、mask 和可复现 artifact。

## 25.2 条件正确

G2 证明模型对 current canonical state 与 canonical desired trajectory 存在方向正确、区域相关、统计支持的响应；desired 与 realized/Oracle 严格分离。

## 25.3 多视角世界一致

G3 证明 joint 模块相对 MV-B0 改善 identity/state/event 与几何一致性，同时保持 per-view floor；不要求不同视角像素相等。

## 25.4 Cross-embodiment 核心贡献成立

G4 在冻结 Shared WM 下区分 seen、adapter-only 与 zero-shot unseen embodiment，并完成 capacity/data/compute control。只有 zero-shot Robot-C 未使用任何视频训练或选择信息时，才使用“zero-shot unseen embodiment”表述。

## 25.5 泛化与鲁棒性成立

G5 Tier 1 达到最终主结果硬要求；Tier 2/3 给出可解释的退化曲线与 operating envelope。

## 25.6 最终判定

方法正确性：

$$
\boxed{G1\land G2\land G3}
$$

核心 cross-embodiment 贡献：

$$
\boxed{G4}
$$

泛化与鲁棒性：

$$
\boxed{G5}
$$

只有在对应 Gate 获得预注册统计支持后，才能形成论文主张。若 E5-A 或 E5-B 无收益，回退 E4 仍是合法结果；若 G2 未通过，则无论视频质量多高，都不能称为 desired world-state-conditioned world model；若 G4 未通过，则不能声称核心 cross-embodiment transfer 成立。

---

# 26. 当前未决项与锁定规则

以下不是架构原则争议，而是必须由 E0/E1 数据统计或资源测量确定的执行参数：

1. canonical trajectory 的目标频率与 horizon $H$；
2. 最大同步 skew 的最终阈值；
3. rotation/translation 数值容差；
4. predicted trajectory uncertainty 的具体表示；
5. G0 各主指标的最小效应与 paired bootstrap 次数；
6. per-view quality floor 与 G3 核心 consistency 指标；
7. moderate/severe calibration noise 的量级；
8. E5 world token 数、通信层与频率；
9. noise mixing 系数 $\alpha$ 的候选集合；
10. 等参数与等算力容差；
11. zero-shot Robot-C 可使用的解析 geometry、appearance 与 calibration 清单；
12. 实验产物目录在仓库中的最终落点；
13. VACE 表达不足并允许 opt-in DiT attention 的客观判据。

这些值应在查看训练集统计、baseline 误差和硬件 profile 后预注册，并冻结在 `protocol.yaml`；不得根据 fixed test 结果倒推。
