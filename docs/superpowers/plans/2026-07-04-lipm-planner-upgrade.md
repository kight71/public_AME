# 3D-LIPM / ICP 落足点规划器升级计划书

> **状态：** 草案（2026-07-04）  
> **前置：** Phase A 本地优化 selector（`select_foothold_v2`）已完成并接入 `PlannerV2`  
> **参考：** 陈龙硕士论文 §4.3.2（3D-LIPM 平地落足点生成）；现有 `2026-07-02-planner-upgrade.md` Phase B/C  
> **Goal：** 用 **基于模型的 ICP/LIPM 名义落足点** 替换当前 **Raibert + phantom + landing horizon 三重外推**，使落足目标与机器人 **当前可执行能力** 对齐，并保留 v2 地形 selector 做 rough 地形投影。

---

## 1. 执行摘要

| 项目 | 内容 |
|------|------|
| **要改什么** | 名义 xy 生成：Raibert prior → LIPM/ICP prior（论文 4-5–4-11） |
| **不改什么** | `select_foothold_v2` 硬/软约束、foot polygon 几何、DTC commit-on-crossover 时序 |
| **新增 env** | `AME-G1-29DOF-DTC-PlannerV2-LIPM-v0`（及 PlaceRew / Play 变体） |
| **预期效果** | `plan_step_mean_dx` 从 ~0.6m 降至 ~0.25–0.45m；placement 闭环可学习 |
| **非目标（本阶段）** | 楼梯语义地图、(4-12) tread 硬约束、full MPC、learned planner |

---

## 2. 问题诊断：是不是「太过理想，机器人追不上」？

**结论：是。** 但需拆成两层——**规划参考系理想化** 与 **策略侧闭环未接通**——二者叠加，表现为 `overlap ≈ 0`、`track_lin_vel` 低、机器人「站着不倒但走不出去」。

### 2.1 规划侧：三重「理想未来」叠加

当前 `PlannerV2`（`use_phantom=True`, `max_leash=0.5`, `raibert_factor=0.3`）在 `_commit_plan` 里对落足点做了 **三层向前假设**：

```
真实 robot root (可能已 stall)
        │
        ▼ ① Phantom：每步按 v_cmd 前进，leash 最多 0.5m
phantom_pos_w
        │
        ▼ ② Landing horizon：再外推 horizon·v_cmd（~0.3–0.6s）
landing_root_pos
        │
        ▼ ③ Raibert：hip_offset + 0.3·t_swing·v_cmd
raibert_xy_w  →  v2 selector  →  target_w
```

**设计意图（注释原文）：** phantom 在 robot stall 时仍前进，避免目标落在身后。  
**实际后果：** 当策略尚未学会跟速、robot 几乎不动时，目标仍被推到 **前方 0.5m + 0.2~0.45m + Raibert 项** 处；脚还在髋下，目标已在远处。

**量化证据（PlaceRew @ iter ~95）：**

| 指标 | 典型值 | 含义 |
|------|--------|------|
| `plan_step_mean_dx` | ~0.6–0.7 m | target 相对 **真实 root** 的前向距离 |
| `error_vel_xy` | ~0.45–0.55 m/s | 实际速度远低于 v_cmd |
| `overlap` | ~0.0002 | 脚踩 target 矩形几乎无重叠 |
| `episode_length` | ~320（PlaceRew） | 能站住，但不等于走得好 |

**对比合理步长：** 论文 4-5 在脚步开始时 `δT = Ts`，`sd = |v|·Ts` → 0.75×0.6 ≈ **0.45 m**（且这是 ICP 动力学步长，不是相对 stall pelvis 的 phantom 偏移）。

### 2.2 策略侧：闭环信号太弱

即使目标合理，当前 `PlannerV2` 还 **主动关掉了摆腿跟踪**：

- `footstep_swing_tracking.weight = 0.0`（为观察 planner 信号而设，见 Phase A plan Task 7）
- policy obs 去掉 `footstep_phase_info` / `swing_side`（简化 obs，但削弱相位感知）
- `overlap` 仅在 contact 时计、量级 ~1e-4，相对 `track_ang_vel` 等可忽略

因此：**不是 selector 坏了，而是「目标太远 + 没有足够强的跟踪/落足 shaping」**，策略学不到「把脚踩到 target 上」。

### 2.3 问题本质（一句话）

> 规划器假设 robot **已经** 以 v_cmd 在理想轨迹上运动；真实 robot **还没跟上**。目标在「理想 future pelvis」坐标系里合理，在「当前真实 pelvis」坐标系里 **不可达**。

LIPM 升级的核心：**名义落足点从当前 CoM/ICP 与上一步 stance foot 出发，步长由 `|v|·δT` 约束，不再用 phantom 叠 horizon。**

---

## 3. 目标架构

### 3.1 数据流（目标态）

```
velocity cmd (vx, vy, wz)
        │
        ▼
┌─────────────────────────────────────┐
│  LIPM/ICP nominal planner (NEW)      │
│  · commit 时刻：真实 root/CoM 状态    │
│  · sd = |v|·δT,  wd = |w|·δT        │
│  · (4-7) ξf, (4-9)(4-10) → p̂_nominal │
│  · 转弯：(4-11) 旋转偏移             │
│  · 可选：Δh 缩放 sd（泛化 4-13）     │
└──────────────┬──────────────────────┘
               │ p̂_xy (world)
               ▼
┌─────────────────────────────────────┐
│  select_foothold_v2 (EXISTING)       │
│  prior: lipm_xy 替换 raibert_xy      │
│  body_pos_w: 当前 root（非 landing）  │
│  + patch / reach / step-height       │
└──────────────┬──────────────────────┘
               ▼
        target_w → obs / overlap / swing_tracking
```

### 3.2 与论文的对应关系

| 论文 | 本计划落地 | 备注 |
|------|-----------|------|
| (4-5)(4-6) sd, wd | `lipm_step_length(v_cmd, delta_T, w)` | δT 来自 phase clock |
| (4-2) 初始 ICP | `xi = com_xy + com_vel_xy / omega0` | z0≈0.78m → ω0≈√(g/z0) |
| (4-7) 步末 ICP | `predict_icp_end(xi0, p_stance, Ts)` | 指数衰减到 stance foot |
| (4-9)(4-10) 落足点 | `lipm_nominal_foothold(...)` | n 奇偶 → 左右脚 y |
| (4-11) 转弯 | `theta = atan2(vy, vx)` 旋转 b | 与现有 wz 外推可并存 |
| (4-12) 楼梯 tread | **本阶段不做** | 由 v2 patch + reach 替代 |
| (4-13) 台阶高度缩步长 | **Phase L1** 泛化为 local Δh | 不限 stair mesh |

### 3.3 关键配置开关

```python
# FootstepPlanCommandCfg 新增
use_lipm_prior: bool = False          # True → LIPM 替代 Raibert prior
use_phantom: bool = True              # LIPM 路径必须 False（互斥）
lipm_com_height: float = 0.78         # 或 online: root_z - terrain_z
lipm_step_width: float = 0.24         # 论文 w，左右髋间距量级
lipm_use_turning: bool = True         # (4-11) vs (4-10)
lipm_sd_terrain_alpha: float = 0.0    # Phase L1: 类似 (4-13) 的 Δh 缩放
```

**LIPM 与 phantom 二选一：** 保留 phantom 仅作 legacy A/B，新 LIPM env 默认 `use_phantom=False`。

---

## 4. 实施任务清单

### Phase L0 — 核心数学 + 接线（平地 / rough，优先）

#### Task L0-1：纯数学模块 `planner_lipm.py`

**文件：**
- Create: `source/ame_locomotion/.../mdp/planner_lipm.py`
- Create: `tests/test_planner_lipm.py`

**内容：**
- [ ] `omega0_from_com_height(z0, g=9.81) -> Tensor`
- [ ] `icp_xy(com_xy, com_vel_xy, omega0) -> Tensor`
- [ ] `predict_icp_end(xi0, p_stance_xy, omega0, Ts) -> Tensor`  — 式 (4-7)
- [ ] `lipm_offset_b(xi_f, xi_0, p_stance_xy, sd, wd, delta_T, omega0) -> Tensor`  — 式 (4-9)
- [ ] `lipm_nominal_foothold(..., swing_side, use_turning) -> Tensor`  — 式 (4-10)/(4-11)
- [ ] `lipm_step_length(v_cmd_xy, delta_T, w_cmd, Ts) -> (sd, wd)`  — 式 (4-5)/(4-6)

**约束：** 仅 `import torch`；与 `planner.py` 相同，Isaac-free。

**单元测试：**
- [ ] v=0.75, Ts=0.6, t=0 → sd ≈ 0.45 m
- [ ] δT=Ts/2 → sd ≈ 0.225 m（步中剩余时间减半，步长减半）
- [ ] 直线 vs 转弯：(4-10) 与 (4-11) 在 vy=0 时一致
- [ ] ICP 传播：已知 ξ0, p → ξf 与手算一致

---

#### Task L0-2：接入 `FootstepPlanCommand`

**文件：**
- Modify: `mdp/commands.py`（`FootstepPlanCommandCfg`, `_commit_plan_v2`）
- Modify: `mdp/foothold_selection.py`（prior 参数 rename/doc：`nominal_xy_w` 泛化名，可选）

**内容：**
- [ ] `FootstepPlanCommandCfg` 增加 LIPM 字段（见 §3.3）
- [ ] 新增 `_lipm_targets_at_commit(env_ids, ...)`：
  - 输入：**真实** `root_pos`, `root_vel`, `last_contact_w[stance_foot]`
  - `delta_T = (1 - phase_at_crossover) * t_step` 或简化为 `t_step`（论文 t=0）
  - 输出 `(lipm_xy_w, body_yaw)` 替代 `_raibert_targets_for_horizons` 的 prior 部分
- [ ] `_commit_plan_v2` 分支：`use_lipm_prior` 时
  - `body_pos_w = root_pos`（当前 pelvis，**非** landing_root）
  - `raibert_xy_w` 槽位传入 `lipm_xy_w`
- [ ] `use_phantom=False` 时 `_commit_plan` 已用真实 root — 与 LIPM 一致

**不改：** crossover commit 时序、plan_buffer 结构、z 投影仍走 selector。

---

#### Task L0-3：Env 变体 + Gym 注册

**文件：**
- Modify: `29dof/velocity_env_cfg_29dof.py`
- Modify: `29dof/__init__.py`

**新增配置函数：** `_configure_planner_lipm_env(env_cfg, *, use_placement_rewards=False)`

| 配置项 | LIPM 默认 | 说明 |
|--------|-----------|------|
| `use_lipm_prior` | True | |
| `use_phantom` | False | 关 phantom |
| `raibert_factor` | 0.0 | LIPM 路径不用 Raibert |
| `footstep_swing_tracking.weight` | **1.0** | 恢复摆腿跟踪 |
| `footstep_phase_info` / `swing_side` obs | **恢复** | 扩展 `_PlannerV2PolicyCfg` 或新 `_PlannerLIPMPolicyCfg` |
| placement rewards | 可选 | PlaceRew-LIPM 变体 |

**Gym IDs：**
- [ ] `AME-G1-29DOF-DTC-PlannerV2-LIPM-v0`
- [ ] `AME-G1-29DOF-DTC-PlannerV2-LIPM-PlaceRew-v0`
- [ ] `-Play-v0` 各一

**Legacy 对照（A/B）：**
- 现有 `PlannerV2` / `PlaceRew` 不动，用于对比。

---

#### Task L0-4：诊断与 smoke 脚本

**文件：**
- Create: `scripts/debug/run_lipm_smoke.sh`
- Optional: extend `monitor_task8b.sh` metrics

**Smoke 检查（无需完整 15k）：**
- [ ] `python -m pytest tests/test_planner_lipm.py`
- [ ] Play 1 env：`plan_step_mean_dx` 打印 < 0.45m（stall 时）
- [ ] 256 env × 200 iter smoke：无 NaN / fallback_rate 爆炸

**Phase L0 验收门（iter 500）：**

| 指标 | 门槛 | 对比 PlannerV2-PlaceRew |
|------|------|-------------------------|
| `plan_step_mean_dx` | < 0.40 m | ~0.6 m |
| `overlap` | > 0.05 | ~0.0002 |
| `track_lin_vel_xy_exp` | > 0.15 | ~0.02 |
| `episode_length` | > 200 | ~320（可略降，但要有前进） |

---

### Phase L1 — 地形泛化（非楼梯专用）

#### Task L1-1：步长随 local Δh 缩放

- [ ] 从 height_scanner 在 nominal xy 处读 `dh = z_nominal - z_stance`
- [ ] `sd_eff = sd * (1 - alpha * clamp(dh - 0.15, min=0))`（泛化式 4-13）
- [ ] 单元测试：dh=0.25 → sd 减小 ~10%（alpha=1）

#### Task L1-2：最小步长与安全裕度

- [ ] `smin = x1 + x2` 概念映射到 `selector_v2` reach + `max_step_dz`
- [ ] `sd = max(sd, smin)` 当 patch 约束导致 nominal 过近台阶边

#### Task L1-3：可选 DCM 软代价

- [ ] `J_dcm = ||xi_xy - p_nominal_xy||²` 加入 selector soft cost（Phase B from 2026-07-02 plan）
- [ ] 权重 `w_dcm` 初值 0.3，smoke A/B

---

### Phase L2 — 楼梯 / 语义（后续，本计划不阻塞 L0）

- [ ] 有 stair semantic（级数 k, d_step, w_step）时启用 (4-12) hard box
- [ ] 每级单点/多点判定 (4-14)(4-17)
- [ ] 与室内语义地图模块对接（论文后续章节）

---

## 5. 策略侧配套（与 LIPM 同步启用）

LIPM 只解决「目标距离合理」；**还必须恢复闭环**：

| 项 | PlannerV2 现状 | LIPM env 目标 |
|----|----------------|---------------|
| `footstep_swing_tracking` | 0.0 | **1.0** |
| `footstep_phase_info` obs | 无 | **有** |
| `footstep_swing_side` obs | 无 | **有** |
| `footstep_placement_overlap` | 0 或 1.0 | PlaceRew 变体 **1.0** |
| `termination_penalty` | 保留 | 保留（勿同时删 alive） |

**原因：** 无 swing_tracking 时，policy 无摆腿期梯度把脚拉向 target；无 phase obs 时难以学相位对齐。

---

## 6. 任务依赖图

```
Task L0-1 (planner_lipm.py + tests)
        │
        ▼
Task L0-2 (commands.py wiring)
        │
        ├──────────────────┐
        ▼                  ▼
Task L0-3 (env + gym)   Task L0-4 (smoke)
        │
        ▼
   [Gate: iter 500 验收]
        │
        ▼
Task L1-1 ──► L1-2 ──► L1-3 (optional DCM cost)
        │
        ▼
   [Gate: iter 3000 A/B vs PlannerV2-LIPM]
        │
        ▼
Phase L2 (stairs semantic — 独立里程碑)
```

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LIPM 参数 z0 不准 | 先用常数 0.78；L1 用 root_z - ray median |
| 步长仍偏大 | δT 用 phase 剩余时间；加 `sd_max` cap |
| 恢复 swing_tracking 后 early training 不稳定 | curriculum：前 500 iter swing_tracking=0.5 → 1.0 |
| v2 reach 仍按 landing frame 写死 | L0-2 明确改 `body_pos_w=root_pos` |
| 与 pretrained 不兼容 | LIPM env 从头训（与 Footstep 相同） |

---

## 8. 与现有文档的关系

| 文档 | 关系 |
|------|------|
| `2026-07-02-planner-upgrade.md` Phase A | **已完成** — v2 selector 保留 |
| 同文档 Phase B (DCM/LIP) | **本计划 = Phase B 的 LIPM prior 落地** |
| 同文档 Phase C (multi-step) | L2 以后 |
| `PLAN.md` Phase 3 MPC/LIP | 本计划是 Phase 3 的 **analytic 第一步**，非 full MPC |
| 陈龙论文 §4.3.2 | LIPM 公式来源；楼梯 (4-12) 延到 L2 |

---

## 9. 建议执行顺序（给人 / 给 agent）

1. **读 + 测：** 跑 `test_planner_lipm.py`（Task L0-1）
2. **接：** `_commit_plan_v2` LIPM 分支（Task L0-2）
3. **配：** `PlannerV2-LIPM-PlaceRew` env（Task L0-3）
4. **验：** Play 看 `plan_step_mean_dx`；256×200 smoke（Task L0-4）
5. **训：** server 上 500 iter 门禁 → 通过再 3k A/B
6. **扩：** L1 地形 Δh、DCM soft cost

**预计工作量：** L0 约 1–2 人天（含测试）；L1 再 1 人天；L2 视语义地图而定。

---

## 10. 附录：当前 vs LIPM 数值示例

假设：`v_cmd=0.75 m/s`, `Ts=0.6 s`, `t_swing=0.3 s`, phantom leash=0.5 m, robot stall。

| 分量 | 当前 PlannerV2 | LIPM @ step start |
|------|----------------|-------------------|
| Phantom offset | up to 0.5 m | **0**（不用 phantom） |
| Horizon extrapolation | ~0.75×0.45≈0.34 m | **0**（不外推 landing pelvis） |
| Raibert push | 0.3×0.3×0.75≈0.07 m | **0** |
| 模型步长 sd | — | **0.75×0.6=0.45 m**（相对 ICP 几何，非 pelvis 偏移叠加以） |
| `plan_step_mean_dx`（stall） | **~0.6 m** | **目标 <0.4 m** |

> 注：LIPM 的 0.45 m 是 **捕获动力学意义下的步长**，落足点相对当前 stance/CoM 的实际 xy 距离由 (4-9)(4-10) 决定，通常 **小于** phantom 叠加以的总偏移。

---

## 11. 决策记录

| 日期 | 决策 |
|------|------|
| 2026-07-04 | 确认根因：phantom+horizon 理想化导致 stall 时目标不可达 |
| 2026-07-04 | 采用 LIPM prior + v2 selector，不单独写楼梯 planner |
| 2026-07-04 | LIPM env 必须恢复 swing_tracking + phase obs |
| TBD | iter 500 门禁通过后，是否将 LIPM 设为 DTC 默认 |
