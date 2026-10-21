# G1 23DoF 连续十平台跨沟任务

尺寸按厘米解释：每个平台沿前进方向长 **90 cm**，横向宽 **60 cm**。
当前 `Instinct-Parkour-Target-Amp-G1-23dof-v0` 的沟壑场景为一条直线排列的 **10 个平台、9 处间隙**。
地形与 CommandsCfg 修改于共享的 `config/parkour_env_cfg.py`，因此引用它的 29DoF Parkour 任务也使用新场景。
机器人、动作维度、AMP 数据集和已有检查点没有改变。

## 地形与路线

| 名称 | 比例 | 用途 |
| --- | --- | --- |
| `square_gaps` | 70% | 连续 10 个粗糙平台，每个 0.90×0.60 m |
| `perlin_rough` | 20% | 粗糙地面行走、到点停止 |
| `perlin_rough_stand` | 10% | 粗糙地面静止站立，线速度和角速度均为零 |

训练配置及 `velocity_ranges` 都只保留这三个名称。
沟壑排在第一列，`--num_envs=1` 播放时分配到平台路线。

```text
前进方向 +X
出生 → [平台1] 间隙 [平台2] 间隙 … 间隙 [平台10] → 终点
        90 cm        90 cm                90 cm
             每个落脚面横向宽 60 cm
```

- 每处间隙在 **10–35 cm** 范围内，以 **2.5 cm** 为步长生成。难度控制允许的最大间隙；每条路线的间隙宽度随机排列，并包含当前允许的最大间隙。最低难度为九处 10 cm 间隙。
- 按所有间隙取最小/最大值计算，路线总长度范围为 **9.90–12.15 m**；默认混合宽度路线的实际长度取决于九处采样值。
- 出生点位于第一平台中心附近，最终目标位于第十平台中心附近，两者 X/Y 扰动均为 ±2.5 cm，落脚 patch 半径 20 cm。
- 自定义 `PlatformCourseTerrainGenerator` 使用与地形网格一致的几何计算，逐块地形设置终点。因此改变间隙宽度不会把目标留在沟内或错误的平台上。
- 平台两侧和间隙均为低处，没有与平台等高的侧边通路；根部离开 60 cm 宽路线或前后端点即失败，落入低处仍会触发根部高度终止。
- 平台等高，低处深度沿用 **0.4–0.6 m**。Perlin 噪声 `noise_scale=[0.01, 0.04]` 随难度增大，垂直分辨率 **1 mm**。噪声缩放参数不等于每点实际高度偏差。
- 场景采用 **10 行×10 列、每块 14×2 m**。14 m 是容纳整段路线的地形长度，2 m 是包含两侧低处的地形宽度；实际平台顶面宽度为 0.60 m。
- 每回合 **60 s**，目标重采样周期 **65 s**，前进速度上限采样范围 **0.35–0.65 m/s**。中间平台不要求停留；最终目标停止距离为 **0.15 m**。
- `perlin_square_gap_terrain` 增加 `layout="platform_course"`；原有 `square_ring` 和单横沟 `platform_gap` 布局仍可用于其他配置。

## 连续跨越与晋级

`Metrics/base_velocity/gaps_crossed` 范围为 **0–9**。只有按顺序到达下一平台，双脚都通过间隙、脚中心在该平台范围内，且至少一只脚接触地面，才增加计数；中途无需停下。

最终 `crossing_success=1` 需要：

1. 已按顺序跨过全部 9 处间隙。
2. 根部距离第十平台目标小于 0.15 m。
3. 双脚均处于最后平台的内部，横向脚中心距边缘至少 6 cm，纵向至少 15 cm，并且双脚竖直接触力均大于 1 N。
4. 身体保持直立（`projected_gravity_b.z < -0.9`），根部相对地形原点高度大于 0.5 m，水平速度小于 0.25 m/s。
5. 上述最终状态连续保持 1 s。

成功后仍在当前回合保持站立；回合结束没有随后跌倒或越界才增加地形难度。未完成整段或随后失败则降级。
站立/普通粗糙地面不计跨沟成功，混合环境日志均值不是纯平台路线成功率。正式评估应只统计路线回合，记录九次跨越进度、整段成功和失败终止。
23DoF Play 同样采用 60 s 时长和训练的高度、越界终止条件，初始难度可覆盖全部十级。

## 续训

从之前的 23DoF 模型继续训练，结果写入独立的平台路线目录：

```bash
./scripts/env.sh scripts/instinct_rl/train.py \
  --task=Instinct-Parkour-Target-Amp-G1-23dof-v0 \
  --headless --livestream=0 --num_envs=512 \
  --resume \
  --load_run=/workspace/instinctlab/logs/instinct_rl/g1_23dof_parkour/20261018_102434 \
  --checkpoint=model_30000.pt \
  --logroot=logs/instinct_rl/g1_23dof_platform_course \
  --max_iterations=10000
```

此命令恢复模型与优化器并额外训练 10,000 次 PPO 迭代，从 30,000 计数到 40,000；轮次数不代表预先保证整段成功。
从头训练时去掉 `--resume`、`--load_run`、`--checkpoint`，去掉 `--max_iterations` 后默认训练 30,000 轮。

## 播放

```bash
./scripts/env.sh source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-23dof-Play-v0 \
  --load_run=/workspace/instinctlab/logs/instinct_rl/g1_23dof_platform_course/<新运行目录> \
  --checkpoint=model_40000.pt \
  --num_envs=1 --livestream=2
```

检查点名称以实际文件为准。

## 配置与运行验证

```bash
./scripts/env.sh scripts/check_parkour_gaps.py \
  --checkpoint=logs/instinct_rl/g1_23dof_parkour/20261018_102434/model_30000.pt \
  --headless --livestream=0
```

检查内容：3 个种子×3 个难度的 90 个平台长度/宽度及 81 处间隙实测；所有地形的首尾平台 patch；顺序计数、跳过平台/悬空/单脚未过沟等反例；实际 PhysX 平台顶面和沟底碰撞；旧模型加载及短程仿真；重置清空进度。
结果位于 `.environment/platform-course-check/`。上一版单横沟的记录仍保留在 `.environment/parkour-gaps-check/`，不能作为新路线的通过率。

这些检查确认配置、几何和训练流程可运行。连续十平台的策略成功率需在专项训练后评估，尚未进行实体机器人验证。
