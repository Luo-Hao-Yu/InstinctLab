# 本机环境

本机 Ubuntu 20.04 / glibc 2.31 不满足 Isaac Sim 5.1 的 glibc 2.35 要求，
因此使用独立 Docker 环境；系统 Python、已有 IsaacLab、instinct_rl 仓库及其他容器保持原样。

## 固定版本

- 基础镜像：本机 `isaac-lab-base:latest`，原始 ID
  `sha256:27432b4101ea6a04a0c20440948f18733997ebb03e21527db5848331f664d0d6`。
- Isaac Sim：镜像内 5.1.0，`VERSION` 原文为
  `5.1.0-rc.19+release.26219.9c81211b.gl`。
- Python：3.11.13；PyTorch：2.7.0+cu128；TorchVision：0.22.0+cu128；NumPy：1.26.0。
- Isaac Lab：README 指定的 `f73c33173801f5f8afea4142482e47b7710c2b75`，源码 VERSION 为 2.3.2。
- Instinct-RL：`c3ef0d5f47834228e49d29e23768f8459abea758`，包版本 1.0.2。
  本机现有 main 要求 Python 3.12、PyTorch 2.11 和 NumPy 2，
  与本项目的 Isaac Sim 5.1 / Isaac Lab 环境冲突，故固定到 Python 3.11 兼容版本。
- InstinctLab：以 editable 模式安装，宿主机代码修改在容器中直接生效。

新镜像修复了基础镜像的 pip 符号链接及若干缺失依赖，并升级 FastAPI 以匹配
Isaac Lab 固定的 Starlette。移除了未使用的可选 `nvidia-srl-usd-to-urdf` 包：
它要求 lxml<5，与 Isaac Lab 的 dex-retargeting 所需 lxml>=5.2.2 无法同时满足。
此环境面向强化学习，未配置 USD 转 URDF 导出功能。

## Parkour 动作数据

已下载用户提供的 [数据包](https://drive.google.com/file/d/10tQylYHdKLVDVnmVHrygLB70neHnPdoF/view)，
压缩包为 `hiking-in-the-wild_Data&Model.zip`（11,905,690 字节）。
解压内容位于项目的 `datasets/parkour`，已排除 macOS 元数据。

- 动作数据：`datasets/parkour/parkour_motion_reference/parkour_motion_without_run_retargetted.npz`。
- 动作筛选：同目录的 `parkour_motion_without_run.yaml`。
- 预训练 ONNX 模型：`datasets/parkour/checkpoints/stand_onboard` 和
  `datasets/parkour/checkpoints/parkour_onboard_preview_stair`。
- 来源及文件校验：`datasets/parkour/SOURCE.json`。
- 原始压缩包：`.environment/downloads/hiking-in-the-wild_Data&Model.zip`。

`g1_parkour_target_amp_cfg.py` 的两个动作路径已改为相对于项目根目录定位，
宿主机和容器均可找到对应数据。容器中动作目录为
`/workspace/instinctlab/datasets/parkour/parkour_motion_reference`，无需添加额外挂载。
数据包含 18,982 帧、29 个 G1 关节，采样率为 50 Hz。
`datasets/` 已加入 `.gitignore`。

训练命令：

```bash
./scripts/env.sh scripts/instinct_rl/train.py \
  --headless --task=Instinct-Parkour-Target-Amp-G1-v0
```

本次只检查数据与路径，未启动正式训练。

## 使用

在项目根目录运行：

```bash
# 进入环境；进入后 python 已指向 Isaac Sim 的 Python 启动脚本
./scripts/env.sh

# 也可直接从宿主机运行 Python 命令
./scripts/env.sh --version
./scripts/env.sh scripts/check_environment.py
./scripts/env.sh scripts/instinct_rl/train.py --help
```

容器名为 `instinctlab-dev`，环境镜像为 `instinctlab-env:local`。
运行入口会自动启动已停止的容器。停止环境：`docker stop instinctlab-dev`。
项目挂载在 `/workspace/instinctlab`，无需启动其他已有容器。

## Parkour ONNX 导出与验证

2026-10-13 已在当前容器安装 `onnxruntime==1.23.2`（CPU 推理版），
并将依赖加入 `docker/Dockerfile.local`；现有镜像需重建后才会包含该依赖。
`pip check` 通过。仅导出使用 `--exportonnx`，加载导出模型播放时再加 `--useonnx`。
导出要求 `--num_envs=1`。
无需播放时使用 `--exportonnx --export_only --headless --livestream=0`，
脚本在模型写出后关闭环境并退出，不进入播放循环。

已检查 `logs/instinct_rl/g1_parkour/20261009_200640/exported/` 中两个模型：

- `0-depth_encoder.onnx`：opset 18，输入 `[1, 8, 18, 32]`，输出 `[1, 128]`。
- `actor.onnx`：opset 12，输入 `[1, 896]`，输出 `[1, 29]`。

两个文件均通过 ONNX 完整结构校验和 ONNX Runtime 加载。
使用 `model_30000.pt` 的权重对照 20 组测试输入，完整策略最大绝对误差为
`1.55e-5`，通过 `atol=1e-4, rtol=1e-4` 的数值检查，项目自带加载函数也通过。
这是离线 CPU 推理验证，未验证实际部署设备或 WebRTC 播放。
记录保存在 `.environment/parkour-onnx-verification.json` 和同名 `.log` 文件中。

可在宿主机项目根目录重新检查：

```bash
./scripts/env.sh .environment/check_parkour_onnx.py
```

## 容器配置

容器以 root 运行，挂载仓库由宿主机用户拥有。已将 `/workspace/instinctlab`
加入当前容器的 Git `safe.directory`，并在 Dockerfile 和 Dev Container 初始化中
保留此设置，避免训练记录 Git 版本时出现 `SHA is empty / dubious ownership`。
只信任这个项目路径。

VS Code 安装 Dev Containers 扩展后，执行 **Dev Containers: Reopen in Container**。
`.devcontainer/devcontainer.json` 已设置 Python 解释器和主要源码索引路径。
仓库原有 `setup_vscode.py` 缺少其所需的模板文件，此处采用容器配置。

## Shadowing 训练数据

README 示例对应任务需要外部动作数据，仓库未附带这些数据。
当前 `plane_shadowing_cfg.py` 中实际生效的路径为：

```text
~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single
```

此处 `~` 在容器中为 `/root`。训练前需将自己的 retargeted NPZ 数据放入项目目录，
并将配置中的 `AmassMotionCfg.path` 改为容器可见的 `/workspace/instinctlab/...` 路径。
同时按任务 README 设置动作名称和选择文件；配置中还存在其他预设数据路径。

```bash
./scripts/env.sh scripts/instinct_rl/train.py \
  --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless
```

GPU 物理验证脚本不依赖外部动作数据。

## 验证结果

2026-10-09 已完成以下检查：

- `pip check`：No broken requirements found。
- 训练入口 `train.py --help` 正常运行。
- Isaac Sim 在 RTX 4090 上成功启动，CUDA 张量运算正常。
- 已注册 14 个 Instinct 任务，包含 README 示例任务。
- 十步 GPU 刚体仿真通过，测试立方体高度从约 0.992 降至 0.787。

完整输出保存在 `.environment/verification.log`，
已安装包清单保存在 `.environment/packages.txt`。
短验证完成后直接结束进程，避免 Kit 后台服务在关闭阶段等待。
Shadowing 完整训练尚未执行，需要先提供并配置动作数据集。

## 重建

固定源码保存在已忽略的 `.environment/IsaacLab` 和 `.environment/instinct_rl`，
完整版本记录在 `.environment/versions.json`，构建日志在 `.environment/build.log`。
在这些目录及基础镜像仍存在的本机重建：

```bash
docker build --network host \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
  --build-arg http_proxy --build-arg https_proxy \
  -f docker/Dockerfile.local -t instinctlab-env:local .
```

如需在其他机器准备源码，分别下载以下固定提交的源码压缩包并解压到上述目录：

- https://github.com/isaac-sim/IsaacLab/archive/f73c33173801f5f8afea4142482e47b7710c2b75.tar.gz
- https://github.com/project-instinct/instinct_rl/archive/c3ef0d5f47834228e49d29e23768f8459abea758.tar.gz

新建运行容器（同名容器不存在时）：

```bash
docker run -d --name instinctlab-dev --runtime=nvidia --gpus all --network host --shm-size=4g \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -v "$PWD:/workspace/instinctlab" \
  --entrypoint sleep instinctlab-env:local infinity
```

`pre-commit` 已随环境安装，可用 `./scripts/env.sh -m pre_commit run --all-files` 手动执行。
该命令可能自动格式化仓库文件。自动提交钩子尚未启用。
