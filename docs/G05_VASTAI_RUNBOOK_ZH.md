# G0.5 12 任务云端训练 Runbook

## 当前已核验输入

- 数据：`data/lerobot_v30_joint`，LeRobot v3，`meta/info.json` 报告 1200 episodes / 12 tasks / 592432 frames。
- Sidecar：`/mnt/pqssd/RoboInter/RoboInterTools/annotations/lerobot_v30_joint.sqlite3`。`lang.video` 是全局 Task，`auto_annotation.segments[]` 提供边界和 `subgoal`。
- 生成的 `subgoal_samples.jsonl` 保留 `task`，并增加 `subtask: Subtask: ...`；窗口尾部不会越过 segment 边界，短尾样本通过 `action_horizon` 提示 mask/pad。`balanced_samples.jsonl` 按 task-uniform → episode-uniform → frame/sample 抽样，避免长任务占优。

## 本机准备（不下载）

```bash
export ROBODOJO_SIDECAR=/mnt/pqssd/RoboInter/RoboInterTools/annotations/lerobot_v30_joint.sqlite3
bash scripts/g05/run_smoke.sh
```

## 一条 SSH 命令启动 VastAI 训练

先在本机准备一次密钥文件，不要提交到 Git：

```bash
mkdir -p ~/.config/robodojo
cp scripts/g05/vast.env.example ~/.config/robodojo/g05_vast.env
chmod 600 ~/.config/robodojo/g05_vast.env
$EDITOR ~/.config/robodojo/g05_vast.env
```

填入 `HF_TOKEN`、可选的 `MODELSCOPE_API_TOKEN`、`WANDB_API_KEY` 和
`SERVERCHAN_SENDKEY`。随后只需输入 SSH 主机和端口：

```bash
bash scripts/g05/launch_vast_train.sh root@<VAST_IP> <SSH_PORT>
```

远端会自动：拉取两个 GitHub fork 分支、拉取 G05 代码、创建 Python 环境、
安装依赖、从 HF 下载 Sidecar、从 ModelScope 下载 RoboDojo 数据和官方 G05
checkpoint、生成 12 任务 manifest、按 `checkpointing_steps` 定期保存、只保留
最近一个 checkpoint，并在训练结束后创建/上传到公开 HF model repo。

Server酱在启动、训练开始、成功和失败时推送；未设置 `SERVERCHAN_SENDKEY`
时仅关闭通知，不影响训练。

## 官方资产与许可证

不要把 gated 权限、HF token 或私钥写入脚本。按 OpenGalaxea/GalaxeaVLA 官方仓库和 G0.5 Community License 操作，先在有权限的机器下载官方 `g05-base`/可续训 pretrained checkpoint、processor、action tokenizer，核对 license/模型 gated 访问条件，再通过 `G05_BASE_ASSETS` 指向本地目录。公开 RoboDojo fm-only checkpoint 仅作为降级评估/初始化参考，不能假设它包含 AR head。

## VastAI 同步

```bash
export VAST_HOST=user@INSTANCE_IP
export VAST_ROOT=/workspace/g05-run
export G05_ROOT=/path/to/GalaxeaVLA-or-G05-checkout
export G05_BASE_ASSETS=/path/to/official/pretrained-assets
export ROBODOJO_SIDECAR=/mnt/pqssd/RoboInter/RoboInterTools/annotations/lerobot_v30_joint.sqlite3
export ROBODOJO_LEROBOT_V30_ROOT=/mnt/pqssd/RoboDojo/data/lerobot_v30_joint
bash scripts/g05/sync_to_vastai.sh
```

同步使用 rsync partial/resume；不会传递 shell 环境中的 token。实例内布局为 `RoboDojo/`、`G05/`、`data/lerobot_v30_joint/`、`annotations/`、`base_assets/`。生成 manifest 后运行：


```bash
ssh "$VAST_HOST" 'cd /workspace/g05-run/RoboDojo && \
  export ROBODOJO_LEROBOT_V30_ROOT=/workspace/g05-run/data/lerobot_v30_joint \
  ROBODOJO_SIDECAR=/workspace/g05-run/annotations/lerobot_v30_joint.sqlite3 \
  G05_ROOT=/workspace/g05-run/G05 \
  G05_BASE_ASSETS=/workspace/g05-run/base_assets; \
  python3 scripts/g05/preflight.py --manifest artifacts/g05_robodojo/subgoal_samples.jsonl'
```

## 训练、续训和回传

```bash
ssh "$VAST_HOST" 'cd /workspace/g05-run/RoboDojo && bash scripts/g05/run_train_vastai.sh'
```

首选 `G05_TRAIN_MODE=ar_fm`，并在外部 G05 配置中加载官方 pretrained assets；若外部 checkout 不认识 `data.subgoal_sidecar` 等 override，先用其官方数据入口把 JSONL join 成训练字段，再移除这些 override。续训：

```bash
export G05_RESUME=/workspace/g05-run/G05/outputs/<run>/checkpoints/<latest>.pt
ssh "$VAST_HOST" 'cd /workspace/g05-run/RoboDojo && bash scripts/g05/run_train_vastai.sh'
bash scripts/g05/sync_back_from_vastai.sh
```

建议用 `tmux`/`systemd` 保活，并每 15–30 分钟执行 `sync_back_from_vastai.sh`；实例回收后从本地最新 checkpoint 重新同步并设置 `G05_RESUME`。对象存储可通过用户自备 `rclone` remote 另行上传，不在仓库保存配置。

## 风险与降级

本机不启动正式训练。AR/FＭ 联合训练依赖外部 G05 checkout 的真实 config/schema 和显存；本适配器无法在没有 G05_ROOT/官方资产时证明单步可训练。若 AR head 或显存不可用，最小风险方案是官方 fm-only checkpoint + 全局 Task，仍保留 Sidecar 统计与边界安全 chunk；不要把 Subtask 覆盖全局 Task。训练日志与 rollout 可通过 `ROBODOJO_G05_RETAIN_COT=1`、`ROBODOJO_G05_COT_LOG=/path/cot.jsonl` 保存 `_cot_text`。
