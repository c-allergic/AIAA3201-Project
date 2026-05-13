# VSR Project (Cleaned)

本版本按课程 PDF 目标做了精简：**Part1 + Part2** 为预训练推理主线；**Part3** 为混合分支（BasicVSR++ + SCST + 融合/后处理）。默认优先使用开源预训练权重，避免从零训练。

> 跨对话窗口快速恢复进度：请先看 `PROJECT_PROGRESS.md`。

## 保留内容

- `models/`
  - `bicubic`（baseline）
  - `srcnn`（Part1）
  - `basicvsr`（Part2）
  - `realesrgan`（Part2）
  - `realesrnet`（Part2，非GAN版本）
- `scripts/inference.py`：统一推理入口（Part1/2）
- `scripts/inference_part3.py`：Part3（`B_only` / `C_hybrid`）
- `scripts/run_full_benchmark.py`：一键推理 + `eval_pipeline` 汇总
- `scripts/download_pretrained.py`：下载开源权重
- `configs/default.yaml`：权重 URL 与运行配置

## 仓库演进说明

- 早期迭代曾删除冗余调试脚本与重复配置；**当前主干已重新集成 Part 3**（见下文与 `PROJECT_PROGRESS.md`）。
- 训练向推理优先：Part1/2 默认使用开源预训练权重；Part3 依赖 HuggingFace / SCST 权重时请优先运行 `scripts/download_pretrained.py`。

## 环境

### GPU：PyTorch 与驱动版本（必读）

`nvidia-smi` 顶栏里的 **CUDA Version**（例如 12.8）表示**驱动支持的最高 CUDA**，并不需要你去装 CUDA 13。若使用 **`pip install torch` 默认包装上了 `2.x+cu130`**（CUDA 13 运行时），而驱动仍为 12.x，会出现 **`CUDA initialization: driver too old`**，`torch.cuda.is_available()` 为 `False`。

推荐与本仓库一并使用的组合：**Python 3.13 + PyTorch 2.6 + CUDA 12.4 轮子**（与驱动 12.x 兼容）：

```bash
pip uninstall -y torch torchvision triton cuda-toolkit cuda-bindings nvidia-cudnn-cu13 2>/dev/null
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, 'cuda=', torch.version.cuda, 'ok=', torch.cuda.is_available())"
```

若有其它带 `nvidia-*-cu13` / `cuda-toolkit` 的旧包残留，可 `pip list | grep -i nvidia` 后逐项卸载，直到 `python -c "import torch; print(torch.cuda.is_available())"` 为 `True`。

更换 PyTorch 后若 `basicsr` 报错，可在上述 GPU 轮子装好后再装：`pip install --no-deps --no-build-isolation basicsr==1.4.2`。

### 依赖安装

```bash
pip install -r requirements.txt
```

如果 `basicsr` 依赖冲突，可先安装核心依赖，再执行：

```bash
pip install --no-deps --no-build-isolation basicsr==1.4.2
```

## 下载开源权重

```bash
python scripts/download_pretrained.py --config configs/default.yaml
```

默认下载：
- SRCNN x4（OpenMMLab）
- BasicVSR x4（OpenMMLab）
- BasicVSR++ x4（OpenMMLab，备用）
- RealESRGAN x4plus（官方 release）
- RealESRNet x4plus（官方 release）

## 推理

输入目录要求为视频帧目录（`.png/.jpg`），例如 `data/BDx4/calendar`。

```bash
python scripts/inference.py \
  --model_name bicubic \
  --input_dir data/BDx4/calendar \
  --output_dir results/calendar_bicubic
```

```bash
python scripts/inference.py \
  --model_name srcnn \
  --input_dir data/BDx4/calendar \
  --output_dir results/calendar_srcnn
```

```bash
python scripts/inference.py \
  --model_name basicvsr \
  --input_dir data/BDx4/calendar \
  --output_dir results/calendar_basicvsr
```

```bash
python scripts/inference.py \
  --model_name realesrgan \
  --input_dir data/BDx4/calendar \
  --output_dir results/calendar_realesrgan
```

```bash
python scripts/inference.py \
  --model_name realesrnet \
  --input_dir data/BDx4/calendar \
  --output_dir results/calendar_realesrnet
```

## 说明

- 本仓库不会修改你的 `data/` 下已下载数据。
- 若要自定义权重路径，可在推理时传 `--checkpoint /path/to/model.pth`。

## Part 3: Hybrid VSR (Direction C + Direction B)

本仓库已接入 Part 3 方案：`BasicVSR++`（保真分支）+ `SCST`（生成分支）+ 不确定性融合 + 时序后处理。

### 参考与出处（严格标注）

- BasicVSR++: Chan et al., CVPR 2022.
- SCST: Shi et al., CVPR 2025, repo: `ssj9596/SCST`（已放在 `third_party/SCST` 并保留出处声明）。
- 可选对照：DOVE (arXiv 2025), Upscale-A-Video (CVPR 2024)。

### Part3 独立 Conda 环境（推荐）

SCST 子进程依赖 **`mmcv`**、**`omegaconf`** 等与 **Python 3.13 / 系统默认环境** 常不兼容；建议在 **Python 3.10** 的干净 conda 环境里跑 Part3。

1. **磁盘**：安装 PyTorch + CUDA + mmcv 建议预留 **≥15GB** 空闲（`df -h`）。
2. **一键安装**（在仓库根目录）：

```bash
bash scripts/setup_part3_env.sh
conda activate vsr_part3
```

3. **运行**（需在激活环境后、`VSR_Project` 根目录）：

```bash
python scripts/inference_part3.py --mode B_only \
  --input_dir data/BIx4/calendar \
  --output_dir results/part3_calendar_B_only \
  --device cuda:0
```

脚本内默认使用 **torch 2.4 + cu124** 与 **mmcv 2.2.0**（OpenMMLab 轮子）；若与本机驱动不匹配，再按需改 `scripts/setup_part3_env.sh`。

### Part3 推理命令

`B_only`（SCST 分支，若缺权重会自动降级到 bicubic 占位输出，保证流程可运行）：

```bash
python scripts/inference_part3.py \
  --mode B_only \
  --input_dir data/BIx4/calendar \
  --output_dir results/part3_calendar_B_only
```

`C_hybrid`（BasicVSR++ + SCST + uncertainty + temporal refine）：

```bash
python scripts/inference_part3.py \
  --mode C_hybrid \
  --input_dir data/BIx4/calendar \
  --output_dir results/part3_calendar_C_hybrid
```

### Part3 轻量训练（可选）

先准备三个目录（同一序列）：
- `fid_dir`: BasicVSR++ 输出
- `gen_dir`: SCST 输出
- `gt_dir`: GT

```bash
python scripts/train_fusion.py \
  --fid_dir results/part3_calendar_fid \
  --gen_dir results/part3_calendar_B_only \
  --gt_dir data/GT/calendar \
  --save_path checkpoints/pretrained/fusion_cnn.pth \
  --steps 2000
```

### Part3 评估命令

```bash
python scripts/eval_pipeline.py \
  --gt_root data/GT \
  --results_root results \
  --methods part3_{seq}_B_only part3_{seq}_C_hybrid \
  --sequences calendar city foliage walk \
  --crop_border 4 \
  --save_json results/eval_part3_bi.json
```

### 一键完整复现（Part1/2 + Part3 + 评估）

在仓库根目录执行（需 GPU/CUDA 环境；SCST 权重不全时会自动退回 bicubic，指标仅供参考）：

```bash
python scripts/run_full_benchmark.py --phase all
```

分阶段（已有人工下载好权重时加 `--skip_download` 可跳过 `download_pretrained`）：

```bash
python scripts/run_full_benchmark.py --phase download
python scripts/run_full_benchmark.py --phase part12 --skip_download
python scripts/run_full_benchmark.py --phase part3 --skip_download
python scripts/run_full_benchmark.py --phase eval --skip_download
```
