#!/usr/bin/env bash
# 创建独立 conda 环境并安装 Part3（SCST + BasicVSR++）依赖。
# 需要：conda、NVIDIA 驱动（CUDA 12.x）、磁盘空闲建议 >= 15GB（torch+mmcv+cuda 库较大）。
#
# 用法：
#   bash scripts/setup_part3_env.sh              # 环境名默认 vsr_part3
#   bash scripts/setup_part3_env.sh my_env_name

set -euo pipefail

ENV_NAME="${1:-vsr_part3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v conda &>/dev/null; then
  echo "未找到 conda，请先安装 Miniconda/Anaconda。"
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "环境已存在: $ENV_NAME （若要重建: conda env remove -n $ENV_NAME -y）"
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

pip install --upgrade pip setuptools wheel

# PyTorch CUDA 12.4（与驱动 12.x 兼容）。优先 2.4 + mmcv 官方轮子较全。
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# mmcv（SCST 自带 basicsr 会 import mmcv.ops）
# 说明：OpenMMLab 暂无 cu124/torch2.4 索引（404）；cu121+torch2.4 的预编译 wheel 与 torch 2.4+cu124 二进制兼容。
# --prefer-binary + 官方 PyPI 索引可避免国内镜像只提供 sdist 导致源码编译失败。
# 仅此步强制官方 PyPI，避免镜像只有 mmcv sdist 而走源码编译。
PIP_INDEX_URL=https://pypi.org/simple \
  pip install mmcv==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html \
  --prefer-binary --no-cache-dir

# SCST / Diffusers 栈（版本贴近上游 SCST；若冲突再微调）
# diffusers 0.25 仍从 huggingface_hub 导入 cached_download（已在 hf_hub>=0.26 / 1.x 移除），须锁定 hf_hub<0.26；
# 不锁定 transformers 时会装到 5.x，常强制 hf_hub 1.x → 与 diffusers 0.25 冲突。
pip install omegaconf==2.3.0
PIP_INDEX_URL=https://pypi.org/simple pip install \
  "diffusers==0.25.0" \
  "accelerate==0.34.2" \
  "transformers>=4.36,<5" \
  "huggingface_hub>=0.19.4,<0.26" \
  safetensors \
  --no-cache-dir

# 可选：xformers 可省显存；与 torch 版本需匹配，装不上可跳过（SCST 已支持无 xformers）。
# pip install xformers || true
pip install einops==0.8.0 numpy Pillow PyYAML opencv-python
pip install imageio-ffmpeg av

# BasicSR（BasicVSR++ / 与 pip basicsr 对齐的推理）
# basicsr 依赖 tb-nightly；清华/CERNet 等镜像常未收录 → pip 报 No matching distribution found。
PIP_INDEX_URL=https://pypi.org/simple pip install basicsr==1.4.2 --no-cache-dir

# torchvision>=0.17 移除 functional_tensor；basicsr 1.4.2 仍从旧路径 import，需在「import basicsr」前可加载。
python - <<'PY'
import pathlib
import site

old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale\n"
new = (
    "try:\n"
    "    from torchvision.transforms.functional_tensor import rgb_to_grayscale\n"
    "except ImportError:\n"
    "    from torchvision.transforms.functional import rgb_to_grayscale\n"
)
for sp in site.getsitepackages():
    p = pathlib.Path(sp) / "basicsr/data/degradations.py"
    if not p.is_file():
        continue
    text = p.read_text()
    if "except ImportError:" in text and "functional_tensor" in text:
        print("basicsr degradations.py 已兼容新版本 torchvision，跳过修补:", p)
        break
    if old not in text:
        raise SystemExit(f"未找到预期语句，请手动检查: {p}")
    p.write_text(text.replace(old, new, 1))
    print("已修补 basicsr degradations.py（torchvision 新 API）:", p)
    break
else:
    raise SystemExit("未找到 site-packages 中的 basicsr/data/degradations.py")
PY

# 评估与其它脚本常用
pip install scikit-image lpips pytorch-fid

echo ""
echo "=== 完成。请激活环境 ==="
echo "  conda activate $ENV_NAME"
echo ""
echo "=== 自检 ==="
python - <<'PY'
import torch
from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
print("cuda:", torch.cuda.is_available(), torch.__version__)
print("BasicVSRPlusPlus import: ok")
PY

echo ""
echo "=== Part3 权重与 SCST 路径（建议在仓库根目录执行）==="
echo "  python scripts/check_part3_ready.py"

echo ""
echo "=== 运行 Part3 示例（仓库根目录）==="
echo "  cd $ROOT"
echo "  conda activate $ENV_NAME"
echo "  python scripts/inference_part3.py --mode B_only --input_dir data/BIx4/calendar --output_dir results/part3_calendar_B_only --device cuda:0"
