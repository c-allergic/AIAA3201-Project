# VSR_Project: Video Super-Resolution Pipeline

Three-stage VSR benchmark: classical interpolation → recurrent alignment-based models → uncertainty-aware hybrid fusion with diffusion priors.

## Project Structure

```
VSR_Project/
├── configs/default.yaml          # Weight URLs and runtime configuration
├── models/
│   ├── baselines.py              # Bicubic, Lanczos3, SRCNN, Temporal Avg
│   ├── basicvsr.py / basicvsr_pp.py   # BasicVSR and BasicVSR++
│   ├── real_esrgan.py            # Real-ESRGAN / Real-ESRNet
│   ├── scst_wrapper.py           # SCST subprocess wrapper (SD 2.1 + ControlNet-Tile)
│   └── uncertainty_fusion.py     # Rule-based fusion weight estimation
├── scripts/
│   ├── inference.py              # Part 1/2 inference
│   ├── inference_part3.py        # Part 3 single-clip inference
│   ├── inference_part3_chunked.py # Part 3 chunked inference for long/high-res clips
│   ├── eval_pipeline.py          # Full-reference evaluation (PSNR, SSIM, LPIPS, tLPIPS, EPE)
│   ├── eval_wild2.py             # No-reference evaluation (tLPIPS, warp-L1, FID)
│   ├── temporal_refine.py        # SpyNet flow-guided temporal blending
│   ├── run_full_benchmark.py     # One-click download → inference → eval
│   ├── download_pretrained.py    # Download all pretrained weights
│   └── setup_part3_env.sh        # Conda environment for Part 3
├── third_party/SCST/             # SCST submodule (CVPR 2025)
├── requirements.txt
└── README.md
```

## Environment Setup

### Part 1 & 2

```bash
conda create -n vsr python=3.10 -y && conda activate vsr
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Part 3

Requires separate environment with `mmcv` and `omegaconf`:

```bash
bash scripts/setup_part3_env.sh
conda activate vsr_part3
```

## Download Pretrained Weights

```bash
python scripts/download_pretrained.py --config configs/default.yaml
```

Downloads: SRCNN, BasicVSR/BasicVSR++, Real-ESRGAN/Real-ESRNet, SpyNet, Stable Diffusion 2.1, ControlNet-Tile, and SCST checkpoints.

## Usage

### Part 1 & 2 Inference

```bash
# Available models: bicubic, lanczos, temporal_avg, srcnn,
#                  basicvsr, basicvsr_pp, realesrgan, realesrnet

python scripts/inference.py --model_name basicvsr_pp   --input_dir data/BIx4/city   --output_dir results/bi_city_basicvsr_pp
```

### Part 3 Inference (`conda activate vsr_part3`)

```bash
# SCST standalone
python scripts/inference_part3.py --mode B_only   --input_dir data/BIx4/city   --output_dir results/part3_city_B_only

# C_hybrid (BasicVSR++ + SCST + fusion + temporal refinement)
python scripts/inference_part3.py --mode C_hybrid   --input_dir data/BIx4/city   --output_dir results/part3_city_C_hybridg0.3

# Chunked inference for long/high-res clips
python scripts/inference_part3_chunked.py   --input_dir data/wild2_lr_frames   --output_dir results/part3_wild2_C_hybrid_g0.3   --chunk_size 8 --overlap 2   --fusion_gen_scale 0.3 --temporal_blend 0.24
```

Key options: `--temporal_mode {localatten,stcm}`, `--scst_steps 20`, `--scst_guidance 5.0`, `--fusion_gen_scale 0.3`, `--temporal_blend 0.24`.

If SCST checkpoints are missing, the wrapper falls back to bicubic for the generative branch (pipeline stays runnable, but metrics must not be reported).

### Evaluation

```bash
# Full-reference (with GT)
python scripts/eval_pipeline.py   --gt_root data/GT --results_root results   --methods bi_city_bicubic bi_city_basicvsr   --sequences city --crop_border 4   --save_json results/eval_results.json

# No-reference (no GT: REDS-sample, Wild Video)
python scripts/eval_wild2.py
```

### One-Click Benchmark

```bash
python scripts/run_full_benchmark.py --phase all
```

## Datasets

- **Vid4**: BI×4 and BD×4 (calendar, city, foliage, walk) — with GT
- **Vimeo-RL**: 4 sequences (00018, 00026, 00031, 00051) — with GT
- **REDS-sample**: 10 sequences (002–029, 100 frames each) — no GT
- **Wild Video V1**: 132-frame 720p clip — no GT
- **Wild Video V2**: 100-frame 240p clip — no GT
- **Vimeo-90K**: 5 selected sequences (00001, 00010, 00019, 00046, 00090) — with GT

All generated outputs are publicly available at [ModelScope](https://www.modelscope.cn/datasets/SheldonLi329/AIAA3201-SR-Project-Videos).

## Full Report

See [Overleaf project](https://www.overleaf.com/project/6a04293048b03f2bd79bf34c) for the complete paper with results, ablation studies, and analysis.

## License

Academic/research use. Third-party components (SCST, Real-ESRGAN, BasicVSR++, Stable Diffusion) retain their original licenses.
