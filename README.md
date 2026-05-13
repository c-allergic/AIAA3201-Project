# VSR_Project: Video Super-Resolution Pipeline

A systematic video super-resolution (VSR) benchmark spanning classical interpolation, recurrent alignment-based models, and a generative hybrid framework with uncertainty-aware fusion.

## Overview

This repository implements a three-part VSR investigation:

| Part | Methods | Description |
|------|---------|-------------|
| **Part 1** | Bicubic, Lanczos3, Temporal Averaging + Unsharp Masking, SRCNN | Classical and shallow-CNN baselines |
| **Part 2** | BasicVSR++, Real-ESRGAN, Real-ESRNet | State-of-the-art recurrent VSR and GAN-based perceptual enhancement |
| **Part 3** | SCST (Stable Diffusion + ControlNet-Tile) + Uncertainty-Aware Fusion | Hybrid pipeline blending fidelity (BasicVSR++) and generative (SCST) branches with pixel-wise uncertainty weights and flow-guided temporal refinement |

## Key Results (BI×4, 4-sequence avg.)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | tLPIPS↓ |
|--------|-------|-------|--------|---------|
| Bicubic | 22.49 | 0.639 | 0.512 | 0.050 |
| SRCNN | 23.16 | 0.690 | 0.388 | 0.069 |
| Real-ESRGAN | 21.02 | 0.595 | **0.282** | 0.092 |
| BasicVSR++ | **25.81** | **0.834** | 0.214 | 0.065 |
| **C_hybrid (g=0.3)** | 25.62 | 0.821 | 0.206 | **0.054** |

C_hybrid (g=0.3) matches BasicVSR++ PSNR while improving temporal consistency (tLPIPS) by **17%**. See the [full report](https://www.overleaf.com/project/6a04293048b03f2bd79bf34c) for complete tables, BD×4, Vimeo-90K, and ablation results.

## Project Structure

```
VSR_Project/
├── configs/default.yaml          # Weight URLs and run configuration
├── models/
│   ├── baselines.py              # Bicubic, Lanczos3, SRCNN, Temporal Avg
│   ├── basicvsr.py               # BasicVSR (bidirectional recurrent)
│   ├── basicvsr_pp.py            # BasicVSR++ (second-order propagation)
│   ├── real_esrgan.py            # Real-ESRGAN / Real-ESRNet
│   ├── scst_wrapper.py           # SCST subprocess wrapper (Stable Diffusion + ControlNet)
│   └── uncertainty_fusion.py     # Rule-based and learned fusion weight estimators
├── scripts/
│   ├── inference.py              # Unified Part 1/2 inference entrypoint
│   ├── inference_part3.py        # Part 3 inference (B_only / C_hybrid modes)
│   ├── eval_pipeline.py          # PSNR/SSIM/LPIPS/tLPIPS/flow evaluation
│   ├── run_full_benchmark.py     # One-click download → inference → eval
│   ├── download_pretrained.py    # Download all pretrained weights
│   ├── setup_part3_env.sh        # Conda environment setup for Part 3
│   ├── train_fusion.py           # Lightweight CNN fusion weight trainer
│   ├── temporal_refine.py        # SpyNet flow-guided temporal blending
│   └── ablation_tr.py            # Temporal refinement ablation
├── third_party/SCST/             # SCST submodule (CVPR 2025)
├── requirements.txt
└── README.md
```

## Environment Setup

### Part 1 & 2 (vsr env)

```bash
conda create -n vsr python=3.10 -y
conda activate vsr
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If `basicsr` conflicts: `pip install --no-deps --no-build-isolation basicsr==1.4.2`

### Part 3 (vsr_part3 env, separate)

Part 3 requires `mmcv` and `omegaconf` which conflict with newer Python/torch versions.

```bash
bash scripts/setup_part3_env.sh
conda activate vsr_part3
```

## Download Pretrained Weights

```bash
python scripts/download_pretrained.py --config configs/default.yaml
```

Downloads: SRCNN x4, BasicVSR x4, BasicVSR++ x4, Real-ESRGAN x4plus, Real-ESRNet x4plus, SpyNet, Stable Diffusion 2.1, ControlNet-Tile, and SCST checkpoints.

## Inference

### Part 1 & 2

```bash
# Bicubic interpolation
python scripts/inference.py --model_name bicubic --input_dir data/BIx4/calendar --output_dir results/bi_calendar_bicubic

# SRCNN
python scripts/inference.py --model_name srcnn --input_dir data/BIx4/calendar --output_dir results/bi_calendar_srcnn

# BasicVSR++
python scripts/inference.py --model_name basicvsr_pp --input_dir data/BIx4/calendar --output_dir results/bi_calendar_basicvsr

# Real-ESRGAN
python scripts/inference.py --model_name realesrgan --input_dir data/BIx4/calendar --output_dir results/bi_calendar_realesrgan

# Real-ESRNet (non-GAN)
python scripts/inference.py --model_name realesrnet --input_dir data/BIx4/calendar --output_dir results/bi_calendar_realesrnet

# Temporal averaging + unsharp masking
python scripts/inference.py --model_name temporal_avg --input_dir data/BIx4/calendar --output_dir results/bi_calendar_temporal_avg
```

Available models: `bicubic`, `lanczos`, `temporal_avg`, `srcnn`, `basicvsr`, `basicvsr_pp`, `realesrgan`, `realesrnet`

### Part 3 (requires `conda activate vsr_part3`)

```bash
# SCST standalone (Direction B only)
python scripts/inference_part3.py --mode B_only \
  --input_dir data/BIx4/calendar \
  --output_dir results/part3_calendar_B_only

# Hybrid fusion (Direction C): BasicVSR++ + SCST + uncertainty fusion + temporal refinement
python scripts/inference_part3.py --mode C_hybrid \
  --input_dir data/BIx4/calendar \
  --output_dir results/part3_calendar_C_hybridg0.3

# Advanced options
python scripts/inference_part3.py --mode C_hybrid \
  --input_dir data/BIx4/calendar \
  --output_dir results/part3_calendar_stcm_g0.3 \
  --temporal_mode stcm \       # STCM instead of LocalAttention
  --scst_steps 20 \            # DDIM sampling steps
  --scst_guidance 5.0 \        # Classifier-free guidance scale
  --fusion_gen_scale 0.3       # Global fusion scale g
```

If SCST checkpoints are missing, the wrapper falls back to bicubic for the generative branch (pipeline stays executable, **but metrics must not be reported**).

## Evaluation

```bash
python scripts/eval_pipeline.py \
  --gt_root data/GT \
  --results_root results \
  --methods bi_calendar_bicubic bi_calendar_srcnn bi_calendar_basicvsr \
  --sequences calendar \
  --crop_border 4 \
  --skip_fid \
  --save_json results/eval_results.json
```

Reports PSNR, SSIM, LPIPS (AlexNet), tLPIPS, Farneback flow EPE, and warp-L1 consistency. A combined overall score (45% frame quality + 55% temporal consistency) is computed via min-max normalization.

## One-Click Benchmark

```bash
# Full pipeline (download → Part1/2 inference → Part3 inference → eval)
python scripts/run_full_benchmark.py --phase all

# Step by step
python scripts/run_full_benchmark.py --phase download
python scripts/run_full_benchmark.py --phase part12 --skip_download
python scripts/run_full_benchmark.py --phase part3 --skip_download
python scripts/run_full_benchmark.py --phase eval --skip_download
```

## Methodology

### Part 1: Classical Baselines
- **Bicubic/Lanczos3**: Per-frame spatial interpolation without temporal modeling
- **SRCNN** (Dong et al., 2015): 3-layer CNN after bicubic pre-upsampling
- **Temporal Averaging**: Gaussian-weighted neighbor averaging with Unsharp Masking

### Part 2: Recurrent VSR
- **BasicVSR++** (Chan et al., CVPR 2022): Bidirectional recurrent network with second-order grid propagation and SpyNet optical flow alignment
- **Real-ESRGAN** (Wang et al., 2021): RRDBNet backbone + PatchGAN discriminator + perceptual loss; high-order degradation pipeline
- **Real-ESRNet**: Same backbone without GAN, fidelity-optimized

### Part 3: Uncertainty-Aware Hybrid Fusion
- **Fidelity branch**: BasicVSR++ output
- **Generative branch**: SCST (Stable Diffusion 2.1 + ControlNet-Tile), 20 DDIM steps
- **Fusion**: Pixel-wise sigmoid weights from temporal variance, inter-branch disagreement, edge strength, and generative instability
- **Temporal refinement**: SpyNet flow-guided neighbor blending (λ=0.24)

```
w = σ(α·var_t + β·|fid - gen| - γ·edge - ζ·var_gen_t) · g
output = (1-w) · fid + w · gen
```

## Datasets

- **BI×4 / BD×4**: calendar (41f), city (34f), foliage (49f), walk (47f)
- **Vimeo-90K**: 5 sequences (00001, 00010, 00019, 00046, 00090)
- **Wild video**: 720p real-world clip, 132 frames

## References

- Dong et al., "Image Super-Resolution Using Deep Convolutional Networks," TPAMI 2015.
- Chan et al., "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment," CVPR 2022.
- Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," ICCV 2021.
- Liu et al., "SCST: Spatio-Temporal Consistent Video Super-Resolution with Diffusion Models," CVPR 2025.
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.
- Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models," ICCV 2023.
- Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018.
- Chu et al., "Temporally Coherent GANs for Video Super-Resolution," ECCV 2020.

## License

This project is for academic/research purposes. Third-party components (SCST, Real-ESRGAN, BasicVSR++, Stable Diffusion) retain their original licenses.
