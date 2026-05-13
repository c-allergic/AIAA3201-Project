"""
VSR Project - Video Super-Resolution
AIAA 3201 - Introduction to Computer Vision, Spring 2026

This package implements a cleaned inference-first pipeline:
- Part 1: Bicubic + SRCNN
- Part 2: BasicVSR + Real-ESRGAN / Real-ESRNet (official pretrained checkpoints)
- Part 3: BasicVSR++ + SCST + uncertainty fusion + temporal refine (`scripts/inference_part3.py`)

一键复现评估：`python scripts/run_full_benchmark.py`
"""

__version__ = "1.0.0"
__author__ = "VSR Project Team"
