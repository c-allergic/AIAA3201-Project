#!/usr/bin/env python3
"""Chunked Part3 C_hybrid inference for high-res videos (e.g., Wild 5K)."""
import argparse, os, sys, shutil, tempfile
import torch

_repo_root = '/home/user/VSR_Project'
sys.path.insert(0, _repo_root)
os.chdir(_repo_root)

from configs import Config
from models import build_model
from models.uncertainty_fusion import build_fusion_training_input, fuse_with_weight, rule_based_weight, FusionWeightCNN
from scripts.temporal_refine import temporal_refine
from utils import ensure_weights, load_frames, save_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--chunk_size', type=int, default=16)
    parser.add_argument('--overlap', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fusion_gen_scale', type=float, default=0.3)
    parser.add_argument('--scst_steps', type=int, default=20)
    parser.add_argument('--scst_guidance', type=float, default=5.0)
    parser.add_argument('--skip_temporal_refine', action='store_true')
    parser.add_argument('--temporal_blend', type=float, default=0.24)
    args = parser.parse_args()

    cfg = Config.from_yaml('configs/default.yaml')
    device = torch.device(args.device)
    ckpts = {}

    ckpts['basicvsr_pp'] = ensure_weights(cfg.weights.root_dir, cfg.weights.urls, 'basicvsr_plusplus_x4')
    ckpts['scst_localatten'] = ensure_weights(cfg.weights.scst_root_dir, cfg.weights.urls, 'scst_localatten_unet')

    all_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.png')])
    print(f'Total frames: {len(all_files)}')

    tmpdir = tempfile.mkdtemp(prefix='chunked_')
    os.makedirs(args.output_dir, exist_ok=True)

    chunk_size = args.chunk_size
    overlap = args.overlap
    stride = chunk_size - overlap

    all_outputs = []
    for start in range(0, len(all_files), stride):
        end = min(start + chunk_size, len(all_files))
        chunk_files = all_files[start:end]
        print(f'Chunk [{start}:{end}] ({len(chunk_files)} frames)...')

        chunk_dir = os.path.join(tmpdir, f'chunk_{start}_{end}')
        os.makedirs(chunk_dir, exist_ok=True)
        for f in chunk_files:
            shutil.copy2(os.path.join(args.input_dir, f), os.path.join(chunk_dir, f))

        lr = load_frames(chunk_dir).to(device)
        print(f'  LR shape: {lr.shape}')

        scst = build_model('scst', scale=cfg.runtime.scale, scst_ckpt_root=cfg.weights.scst_root_dir).to(device).eval()
        scst.temporal_mode = 'localatten'
        scst.num_inference_steps = args.scst_steps
        scst.guidance_scale = args.scst_guidance
        scst.seed = 42
        scst.load_checkpoint(ckpts['scst_localatten'])

        with torch.no_grad():
            sr_gen = scst(lr).clamp(0.0, 1.0).to(device)
            if sr_gen.dim() == 4:
                sr_gen = sr_gen.unsqueeze(0)

        try:
            bvsrpp = build_model('basicvsr_pp', scale=cfg.runtime.scale).to(device).eval()
            bvsrpp.load_checkpoint(ckpts['basicvsr_pp'])
        except Exception:
            bvsrpp = build_model('bicubic', scale=cfg.runtime.scale).to(device).eval()

        with torch.no_grad():
            sr_fid = bvsrpp(lr).clamp(0.0, 1.0)
            if sr_fid.dim() == 4:
                sr_fid = sr_fid.unsqueeze(0)

            sr_gen = sr_gen.to(device)
            sr_fid = sr_fid.to(device)
            w, _ = rule_based_weight(sr_fid, sr_gen, alpha=4.0, beta=6.0, gamma=3.0, zeta=2.5)
            w = torch.clamp(w * args.fusion_gen_scale, 0.0, 1.0)
            out = fuse_with_weight(sr_fid, sr_gen, w)
            if not args.skip_temporal_refine:
                out = temporal_refine(out, blend=args.temporal_blend)

        all_outputs.append(out.cpu())
        del scst, bvsrpp, lr, sr_gen, sr_fid, out
        torch.cuda.empty_cache()
        shutil.rmtree(chunk_dir, ignore_errors=True)

    # Blend overlapping regions
    print('Blending...')
    final = all_outputs[0]
    for i in range(1, len(all_outputs)):
        prev = final
        curr = all_outputs[i]
        prev_t = prev.shape[1]
        curr_t = curr.shape[1]

        if i == 1:
            final = torch.cat([prev[:, :prev_t-overlap//2], curr[:, overlap//2:]], dim=1)
        else:
            final = torch.cat([final, curr[:, overlap//2:]], dim=1)

    print(f'Final output: {final.shape}')
    save_frames(final, args.output_dir)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f'Done: {args.output_dir}')


if __name__ == '__main__':
    main()
