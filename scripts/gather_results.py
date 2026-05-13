#!/usr/bin/env python3
"""Gather all eval results for report."""
import json, sys

files = {
    "full": "results/eval_full_4seq.json",
    "part3": "results/eval_part3_final.json",
    "vimeo": "results/eval_vimeo_5seq.json",
}
results = {}
for name, path in files.items():
    try:
        with open(path) as f:
            data = json.load(f)
        results[name] = data
        print("=" * 90)
        print("  %s" % name.upper())
        print("-" * 90)
        print("%-48s %8s %8s %8s %8s %8s" % ("Method", "PSNR", "SSIM", "LPIPS", "tLPIPS", "flow_epe"))
        print("-" * 90)
        for method in data["summary"]:
            s = data["summary"][method]
            print("%-48s %8.2f %8.4f %8.4f %8.4f %8.4f" % (
                method, s["psnr"], s["ssim"], s["lpips"], s["tlpips"], s["flow_epe"]))
        print()
    except Exception as e:
        print("%s: %s" % (name, e), file=sys.stderr)
