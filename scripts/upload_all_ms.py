#!/usr/bin/env python3
"""Upload all mandatory dataset results to ModelScope."""
import os
import sys

from modelscope.hub.api import HubApi

REPO_ID = "SheldonLi329/AIAA3201-SR-Project-Videos"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.environ.get("VSR_RESULTS_DIR", os.path.join(_ROOT, "results"))


def _load_token() -> str:
    token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if token:
        return token.strip()
    token_file = os.environ.get(
        "MODELSCOPE_TOKEN_FILE",
        os.path.join(_ROOT, "ms_token.txt"),
    )
    if os.path.isfile(token_file):
        with open(token_file, encoding="utf-8") as f:
            token = f.read().strip()
        if token:
            return token
    print(
        "ModelScope token required. Set MODELSCOPE_API_TOKEN or create ms_token.txt "
        "(one line, gitignored) in the project root.",
        file=sys.stderr,
    )
    sys.exit(1)


TOKEN = _load_token()

api = HubApi()
api.login(access_token=TOKEN)

total_dirs = 0
total_ok = 0
total_fail = 0


def upload_one(local_dir, remote_path, label):
    global total_dirs, total_ok, total_fail
    if not os.path.isdir(local_dir):
        print(f"  SKIP (missing): {local_dir}")
        return
    n = len([f for f in os.listdir(local_dir) if f.endswith((".png", ".jpg", ".mp4", ".jpeg"))])
    print(f"[{label}] {n} files -> {remote_path}")
    try:
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=local_dir,
            path_in_repo=remote_path,
            repo_type="dataset",
            token=TOKEN,
            commit_message=f"Upload: {remote_path}",
        )
        total_ok += 1
        print("  OK")
    except Exception as e:
        total_fail += 1
        print(f"  FAIL: {e}")
    total_dirs += 1


# ============================================================
# vimeo-RL P1&P2: 4 seqs x 8 methods
# ============================================================
seqs = ["00018", "00026", "00031", "00051"]
methods_p1 = ["bicubic", "lanczos", "srcnn", "temporal_avg", "basicvsr", "basicvsr_pp", "realesrgan", "realesrnet"]

print("\n" + "=" * 60)
print("vimeo-RL P1&P2")
print("=" * 60)
for seq in seqs:
    for m in methods_p1:
        d = os.path.join(BASE, f"vimeo_rl_{seq}_{m}")
        upload_one(d, f"vimeo-RL/P1_P2/vimeo_rl_{seq}_{m}", f"vimeo-RL/{seq}/{m}")

# ============================================================
# vimeo-RL Part3
# ============================================================
print("\n" + "=" * 60)
print("vimeo-RL Part3")
print("=" * 60)
for seq in seqs:
    d = os.path.join(BASE, f"part3_vimeo_rl_{seq}_C_hybrid_g0.3")
    upload_one(d, f"vimeo-RL/Part3/part3_vimeo_rl_{seq}_C_hybrid_g0.3", f"vimeo-RL-P3/{seq}")

# ============================================================
# REDS-sample P1&P2: 10 seqs x 8 methods
# ============================================================
reds_seqs = ["002", "007", "010", "012", "013", "018", "025", "027", "028", "029"]

print("\n" + "=" * 60)
print("REDS-sample P1&P2")
print("=" * 60)
for seq in reds_seqs:
    for m in methods_p1:
        d = os.path.join(BASE, f"reds_{seq}_{m}")
        upload_one(d, f"REDS-sample/P1_P2/reds_{seq}_{m}", f"REDS/{seq}/{m}")

# ============================================================
# REDS-sample Part3
# ============================================================
print("\n" + "=" * 60)
print("REDS-sample Part3")
print("=" * 60)
for seq in reds_seqs:
    d = os.path.join(BASE, f"part3_reds_{seq}_C_hybrid_g0.3")
    upload_one(d, f"REDS-sample/Part3/part3_reds_{seq}_C_hybrid_g0.3", f"REDS-P3/{seq}")

# ============================================================
# Wild V1 P1&P2: 8 methods
# ============================================================
print("\n" + "=" * 60)
print("Wild V1 P1&P2")
print("=" * 60)
for m in methods_p1:
    d = os.path.join(BASE, f"wild_{m}")
    upload_one(d, f"Wild_V1/P1_P2/wild_{m}", f"WildV1/{m}")

# ============================================================
# Wild V2 P1&P2: 3 methods
# ============================================================
print("\n" + "=" * 60)
print("Wild V2 P1&P2")
print("=" * 60)
for m in ["bicubic", "basicvsr_pp", "realesrgan"]:
    d = os.path.join(BASE, f"wild2_{m}")
    upload_one(d, f"Wild_V2/P1_P2/wild2_{m}", f"WildV2/{m}")

# ============================================================
# demo_videos: 120 mp4 files, uploaded as individual files
# ============================================================
print("\n" + "=" * 60)
print("Demo videos")
print("=" * 60)
demo_dir = os.path.join(BASE, "demo_videos")
if os.path.isdir(demo_dir):
    upload_one(demo_dir, "demo_videos", "demo_videos")

print("\n" + "=" * 60)
print(f"DONE: {total_ok} ok, {total_fail} fail, {total_dirs} total dirs")
print("=" * 60)
