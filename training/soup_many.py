"""Uniform (or weighted) weight average of several fine-tuned checkpoints from the same init.

    python training/soup_many.py --out .work/soup .work/ckpt_v3 .work/ckpt_v4:2 ...
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    parts = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) if ":" in p else (p, 1.0) for p in a.ckpts]
    tot = sum(w for _, w in parts)
    acc = None
    for path, w in parts:
        sd = load_file(os.path.join(path, "model.safetensors"))
        if acc is None:
            acc = {k: v.float() * (w / tot) for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k] += v.float() * (w / tot)
    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in acc.items()}, os.path.join(a.out, "model.safetensors"))
    first = parts[0][0]
    for d in ("encoder", "tokenizer"):
        shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
        shutil.copytree(os.path.join(first, d), os.path.join(a.out, d))
    cfg = json.load(open(os.path.join(first, "cbjev_config.json")))
    cfg["soup"] = {os.path.basename(p.rstrip("/")): w / tot for p, w in parts}
    json.dump(cfg, open(os.path.join(a.out, "cbjev_config.json"), "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
