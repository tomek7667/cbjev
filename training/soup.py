"""Weight interpolation between a fine-tuned checkpoint and the Laya checkpoint it started from.

    python training/soup.py --ft .work/ckpt_v3 --base laya --alpha 0.6 --out .work/soup

W = alpha * fine-tuned + (1 - alpha) * base (WiSE-FT). Works because the shared layout feeds a Laya
checkpoint exactly the rows it was trained on, so both ends of the line are sensible models; the
middle typically keeps most of the fine-tuning gain and gives back what fine-tuning forgot.
Several bases can be mixed: --base laya:0.5,laya/typed-decisions:0.5.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cbjev.agent import resolve_checkpoint  # noqa: E402
from cbjev.checkpoints import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True)
    ap.add_argument("--base", default="laya")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ft = load_file(os.path.join(a.ft, "model.safetensors"))
    bases = []
    for part in a.base.split(","):
        name, _, w = part.partition(":")
        repo, _, sub = name.partition("/")
        bases.append((load_file(os.path.join(resolve_checkpoint(resolve(repo), sub or None), "model.safetensors")),
                      float(w or 1.0)))
    tot = sum(w for _, w in bases)
    out = {}
    for k, v in ft.items():
        b = sum(w * bs[k].float() for bs, w in bases if k in bs) / tot if all(k in bs for bs, _ in bases) else v.float()
        out[k] = (a.alpha * v.float() + (1 - a.alpha) * b).to(torch.bfloat16).contiguous()
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"))
    for d in ("encoder", "tokenizer"):
        if os.path.exists(os.path.join(a.out, d)):
            shutil.rmtree(os.path.join(a.out, d))
        shutil.copytree(os.path.join(a.ft, d), os.path.join(a.out, d))
    cfg = json.load(open(os.path.join(a.ft, "cbjev_config.json")))
    cfg["soup"] = {"fine_tuned": os.path.basename(a.ft.rstrip("/")), "base": a.base, "alpha": a.alpha}
    json.dump(cfg, open(os.path.join(a.out, "cbjev_config.json"), "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
