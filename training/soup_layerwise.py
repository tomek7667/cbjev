"""Layer-wise learned merge of a fine-tuned checkpoint with the Laya checkpoints (AdaMerging-style).

    python training/soup_layerwise.py --ft .work/ckpt_v7 --out .work/merge_lw --data .work/data_v2

The flat soups use one coefficient for all 395M parameters. Here every block gets its own pair:

    W_l = B_l + a_l * (FT_l - B_l) + b_l * (L_l - B_l)

with B = laya-typed-decisions (the fine-tune's starting point), FT = the cbjev fine-tune and
L = the root Laya checkpoint. The ~60 scalars are fitted by gradient descent on the *dev split of
the training sources* (never on benchmark cases) through a functional forward in the shared layout,
then baked into an ordinary checkpoint, so inference cost does not change.
"""
import argparse
import json
import os
import random
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file
from torch.func import functional_call

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cbjev.agent import _tokens_for, resolve_checkpoint  # noqa: E402
from cbjev.model import DecisionNet, EncoderSpec  # noqa: E402
from train import Rows, batches, collate, question_logprobs, read_jsonl  # noqa: E402


def group(k: str) -> int:
    if k.startswith("encoder.layers."):
        return 1 + int(k.split(".")[2])
    if k.startswith("encoder.embeddings"):
        return 0
    return -1          # final norm, head, type embedding, scorer: last group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, ".work", "data_v2"))
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")

    base_dir = resolve_checkpoint("convaiinnovations/laya", "typed-decisions")
    root_dir = resolve_checkpoint("convaiinnovations/laya")
    sd0 = load_file(os.path.join(base_dir, "model.safetensors"))
    sft = load_file(os.path.join(a.ft, "model.safetensors"))
    slaya = load_file(os.path.join(root_dir, "model.safetensors"))
    keys = [k for k in sft if k in sd0 and k in slaya and sft[k].is_floating_point()]
    B = {k: sd0[k].to(dev, torch.float32) for k in keys}
    TF = {k: sft[k].to(dev, torch.float32) - B[k] for k in keys}
    TL = {k: slaya[k].to(dev, torch.float32) - B[k] for k in keys}
    del sd0, slaya
    ngroups = 1 + max(group(k) for k in keys) + 1
    gid = {k: (group(k) if group(k) >= 0 else ngroups - 1) for k in keys}

    with open(os.path.join(a.ft, "encoder", "config.json")) as f:
        spec = EncoderSpec.from_hf(json.load(f))
    cfg = json.load(open(os.path.join(a.ft, "cbjev_config.json")))
    net = DecisionNet(spec, head_layers=int(cfg.get("head_layers", 2))).to(dev).eval()
    net.shared = cfg.get("layout") == "shared"
    net.encoder.checkpointing = False
    for p in net.parameters():
        p.requires_grad_(False)
    buffers = {k: v for k, v in net.named_buffers()}

    tokens = _tokens_for(os.path.join(a.ft, "tokenizer"))
    cases = [c for c in read_jsonl(os.path.join(a.data, "dev.jsonl")) if any(g is not None for g in c["gold"].values())]
    random.Random(a.seed).shuffle(cases)
    fit, hold = cases[: int(len(cases) * 0.8)], cases[int(len(cases) * 0.8):]
    mk = lambda cs: Rows(cs, tokens, int(cfg.get("max_state_tokens", 1024)), int(cfg.get("max_question_tokens", 768)),  # noqa: E731
                         0.0, a.seed, net.shared).build(0)
    fit_rows, hold_rows = mk(fit), mk(hold)
    print("fit rows %d, held-out rows %d, %d groups" % (len(fit_rows), len(hold_rows), ngroups), flush=True)

    ca = torch.ones(ngroups, device=dev, requires_grad=True)
    cb = torch.zeros(ngroups, device=dev, requires_grad=True)
    opt = torch.optim.Adam([ca, cb], lr=a.lr)

    def weights():
        return {k: B[k] + ca[gid[k]] * TF[k] + cb[gid[k]] * TL[k] for k in keys}

    def loss_on(rows, sel):
        x, (qrow, qs, qk, T, W) = collate(rows, sel, tokens.pad, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = functional_call(net, {**weights(), **buffers}, x, {"packed": True})
        lp, m = question_logprobs(z, qrow, qs, qk)
        nll = -(T * lp.masked_fill(~m, 0)).sum(-1)
        acc = (lp.argmax(-1) == T.argmax(-1)).float()
        return (nll * W).sum() / W.sum(), acc.sum().item(), len(acc)

    def evaluate(rows):
        with torch.no_grad():
            tot = n = hit = 0.0
            for sel in batches(rows, a.budget, random.Random(0)):
                l, h, c = loss_on(rows, sel)
                tot += l.item() * c
                hit += h
                n += c
        return tot / n, hit / n

    print("start  held-out nll %.4f acc %.4f" % evaluate(hold_rows), flush=True)
    rng = random.Random(a.seed)
    stream = []
    for step in range(a.steps):
        if not stream:
            stream = batches(fit_rows, a.budget, rng)
        l, _, _ = loss_on(fit_rows, stream.pop())
        opt.zero_grad()
        l.backward()
        opt.step()
        if (step + 1) % 60 == 0:
            print("step %d  held-out nll %.4f acc %.4f" % ((step + 1,) + evaluate(hold_rows)), flush=True)
    print("a (fine-tune) :", [round(v, 2) for v in ca.tolist()])
    print("b (root laya) :", [round(v, 2) for v in cb.tolist()])

    with torch.no_grad():
        W = weights()
        out = {k: W[k].to(torch.bfloat16).cpu().contiguous() for k in keys}
    for k, v in sft.items():
        out.setdefault(k, v)
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"))
    for d in ("encoder", "tokenizer"):
        shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
        shutil.copytree(os.path.join(a.ft, d), os.path.join(a.out, d))
    cfg["merge"] = {"method": "layer-wise", "a": [round(v, 4) for v in ca.tolist()],
                    "b": [round(v, 4) for v in cb.tolist()]}
    json.dump(cfg, open(os.path.join(a.out, "cbjev_config.json"), "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
