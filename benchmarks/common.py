"""Engine factory shared by the benchmark scripts.

Every engine is a callable ``(state, questions) -> Jev-shaped response``. Laya engines need the
`laya` package; the fast Laya path also needs TileLang and a CUDA toolkit (nvcc).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LAYA_REPO = "convaiinnovations/laya"
ENGINES = ("laya", "laya-fast", "laya-td", "laya-ml", "cbjev", "cbjev-classic", "cbjev-ml")


def _laya(sub=None, fast=False):
    import laya
    if fast:
        # TileLang picks the first GPU backend it finds; pin it to CUDA on mixed AMD/NVIDIA hosts
        from tilelang import tvm
        tvm.target.Target("cuda").__enter__()
    ag = laya.load(LAYA_REPO, device="cuda", subfolder=sub, fast=fast)
    if fast and ag._fast is None:
        raise RuntimeError("laya fast path did not build (needs tilelang + nvcc)")
    return ag


def load_agent(name):
    """Engine names: laya, laya-fast, laya-td, laya-ml, cbjev, cbjev-classic, cbjev:<path>;
    a cbjev name may end in @v2 to answer with option-order voting."""
    import cbjev
    if "@v" in name:
        base, votes = name.rsplit("@v", 1)
        ag = load_agent(base)
        ag.order_votes = int(votes)
        return ag
    if name == "laya":
        return _laya()
    if name == "laya-fast":
        return _laya(fast=True)
    if name == "laya-td":
        return _laya("typed-decisions")
    if name == "laya-ml":
        return _laya("multilingual")
    if name == "cbjev-classic":
        return cbjev.load("laya", device="cuda")
    if name.startswith("cbjev:"):
        return cbjev.load(name.split(":", 1)[1], device="cuda")
    return cbjev.load(name, device="cuda")


def make_engine(name):
    ag = load_agent(name)
    return lambda state, qs: ag.predict(state, qs)
