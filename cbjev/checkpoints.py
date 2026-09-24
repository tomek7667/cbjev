"""Short names for checkpoints. Anything not listed is used as a path or hub repo id as-is.

A name resolves to a local directory under CBJEV_HOME when one exists (e.g. weights you trained
yourself with training/), and to the published Hugging Face repo otherwise.
"""
import os

# Local weights, if any. Override with CBJEV_HOME or pass a path to cbjev.load().
HOME = os.environ.get("CBJEV_HOME", os.path.expanduser("~/.cache/cbjev"))

HUB = "0010101010-1/cbjev"

# name -> (local directory, hub repo, subfolder in that repo)
NAMES = {
    "cbjev": (os.path.join(HOME, "cbjev"), HUB, None),
    "cbjev-multilingual": (os.path.join(HOME, "cbjev-multilingual"), HUB, "multilingual"),
    # the Laya family runs unchanged in classic layout
    "laya": (None, "convaiinnovations/laya", None),
}


def locate(name: str):
    """(path or repo id, subfolder) for a checkpoint name, path or repo id."""
    if name not in NAMES:
        return name, None
    local, repo, sub = NAMES[name]
    if local and os.path.isfile(os.path.join(local, "model.safetensors")):
        return local, None
    return repo, sub


def resolve(name: str) -> str:
    """Path or repo id only, for callers that pass their own subfolder."""
    return locate(name)[0]
