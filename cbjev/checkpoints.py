"""Short names for checkpoints. Anything not listed is used as a path or hub repo id as-is."""
import os

# Where the trained cbjev weights live. Override with CBJEV_HOME or pass a path to cbjev.load().
HOME = os.environ.get("CBJEV_HOME", os.path.expanduser("~/.cache/cbjev"))

NAMES = {
    "cbjev": os.path.join(HOME, "cbjev"),
    "cbjev-multilingual": os.path.join(HOME, "cbjev-multilingual"),
    # the Laya family runs unchanged in classic layout
    "laya": "convaiinnovations/laya",
}


def resolve(name: str) -> str:
    return NAMES.get(name, name)
