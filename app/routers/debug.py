from fastapi import APIRouter
import sys, os, platform, resource
from typing import Dict

try:
    from app.retrieval import get_embeddings_and_meta
    _HAS_RETRIEVAL = True
except Exception:
    _HAS_RETRIEVAL = False
    get_embeddings_and_meta = None  # type: ignore

router = APIRouter(tags=["debug"])

def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns kilobytes.
    return round(usage/1e6, 2) if usage > 1e9 else round(usage/1024, 2)

@router.get("/debug/mem")
async def debug_mem() -> Dict:
    info = {
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rss_mb": _rss_mb(),
        "env": {
            "POKERRAG_STEP": os.getenv("POKERRAG_STEP", None),
        },
        "embeddings": {
            "loaded": False,
            "shape": None,
            "dtype": None,
            "is_memmap": None,
        }
    }
    if _HAS_RETRIEVAL and get_embeddings_and_meta:
        try:
            E, M = get_embeddings_and_meta()
            info["embeddings"].update({
                "loaded": True,
                "shape": tuple(getattr(E, "shape", [])),
                "dtype": str(getattr(E, "dtype", "")),
                "is_memmap": E.__class__.__name__ == "memmap",
            })
        except Exception as e:
            info["embeddings"]["error"] = str(e)
    return info
