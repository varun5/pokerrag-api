from fastapi import APIRouter
import sys, os, platform, resource
from typing import Dict

try:
    from app.retrieval import get_embeddings_and_meta
    _HAS_RETRIEVAL = True
except Exception:
    _HAS_RETRIEVAL = False
    get_embeddings_and_meta = None  # type: ignore

try:
    from app.llm import LLMClient
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False
    LLMClient = None  # type: ignore

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

@router.get("/debug/llm")
async def debug_llm() -> Dict:
    info = {
        "llm_available": _HAS_LLM,
        "env_vars": {
            "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "NOT_SET",
            "GROQ_API_KEY": "SET" if os.getenv("GROQ_API_KEY") else "NOT_SET",
        },
        "llm_config": None,
        "error": None
    }
    
    if _HAS_LLM and LLMClient:
        try:
            llm_client = LLMClient()
            info["llm_config"] = {
                "use_openai": llm_client.use_openai,
                "use_groq": llm_client.use_groq,
                "use_tinyllama": llm_client.use_tinyllama,
                "fallback_active": not (llm_client.use_openai or llm_client.use_groq or llm_client.use_tinyllama),
            }
        except Exception as e:
            info["error"] = str(e)
    
    return info
