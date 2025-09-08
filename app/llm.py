import os, re
from typing import List, Dict, Any

# Optional OpenAI import (works if you install openai>=1.x)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class LLMClient:
    """
    - If OPENAI_API_KEY is set (and openai is installed), call OpenAI.
    - Otherwise, fall back to a tiny extractive heuristic so CI/tests pass without secrets.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = DEFAULT_MODEL
        self.use_openai = bool(self.api_key and OpenAI is not None)
        if self.use_openai:
            self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        if self.use_openai:
            # Chat Completions with minimal temperature for determinism
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            return (resp.choices[0].message.content or "").strip()

        # ----- Fallback: light extractive answer -----
        # Prefer a concise definition from the FIRST passage
        ans = None
        try:
            block = prompt.split("Use only these passages:", 1)[1]
            import re as _re
            m = _re.search(r"\[1\]\s*\([^)]*\)\s*(.+)", block, flags=_re.S)
            if m:
                passage = m.group(1).strip()
                sents = _re.split(r'(?<=[.!?])\s+', passage)
                pick = [x for x in sents if any(kw in x.lower() for kw in [" is ", " are ", " means ", " defined "])]
                ans = (" ".join(pick[:2]) if pick else " ".join(sents[:2])).strip()
        except Exception:
            ans = None

        if not ans:
            import re as _re
            sents = _re.split(r'(?<=[.!?])\s+', prompt)
            pick = [t for t in sents if any(kw in t.lower() for kw in [" is ", " are ", " means ", " defined "])]
            ans = (" ".join(pick[:2]) if pick else " ".join(sents[:2])).strip()

        # Clean leftover SYSTEM/USER noise
        ans = re.sub(r"\bSYSTEM:.*?\bUSER:", "", ans, flags=re.S).strip()
        ans = re.sub(r"\bSYSTEM:.*$", "", ans, flags=re.S).strip()
        ans = re.sub(r"\bUSER:.*$", "", ans, flags=re.S).strip()

        # Ensure at least one bracket citation if the prompt had numbered passages
        if "[1]" in prompt and "[1]" not in ans:
            ans = ans.rstrip(". ") + " [1]"

        return ans or "I don't know from the provided docs."
