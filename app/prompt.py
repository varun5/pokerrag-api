from textwrap import dedent
import re

# System guidance for the model
SYSTEM = (
    "You are a concise assistant. Answer strictly from the provided passages. "
    "If the answer is not present, say you don’t know."
)

# Hard limits to keep prompts tidy
MAX_PASSAGES = 8          # don't flood the model
MAX_CHARS_PER_PASSAGE = 1200  # truncate long chunks to avoid token bloat

def _trim(text: str, max_chars: int) -> str:
    t = text.strip().replace('\r', ' ')
    return (t[: max_chars - 1] + '…') if len(t) > max_chars else t

def _sanitize(text: str) -> str:
    # remove markdown links [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # remove bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

    t = text.strip().replace("\r", " ")
    return (t[: max_chars - 1] + "…") if len(t) > max_chars else t

def build_prompt(question: str, passages: list[dict]) -> str:
    """
    Build a single-string prompt that:
      - Lists numbered passages with (doc_id) tags
      - Asks for a short answer
      - Instructs the model to cite with [1], [2], etc.
    passages: [{ "text": str, "metadata": {"doc_id": str, ...}}, ...]
    """
    # Limit and trim passages
    passages = passages[:MAX_PASSAGES]
    lines = []
    for i, p in enumerate(passages, start=1):
        doc = (p.get("metadata") or {}).get("doc_id", "doc")
        raw = p.get('text', '')
        clean = _sanitize(raw)
        text = _trim(clean, MAX_CHARS_PER_PASSAGE)
        lines.append(f"[{i}] ({doc}) {text}")

    context = "\n\n".join(lines)
    prompt = dedent(f"""
    SYSTEM:
    {SYSTEM}

    USER:
    Question: {question}

    Use only these passages:
    {context}

    Instructions for your reply:
    - Answer in 1–4 sentences when possible.
    - Cite relevant passage numbers in square brackets like [1], [2].
    - If uncertain or unsupported by the passages, say:
      "I don't know from the provided docs."
    """).strip()
    return prompt
