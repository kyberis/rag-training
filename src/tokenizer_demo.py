"""
Tokenization visualizer: no OpenAI calls, no embeddings — just shows how
raw text is actually split into tokens before it ever reaches the model,
using the real encoding gpt-4o-mini uses (o200k_base). This is the level
below chunking: chunking counts words as a proxy for tokens (see
chunking.py's docstring, and README.md section 1.2 for why), and this
module makes that proxy a measured ratio instead of an assumption.
"""
from __future__ import annotations

from . import config

import tiktoken

_encoding_cache: dict[str, "tiktoken.Encoding"] = {}


def _get_encoding(encoding_name: str | None = None):
    name = encoding_name or config.TOKENIZER_ENCODING_NAME
    if name not in _encoding_cache:
        _encoding_cache[name] = tiktoken.get_encoding(name)
    return _encoding_cache[name]


def tokenize_text(text: str, encoding_name: str | None = None) -> dict:
    """Splits text into tokens and returns enough detail to render each
    token as a colored span plus the word-count-vs-token-count ratio.
    """
    enc = _get_encoding(encoding_name)
    ids = enc.encode(text)
    tokens = [
        {"id": token_id, "text": enc.decode([token_id])}
        for token_id in ids
    ]
    n_words = len(text.split())
    n_tokens = len(ids)
    return {
        "encoding": enc.name,
        "tokens": tokens,
        "n_tokens": n_tokens,
        "n_words": n_words,
        "words_per_token": round(n_words / n_tokens, 2) if n_tokens else 0.0,
    }
