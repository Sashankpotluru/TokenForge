"""Unicode scalar normalization and shared text boundaries."""

import re
import string
import unicodedata
from collections.abc import Iterator

WHITE_SPACE = "\u0009-\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028-\u2029\u202f\u205f\u3000"
_RUNS = re.compile(f"[{WHITE_SPACE}]+|[^{WHITE_SPACE}]+")
_WHITE = re.compile(f"[{WHITE_SPACE}]")


def normalize(text: str, normalization: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a Unicode string")
    text.encode("utf-8")  # Reject lone surrogates, including inside unused training input.
    return unicodedata.normalize("NFC", text) if normalization == "NFC" else text


def is_punctuation(char: str) -> bool:
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def pretokens(text: str, algorithm: str) -> Iterator[tuple[str, int, int]]:
    """Yield (text, start, end) in Unicode scalar positions; input is already normalized."""
    if algorithm not in ("bpe", "wordpiece"):
        raise ValueError("algorithm must be bpe or wordpiece")
    for match in _RUNS.finditer(text):
        run = match.group()
        start = match.start()
        if algorithm == "bpe":
            yield run, start, match.end()
        elif not _WHITE.match(run):
            for i, char in enumerate(run, match.start()):
                if is_punctuation(char):
                    if start < i:
                        yield text[start:i], start, i
                    yield char, i, i + 1
                    start = i + 1
            if start < match.end():
                yield text[start : match.end()], start, match.end()
