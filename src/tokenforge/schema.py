"""Wire format validation shared by training, loading, and runtime construction."""

import re
from typing import Any

from .pretokenize import WHITE_SPACE, is_punctuation

SPECIAL = ("[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]")
MAX_VOCAB = 100_000
MAX_MERGES = 200_000
TRACE_LIMIT = 256
MAX_FILE = 32 * 1024 * 1024
MAX_PIECE = 16_384  # Bytes for BPE, Unicode scalars for WordPiece.
MAX_COUNT = 2**63 - 1
_HEX = re.compile(r"(?:[0-9a-f]{2})+")
_SPACE = re.compile(f"[{WHITE_SPACE}]")
_FIELDS = {
    "version",
    "algorithm",
    "normalization",
    "base_size",
    "vocab",
    "merges",
    "training",
    "trace",
    "trace_truncated",
}


def integer(value: object, name: str, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def parameters(
    algorithm: str, normalization: str, vocab_size: int, min_frequency: int, max_merges: int | None
) -> None:
    if algorithm not in ("bpe", "wordpiece"):
        raise ValueError("algorithm must be bpe or wordpiece")
    if normalization not in ("none", "NFC"):
        raise ValueError("normalization must be none or NFC")
    integer(vocab_size, "vocab_size", 261 if algorithm == "bpe" else 5, MAX_VOCAB)
    integer(min_frequency, "min_frequency", 1, MAX_COUNT)
    if max_merges is not None:
        integer(max_merges, "max_merges", 0, MAX_MERGES)


def merged_piece(left: str, right: str, algorithm: str) -> str:
    return left + (right.removeprefix("##") if algorithm == "wordpiece" else right)


def valid_model(d: Any) -> None:
    if not isinstance(d, dict) or d.keys() != _FIELDS:
        raise ValueError("invalid model fields")
    integer(d["version"], "version", 1, 1)
    p = d["training"]
    if not isinstance(p, dict) or p.keys() != {"vocab_size", "min_frequency", "max_merges"}:
        raise ValueError("invalid training parameters")
    algo = d["algorithm"]
    parameters(algo, d["normalization"], p["vocab_size"], p["min_frequency"], p["max_merges"])
    vocab = d["vocab"]
    if not isinstance(vocab, list) or not 5 <= len(vocab) <= p["vocab_size"]:
        raise ValueError("invalid vocabulary length")
    if vocab[:5] != list(SPECIAL):
        raise ValueError("invalid special IDs")
    base = integer(d["base_size"], "base_size", 5, len(vocab))
    if algo == "bpe" and (base != 261 or vocab[5:261] != [f"{b:02x}" for b in range(256)]):
        raise ValueError("invalid byte alphabet")
    total = 0
    for i, piece in enumerate(vocab[5:], 5):
        if not isinstance(piece, str) or not 1 <= len(piece) <= 2 * MAX_PIECE:
            raise ValueError("invalid token size or type")
        total += len(piece.encode())
        if total > MAX_FILE:
            raise ValueError("vocabulary exceeds model size limit")
        if algo == "bpe":
            if not _HEX.fullmatch(piece):
                raise ValueError("invalid byte hex")
        else:
            surface = piece.removeprefix("##")
            if not 1 <= len(surface) <= MAX_PIECE or _SPACE.search(surface):
                raise ValueError("invalid WordPiece surface")
            if (i < base and len(surface) != 1) or (i >= base and len(surface) == 1):
                raise ValueError("invalid WordPiece alphabet")
            if (len(surface) > 1 or surface != piece) and any(map(is_punctuation, surface)):
                raise ValueError("invalid punctuation in WordPiece token")
    if algo == "wordpiece" and vocab[5:base] != sorted(vocab[5:base]):
        raise ValueError("WordPiece alphabet must be in Unicode sort order")
    if len(set(vocab)) != len(vocab):
        raise ValueError("duplicate token")
    merges = d["merges"]
    limit = MAX_MERGES if p["max_merges"] is None else p["max_merges"]
    if not isinstance(merges, list) or len(merges) > limit:
        raise ValueError("invalid merges length")
    trace = d["trace"]
    if not isinstance(trace, list) or len(trace) != min(TRACE_LIMIT, len(merges)):
        raise ValueError("invalid trace length")
    if d["trace_truncated"] is not (len(merges) > TRACE_LIMIT):
        raise ValueError("invalid trace_truncated")
    next_id = base
    seen: set[tuple[int, int]] = set()
    for rank, rule in enumerate(merges):
        if not isinstance(rule, list) or len(rule) != 3:
            raise ValueError("invalid merge record")
        a = integer(rule[0], "left ID", 5, next_id - 1)
        b = integer(rule[1], "right ID", 5, next_id - 1)
        c = integer(rule[2], "result ID", 5, min(len(vocab) - 1, next_id))
        if (a, b) in seen or merged_piece(vocab[a], vocab[b], algo) != vocab[c]:
            raise ValueError("duplicate pair or invalid merge concatenation")
        if algo == "wordpiece" and not vocab[b].startswith("##"):
            raise ValueError("right WordPiece must be a continuation")
        seen.add((a, b))
        next_id += int(c == next_id)
        if rank < len(trace):
            row = trace[rank]
            if not isinstance(row, dict) or row.keys() != {
                "rank",
                "left",
                "right",
                "result",
                "count",
                "score_num",
                "score_den",
            }:
                raise ValueError("invalid trace fields")
            for name, expected in (("rank", rank), ("left", a), ("right", b), ("result", c)):
                integer(row[name], f"trace {name}", expected, expected)
            count = integer(row["count"], "trace count", p["min_frequency"], MAX_COUNT)
            integer(row["score_num"], "trace numerator", count, count)
            if algo == "bpe":
                integer(row["score_den"], "trace denominator", 1, 1)
            else:
                integer(row["score_den"], "trace denominator", count**2, MAX_COUNT**2)
    if next_id != len(vocab):
        raise ValueError("unreachable vocabulary token")
