"""Row-level evidence, corpus-weighted metrics and timing units."""

from statistics import median

from tokenforge.pretokenize import pretokens


def word_spans(text):
    byte = previous = 0
    spans = []
    for surface, start, end in pretokens(text, "wordpiece"):
        byte += len(text[previous:start].encode())
        stop = byte + len(surface.encode())
        spans.append((surface, byte, stop))
        byte, previous = stop, end
    return spans


def observe_model(text, spans, model, adapter):
    encoding = model.encode(text)
    ids = encoding.ids
    if ids != adapter.encode(text):
        raise ValueError("Hugging Face ID mismatch")
    decoded = model.decode(ids)
    exact = decoded == text
    evidence = {"tokens": len(ids), "exact": exact, "hf_ids_match": True}
    if model.algorithm == "bpe":
        if not exact:
            raise ValueError("BPE failed exact round trip")
        return evidence
    unknown = {(t.start, t.end) for t in encoding.tokens if t.unknown}
    if not unknown.issubset({(start, end) for _, start, end in spans}):
        raise ValueError("Unknown token does not cover a complete pretoken")
    canonical = " ".join(
        "[UNK]" if (start, end) in unknown else surface for surface, start, end in spans
    )
    if decoded != canonical:
        raise ValueError("WordPiece failed canonical reconstruction")
    return {
        **evidence,
        "unknown_words": len(unknown),
        "unknown_bytes": sum(end - start for start, end in unknown),
        "canonical": True,
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(rows, key, algorithm):
    observations = [r["models"][key] for r in rows]
    tokens = sum(o["tokens"] for o in observations)
    nbytes = sum(r["bytes"] for r in rows)
    scalars = sum(r["scalars"] for r in rows)
    wp = algorithm == "wordpiece"
    words = sum(r["words"] for r in rows)
    content = sum(r["word_bytes"] for r in rows)
    unknown = sum(o["unknown_words"] for o in observations) if wp else 0
    lost = sum(o["unknown_bytes"] for o in observations) if wp else 0
    same_model = algorithm in ("bpe", "wordpiece")
    return {
        "rows": len(rows),
        "bytes": nbytes,
        "scalars": scalars,
        "tokens": tokens,
        "bytes_per_token": ratio(nbytes, tokens),
        "scalars_per_token": ratio(scalars, tokens),
        "exact_roundtrips": sum(o["exact"] for o in observations),
        "canonical_matches": sum(o["canonical"] for o in observations) if wp else None,
        "parity_rows": sum(o["hf_ids_match"] for o in observations) if same_model else None,
        "words": words if wp else None,
        "unknown_words": unknown if wp else None,
        "unknown_word_rate": ratio(unknown, words) if wp else None,
        "unknown_token_rate": ratio(unknown, tokens) if wp else None,
        "word_coverage": ratio(words - unknown, words) if wp else None,
        "unknown_bytes": lost if wp else None,
        "word_bytes": content if wp else None,
        "word_byte_coverage": ratio(content - lost, content) if wp else None,
        "known_bytes_per_known_token": ratio(content - lost, tokens - unknown) if wp else None,
    }


def timing_rates(nbytes, tokens, runs_ns):
    ns = median(runs_ns)
    if ns <= 0:
        raise ValueError("timing must have a positive duration")
    return {
        "median_bytes_per_second": nbytes * 1e9 / ns,
        "median_tokens_per_second": tokens * 1e9 / ns,
        "repetitions": len(runs_ns),
    }
