"""Greedy whole-pretoken WordPiece segmentation."""

from collections.abc import Mapping


def segment(word: str, vocab: Mapping[str, int], max_length: int) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    start = 0
    while start < len(word):
        prefix = "##" if start else ""
        for end in range(min(len(word), start + max_length), start, -1):
            surface = word[start:end]
            token = vocab.get(prefix + surface)
            if token is not None:
                tokens.append((token, surface))
                start = end
                break
        else:
            return [(1, word)]
    return tokens
