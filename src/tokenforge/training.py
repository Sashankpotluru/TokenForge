"""Train on weighted unique pretokens with an inverted index of adjacent pairs."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from itertools import pairwise

from . import bpe
from .model import Tokenizer
from .pretokenize import normalize, pretokens
from .schema import MAX_MERGES, MAX_PIECE, SPECIAL, TRACE_LIMIT, merged_piece, parameters

type Pair = tuple[int, int]


def _letters(word: str) -> Iterator[str]:
    yield word[0]
    for c in word[1:]:
        yield "##" + c


class _Trainer:
    def __init__(self, words: list[tuple[list[int], int]], algorithm: str) -> None:
        self.words = words
        self.algorithm = algorithm
        self.members: dict[Pair, set[int]] = defaultdict(set)
        self.pairs: Counter[Pair] = Counter()
        self.symbols: Counter[int] = Counter()
        for i, (ids, weight) in enumerate(words):
            self._index(i, ids, weight, add=True)

    def _index(self, i: int, ids: list[int], weight: int, *, add: bool) -> None:
        change = weight if add else -weight
        if self.algorithm == "wordpiece":
            for token in ids:
                self.symbols[token] += change
        for pair in pairwise(ids):
            self.pairs[pair] += change
            if add:
                self.members[pair].add(i)
            else:
                self.members[pair].discard(i)
                if not self.pairs[pair]:
                    del self.pairs[pair]
                    del self.members[pair]

    def best(self, minimum: int) -> tuple[Pair, int, int] | None:
        winner: Pair | None = None
        best_count, best_den = 0, 1
        for pair, count in self.pairs.items():
            if count < minimum:
                continue
            a, b = pair
            den = 1 if self.algorithm == "bpe" else self.symbols[a] * self.symbols[b]
            # Cross multiplication gives exact rational ordering, including ties.
            difference = count * best_den - best_count * den
            if difference > 0 or (difference == 0 and winner is not None and pair < winner):
                winner, best_count, best_den = pair, count, den
        return (winner, best_count, best_den) if winner is not None else None

    def replace(self, pair: Pair, result: int) -> None:
        for i in tuple(self.members[pair]):
            old, weight = self.words[i]
            self._index(i, old, weight, add=False)
            new = []
            pos = 0
            while pos < len(old):
                if pos + 1 < len(old) and (old[pos], old[pos + 1]) == pair:
                    new.append(result)
                    pos += 2
                else:
                    new.append(old[pos])
                    pos += 1
            self.words[i] = (new, weight)
            self._index(i, new, weight, add=True)


def train(
    texts: Iterable[str],
    *,
    algorithm: str,
    vocab_size: int,
    min_frequency: int = 2,
    normalization: str = "none",
    max_merges: int | None = None,
) -> Tokenizer:
    if normalization == "nfc":
        normalization = "NFC"
    parameters(algorithm, normalization, vocab_size, min_frequency, max_merges)
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of documents, e.g. [text]")
    words: Counter[str] = Counter()
    seen_document = False
    for text in texts:
        seen_document = True
        words.update(word for word, _, _ in pretokens(normalize(text, normalization), algorithm))
    if not seen_document:
        raise ValueError("training requires at least one document")
    vocab = list(SPECIAL)
    if algorithm == "bpe":
        vocab += [f"{i:02x}" for i in range(256)]
    else:
        vocab += sorted({letter for word in words for letter in _letters(word)})
    if len(vocab) > vocab_size:
        raise ValueError(f"vocab_size {vocab_size} cannot hold the {len(vocab)} base tokens")
    base = len(vocab)
    lookup = {s: i for i, s in enumerate(vocab)}
    weighted = []
    for word, weight in words.items():
        ids = (
            [n + bpe.BYTE_START for n in word.encode()]
            if algorithm == "bpe"
            else [lookup[s] for s in _letters(word)]
        )
        weighted.append((ids, weight))
    trainer = _Trainer(weighted, algorithm)
    merges: list[list[int]] = []
    trace: list[dict[str, int]] = []
    limit = MAX_MERGES if max_merges is None else max_merges
    while len(vocab) < vocab_size and len(merges) < limit:
        best = trainer.best(min_frequency)
        if best is None:
            break
        (left, right), count, den = best
        piece = merged_piece(vocab[left], vocab[right], algorithm)
        if len(piece) > (2 * MAX_PIECE if algorithm == "bpe" else MAX_PIECE + 2):
            raise ValueError("learned token exceeds model size limit")
        result = lookup.get(piece)
        if result is None:
            result = len(vocab)
            vocab.append(piece)
            lookup[piece] = result
        rank = len(merges)
        merges.append([left, right, result])
        if rank < TRACE_LIMIT:
            trace.append(
                dict(
                    rank=rank,
                    left=left,
                    right=right,
                    result=result,
                    count=count,
                    score_num=count,
                    score_den=den,
                )
            )
        trainer.replace((left, right), result)
    return Tokenizer.from_dict(
        dict(
            version=1,
            algorithm=algorithm,
            normalization=normalization,
            vocab=vocab,
            base_size=base,
            merges=merges,
            trace=trace,
            trace_truncated=len(merges) > TRACE_LIMIT,
            training=dict(
                vocab_size=vocab_size, min_frequency=min_frequency, max_merges=max_merges
            ),
        )
    )
