"""Validated tokenizer models, JSON artifacts and byte-offset token records."""

import json
import os
import re
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Self

from . import bpe, wordpiece
from .pretokenize import normalize, pretokens
from .schema import MAX_FILE, SPECIAL, integer, valid_model

_MARKERS = re.compile("|".join(map(re.escape, SPECIAL)))


@dataclass(frozen=True, slots=True)
class Token:
    id: int
    piece: str
    bytes_hex: str | None
    start: int | None
    end: int | None
    special: bool = False
    unknown: bool = False


@dataclass(slots=True)
class Encoding:
    normalized_text: str
    tokens: list[Token]

    @property
    def ids(self) -> list[int]:
        return [token.id for token in self.tokens]

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_text": self.normalized_text,
            "ids": self.ids,
            "tokens": [asdict(token) for token in self.tokens],
        }


def _regions(text: str, allowed: set[str]) -> Iterator[tuple[str, bool]]:
    start = 0
    if allowed:
        for match in _MARKERS.finditer(text):
            if match.group() in allowed:
                yield text[start : match.start()], False
                yield match.group(), True
                start = match.end()
    yield text[start:], False


def _special(idx: int) -> Token:
    return Token(idx, SPECIAL[idx], None, None, None, special=True)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON key")
    return obj


class Tokenizer:
    def __init__(self, data: dict[str, Any]) -> None:
        valid_model(data)
        self._data = deepcopy(data)
        self._vocab: tuple[str, ...] = tuple(data["vocab"])
        self._rules = {(a, b): (rank, c) for rank, (a, b, c) in enumerate(data["merges"])}
        self._wp_vocab = {p: i for i, p in enumerate(self._vocab) if i >= 5}
        self._max_length = max((len(p.removeprefix("##")) for p in self._wp_vocab), default=0)
        self._bytes = {
            i: bytes.fromhex(p) for p, i in self._wp_vocab.items() if self.algorithm == "bpe"
        }
        self._display = {
            i: raw.decode("utf-8", "backslashreplace") for i, raw in self._bytes.items()
        }

    @property
    def algorithm(self) -> str:
        return str(self._data["algorithm"])

    @property
    def normalization(self) -> str:
        return str(self._data["normalization"])

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data)

    def save(self, path: str | os.PathLike[str]) -> None:
        encoded = (
            json.dumps(self._data, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        if len(encoded) > MAX_FILE:
            raise ValueError("model exceeds file size limit")
        destination = Path(path)
        tmp = NamedTemporaryFile(dir=destination.parent, delete=False)
        temp_path = Path(tmp.name)
        try:
            with tmp:
                tmp.write(encoded)
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        with open(path, "rb") as f:
            return cls._load_stream(f)

    @classmethod
    def _load_stream(cls, stream: BinaryIO) -> Self:
        raw = stream.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE:
            raise ValueError("model exceeds file size limit")
        try:
            data = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
        except RecursionError as exc:
            raise ValueError("model JSON nesting is too deep") from exc
        return cls.from_dict(data)

    def encode(
        self,
        text: str,
        *,
        allowed_special: Iterable[str] = (),
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Encoding:
        if type(add_bos) is not bool or type(add_eos) is not bool:
            raise TypeError("add_bos and add_eos must be booleans")
        if isinstance(allowed_special, str):
            raise TypeError("allowed_special must be a collection of marker strings")
        allowed = set(allowed_special)
        if not allowed.issubset(SPECIAL):
            raise ValueError("allowed_special contains an unknown marker")
        text = normalize(text, self.normalization)
        tokens = [_special(2)] if add_bos else []
        byte = 0
        for region, special in _regions(text, allowed):
            if special:
                tokens.append(_special(SPECIAL.index(region)))
                byte += len(region)  # All reserved markers are ASCII.
                continue
            previous_end = 0
            for word, start, end in pretokens(region, self.algorithm):
                byte += len(region[previous_end:start].encode())
                if self.algorithm == "bpe":
                    for i in bpe.encode_heap(word.encode(), self._rules):
                        stop = byte + len(self._bytes[i])
                        tokens.append(Token(i, self._display[i], self._vocab[i], byte, stop))
                        byte = stop
                else:
                    for i, surface in wordpiece.segment(word, self._wp_vocab, self._max_length):
                        stop = byte + len(surface.encode())
                        tokens.append(Token(i, self._vocab[i], None, byte, stop, unknown=(i == 1)))
                        byte = stop
                previous_end = end
            byte += len(region[previous_end:].encode())
        if add_eos:
            tokens.append(_special(3))
        return Encoding(text, tokens)

    def decode(
        self, ids: Iterable[int], *, skip_special_tokens: bool = False, errors: str = "strict"
    ) -> str:
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a boolean")
        if errors not in ("strict", "replace"):
            raise ValueError("errors must be strict or replace")
        if isinstance(ids, (str, bytes, bytearray, dict)):
            raise TypeError("ids must be a sequence of integer token IDs")
        words: list[str] = []
        byte_parts: list[bytes] = []
        for item in ids:
            i = integer(item, "token ID", 0, self.vocab_size - 1)
            if skip_special_tokens and i < 5 and i != 1:
                continue
            piece = self._vocab[i]
            if self.algorithm == "bpe":
                byte_parts.append(self._bytes[i] if i >= 5 else piece.encode())
            else:
                if words and not piece.startswith("##"):
                    words.append(" ")
                words.append(piece.removeprefix("##"))
        if self.algorithm == "bpe":
            return b"".join(byte_parts).decode("utf-8", errors)
        return "".join(words)
