"""Byte encoders. A rule maps an adjacent ID pair to (rank, result ID)."""

from collections.abc import Mapping
from heapq import heappop, heappush

type Rules = Mapping[tuple[int, int], tuple[int, int]]
BYTE_START = 5


def encode_reference(data: bytes, rules: Rules) -> list[int]:
    """Simple specification: replace the lowest rank, leftmost position each time."""
    ids = [b + BYTE_START for b in data]
    while True:
        options = (
            (rule[0], i, rule[1])
            for i, pair in enumerate(zip(ids, ids[1:], strict=False))
            if (rule := rules.get(pair)) is not None
        )
        choice = min(options, default=None)
        if choice is None:
            return ids
        _, i, result = choice
        ids[i : i + 2] = [result]


def encode_heap(data: bytes, rules: Rules) -> list[int]:
    """Keep neighbors in arrays; only links incident to a replacement can gain rules."""
    size = len(data)
    ids = [b + BYTE_START for b in data]
    if size < 2 or not rules:
        return ids
    before = list(range(-1, size - 1))
    after = list(range(1, size + 1))
    queue: list[tuple[int, int, int, int]] = []

    def enqueue(left: int) -> None:
        if left == -1 or (right := after[left]) == size:
            return
        rule = rules.get((ids[left], ids[right]))
        if rule is not None:
            heappush(queue, (rule[0], left, right, rule[1]))

    for i in range(size - 1):
        enqueue(i)
    while queue:
        rank, left, right, result = heappop(queue)
        if after[left] != right or rules.get((ids[left], ids[right])) != (rank, result):
            continue
        ids[left] = result
        ids[right] = -1
        after[left] = after[right]
        if after[right] < size:
            before[after[right]] = left
        enqueue(before[left])
        enqueue(left)
    return [idx for idx in ids if idx >= 0]
