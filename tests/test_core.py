import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import tokenforge as tf
from tokenforge import train
from tokenforge.pretokenize import pretokens


@pytest.mark.parametrize("algorithm", ["bpe", "wordpiece"])
def test_weighting_overlap_ties_and_artifact_reordering(algorithm, tmp_path):
    corpus = ["aaaaa", "bc", "bc", "de", "de"]
    kw = dict(algorithm=algorithm, vocab_size=300, min_frequency=2)
    m = train(corpus, **kw)
    d = m.to_dict()
    first = d["trace"][0]
    if algorithm == "bpe":
        # Four overlapping aa pairs in one pretoken, only two non-overlapping replacements.
        assert (first["left"], first["right"], first["count"]) == (102, 102, 4)
        assert first["score_num"] / first["score_den"] == 4
        second = d["trace"][1]
        assert (second["left"], second["right"], second["count"]) == (103, 104, 2)
    else:
        assert d["vocab"][first["result"]] == "bc"  # bc and de both score 2/(2*2)
        assert first["count"] == 2 and first["score_num"] / first["score_den"] == 0.5
    p, q = tmp_path / "one", tmp_path / "two"
    m.save(p)
    train(reversed(corpus), **kw).save(q)
    assert p.read_bytes() == q.read_bytes()
    assert tf.Tokenizer.load(p).to_dict() == d


def test_byte_base_offsets_nfc_and_specials():
    m = train([""], algorithm="bpe", vocab_size=261, normalization="NFC")
    assert m.vocab_size == 261
    e = m.encode("e\u0301[BOS]!", allowed_special=["[BOS]"], add_bos=True, add_eos=True)
    assert e.normalized_text == "é[BOS]!"
    assert e.ids == [2, 200, 174, 2, 38, 3]
    assert [(t.start, t.end) for t in e.tokens] == [
        (None, None),
        (0, 1),
        (1, 2),
        (None, None),
        (7, 8),
        (None, None),
    ]
    assert e.tokens[1].bytes_hex == "c3" and "�" not in e.tokens[1].piece
    assert e.to_dict()["tokens"][3] == dict(
        id=2, piece="[BOS]", bytes_hex=None, start=None, end=None, special=True, unknown=False
    )
    assert m.encode("[BOS]").ids != [2]
    with pytest.raises(ValueError):
        m.encode("", allowed_special=["[TYPO]"])
    with pytest.raises(ValueError):
        m.encode("\ud800")


def test_wordpiece_unknown_rolls_back_and_joins_canonically():
    m = train(["tea! tea"], algorithm="wordpiece", vocab_size=30, min_frequency=1)
    e = m.encode(" tea!\u00a0tea茶 [UNK]", allowed_special=["[UNK]"])
    assert [t.piece for t in e.tokens] == ["tea", "!", "[UNK]", "[UNK]"]
    assert [(t.start, t.end, t.unknown, t.special) for t in e.tokens] == [
        (1, 4, False, False),
        (4, 5, False, False),
        (7, 13, True, False),
        (None, None, False, True),
    ]
    assert m.decode(e.ids, skip_special_tokens=True) == "tea ! [UNK] [UNK]"


def test_wordpiece_greedy_continuations_and_explicit_whitespace():
    m = train(
        ["ab abc abc xab a\x1cb a\x1cb $x!"], algorithm="wordpiece", vocab_size=50, min_frequency=2
    )
    assert [t.piece for t in m.encode("abcab").tokens] == ["abc", "##a", "##b"]
    assert [t.piece for t in m.encode("a\x1cb\u0085$x!").tokens] == ["a\x1cb", "$", "x", "!"]


@pytest.mark.parametrize("algorithm,minimum", [("bpe", 261), ("wordpiece", 5)])
def test_empty_text_and_empty_iterable(algorithm, minimum):
    assert train([""], algorithm=algorithm, vocab_size=minimum).encode("").ids == []
    with pytest.raises(ValueError):
        train([], algorithm=algorithm, vocab_size=minimum)
    with pytest.raises(ValueError):
        train(["ab"], algorithm=algorithm, vocab_size=minimum - 1)
    m = train(["  "], algorithm=algorithm, vocab_size=minimum)
    assert m.decode(m.encode("  ").ids) == ("  " if algorithm == "bpe" else "")


@pytest.mark.parametrize("ids", [[True], [-1], [9999], [1.5], ["3"]])
def test_decode_rejects_invalid_ids(ids):
    m = train([""], algorithm="bpe", vocab_size=261)
    with pytest.raises((TypeError, ValueError)):
        m.decode(ids)


def test_decode_incomplete_utf8_and_arbitrary_unknown_ids():
    m = train([""], algorithm="bpe", vocab_size=261)
    with pytest.raises(UnicodeDecodeError):
        m.decode([260])
    assert m.decode([260], errors="replace") == "�"
    assert m.decode([1, 2], skip_special_tokens=True) == "[UNK]"
    with pytest.raises(ValueError):
        m.decode([], errors="ignore")


def test_public_api_and_unicode_fixtures():
    m = train(["abc aa    🥁"], algorithm="bpe", vocab_size=300, min_frequency=1)
    assert isinstance(m, tf.Tokenizer)
    for s in json.loads((Path(__file__).parent / "fixtures/unicode.json").read_text()):
        e = m.encode(s)
        assert isinstance(e, tf.Encoding)
        assert m.decode(e.ids) == s
        assert b"".join(bytes.fromhex(t.bytes_hex) for t in e.tokens) == s.encode()


@given(st.text(), st.lists(st.text(alphabet="abcdeä🦦 ", max_size=25), min_size=1, max_size=6))
def test_heap_reference_and_unicode_roundtrip(s, texts):
    from tokenforge.bpe import encode_heap, encode_reference

    m = train(texts, algorithm="bpe", vocab_size=310, min_frequency=1)
    rules = {(a, b): (rank, c) for rank, (a, b, c) in enumerate(m.to_dict()["merges"])}
    for x in [s, *texts]:
        assert encode_heap(x.encode(), rules) == encode_reference(x.encode(), rules)
        e = m.encode(x)
        assert m.decode(e.ids) == x
        assert [t.bytes_hex for t in e.tokens] == [
            x.encode()[t.start : t.end].hex() for t in e.tokens
        ]


@pytest.mark.parametrize(
    "key,bad",
    [
        ("version", True),
        ("algorithm", ["bpe"]),
        ("normalization", 0),
        ("base_size", 10**100),
        ("merges", [[102, 102, 99999]]),
        ("merges", [[261, 102, 261]]),
        ("merges", [[102, 103, 261]]),
        ("trace", []),
        ("trace_truncated", 1),
        ("training", {"vocab_size": 10**100}),
    ],
)
def test_model_rejects_corruption(key, bad):
    m = train(["aaa"], algorithm="bpe", vocab_size=262, min_frequency=1)
    d = m.to_dict()
    d[key] = bad
    with pytest.raises((ValueError, TypeError)):
        tf.Tokenizer.from_dict(d)


def test_reused_byte_sequence_rule():
    m = train([""], algorithm="bpe", vocab_size=310, min_frequency=1)
    d = m.to_dict()
    d["vocab"] += ["6162", "6263", "616263"]
    d["merges"] = [[102, 103, 261], [103, 104, 262], [261, 104, 263], [102, 262, 263]]
    d["trace"] = [
        dict(rank=i, left=a, right=b, result=c, count=1, score_num=1, score_den=1)
        for i, (a, b, c) in enumerate(d["merges"])
    ]
    d["training"]["vocab_size"] = len(d["vocab"])
    assert tf.Tokenizer.from_dict(d).encode("abc").ids == [263]
    d["vocab"][-1] = "6162"
    with pytest.raises(ValueError):
        tf.Tokenizer.from_dict(d)


def test_impossible_wordpiece_frequency_denominator():
    m = train(["xy xy"], algorithm="wordpiece", vocab_size=10)
    d = m.to_dict()
    d["trace"][0]["score_den"] = 1
    with pytest.raises(ValueError):
        tf.Tokenizer.from_dict(d)


def test_wordpiece_alphabet_order_and_invalid_continuations():
    m = train(["ab ab $"], algorithm="wordpiece", vocab_size=12, max_merges=0)
    d = m.to_dict()
    d["vocab"][5:7] = reversed(d["vocab"][5:7])
    with pytest.raises(ValueError):
        tf.Tokenizer.from_dict(d)
    d = m.to_dict()
    d["vocab"][5] = "##$"
    with pytest.raises(ValueError):
        tf.Tokenizer.from_dict(d)


def test_atomic_failure_leaves_original_and_duplicate_json_rejected(tmp_path):
    from unittest.mock import patch

    target = tmp_path / "saved.json"
    target.write_bytes(b"original")
    m = train([""], algorithm="bpe", vocab_size=261)
    with patch("tokenforge.model.os.replace", side_effect=OSError("full")), pytest.raises(OSError):
        m.save(target)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]
    target.write_text('{"version":1,"version":0}')
    with pytest.raises(ValueError, match="duplicate"):
        tf.Tokenizer.load(target)


def test_trace_is_bounded_and_deep_copies_preserve_the_model():
    m = train([f"word{i}xyz" for i in range(350)], algorithm="bpe", vocab_size=620, min_frequency=1)
    d = m.to_dict()
    assert len(d["merges"]) > 256
    assert len(d["trace"]) == 256 and d["trace_truncated"] is True
    d["vocab"][5] = "changed"
    assert m.decode([5]) == "\x00"


@pytest.mark.parametrize(
    "kw",
    [
        {"vocab_size": True},
        {"vocab_size": -1},
        {"min_frequency": False},
        {"min_frequency": 0},
        {"normalization": "NFD"},
        {"algorithm": "unknown"},
        {"max_merges": -1},
        {"max_merges": True},
    ],
)
def test_invalid_training_parameters(kw):
    options = {"algorithm": "wordpiece", "vocab_size": 40, **kw}
    with pytest.raises(ValueError):
        train(["a"], **options)


def test_ranked_merges_stay_within_explicit_whitespace_runs():
    model = train(["a b a b a b"], algorithm="bpe", vocab_size=300, min_frequency=1)
    assert model.encode("a b").ids == [102, 37, 103]
    whitespace = (
        "\t\n\v\f\r \u0085\xa0\u1680\u2000\u2001\u2002\u2003"
        "\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
    )
    for space in whitespace:
        assert list(pretokens("a" + space + "b", "bpe")) == [
            ("a", 0, 1),
            (space, 1, 2),
            ("b", 2, 3),
        ]
    assert list(pretokens("a\x1cb", "bpe")) == [("a\x1cb", 0, 3)]


def test_long_known_word_has_no_wordpiece_length_cutoff():
    model = train(["aa"], algorithm="wordpiece", vocab_size=12, max_merges=0)
    text = "a" * 12001
    assert model.decode(model.encode(text).ids) == text


def test_observed_alphabet_must_fit():
    with pytest.raises(ValueError, match="base"):
        train(["abc"], algorithm="wordpiece", vocab_size=7)
