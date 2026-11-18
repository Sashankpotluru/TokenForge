import json

import pytest

from tokenforge import train
from tokenforge.schema import SPECIAL

pytest.importorskip("tokenizers")
from benchmarks.adapters import HFAdapter
from benchmarks.data import ROOT


@pytest.mark.parametrize("algorithm", ["bpe", "wordpiece"])
def test_hf_parity_on_markers_byte_alphabet_and_long_words(algorithm):
    fixtures = json.loads((ROOT / "tests/fixtures/unicode.json").read_text())
    literal = "[UNK][MASK]"
    # Combining characters and extended Latin exercise both byte remapping ranges.
    corpus = [literal, "aaa aa a aé — ÿ ¡ ﬁ", "a" * 120, *fixtures]
    model = train(corpus, algorithm=algorithm, vocab_size=512, normalization="NFC", min_frequency=1)
    adapter = HFAdapter(model)
    samples = [*corpus, "a" * 121, literal + "foreign𒁃" + literal, "\u001c\ufeff", "e\u0301"]
    for text in samples:
        for allowed in [(), ("[UNK]",), SPECIAL]:
            opts = dict(allowed_special=allowed, add_bos=True, add_eos=True)
            assert adapter.encode(text, **opts) == model.encode(text, **opts).ids
    if algorithm == "bpe":
        assert len(adapter.encode(literal)) == 1
        assert adapter.encode(literal, allowed_special=SPECIAL) == [1, 4]
    else:
        assert 1 not in adapter.encode("a" * 121)
