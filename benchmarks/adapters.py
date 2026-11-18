"""Reference runtimes. Hugging Face receives exactly the selected TokenForge model."""

import hashlib
import os
import re

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers

from tokenforge.pretokenize import normalize, pretokens
from tokenforge.schema import SPECIAL

from .data import CACHE


def byte_alphabet():
    # The GPT-2 byte alphabet used by HF ByteLevel. Printable ranges keep their
    # code points. Assign the remaining bytes fresh code points in byte order.
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    alphabet = {i: chr(i) for i in visible}
    for offset, i in enumerate(i for i in range(256) if i not in visible):
        alphabet[i] = chr(256 + offset)
    return alphabet


class HFAdapter:
    def __init__(self, model):
        data = model.to_dict()
        self.algorithm = model.algorithm
        self.normalization = model.normalization
        if self.algorithm == "bpe":
            alphabet = byte_alphabet()
            # Private-use symbols can never be produced by the byte alphabet.
            pieces = [chr(0xE000 + i) for i in range(len(SPECIAL))]
            pieces.extend(
                "".join(alphabet[b] for b in bytes.fromhex(piece)) for piece in data["vocab"][5:]
            )
            vocab = {piece: i for i, piece in enumerate(pieces)}
            pairs = [(pieces[a], pieces[b]) for a, b, _ in data["merges"]]
            self.engine = HFTokenizer(models.BPE(vocab, pairs, cache_capacity=0))
            self.engine.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False
            )
        else:
            vocab = {piece: i for i, piece in enumerate(data["vocab"])}
            self.engine = HFTokenizer(
                models.WordPiece(
                    vocab,
                    unk_token="[UNK]",
                    continuing_subword_prefix="##",
                    max_input_chars_per_word=2_147_483_647,
                )
            )

    def encode(self, text, *, allowed_special=(), add_bos=False, add_eos=False):
        text = normalize(text, self.normalization)
        ids = [2] if add_bos else []
        if isinstance(allowed_special, str):
            raise TypeError("expected a collection of reserved markers")
        allowed_special = set(allowed_special)
        if not allowed_special <= set(SPECIAL):
            raise ValueError("expected a collection of reserved markers")
        parts = (
            re.split("(" + "|".join(map(re.escape, sorted(allowed_special))) + ")", text)
            if allowed_special
            else [text]
        )
        for i, part in enumerate(parts):
            if i % 2:
                ids.append(SPECIAL.index(part))
            else:
                words = [word for word, _, _ in pretokens(part, self.algorithm)]
                if words:
                    ids.extend(
                        self.engine.encode(
                            words, is_pretokenized=True, add_special_tokens=False
                        ).ids
                    )
        if add_eos:
            ids.append(3)
        return ids


def cl100k(manifest, cache=CACHE):
    import tiktoken

    spec = next(f for f in manifest["files"] if f["path"] == "cl100k_base.tiktoken")
    # The official loader keys its local cache by URL SHA-1 and verifies content
    # with its own expected SHA-256. Populate it from the already-verified asset.
    cache_dir = cache / "tiktoken"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha1(spec["url"].encode()).hexdigest()
    (cache_dir / key).write_bytes((cache / spec["path"]).read_bytes())
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    return tiktoken.get_encoding("cl100k_base")
