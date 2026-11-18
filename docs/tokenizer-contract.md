# TokenForge tokenizer contract

## Python API

Python 3.12 or later. The core uses the standard library.

```python
from tokenforge import Encoding, Tokenizer, train

model = train(
    ["one tea", "two teas"],
    algorithm="bpe",
    vocab_size=300,
    min_frequency=2,
    normalization="none",
    max_merges=None,
)
encoded = model.encode("tea 🫖", allowed_special=(), add_bos=False, add_eos=False)
assert isinstance(encoded, Encoding)
assert model.decode(encoded.ids, skip_special_tokens=False, errors="strict") == "tea 🫖"
model.save("tea.json")
assert Tokenizer.load("tea.json").to_dict() == model.to_dict()
```

All parameters after the first argument to `train`, `encode`, and `decode` are keyword-only. Training requires an iterable containing at least one document (use `[text]` for one string). Documents may be empty. `algorithm` is `bpe` or `wordpiece`. `normalization` is `none` or `NFC`. Training also accepts the alias `nfc` and serializes it as `NFC`. Lone surrogates raise a Unicode error in training and encoding. Model properties `algorithm`, `normalization` and `vocab_size` expose the algorithm, canonical normalization and realized ID count.

### Text boundaries and training

BPE starts with the five reserved IDs followed by all 256 byte values. Its pretokens alternate between maximal whitespace and non-whitespace runs. WordPiece discards whitespace, isolates every ASCII punctuation character and every Unicode character in general category P, and groups the other characters into runs. Both use the Unicode White_Space set U+0009–000D, 0020, 0085, 00A0, 1680, 2000–200A, 2028–2029, 202F, 205F, 3000. WordPiece punctuation classification and optional NFC follow the Python runtime's Unicode database. `tokenforge.pretokenize.pretokens(normalized_text, algorithm)` yields `(surface, start, end)` in **Unicode scalar positions**. Public encoding offsets use **UTF-8 bytes**.

Each document is normalized and split independently. Pair and symbol counts include the frequency of each distinct pretoken. Pair counts include overlapping occurrences. Each selected pair is replaced left-to-right without overlap. Eligible pairs meet `min_frequency`. BPE maximizes pair frequency. WordPiece maximizes the exact ratio `pair_count / (left_count * right_count)`, a deterministic educational heuristic. Google's original training procedure is unpublished. Both implementations break equal scores by the lexicographically smallest pair of integer IDs.

The WordPiece base alphabet contains observed initials as plain scalar strings and observed noninitials prefixed by `##`. IDs after the specials follow Python's Unicode string sort order on these complete symbol strings. The vocabulary budget includes specials and all base tokens. Too-small budgets raise `ValueError`. A corpus of only empty documents yields the base vocabulary. Whitespace-only documents supply ordinary whitespace to BPE and no pretokens to WordPiece. A run stops at the target, the merge limit, or exhaustion of eligible pairs. The realized count can be smaller than the target.

### Encoding and decoding

BPE replaces the lowest-ranked adjacent pair in a pretoken, breaking rank ties by the leftmost location. The heap encoder maintains neighbor links. The independent reference encoder rescans the current sequence for the next operation. Both operate on byte IDs, which permits splitting a multibyte scalar between output tokens.

WordPiece greedily selects the longest match at each position, requiring the `##` prefix after the first piece. If any suffix fails, the entire pretoken emits one `[UNK]`. There is no input-length cutoff. Candidate searches are bounded by the longest vocabulary surface. Hugging Face adapter implementations should raise `max_input_chars_per_word` to at least the longest tested pretoken (for example, 2,147,483,647) and use TokenForge's boundaries.

`Encoding.normalized_text` is the actual text used for encoding. `Encoding.tokens` is an ordered list of records and `Encoding.ids` gives their IDs in the same order. `Encoding.to_dict()` returns those three fields as JSON-ready values.

| Record field | Meaning |
|---|---|
| `id` | Integer vocabulary index |
| `piece` | For ordinary BPE, UTF-8 display text with malformed bytes shown as literal backslash-x escapes. For WordPiece and specials, the vocabulary string |
| `bytes_hex` | Lowercase byte hex for ordinary BPE tokens, otherwise null. Use this field for exact bytes |
| `start`, `end` | Half-open UTF-8 byte offsets in the normalized input, or both null for a recognized or inserted special |
| `special` | True for explicitly recognized markers or inserted BOS/EOS |
| `unknown` | True only for whole-pretoken WordPiece fallback, which carries the full pretoken byte span |

IDs 0–4 are `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, `[MASK]` in order. Literal marker strings follow ordinary pretokenization by default. Supply a collection such as `allowed_special={"[MASK]"}` to recognize selected literals exactly. `add_bos` and `add_eos` insert their markers independently. When `skip_special_tokens=True`, decode skips IDs 0, 2, 3 and 4. ID 1 stays visible because the IDs alone cannot distinguish a recognized special from a whole-word fallback.

BPE decoding concatenates bytes before decoding as UTF-8. Default encoding and decoding reconstruct the normalized input exactly when BOS/EOS are disabled. `errors="strict"` raises for malformed byte streams, while `"replace"` inserts Unicode replacement characters. Negative, out-of-vocabulary and non-integer IDs, including booleans, raise errors.

WordPiece canonical decoding joins continuations to the preceding surface and separates all initial pieces with one ASCII space, including punctuation. A leading continuation in an arbitrary ID sequence starts a word with its surface. Canonical output loses original spacing and substitutes `[UNK]` for unrepresentable words. Accuracy and compression reports should count those unknown records and their original byte spans.

## JSON model version 1

`Tokenizer.to_dict()` returns a detached copy. `Tokenizer.from_dict(data)` validates and detaches a JSON-compatible model. `save(path)` writes canonical compact UTF-8 JSON with sorted object keys, literal non-ASCII characters and a single trailing LF, then atomically replaces the destination. `load(path)` reads and validates the artifact. Corpus reordering produces byte-identical artifacts for a given Python Unicode database and fixed parameters.

| Top-level field | Type and semantics |
|---|---|
| `version` | Integer 1 |
| `algorithm`, `normalization` | Canonical names described above |
| `vocab` | Array of unique strings. Array index is token ID. The first five entries are reserved marker literals. Ordinary BPE strings are lowercase even-length hex. IDs 5 through 260 equal byte value plus five. WordPiece tokens use literal surfaces and the continuation prefix |
| `base_size` | Number of initial tokens including specials, before any merges |
| `merges` | Ranked list of three-integer arrays `[left, right, result]`. Rank starts at zero. Operands must already exist. A new result uses the next available ID. An existing ID is valid when a different pair spells the same result. WordPiece joins the left token with the right token's continuation surface |
| `training` | Object with integer `vocab_size` (target), positive integer `min_frequency`, and integer or null `max_merges` (zero trains the base only) |
| `trace` | First 256 choices, or all choices for shorter runs. Each object contains integer `rank`, `left`, `right`, `result`, `count`, `score_num`, `score_den`. Look up the three symbols by ID in `vocab`. `count` is weighted overlapping pair frequency at selection. The numerator equals count. The denominator is 1 for BPE and the product of the two weighted symbol frequencies for WordPiece |
| `trace_truncated` | Boolean equal to whether more choices exist than fit in `trace`. The full ranked rule list is always in `merges` |

Version 1 limits vocabulary to 100,000 IDs, rules to 200,000, model files to 32 MiB, and individual ordinary pieces to 16,384 bytes (BPE) or scalars excluding the prefix (WordPiece). Frequencies are positive signed-64-bit values and the WordPiece denominator can reach their squared maximum. Counts, lengths, sequential reachability, concatenations, schema fields, and reserved/ordinary alphabets are checked on import. JSON load rejects duplicate keys. The trace records training observations; authenticating those observations against the corpus requires retraining.

## CLI

All commands write JSON to stdout and failures to stderr with a nonzero exit status. Input files use UTF-8, preserving CRLF. Omit `--input` or pass `-` to read stdin. Repeated training inputs are separate documents.

```sh
tokenforge train --algorithm bpe --vocab-size 300 --input corpus.txt --output model.json
printf 'Tea!' | tokenforge encode --model model.json
printf '[89,106,102,38]' | tokenforge decode --model model.json
tokenforge inspect --model model.json
tokenforge inspect < model.json
```

`train` and `inspect` return the summary fields `version`, `algorithm`, `normalization`, `vocab_size` (realized), `base_size`, `merge_count`, `training`, `trace`, and `trace_truncated`. `encode` emits the encoding object and `decode` returns an object with a `text` string.
