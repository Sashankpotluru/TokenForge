# Evaluation

Each tokenizer trained independently on 249,386 UTF-8 bytes across 540 WikiText rows. At 4,096 vocabulary entries, byte BPE produced 618,418 tokens on the WikiText test set and reconstructed every input. WordPiece produced 720,989 tokens, including 382 whole-pretoken unknowns. On four CPython files its unknown rate rose from 0.15% to 11.08%. [Measurements and machine metadata](../results/report.json) · [Per-row evidence](../results/rows.jsonl.gz)

| Model | Vocabulary | WikiText tokens | Bytes/token | WikiText pretoken coverage | CPython tokens | Bytes/token | CPython pretoken coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Byte BPE | 1,024 | 731,824 | 1.760 | All input bytes | 110,260 | 1.398 | All input bytes |
| Byte BPE | 2,048 | 666,083 | 1.933 | All input bytes | 104,933 | 1.469 | All input bytes |
| Byte BPE | 4,096 | 618,418 | 2.082 | All input bytes | 100,493 | 1.534 | All input bytes |
| WordPiece | 1,024 | 922,084 | 1.396 | 99.847% | 102,023 | 1.511 | 88.923% |
| WordPiece | 2,048 | 852,585 | 1.510 | 99.847% | 96,695 | 1.594 | 88.923% |
| WordPiece | 4,096 | 720,989 | 1.786 | 99.847% | 86,657 | 1.778 | 88.923% |
| Pretrained cl100k_base | 100,277¹ | 289,165 | 4.453 | All input bytes | 37,931 | 4.063 | All input bytes |

¹ tiktoken reports a vocabulary ID space of 100,277, which includes reserved IDs and gaps. cl100k_base has its own training corpus, vocabulary, regex boundaries and special-token scheme. These counts give a deployment reference. They cannot isolate a training algorithm’s effect.

The WordPiece columns count raw input bytes per emitted token, including discarded whitespace and text replaced by `[UNK]`. At 4,096 entries, counting **only represented, non-whitespace bytes per known token** gives 1.441 on WikiText and 1.267 on CPython. All three WordPiece vocabularies shared the same base alphabet and had the same whole-pretoken coverage in this run. Each matched canonical decoding for 2,891 WikiText rows and four code files. Zero decoded strings in either real domain matched their original input exactly. [Full count and coverage details](../results/report.json)

## Inputs and split

| Domain | Unit | Included examples | UTF-8 bytes |
| --- | --- | ---: | ---: |
| WikiText-2 raw v1, official test split | Nonempty Parquet text row | 2,891 | 1,287,656 |
| CPython v3.13.0 | Whole file | 4 | 154,110 |
| Authored Unicode stress cases | String, including the empty string | 8 | 70 |

WikiText uses Salesforce's revision `b08601e04326c79dfdd32d625aee71d232d685c3`. We remove only empty strings, preserving every other byte, including whitespace-only rows and line endings. From 36,718 official training rows we exclude 1,513 nonempty rows whose complete text equals a nonempty official test row, counted over the entire source train split. We take remaining rows in file order and stop before the first row that would exceed 250,000 bytes. We keep duplicate rows within a split. Matching operates at the whole-row level and can miss partial overlap. All 2,891 nonempty rows of the 4,358-row official test split enter the held-out counts. [Selected and excluded row IDs, hashes and byte lengths](../results/training-rows.json)

We independently train BPE and WordPiece at targets of 1,024, 2,048 and 4,096 IDs, including the five reserved IDs. Each fit uses minimum pair frequency two, no normalization and no explicit merge cap. All six attained their targets. The targets, budget and data selection were fixed before measuring held-out performance. Source revision, Unicode database version and tokenizer settings affect reproducibility. [Tokenizer and training contract](tokenizer-contract.md)

CPython's official GitHub tag endpoint resolves `v3.13.0` to `60403a5409ff2c3f3b07dd2ca91a7a3e096839c7`. The four inputs are `Lib/functools.py`, `Lib/collections/__init__.py`, `Lib/pathlib/__init__.py` and `Lib/statistics.py`. Each file is decoded directly from its pinned UTF-8 bytes. These files serve as a second domain for tokenizers trained only on WikiText. The stress cases exercise combining marks, emoji, right-to-left text, non-ASCII whitespace, control characters and literal marker spelling. Their counts stay separate from the two real domains. [Source URLs and SHA-256 digests](../benchmarks/sources.json) · [Stress strings](../tests/fixtures/unicode.json)

## Metric definitions

- **Bytes/token and scalars/token** divide the corresponding corpus totals by total emitted tokens. Unicode scalars count each code point individually, including combining marks. Row-level ratios receive no separate weight.
- **WordPiece coverage** counts pretokens, including punctuation. An unsegmentable pretoken emits one `[UNK]` spanning that entire input. Unknown-pretoken rate divides those fallbacks by all pretokens. Unknown-token rate divides the same count by emitted tokens. Byte coverage divides representable pretoken bytes by all pretoken bytes, excluding discarded whitespace. The adjusted bytes/token metric divides representable pretoken bytes by known emitted tokens. A zero denominator yields JSON null.
- **Reconstruction** compares original UTF-8 text with decoded text. A second WordPiece check independently joins input pretokens with one ASCII space and substitutes a placeholder at each verified, whole-pretoken unknown span. The two counts appear as `exact_roundtrips` and `canonical_matches`.
- **Same-model parity** requires identical TokenForge and Hugging Face ID sequences on each complete input. Each model also has 56 fixture checks combining literal markers, explicit per-marker allowlists and BOS/EOS insertion. The run raises on any ID disagreement, invalid unknown span or failed required reconstruction. Raw measurements record matched rows and the expected row count.

## Runtime protocol

Both controlled runtimes consume the same trained vocabulary, IDs and ranked merges. Hugging Face Tokenizers uses its native BPE or WordPiece model. For BPE, we translate model bytes to the HF ByteLevel alphabet, keep reserved IDs outside that alphabet, disable the model cache, and turn off ByteLevel's space prefix and regex splitting. For WordPiece the maximum pretoken length is 2,147,483,647 characters. Python pretokenization applies the same documented boundaries to both models. Unicode punctuation classification follows the recorded Python Unicode database. [Adapter](../benchmarks/adapters.py)

Timings start with each original input string. TokenForge builds its diagnostic records, byte offsets and ID list. The HF adapter normalizes, splits, constructs its pretokenized input, calls the native implementation and extracts the IDs. The timed loop counts the returned list lengths. These end-to-end interfaces do different amounts of record construction. The measured throughput reflects implementation and interface overhead on this machine. The tiktoken reference uses its own `encode_ordinary` method with marker strings treated as plain text.

We time the first 32 retained WikiText test rows (12,999 bytes), the first CPython file (`functools.py`, 39,558 bytes), and separately all eight stress strings. The full-corpus counts above include every held-out input. Each runtime receives one unmeasured pass over its subset, then five serial measured passes with `perf_counter_ns()`. We preload models, disable HF parallelism, set the Rayon thread count to one, and exclude downloading, file I/O, training, decoding and parity checks from the timing interval. Each pass must produce the previously observed per-row token counts. Reported throughput divides bytes or tokens per pass by the median wall-clock seconds across the five passes. Individual times, subsets, CPU, package versions and code revision are retained. Small subsets, fixed execution order and a single machine limit generalization. [Five-run timings](../results/report.json) · [Median rates used by the workbench](../results/site-summary.json)

On the recorded Apple M4 Pro with CPython 3.13.2, the 4,096-entry BPE model processed 758,927 WikiText bytes/s through TokenForge and 1,294,512 bytes/s through the HF adapter. On the code timing subset the corresponding rates were 807,185 and 1,905,159 bytes/s. Both interfaces returned identical IDs on every held-out input. [Measured median rates](../results/site-summary.json)

## Reproduce

From the repository root, use Python 3.13.2 to match the recorded runtime and Unicode database.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-benchmark.txt
python -m pip install -e .
python -m benchmarks.download
python -m benchmarks.measure
python -m benchmarks.evidence
```

The downloader checks the release tag online, fetches missing files into ignored `data/cache/`, and validates SHA-256 and byte length for every file, including existing cache entries. A damaged entry produces an error naming the file. Delete that single cached file and rerun the downloader to replace it. With a populated cache, `python -m benchmarks.download --offline` verifies all files without contacting any host. Measurement uses the verified cache. The cl100k_base rank file must hash to `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`. [Source manifest](../benchmarks/sources.json)

`sh scripts/reproduce.sh` runs download, measurement and verification in sequence using the active environment. Add `--offline` for cached inputs. The measure command checks that the imported TokenForge package comes from this checkout’s `src/tokenforge/` directory. If another installation shadows it, run `python -m pip install -e .` from the repository root in the active environment. Measurement requires committed measurement code, trains six models independently, verifies the staged evidence and then replaces the generated files in `models/` and `results/`. Run it in a fresh checkout to preserve a previous run. Model JSON, row identities and counts are intended to reproduce under the recorded runtime. Training times, throughput, environment and source commit can change, as can the report and summary hashes that bind those fields.

To audit the release without downloading data or installing the benchmark dependencies, install the core with `python -m pip install -e .` and run `python -m benchmarks.evidence`. It checks model/report hashes, validates model JSON, recomputes counts and rates from row observations, and compares the site summary with the report. Re-encoding the original inputs requires the measurement environment and source files.

| Artifact | Contents |
| --- | --- |
| [Source manifest](../benchmarks/sources.json) | Exact URLs, source revisions, bytes and SHA-256 for all eight required downloads |
| [Training selection](../results/training-rows.json) | Selected and excluded zero-based source row indices, byte/scalar lengths and row hashes |
| [Compact observations](../results/rows.jsonl.gz) | One gzipped JSON line per evaluated input with domain, row ID, hash, counts, unknown loss and reconstruction/parity outcomes |
| [Measured report](../results/report.json) | Model files, actual sizes and hashes, input and evidence hashes, hardware/software, aggregate counts and raw repetition times |
| [Workbench summary](../results/site-summary.json) | Real-domain counts, coverage and median throughput with model/report digests |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Source attribution and separate terms for the learned model artifacts |
