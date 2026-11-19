# Third-party notices

## WikiText and the six trained models

The six JSON models under `models/` derive their vocabularies, merge rules and training traces from WikiText-2 raw English Wikipedia text. Wikipedia contributors authored the articles. Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher introduced the WikiText benchmark. TokenForge selected a bounded subset, removed exact test-row matches and trained new byte BPE and WordPiece vocabularies. The source row indices and hashes identify the training inputs. [Dataset release](https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset/) · [Original paper](https://arxiv.org/abs/1609.07843) · [Training row manifest](results/training-rows.json)

The pinned dataset card's metadata lists CC BY-SA 3.0 and GFDL. Its licensing section names CC BY-SA 4.0. The trained model artifacts are distributed under CC BY-SA 4.0 with attribution to the authors above and preservation of the upstream notices, including the GFDL notice for Wikipedia material. Retain this notice with redistributed model files and identify further modifications. The repository MIT license covers original software. [Pinned dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html)

## CPython source samples

The benchmark reads four original standard-library files from CPython v3.13.0, copyright the Python Software Foundation and other contributors. The downloader verifies the release's license file alongside the source. The published evaluation contains file paths, hashes and measurements. [Pinned CPython license and historical notices](https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/LICENSE) · [Source manifest](benchmarks/sources.json)

## Evaluation dependencies

| Component | Use | Upstream terms |
| --- | --- | --- |
| Hugging Face Tokenizers | Independent runtime for the same six trained models | [Apache License 2.0](https://github.com/huggingface/tokenizers/blob/main/LICENSE) |
| OpenAI tiktoken | Separate pretrained cl100k_base reference. Its ranks are downloaded into the local cache | [MIT license for the software](https://github.com/openai/tiktoken/blob/main/LICENSE) · [Encoding definitions](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py) |
| Apache Arrow / PyArrow | Read the two pinned Parquet inputs | [Apache License 2.0 and third-party notices](https://github.com/apache/arrow/blob/main/LICENSE.txt) |
| Truststore | Validate HTTPS source downloads with platform certificate authorities | [MIT license](https://github.com/sethmlarson/truststore/blob/main/LICENSE) |
| NumPy | Python array support loaded by the Hugging Face pretokenized-input binding | [NumPy license and bundled notices](https://numpy.org/doc/stable/license.html) |

The benchmark environment installs the upstream distributions with their own notices. The six model JSON files contain TokenForge's training output. The raw WikiText files, CPython files and cl100k_base ranks are obtained separately. [Dependency lock](requirements-benchmark.txt) · [Source URLs, expected byte lengths and SHA-256 digests](benchmarks/sources.json)

## Browser runtime

The workbench loads Pyodide 0.29.3 from jsDelivr. Pyodide is distributed under the Mozilla Public License 2.0 and includes CPython and other components with their own terms. [Pyodide license](https://github.com/pyodide/pyodide/blob/0.29.3/LICENSE) · [Pinned runtime distribution](https://cdn.jsdelivr.net/pyodide/v0.29.3/full/)
