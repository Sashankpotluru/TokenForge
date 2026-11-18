"""Pinned source intake and row selection. Input text never enters public observations."""

import hashlib
import json
import ssl
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
SOURCES = ROOT / "benchmarks/sources.json"
BUDGET = 250_000
TARGETS = (1024, 2048, 4096)
ALGORITHMS = ("bpe", "wordpiece")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_info(path, root=ROOT):
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "bytes": len(data), "sha256": sha(data)}


def check_file(path, spec):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {spec['path']}. Run python -m benchmarks.download")
    data = path.read_bytes()
    if len(data) != spec["bytes"] or sha(data) != spec["sha256"]:
        raise ValueError(f"Size or SHA-256 mismatch: {spec['path']}")


def fetch_sources(cache=CACHE, *, offline=False):
    manifest = json.loads(SOURCES.read_text())
    if not offline:
        import truststore

        tls = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        request = Request(manifest["cpython_commit_api"], headers={"User-Agent": "TokenForge"})
        with urlopen(request, timeout=60, context=tls) as response:
            if json.load(response)["sha"] != manifest["cpython_revision"]:
                raise ValueError("CPython tag no longer resolves to pinned commit")
    for spec in manifest["files"]:
        dest = cache / spec["path"]
        if not dest.exists() and not offline:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                staged = Path(tmp.name)
            try:
                with urlopen(spec["url"], timeout=90, context=tls) as resp:
                    staged.write_bytes(resp.read(spec["bytes"] + 1))
                check_file(staged, spec)
                staged.replace(dest)
            finally:
                staged.unlink(missing_ok=True)
        check_file(dest, spec)
    return manifest


@dataclass(frozen=True)
class Row:
    id: int | str
    text: str

    def identity(self):
        raw = self.text.encode()
        return {"id": self.id, "sha256": sha(raw), "bytes": len(raw), "scalars": len(self.text)}


def split_rows(train, test, budget=BUDGET):
    """Drop exact test matches, then take a prefix ending before the first overflow."""
    test_rows = [Row(i, text) for i, text in enumerate(test) if text != ""]
    test_texts = {row.text for row in test_rows}
    selected, excluded = [], []
    used = 0
    stopped = False
    for i, text in enumerate(train):
        if text == "":
            continue
        row = Row(i, text)
        if text in test_texts:
            excluded.append(row.identity())
        elif not stopped:
            size = len(text.encode())
            if used + size > budget:
                stopped = True
            else:
                selected.append(row)
                used += size
    return (
        selected,
        test_rows,
        {
            "budget_bytes": budget,
            "selected_bytes": used,
            "selected": [r.identity() for r in selected],
            "excluded_exact_matches": excluded,
            "train_source_rows": len(train),
            "test_source_rows": len(test),
        },
    )


def read_data(manifest, cache=CACHE):
    import pyarrow.parquet as pq

    def parquet(split):
        path = cache / f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
        return pq.read_table(path, columns=["text"], use_threads=False)["text"].to_pylist()

    train, test, split = split_rows(parquet("train"), parquet("test"))
    code = [
        Row(f["path"], (cache / f["path"]).read_bytes().decode())
        for f in manifest["files"]
        if f["path"].startswith("Lib/")
    ]
    fixtures = json.loads((ROOT / "tests/fixtures/unicode.json").read_text())
    return (
        train,
        {"wikitext": test, "cpython": code, "unicode": [Row(i, s) for i, s in enumerate(fixtures)]},
        split,
    )
