"""Train six models and measure the pinned corpora. Requires a previous download."""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"

import platform
import shutil
import subprocess
import sys
import unicodedata
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import tokenforge

from .data import ALGORITHMS, CACHE, ROOT, SOURCES, TARGETS, fetch_sources, file_info, read_data

SUBSET_ROWS = {"wikitext": 32, "cpython": 1, "unicode": 8}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_checkout_core():
    expected = (ROOT / "src/tokenforge/__init__.py").resolve()
    if Path(tokenforge.__file__).resolve() != expected:
        raise ValueError(
            "Benchmark must import this checkout's TokenForge. From the repository root, "
            "run python -m pip install -e . with the same interpreter and retry."
        )


def environment():
    require_checkout_core()
    paths = ["src", "benchmarks", "tests/fixtures", "pyproject.toml", "requirements-benchmark.txt"]
    if git("status", "--porcelain", "--", *paths):
        raise ValueError("Commit the benchmark, core, inputs and dependencies before measuring")
    packages = ("pyarrow", "numpy", "tokenizers", "tiktoken")
    versions = {name: version(name) for name in packages}
    locked = (ROOT / "requirements-benchmark.txt").read_text().splitlines()
    for name, value in versions.items():
        if f"{name}=={value}" not in locked:
            raise ValueError(f"Benchmark dependency differs from the lock: {name}")
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "worktree_dirty_at_start": bool(git("status", "--porcelain")),
        "measurement_sources_dirty": False,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "unicode_version": unicodedata.unidata_version,
        "os": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "machine": platform.machine(),
        "cpu": cpu,
        "packages": versions,
        "tokenizers_parallelism": False,
        "rayon_num_threads": 1,
        "benchmark_lock": file_info(ROOT / "requirements-benchmark.txt"),
    }


def timed(encode, subset, expected_counts):
    def once():
        return [len(encode(row.text)) for row in subset]

    if once() != expected_counts:  # one full warm-up, excluded
        raise ValueError("Warm-up token counts differ from full evaluation")
    runs = []
    for _ in range(5):
        start = perf_counter_ns()
        counts = once()
        elapsed = perf_counter_ns() - start
        if counts != expected_counts:
            raise ValueError("Timed token counts changed")
        runs.append(elapsed)
    return {
        "runs_ns": runs,
        "tokens": sum(expected_counts),
        "bytes": sum(len(row.text.encode()) for row in subset),
    }


def special_parity(model, hf, fixtures):
    from tokenforge.schema import SPECIAL

    matched = 0
    for row in fixtures:
        # Include every literal even if the base fixture itself has none.
        text = "".join(SPECIAL) + row.text + "".join(reversed(SPECIAL))
        for allowed in [(), *[(s,) for s in SPECIAL], SPECIAL]:
            opts = {"allowed_special": allowed, "add_bos": True, "add_eos": True}
            if model.encode(text, **opts).ids != hf.encode(text, **opts):
                raise ValueError(f"Special ID mismatch on fixture {row.id}")
            matched += 1
    return matched


def run(stage, provenance, manifest):
    from tokenforge import train

    from .adapters import HFAdapter, cl100k
    from .evidence import (
        DOMAINS,
        make_summary,
        training_info,
        verify,
        write_json,
        write_observations,
    )
    from .metrics import aggregate, observe_model, word_spans

    models, results = stage / "models", stage / "results"
    models.mkdir()
    results.mkdir()
    selected, domains, split = read_data(manifest)
    write_json(results / "training-rows.json", split)
    observations = {}
    spans = {}
    for domain, rows in domains.items():
        spans[domain] = [word_spans(row.text) for row in rows]
        observations[domain] = [
            {
                **row.identity(),
                "domain": domain,
                "words": len(words),
                "word_bytes": sum(end - start for _, start, end in words),
                "models": {},
            }
            for row, words in zip(rows, spans[domain], strict=True)
        ]
    report = {
        "version": 1,
        "provenance": provenance,
        "sources_manifest": file_info(SOURCES),
        "sources": manifest,
        "unicode_fixtures": file_info(ROOT / "tests/fixtures/unicode.json"),
        "training": training_info(split),
        "training_rows": file_info(results / "training-rows.json", stage),
        "timing_subset": {d: [r.id for r in rows[: SUBSET_ROWS[d]]] for d, rows in domains.items()},
        "models": [],
    }

    def evaluate(entry, observe, encoders):
        key = entry["key"]
        entry.update(datasets={}, timing={})
        for domain, rows in domains.items():
            print(f"Evaluating {key} / {domain} ({len(rows)} rows)", flush=True)
            records = observations[domain]
            for row, words, record in zip(rows, spans[domain], records, strict=True):
                try:
                    record["models"][key] = observe(row.text, words)
                except ValueError as exc:
                    raise ValueError(f"{key}/{domain}/{row.id}: {exc}") from exc
            entry["datasets"][domain] = aggregate(records, key, entry["algorithm"])
            n = SUBSET_ROWS[domain]
            counts = [r["models"][key]["tokens"] for r in records[:n]]
            entry["timing"][domain] = {
                runtime: timed(fn, rows[:n], counts) for runtime, fn in encoders.items()
            }

    for algorithm in ALGORITHMS:
        for target in TARGETS:
            key = f"{algorithm}-{target}"
            print(f"Training {key} on {len(selected)} rows", flush=True)
            start = perf_counter_ns()
            model = train(
                (r.text for r in selected),
                algorithm=algorithm,
                vocab_size=target,
                min_frequency=2,
                normalization="none",
            )
            duration = perf_counter_ns() - start
            path = models / f"{key}.json"
            model.save(path)
            hf = HFAdapter(model)
            entry = {
                **file_info(path, stage),
                "key": key,
                "algorithm": algorithm,
                "target": target,
                "vocab_size": model.vocab_size,
                "training_seconds": duration / 1e9,
                "special_parity_cases": special_parity(model, hf, domains["unicode"]),
            }
            evaluate(
                entry,
                lambda text, words, m=model, h=hf: observe_model(text, words, m, h),
                {"tokenforge": lambda text, m=model: m.encode(text).ids, "huggingface": hf.encode},
            )
            report["models"].append(entry)

    reference = cl100k(manifest)

    def observe_reference(text, words):
        ids = reference.encode_ordinary(text)
        if reference.decode(ids) != text:
            raise ValueError("Reference failed exact round trip")
        return {"tokens": len(ids), "exact": True}

    entry = {
        "key": "cl100k_base",
        "algorithm": "pretrained_reference",
        "vocab_size": reference.n_vocab,
    }
    evaluate(entry, observe_reference, {"tiktoken": reference.encode_ordinary})
    report["reference"] = entry
    path = results / "rows.jsonl.gz"
    write_observations(path, (row for d in DOMAINS for row in observations[d]))
    report["observations"] = file_info(path, stage)
    write_json(results / "report.json", report)
    write_json(
        results / "site-summary.json",
        make_summary(report, file_info(results / "report.json", stage)),
    )
    verify(stage)


def publish(stage, root):
    """Replace staged files, rolling back completed replacements on publication errors."""
    backups = root / ".tokenforge-publish"
    try:
        backups.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Publication in progress or recovery needed at {backups}") from exc

    rollback = ExitStack()
    try:
        for directory in ("models", "results"):
            destination = root / directory
            if not destination.exists():
                destination.mkdir()
                rollback.callback(destination.rmdir)
            (backups / directory).mkdir()
            for artifact in sorted((stage / directory).iterdir()):
                target = destination / artifact.name
                backup = backups / directory / artifact.name
                existed = target.exists(follow_symlinks=False)
                if existed:
                    shutil.copy2(target, backup, follow_symlinks=False)
                artifact.replace(target)
                if existed:
                    rollback.callback(backup.replace, target)
                else:
                    rollback.callback(target.unlink)
    except Exception:
        try:
            rollback.close()
        except OSError as exc:
            message = f"Rollback incomplete. Recover remaining backups from {backups}"
            raise RuntimeError(message) from exc
        shutil.rmtree(backups)
        raise
    shutil.rmtree(backups)


def main():
    provenance = environment()
    manifest = fetch_sources(offline=True)
    with TemporaryDirectory(dir=CACHE) as temp:
        stage = Path(temp)
        run(stage, provenance, manifest)
        publish(stage, ROOT)
    print("Verified and wrote models/ and results/.", flush=True)


if __name__ == "__main__":
    main()
