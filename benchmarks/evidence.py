"""Serialize, summarize and verify published evidence without loading the datasets."""

import gzip
import json

from tokenforge import Tokenizer

from .data import ALGORITHMS, ROOT, SOURCES, TARGETS, check_file, file_info
from .metrics import aggregate, timing_rates

REAL = ("wikitext", "cpython")
DOMAINS = (*REAL, "unicode")


def json_bytes(value, *, pretty=False):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode()


def write_json(path, value):
    path.write_bytes(json_bytes(value, pretty=True))


def write_observations(path, observations):
    with (
        path.open("wb") as output,
        gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed,
    ):
        for row in observations:
            compressed.write(json_bytes(row))


def make_summary(report, report_info):
    def compact(model):
        return {
            **{k: v for k, v in model.items() if k not in ("datasets", "timing")},
            "datasets": {domain: model["datasets"][domain] for domain in REAL},
            "timing": {
                domain: {
                    runtime: timing_rates(t["bytes"], t["tokens"], t["runs_ns"])
                    for runtime, t in model["timing"][domain].items()
                }
                for domain in REAL
            },
        }

    return {
        "version": 1,
        "report": report_info,
        "models": [compact(m) for m in report["models"]],
        "reference": compact(report["reference"]),
        "datasets": {
            d: {k: report["reference"]["datasets"][d][k] for k in ("rows", "bytes", "scalars")}
            for d in REAL
        },
        "training_bytes": report["training"]["selected_bytes"],
        "limits": [
            "Training uses a WikiText prefix within 250,000 bytes after exact test-row exclusion.",
            "WordPiece discards whitespace and replaces entire unknown pretokens.",
            "cl100k_base is separately pretrained with a larger vocabulary.",
            "Runtime timings include each interface's text processing and ID extraction.",
        ],
    }


def verify_model(root, entry, algorithm, target):
    key = entry["key"]
    path = root / "models" / f"{key}.json"
    descriptor = file_info(path, root)
    if descriptor != {field: entry.get(field) for field in descriptor}:
        raise ValueError(f"Model descriptor mismatch: {descriptor['path']}")
    if entry["algorithm"] != algorithm:
        raise ValueError(f"{key}: report algorithm mismatch")
    if type(entry["target"]) is not int or entry["target"] != target:
        raise ValueError(f"{key}: report target mismatch")
    model = Tokenizer.load(path)
    if model.algorithm != algorithm:
        raise ValueError(f"{key}: model algorithm mismatch")
    if model.normalization != "none":
        raise ValueError(f"{key}: normalization mismatch")
    training = {"vocab_size": target, "min_frequency": 2, "max_merges": None}
    if model.to_dict()["training"] != training:
        raise ValueError(f"{key}: training parameters mismatch")
    if type(entry["vocab_size"]) is not int or entry["vocab_size"] != model.vocab_size:
        raise ValueError(f"{key}: vocabulary size mismatch")


def verify(root=ROOT):
    report_path = root / "results/report.json"
    summary = json.loads((root / "results/site-summary.json").read_bytes())
    check_file(report_path, summary["report"])
    report = json.loads(report_path.read_bytes())
    expected_models = {f"{a}-{v}": (a, v) for a in ALGORITHMS for v in TARGETS}
    observed_keys = [m["key"] for m in report["models"]]
    if observed_keys != list(expected_models) or report["reference"]["key"] != "cl100k_base":
        raise ValueError("Unexpected model inventory")
    check_file(SOURCES, report["sources_manifest"])
    if report["sources"] != json.loads(SOURCES.read_bytes()):
        raise ValueError("Source inventory mismatch")
    check_file(ROOT / "tests/fixtures/unicode.json", report["unicode_fixtures"])
    check_file(ROOT / "requirements-benchmark.txt", report["provenance"]["benchmark_lock"])
    for name, suffix in (
        ("training_rows", "training-rows.json"),
        ("observations", "rows.jsonl.gz"),
    ):
        check_file(root / "results" / suffix, report[name])
    split = json.loads((root / "results/training-rows.json").read_bytes())
    if sum(row["bytes"] for row in split["selected"]) != split["selected_bytes"]:
        raise ValueError("Training byte count mismatch")
    if report["training"] != training_info(split):
        raise ValueError("Training subset summary mismatch")
    rows = {domain: [] for domain in DOMAINS}
    with gzip.open(root / "results/rows.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["domain"]].append(row)
    for model in [*report["models"], report["reference"]]:
        key = model["key"]
        expected_runtimes = {"tiktoken"} if key == "cl100k_base" else {"tokenforge", "huggingface"}
        if key in expected_models:
            verify_model(root, model, *expected_models[key])
        for domain in DOMAINS:
            domain_rows = rows[domain]
            totals = aggregate(domain_rows, key, model["algorithm"])
            if totals != model["datasets"][domain]:
                raise ValueError(f"{key}/{domain}: aggregate mismatch")
            timed = report["timing_subset"][domain]
            by_id = {r["id"]: r for r in domain_rows}
            subset = [by_id[i] for i in timed]
            subset_bytes = sum(r["bytes"] for r in subset)
            subset_tokens = sum(r["models"][key]["tokens"] for r in subset)
            timings = model["timing"][domain]
            if set(timings) != expected_runtimes:
                raise ValueError(f"{key}/{domain}: unexpected timing runtimes")
            for t in timings.values():
                if (
                    t["tokens"] != subset_tokens
                    or t["bytes"] != subset_bytes
                    or len(t["runs_ns"]) != 5
                    or min(t["runs_ns"]) <= 0
                ):
                    raise ValueError(f"{key}/{domain}: invalid timing evidence")
    if summary != make_summary(report, file_info(report_path, root)):
        raise ValueError("Site summary disagrees with report")
    return report


def training_info(split):
    return {
        **{
            k: split[k]
            for k in ("budget_bytes", "selected_bytes", "train_source_rows", "test_source_rows")
        },
        "selected_rows": len(split["selected"]),
        "excluded_rows": len(split["excluded_exact_matches"]),
    }


if __name__ == "__main__":
    verified = verify()
    print(f"Verified {len(verified['models'])} models, report hashes and row aggregates.")
