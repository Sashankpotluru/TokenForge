import json
import shutil

import pytest

from benchmarks.data import ROOT, check_file, file_info, split_rows
from benchmarks.evidence import make_summary, verify, write_json
from benchmarks.metrics import aggregate, timing_rates


def test_duplicate_filter_precedes_whole_row_byte_prefix():
    selected, test, meta = split_rows(
        ["", "dupe", "é", "a", "long", "x", "dupe"], ["dupe", "", " "], 4
    )
    assert [r.id for r in selected] == [2, 3]
    assert [r.id for r in test] == [0, 2]
    assert meta["selected_bytes"] == 3
    assert [r["id"] for r in meta["excluded_exact_matches"]] == [1, 6]
    assert meta["excluded_exact_matches"][0]["sha256"] == test[0].identity()["sha256"]


def test_source_hash_checked_even_on_same_size(tmp_path):
    p = tmp_path / "raw"
    p.write_bytes(b"original")
    spec = file_info(p, tmp_path)
    p.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        check_file(p, spec)


def test_metrics_weight_counts_and_exclude_unknown_bytes_and_whitespace():
    def row(size, scalars, words, content, tokens, unknown, lost, exact):
        return {
            "bytes": size,
            "scalars": scalars,
            "words": words,
            "word_bytes": content,
            "models": {
                "wp": {
                    "tokens": tokens,
                    "unknown_words": unknown,
                    "unknown_bytes": lost,
                    "canonical": True,
                    "exact": exact,
                    "hf_ids_match": True,
                }
            },
        }

    rows = [row(7, 4, 1, 6, 1, 1, 6, False), row(10, 10, 3, 8, 4, 0, 0, True)]
    result = aggregate(rows, "wp", "wordpiece")
    assert result["bytes_per_token"] == 17 / 5
    assert result["known_bytes_per_known_token"] == 8 / 4
    assert result["unknown_token_rate"] == 1 / 5
    assert result["word_coverage"] == 3 / 4
    assert result["word_byte_coverage"] == 8 / 14
    assert result["scalars_per_token"] == 14 / 5
    assert (result["exact_roundtrips"], result["canonical_matches"]) == (1, 2)
    assert aggregate(rows[:1], "wp", "wordpiece")["known_bytes_per_known_token"] is None


def test_timing_rates_use_median_nanoseconds_and_decimal_bytes():
    assert timing_rates(1000, 400, [1, 2_000_000, 4_000_000, 1_000_000, 10_000_000]) == {
        "median_bytes_per_second": 500_000,
        "median_tokens_per_second": 200_000,
        "repetitions": 5,
    }


@pytest.mark.skipif(not (ROOT / "results/report.json").is_file(), reason="measured bundle absent")
def test_published_evidence_and_model_report_corruption(tmp_path):
    verify()
    for directory in ("results", "models"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    model = tmp_path / "models/bpe-1024.json"
    model.write_bytes(model.read_bytes().replace(b"261", b"262", 1))
    with pytest.raises(ValueError, match="mismatch: models/bpe-1024.json"):
        verify(tmp_path)
    report = tmp_path / "results/report.json"
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(ValueError, match="mismatch: results/report.json"):
        verify(tmp_path)


@pytest.mark.skipif(not (ROOT / "results/report.json").is_file(), reason="measured bundle absent")
def test_independent_aggregate_check_survives_rehashed_report(tmp_path):
    for directory in ("results", "models"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    path = tmp_path / "results/report.json"
    summary_path = tmp_path / "results/site-summary.json"
    report = json.loads(path.read_bytes())
    report["models"][0]["datasets"]["wikitext"]["tokens"] += 1
    path.write_text(json.dumps(report))
    summary = json.loads(summary_path.read_bytes())
    summary["report"] = file_info(path, tmp_path)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="aggregate mismatch"):
        verify(tmp_path)


@pytest.fixture
def mutable_bundle(tmp_path):
    for directory in ("results", "models"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    return tmp_path, json.loads((tmp_path / "results/report.json").read_bytes())


def rebuild_summary(root, report):
    path = root / "results/report.json"
    write_json(path, report)
    write_json(root / "results/site-summary.json", make_summary(report, file_info(path, root)))


def test_reject_wrong_model_path_after_rebuilding_summary(mutable_bundle):
    root, report = mutable_bundle
    report["models"][0]["path"] = "models/wordpiece-1024.json"
    rebuild_summary(root, report)
    with pytest.raises(ValueError, match="descriptor mismatch: models/bpe-1024.json"):
        verify(root)


@pytest.mark.parametrize(
    ("reference", "domain", "runtime", "extra"),
    [
        (False, "wikitext", "huggingface", False),
        (False, "unicode", "tokenforge", False),
        (True, "cpython", "tiktoken", False),
        (True, "wikitext", "huggingface", True),
    ],
)
def test_reject_incomplete_or_unexpected_runtime_inventory(
    mutable_bundle, reference, domain, runtime, extra
):
    root, report = mutable_bundle
    model = report["reference"] if reference else report["models"][0]
    timings = model["timing"][domain]
    if extra:
        timings[runtime] = timings["tiktoken"]
    else:
        del timings[runtime]
    rebuild_summary(root, report)
    with pytest.raises(ValueError, match=f"{model['key']}/{domain}: unexpected timing runtimes"):
        verify(root)
