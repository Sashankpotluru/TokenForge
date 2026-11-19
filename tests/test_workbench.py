import json
import os
import subprocess
import sys
import zipfile

import pytest

from scripts.build_site import PUBLIC, ROOT, WEB, build, bundle
from web import bridge


def test_benchmark_rejects_installed_core_outside_measured_checkout(tmp_path):
    # A different checkout or old wheel can resolve before the editable package.
    package = tmp_path / "tokenforge"
    package.mkdir()
    (package / "__init__.py").write_text("")
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.measure"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "python -m pip install -e ." in result.stderr


def test_bundle_imports_identical_sources_and_bridge_in_isolation(tmp_path):
    path = tmp_path / "core.zip"
    bundle(path)
    with zipfile.ZipFile(path) as archive:
        for file in (ROOT / "src/tokenforge").glob("*.py"):
            assert archive.read("tokenforge/" + file.name) == file.read_bytes()
    run = subprocess.check_output(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import tokenforge, workbench_bridge;"
            " print(tokenforge.__file__); print(workbench_bridge.__file__)",
            str(path),
        ],
        cwd=tmp_path,
        text=True,
    )
    assert all(str(path) in line for line in run.splitlines())


def test_build_inventory_and_repeatability(tmp_path):
    out = tmp_path / "site"
    build(out)

    def snapshot():
        return {
            p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()
        }

    first = snapshot()
    expected = {*WEB, *PUBLIC, "manifest.json", "core.zip", ".nojekyll"}
    model_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "models").glob("*.json")}
    assert set(first) == expected | model_files
    assert b"<!-- specimen -->" not in first["index.html"]
    build(out)
    assert first == snapshot()
    (out / "personal.txt").write_text("keep")
    with pytest.raises(ValueError, match="unrelated files"):
        build(out)
    assert (out / "personal.txt").read_text() == "keep"


def test_bridge_preserves_full_decode_for_truncated_multibyte_and_literal_data():
    key = "bpe-1024"
    bridge.register_model(key, (ROOT / "models/bpe-1024.json").read_text())
    source = '𒁃[MASK]"\n' * 100
    result = json.loads(bridge.handle(json.dumps({"kind": "encode", "key": key, "text": source})))
    assert result["decoded"] == source
    assert result["count"] > len(result["tokens"]) == 160
    assert result["tokens"][0]["start"] == 0
    assert all(not t["special"] for t in result["tokens"])
    assert result["bytes"] == len(source.encode())


def test_bridge_training_and_unknowns():
    key = "wordpiece-1024"
    bridge.register_model(key, (ROOT / "models/wordpiece-1024.json").read_text())
    result = json.loads(bridge.handle(json.dumps({"kind": "encode", "key": key, "text": "🪶"})))
    assert result["decoded"] == "[UNK]"
    assert result["unknown"] == 1
    assert result["tokens"][0]["end"] == 4
    for algorithm in ("wordpiece", "bpe"):
        request = {"kind": "train", "algorithm": algorithm, "text": "abcdefghijklmno " * 3}
        result = json.loads(bridge.handle(json.dumps(request)))
        assert len(result["trace"]) == 12
        assert all(s["count"] >= 2 for s in result["trace"])
        assert result["size"] >= result["base"]
        with pytest.raises(ValueError, match="1,000"):
            bridge.handle(json.dumps({**request, "text": "🪶" * 1001}))
