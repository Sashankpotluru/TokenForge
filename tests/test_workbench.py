import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from scripts import build_site
from scripts.build_site import PUBLIC, ROOT, WEB, build, bundle
from tokenforge import train
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


def test_bundle_imports_only_release_sources_and_bridge_in_isolation(tmp_path, monkeypatch):
    package = tmp_path / "src/tokenforge"
    shutil.copytree(ROOT / "src/tokenforge", package, ignore=shutil.ignore_patterns("__pycache__"))
    (package / ".local-notes.py").write_text("hidden scratch content")
    (package / "experiment.py").write_text("unreleased working code")
    (tmp_path / "web").mkdir()
    shutil.copyfile(ROOT / "web/bridge.py", tmp_path / "web/bridge.py")
    monkeypatch.setattr(build_site, "ROOT", tmp_path)
    path = tmp_path / "core.zip"
    bundle(path)
    expected_modules = {
        "__init__.py",
        "bpe.py",
        "cli.py",
        "model.py",
        "pretokenize.py",
        "schema.py",
        "training.py",
        "wordpiece.py",
    }
    expected_members = {"tokenforge/" + module for module in expected_modules}
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == sorted(expected_members | {"workbench_bridge.py"})
        for module in expected_modules:
            assert archive.read("tokenforge/" + module) == (package / module).read_bytes()
        assert archive.read("workbench_bridge.py") == (tmp_path / "web/bridge.py").read_bytes()
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


def test_bridge_registration_uses_requested_training_target(monkeypatch):
    monkeypatch.setattr(bridge, "MODELS", {})
    model = train(["x"], algorithm="bpe", vocab_size=1024)
    assert model.vocab_size == 261
    payload = json.dumps(model.to_dict())
    for wrong_key in ("bpe-261", "bpe-2048", "wordpiece-1024"):
        with pytest.raises(ValueError, match="differs from the selected model"):
            bridge.register_model(wrong_key, payload)
    assert not bridge.MODELS
    bridge.register_model("bpe-1024", payload)
    request = {"kind": "encode", "key": "bpe-1024", "text": "x"}
    result = json.loads(bridge.handle(json.dumps(request)))
    assert result["decoded"] == "x"
