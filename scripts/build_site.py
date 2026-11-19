"""Build the static Pages directory using only local, verified inputs."""

import hashlib
import json
import shutil
import sys
import zipfile
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from benchmarks.evidence import verify  # noqa: E402
from tokenforge import Tokenizer  # noqa: E402

WEB = ("index.html", "style.css", "app.mjs", "controller.mjs", "worker.mjs", "runtime.mjs")
PUBLIC = (
    "docs/architecture.svg",
    "results/site-summary.json",
    "results/report.json",
    "results/rows.jsonl.gz",
    "results/training-rows.json",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
)


def descriptor(path, root):
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def bundle(output):
    members = [(p, f"tokenforge/{p.name}") for p in sorted((ROOT / "src/tokenforge").glob("*.py"))]
    members.append((ROOT / "web/bridge.py", "workbench_bridge.py"))
    with zipfile.ZipFile(output, "w") as archive:
        for path, name in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build(destination):
    verify()
    summary = json.loads((ROOT / "results/site-summary.json").read_bytes())
    model_paths = [entry["path"] for entry in summary["models"]]
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "public"
        stage.mkdir()
        for name in WEB:
            shutil.copyfile(ROOT / "web" / name, stage / name)
        for name in (*PUBLIC, *model_paths):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        bundle(stage / "core.zip")
        models = {
            entry["key"]: descriptor(stage / entry["path"], stage) for entry in summary["models"]
        }
        manifest = {
            "version": 1,
            "core": descriptor(stage / "core.zip", stage),
            "summary": descriptor(stage / "results/site-summary.json", stage),
            "models": models,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        model = Tokenizer.load(stage / "models/bpe-4096.json")
        specimen = "".join(
            f"<span><b>{escape(t.piece)}</b><small>{t.id}</small></span>"
            for t in model.encode("tokenization").tokens
        )
        page = stage / "index.html"
        page.write_text(page.read_text().replace("<!-- specimen -->", specimen))
        (stage / ".nojekyll").touch()
        if destination.exists():
            # Replace only our own previous build. Never delete unrelated content.
            managed = {path.relative_to(stage) for path in stage.rglob("*")}
            existing = {path.relative_to(destination) for path in destination.rglob("*")}
            if not (destination / "manifest.json").is_file() or existing - managed:
                raise ValueError(f"Move unrelated files out of {destination} before rebuilding")
            destination.rename(Path(temp) / "previous")
        try:
            stage.rename(destination)
        except OSError:
            previous = Path(temp) / "previous"
            if previous.exists():
                previous.rename(destination)
            raise


if __name__ == "__main__":
    build(ROOT / "site")
    print("Built and verified site/ for project-relative hosting.")
