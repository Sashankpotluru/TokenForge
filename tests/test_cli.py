import json
import subprocess
import sys


def cli(*args, stdin=""):
    return subprocess.run(
        [sys.executable, "-m", "tokenforge.cli", *map(str, args)],
        input=stdin,
        text=True,
        capture_output=True,
    )


def test_cli_train_encode_decode_inspect_and_bad_ids(tmp_path):
    model = tmp_path / "model.json"
    trained = cli(
        "train", "--algorithm", "bpe", "--vocab-size", "269", "--output", model, stdin="hé hé"
    )
    assert trained.returncode == 0, trained.stderr
    assert json.loads(trained.stdout)["algorithm"] == "bpe"
    enc = json.loads(cli("encode", "--model", model, stdin="hé\n").stdout)
    assert (
        json.loads(cli("decode", "--model", model, stdin=json.dumps(enc["ids"])).stdout)["text"]
        == "hé\n"
    )
    assert json.loads(cli("inspect", "--model", model).stdout)["vocab_size"] >= 261
    bad = cli("decode", "--model", model, stdin="[false]")
    assert bad.returncode != 0 and bad.stderr.strip() and "Traceback" not in bad.stderr
    path = tmp_path / "text.txt"
    path.write_bytes(b"a\r\nb")
    assert (
        json.loads(cli("encode", "--input", path, "--model", model).stdout)["normalized_text"]
        == "a\r\nb"
    )


def test_inspect_model_on_stdin(tmp_path):
    model = tmp_path / "wordpiece.json"
    run = cli("train", "--algorithm", "wordpiece", "--vocab-size", "5", "--output", model)
    assert run.returncode == 0, run.stderr
    run = cli("inspect", stdin=model.read_text())
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["base_size"] == 5
