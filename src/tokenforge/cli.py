"""Command-line interface. Read stdin or UTF-8 files without translating newlines."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import Tokenizer, train
from .schema import SPECIAL


def _read(path: str) -> str:
    raw = sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
    return raw.decode("utf-8")


def _summary(model: Tokenizer) -> dict[str, Any]:
    data = model.to_dict()
    return {
        "version": data["version"],
        "algorithm": model.algorithm,
        "vocab_size": model.vocab_size,
        "normalization": model.normalization,
        "base_size": data["base_size"],
        "merge_count": len(data["merges"]),
        "training": data["training"],
        "trace": data["trace"],
        "trace_truncated": data["trace_truncated"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenforge")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="train on one or more UTF-8 documents")
    fit.add_argument("--algorithm", required=True, choices=["bpe", "wordpiece"])
    fit.add_argument("--vocab-size", required=True, type=int)
    fit.add_argument("--min-frequency", type=int, default=2)
    fit.add_argument("--normalization", choices=["none", "NFC", "nfc"], default="none")
    fit.add_argument("--max-merges", type=int)
    fit.add_argument(
        "--input", action="append", help="one document per file (repeatable, default stdin)"
    )
    fit.add_argument("--output", required=True, help="destination model JSON path")
    encode = commands.add_parser("encode", help="encode a UTF-8 input")
    encode.add_argument("--allow-special", choices=SPECIAL, action="append", default=[])
    encode.add_argument("--add-bos", action="store_true")
    encode.add_argument("--add-eos", action="store_true")
    decode = commands.add_parser("decode", help="decode a JSON array of token IDs")
    decode.add_argument("--skip-special-tokens", action="store_true")
    decode.add_argument("--errors", choices=["strict", "replace"], default="strict")
    inspect = commands.add_parser("inspect", help="summarize a trained model")
    inspect.add_argument("--model", default="-", help="model JSON file, or stdin (default)")
    for sub in (encode, decode):
        sub.add_argument("--model", required=True, help="source model JSON path")
    for sub in (encode, decode):
        sub.add_argument("--input", default="-", help="UTF-8 file path or '-' for stdin")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            model = train(
                (_read(path) for path in (args.input or ["-"])),
                algorithm=args.algorithm,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                normalization=args.normalization,
                max_merges=args.max_merges,
            )
            model.save(args.output)
            output = _summary(model)
        else:
            if args.command in {"encode", "decode"} and args.model == args.input == "-":
                raise ValueError("model and input cannot both read from stdin")
            model = (
                Tokenizer._load_stream(sys.stdin.buffer)
                if args.model == "-"
                else Tokenizer.load(args.model)
            )
            if args.command == "encode":
                output = model.encode(
                    _read(args.input),
                    allowed_special=args.allow_special,
                    add_bos=args.add_bos,
                    add_eos=args.add_eos,
                ).to_dict()
            elif args.command == "decode":
                ids = json.loads(_read(args.input))
                if not isinstance(ids, list):
                    raise ValueError("decode input must be a JSON array of integer IDs")
                output = {
                    "text": model.decode(
                        ids, skip_special_tokens=args.skip_special_tokens, errors=args.errors
                    )
                }
            else:
                output = _summary(model)
        print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        print(f"tokenforge: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
