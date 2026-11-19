"""Small JSON interface to the shared Python core in the browser worker."""

import json
from dataclasses import asdict

from tokenforge import Tokenizer, train

MODELS = {}
TOKEN_LIMIT = 160
MERGE_LIMIT = 12


def register_model(key, payload):
    data = json.loads(payload)
    model = Tokenizer.from_dict(data)
    if key != f"{model.algorithm}-{data['training']['vocab_size']}":
        raise ValueError("Loaded model differs from the selected model")
    MODELS[key] = model


def checked_text(request, limit):
    text = request["text"]
    if not isinstance(text, str) or len(text) > limit:
        raise ValueError(f"Use at most {limit:,} Unicode scalars")
    text.encode("utf-8", errors="strict")
    return text


def handle(payload):
    request = json.loads(payload)
    if request["kind"] == "encode":
        text = checked_text(request, 12_000)
        model = MODELS[request["key"]]
        encoding = model.encode(text)
        return json.dumps(
            {
                "tokens": [asdict(token) for token in encoding.tokens[:TOKEN_LIMIT]],
                "count": len(encoding.tokens),
                "unknown": sum(token.unknown for token in encoding.tokens),
                "bytes": len(text.encode()),
                "decoded": model.decode(encoding.ids),
            }
        )
    if request["kind"] != "train":
        raise ValueError("Unknown workbench operation")
    text = checked_text(request, 1_000)
    algorithm = request["algorithm"]
    data = train(
        [text], algorithm=algorithm, vocab_size=2_048, max_merges=MERGE_LIMIT, min_frequency=2
    ).to_dict()

    def display(index):
        piece = data["vocab"][index]
        if algorithm == "bpe":
            return bytes.fromhex(piece).decode("utf-8", errors="backslashreplace")
        return piece

    return json.dumps(
        {
            "size": len(data["vocab"]),
            "base": data["base_size"],
            "trace": [
                {
                    **step,
                    "pair": [display(step["left"]), display(step["right"])],
                    "piece": display(step["result"]),
                }
                for step in data["trace"]
            ],
        }
    )
