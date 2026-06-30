#!/usr/bin/env python3
"""Tokenize text with HuggingFace tokenizers and dump token IDs.

Usage:
    python tools/tokenizer.py "The capital of France is" --model meta-llama/Llama-3.2-1B
    python tools/tokenizer.py --decode --ids 791 6864 315 9822 374 --model meta-llama/Llama-3.2-1B

The C++ engine reads and writes plain text files of space-delimited token IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def load_tokenizer(model: str) -> Tokenizer:
    tokenizer_path = hf_hub_download(model, "tokenizer.json")
    return Tokenizer.from_file(tokenizer_path)


def format_token_ids(ids: list[int]) -> str:
    return " ".join(str(token_id) for token_id in ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize text or decode token IDs using a HuggingFace tokenizer.",
    )
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="HuggingFace model or tokenizer name")
    parser.add_argument("--decode", action="store_true", help="Decode mode: token IDs to text")
    parser.add_argument("--ids", nargs="+", type=int, help="Token IDs to decode")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)

    if args.decode:
        if not args.ids:
            raise SystemExit("--decode requires --ids")

        text = tokenizer.decode(args.ids)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)
        return

    if not args.text:
        raise SystemExit("Provide text to tokenize")

    ids = tokenizer.encode(args.text).ids
    output = format_token_ids(ids)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
        print(f"Wrote {len(ids)} tokens to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
