#!/usr/bin/env python3
"""Greedy evals whose official generation budget fits in a 4096-token context.

Tasks, matching the Llama 3.2 1B Instruct card where the lengths fit:
  ARC-Challenge  0-shot, generate the letter, up to 100 tokens
  MMLU           5-shot from the dev split, generate the letter, up to 10 tokens
  GSM8K          8-shot chain-of-thought from the train split, up to 1024 tokens

Anything whose prompt does not leave room inside 4096 tokens is recorded and
left out of the score. MATH, GPQA, and the long-context tasks need more than
this window, so they are not in this run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

TOKENIZER_PATH = Path("/home/ameyades/models/Llama-3.2-1B-Instruct/tokenizer.json")
MODEL_PATH = Path("/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors")
ENGINE = Path("/home/ameyades/cuda-cublas-json-project/build/cuda_cublas_json")
MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 1024

LETTER = re.compile(r"[A-J]")
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def chat_user(user_text: str) -> str:
    # Official Llama 3.2 instruct header. The longer engine system prompt
    # ("Do not repeat words") makes this 1B model stop after a nonsense word.
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    # The chat string already contains <|begin_of_text|>.
    return tokenizer.encode(text, add_special_tokens=False).ids


def choice_block(question: str, choices: list[str], answer: str | None = None) -> str:
    lines = [question.strip()]
    for index, choice in enumerate(choices):
        lines.append(f"{chr(ord('A') + index)}. {choice}")
    if answer is None:
        lines.append("Answer:")
    else:
        lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def first_letter(text: str, allowed: set[str]) -> str:
    for match in LETTER.finditer(text.upper()):
        if match.group(0) in allowed:
            return match.group(0)
    return ""


def normalize_number(text: str) -> str | None:
    match = NUMBER.search(text.replace(",", ""))
    if match is None:
        return None
    number = match.group(0)
    if number.endswith("."):
        number = number[:-1]
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def predicted_number(text: str) -> str | None:
    tail = text.split("####")[-1] if "####" in text else text
    numbers = NUMBER.findall(tail.replace(",", ""))
    if not numbers:
        return None
    return normalize_number(numbers[-1])


def gold_gsm(answer: str) -> str | None:
    tail = answer.split("####")[-1]
    return normalize_number(tail)


def build_arc(tokenizer: Tokenizer, limit: int | None) -> list[dict]:
    split = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row_index, row in enumerate(split):
        if limit is not None and len(items) >= limit:
            break
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        # Some rows label choices 1..4 instead of A..D.
        letters = []
        choices = []
        for label, text in zip(labels, texts):
            letter = label.strip().upper()
            if letter.isdigit():
                letter = chr(ord("A") + int(letter) - 1)
            letters.append(letter)
            choices.append(text)
        body = choice_block(row["question"], choices) + "\nReply with only the letter."
        tokens = encode(tokenizer, chat_user(body))
        items.append(
            {
                "id": f"arc/{row_index}",
                "task": "arc",
                "tokens": tokens,
                "max_new": 100,
                "gold": row["answerKey"].strip().upper(),
                "allowed": letters,
            }
        )
    return items


def build_mmlu(tokenizer: Tokenizer, limit: int | None) -> list[dict]:
    subjects = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    items = []
    for subject in subjects:
        dev = list(load_dataset("cais/mmlu", subject, split="dev"))
        test = list(load_dataset("cais/mmlu", subject, split="test"))
        shots = []
        for shot in dev[:5]:
            shots.append(choice_block(shot["question"], list(shot["choices"]), chr(ord("A") + int(shot["answer"]))))
        header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}."
        for row_index, row in enumerate(test):
            if limit is not None and len(items) >= limit:
                return items
            question = choice_block(row["question"], list(row["choices"])) + "\nReply with only the letter."
            body = header + "\n\n" + "\n\n".join(shots + [question])
            tokens = encode(tokenizer, chat_user(body))
            items.append(
                {
                    "id": f"mmlu/{subject}/{row_index}",
                    "task": "mmlu",
                    "subject": subject,
                    "tokens": tokens,
                    "max_new": 10,
                    "gold": chr(ord("A") + int(row["answer"])),
                    "allowed": ["A", "B", "C", "D"],
                }
            )
    return items


def build_gsm8k(tokenizer: Tokenizer, limit: int | None) -> list[dict]:
    train = list(load_dataset("openai/gsm8k", "main", split="train"))
    test = list(load_dataset("openai/gsm8k", "main", split="test"))
    shots = []
    for shot in train[:8]:
        shots.append(f"Q: {shot['question'].strip()}\nA: {shot['answer'].strip()}")
    prefix = "\n\n".join(shots)
    items = []
    for row_index, row in enumerate(test):
        if limit is not None and len(items) >= limit:
            break
        body = prefix + f"\n\nQ: {row['question'].strip()}\nA:"
        tokens = encode(tokenizer, chat_user(body))
        items.append(
            {
                "id": f"gsm8k/{row_index}",
                "task": "gsm8k",
                "tokens": tokens,
                "max_new": MAX_NEW_TOKENS,
                "gold": gold_gsm(row["answer"]),
            }
        )
    return items


def write_requests(path: Path, items: list[dict]) -> None:
    with path.open("w") as handle:
        for item in items:
            if len(item["tokens"]) >= MAX_SEQ_LEN:
                continue
            room = min(item["max_new"], MAX_NEW_TOKENS, MAX_SEQ_LEN - len(item["tokens"]))
            if room < 1:
                continue
            handle.write(
                json.dumps({"id": item["id"], "tokens": item["tokens"], "max_new": room}) + "\n"
            )


def load_results(path: Path) -> dict[str, dict]:
    found = {}
    if not path.exists():
        return found
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            found[row["id"]] = row
    return found


def score(items: list[dict], results: dict[str, dict], tokenizer: Tokenizer) -> dict:
    per_task: dict[str, list[bool]] = {}
    per_subject: dict[str, list[bool]] = {}
    skipped = 0
    missing = 0
    for item in items:
        row = results.get(item["id"])
        if row is None:
            missing += 1
            continue
        if "error" in row or item["prompt_tokens"] >= MAX_SEQ_LEN:
            skipped += 1
            continue
        text = tokenizer.decode(row.get("tokens") or [])
        task = item["task"]
        if task == "gsm8k":
            correct = predicted_number(text) is not None and predicted_number(text) == item["gold"]
        else:
            allowed = set(item["allowed"])
            pred = first_letter(text, allowed)
            gold = item["gold"]
            if gold.isdigit():
                gold = chr(ord("A") + int(gold) - 1)
            correct = pred == gold
        per_task.setdefault(task, []).append(correct)
        if task == "mmlu":
            per_subject.setdefault(item["subject"], []).append(correct)
    summary = {"skipped_or_error": skipped, "missing": missing, "tasks": {}}
    for task, flags in per_task.items():
        summary["tasks"][task] = {
            "n": len(flags),
            "accuracy": (sum(flags) / len(flags)) if flags else None,
        }
    if per_subject:
        macros = [sum(flags) / len(flags) for flags in per_subject.values() if flags]
        summary["tasks"]["mmlu"]["macro_accuracy"] = sum(macros) / len(macros) if macros else None
        summary["tasks"]["mmlu"]["subjects"] = len(macros)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="arc,mmlu,gsm8k")
    parser.add_argument("--limit", type=int, default=None, help="Max items per task, for a smoke run")
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/nvme/cuda-eval"))
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--launch-engine",
        action="store_true",
        help="Run the engine from an already written items.jsonl. Used after prepare exits so the dataset is not resident.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    meta_path = args.out_dir / "items.jsonl"
    requests = args.out_dir / "requests.jsonl"
    results = args.out_dir / "results.jsonl"

    if not args.launch_engine and not args.score_only:
        selected = [name.strip() for name in args.tasks.split(",") if name.strip()]
        builders = {"arc": build_arc, "mmlu": build_mmlu, "gsm8k": build_gsm8k}
        items: list[dict] = []
        for name in selected:
            started = time.time()
            built = builders[name](tokenizer, args.limit)
            items.extend(built)
            print(f"prepared {name}: {len(built)} items in {time.time() - started:.1f}s", flush=True)
        lengths = [len(item["tokens"]) for item in items]
        if lengths:
            print(
                f"prompt tokens min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.0f}",
                flush=True,
            )
        with meta_path.open("w") as handle:
            for item in items:
                stored = {key: value for key, value in item.items() if key != "tokens"}
                stored["prompt_tokens"] = len(item["tokens"])
                handle.write(json.dumps(stored) + "\n")
        done = load_results(results)
        pending = [item for item in items if item["id"] not in done]
        write_requests(requests, pending)
        print(f"pending {len(pending)} already_done {len(items) - len(pending)}", flush=True)
        # Replace this process so the datasets and token lists are gone before the GPU alloc.
        relaunch = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--launch-engine",
            "--tasks",
            args.tasks,
            "--out-dir",
            str(args.out_dir),
        ]
        os.execv(sys.executable, relaunch)

    items = []
    with meta_path.open() as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    if args.launch_engine:
        pending = sum(1 for _ in requests.open()) if requests.exists() else 0
        if pending:
            max_prompt = 1
            for item in items:
                prompt_len = int(item["prompt_tokens"])
                if 0 < prompt_len < MAX_SEQ_LEN:
                    max_prompt = max(max_prompt, prompt_len)
            command = [
                str(ENGINE),
                str(MODEL_PATH),
                "--eval-jsonl",
                str(requests),
                "--eval-out",
                str(results),
                "--eval-max-prompt",
                str(max_prompt),
            ]
            print("running", " ".join(command), flush=True)
            subprocess.run(command, check=True)
    summary = score(items, load_results(results), tokenizer)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
