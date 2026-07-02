import argparse
import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "finance_dpo_dataset.jsonl"
REQUIRED_FIELDS = ("prompt", "chosen", "rejected")
KNOWN_TASK_TYPES = {"intent_classification", "qa", "nl2sql", "compliance"}


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield line_number, json.loads(text)
            except json.JSONDecodeError as exc:
                yield line_number, {"__json_error__": str(exc)}


def short_text(text, limit=120):
    text = str(text).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def ratio(a, b):
    return len(a) / max(len(b), 1)


def check_dpo_data(data_path, max_examples):
    errors = []
    warnings = []
    task_counts = Counter()
    db_counts = Counter()
    ids = Counter()
    prompt_counts = Counter()
    pair_counts = Counter()
    total = 0
    identical_pairs = 0
    extreme_length_pairs = 0

    for line_number, row in iter_jsonl(data_path):
        total += 1
        if "__json_error__" in row:
            errors.append(f"line {line_number}: invalid JSON: {row['__json_error__']}")
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"line {line_number}: missing required fields: {missing}")

        empty = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if empty:
            errors.append(f"line {line_number}: empty required fields: {empty}")

        row_id = str(row.get("id", "")).strip()
        if row_id:
            ids[row_id] += 1

        task_type = row.get("task_type", "")
        if task_type:
            task_counts[task_type] += 1
            if task_type not in KNOWN_TASK_TYPES:
                warnings.append(f"line {line_number}: unknown task_type={task_type!r}")
        else:
            warnings.append(f"line {line_number}: missing task_type")

        db_id = row.get("db_id", "")
        if db_id:
            db_counts[db_id] += 1

        prompt = str(row.get("prompt", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        prompt_counts[prompt] += 1
        pair_counts[(prompt, chosen, rejected)] += 1

        if prompt and ("<|im_start|>user" not in prompt or "<|im_start|>assistant" not in prompt):
            warnings.append(f"line {line_number}: prompt does not look like chat template: {short_text(prompt)}")

        if chosen and rejected and chosen == rejected:
            identical_pairs += 1
            errors.append(f"line {line_number}: chosen and rejected are identical")

        if chosen and rejected:
            length_ratio = ratio(chosen, rejected)
            if length_ratio >= 5 or length_ratio <= 0.2:
                extreme_length_pairs += 1
                warnings.append(
                    f"line {line_number}: chosen/rejected length ratio is extreme "
                    f"({len(chosen)}/{len(rejected)}): chosen={short_text(chosen)} rejected={short_text(rejected)}"
                )

        source_input = str(row.get("source_input", "")).strip()
        if source_input and source_input not in prompt:
            warnings.append(f"line {line_number}: source_input is not found in prompt")

    duplicate_ids = sum(count - 1 for count in ids.values() if count > 1)
    if duplicate_ids:
        errors.append(f"found {duplicate_ids} duplicate ids")

    duplicate_pairs = sum(count - 1 for count in pair_counts.values() if count > 1)
    if duplicate_pairs:
        warnings.append(f"found {duplicate_pairs} duplicate DPO triples by prompt/chosen/rejected")

    repeated_prompts = sum(count - 1 for count in prompt_counts.values() if count > 1)
    if repeated_prompts:
        warnings.append(f"found {repeated_prompts} repeated prompts; this may be expected if multiple preferences exist")

    print("DPO data check")
    print(f"- file: {data_path}")
    print(f"- total rows: {total}")
    print(f"- task counts: {dict(task_counts)}")
    print(f"- nl2sql db counts: {dict(db_counts)}")
    print(f"- identical chosen/rejected pairs: {identical_pairs}")
    print(f"- extreme chosen/rejected length pairs: {extreme_length_pairs}")
    print(f"- warnings: {len(warnings)}")
    print(f"- errors: {len(errors)}")

    if warnings:
        print("\nWarnings:")
        for item in warnings[:max_examples]:
            print(f"- {item}")
        if len(warnings) > max_examples:
            print(f"- ... {len(warnings) - max_examples} more")

    if errors:
        print("\nErrors:")
        for item in errors[:max_examples]:
            print(f"- {item}")
        if len(errors) > max_examples:
            print(f"- ... {len(errors) - max_examples} more")

    return 1 if errors else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Check finance DPO JSONL preference data quality.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to DPO JSONL data.")
    parser.add_argument("--max-examples", type=int, default=20, help="Max warning/error examples to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"DPO data file not found: {args.data}")
    raise SystemExit(check_dpo_data(args.data, args.max_examples))


if __name__ == "__main__":
    main()
