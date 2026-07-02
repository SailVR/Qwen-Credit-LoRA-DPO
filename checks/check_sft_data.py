import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "finance_sft_dataset.jsonl"
DEFAULT_DB_DIR = BASE_DIR / "finance_dbs"
REQUIRED_FIELDS = ("task_type", "instruction", "input", "output")
KNOWN_TASK_TYPES = {"intent_classification", "qa", "nl2sql", "compliance"}
DANGEROUS_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH)\b", re.I)


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


def validate_sql(row, line_number, db_dir):
    errors = []
    warnings = []
    db_id = row.get("db_id", "")
    sql = (row.get("output") or "").strip()

    if not db_id:
        errors.append(f"line {line_number}: nl2sql row missing db_id")
        return errors, warnings

    db_path = db_dir / f"{db_id}.sqlite3"
    if not db_path.exists():
        errors.append(f"line {line_number}: database not found for db_id={db_id}: {db_path}")
        return errors, warnings

    if not sql:
        errors.append(f"line {line_number}: nl2sql output is empty")
        return errors, warnings

    if DANGEROUS_SQL.search(sql):
        errors.append(f"line {line_number}: nl2sql output contains dangerous SQL keyword: {short_text(sql)}")
        return errors, warnings

    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.I):
        warnings.append(f"line {line_number}: nl2sql output is not SELECT/WITH: {short_text(sql)}")

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(sql).fetchmany(3)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        errors.append(f"line {line_number}: SQL execution failed on {db_id}: {exc}; sql={short_text(sql)}")

    return errors, warnings


def check_sft_data(data_path, db_dir, max_examples):
    errors = []
    warnings = []
    task_counts = Counter()
    db_counts = Counter()
    duplicate_keys = Counter()
    total = 0
    nl2sql_total = 0
    nl2sql_valid = 0

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

        task_type = row.get("task_type", "")
        task_counts[task_type] += 1
        if task_type not in KNOWN_TASK_TYPES:
            warnings.append(f"line {line_number}: unknown task_type={task_type!r}")

        key = (
            task_type,
            row.get("instruction", "").strip(),
            row.get("input", "").strip(),
            row.get("output", "").strip(),
        )
        duplicate_keys[key] += 1

        if task_type == "nl2sql":
            nl2sql_total += 1
            db_counts[row.get("db_id", "")] += 1
            sql_errors, sql_warnings = validate_sql(row, line_number, db_dir)
            errors.extend(sql_errors)
            warnings.extend(sql_warnings)
            if not sql_errors:
                nl2sql_valid += 1

    duplicate_count = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    if duplicate_count:
        warnings.append(f"found {duplicate_count} duplicate SFT rows by task/instruction/input/output")

    print("SFT data check")
    print(f"- file: {data_path}")
    print(f"- total rows: {total}")
    print(f"- task counts: {dict(task_counts)}")
    print(f"- nl2sql db counts: {dict(db_counts)}")
    print(f"- nl2sql executable: {nl2sql_valid}/{nl2sql_total}")
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
    parser = argparse.ArgumentParser(description="Check finance SFT JSONL data quality.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to SFT JSONL data.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="Path to SQLite database directory.")
    parser.add_argument("--max-examples", type=int, default=20, help="Max warning/error examples to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"SFT data file not found: {args.data}")
    if not args.db_dir.exists():
        raise SystemExit(f"SQLite database directory not found: {args.db_dir}")
    raise SystemExit(check_sft_data(args.data, args.db_dir, args.max_examples))


if __name__ == "__main__":
    main()
