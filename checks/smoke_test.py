import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "finance_dbs"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REQUIRED_FILES = [
    DATA_DIR / "finance_sft_dataset.jsonl",
    DATA_DIR / "train_format.json",
    DATA_DIR / "eval_format.json",
    DATA_DIR / "finance_dpo_dataset.jsonl",
    DATA_DIR / "dpo_train.jsonl",
    DATA_DIR / "dpo_eval.jsonl",
    BASE_DIR / "app.py",
    BASE_DIR / "train_lora.py",
    BASE_DIR / "train_dpo.py",
]

REQUIRED_DATABASES = [
    DB_DIR / "credit_card_db.sqlite3",
    DB_DIR / "loan_db.sqlite3",
    DB_DIR / "wealth_db.sqlite3",
    DB_DIR / "risk_control_db.sqlite3",
]


def ok(message):
    print(f"[OK] {message}")


def fail(message):
    print(f"[FAIL] {message}")


def warn(message):
    print(f"[WARN] {message}")


def read_first_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)
    raise ValueError("file is empty")


def check_files():
    errors = []
    for path in REQUIRED_FILES:
        if path.exists():
            ok(f"found {path.relative_to(BASE_DIR)}")
        else:
            fail(f"missing {path.relative_to(BASE_DIR)}")
            errors.append(str(path))
    return errors


def check_jsonl_shapes():
    errors = []
    checks = [
        (DATA_DIR / "finance_sft_dataset.jsonl", {"task_type", "instruction", "input", "output"}),
        (DATA_DIR / "finance_dpo_dataset.jsonl", {"prompt", "chosen", "rejected"}),
        (DATA_DIR / "dpo_train.jsonl", {"prompt", "chosen", "rejected"}),
        (DATA_DIR / "dpo_eval.jsonl", {"prompt", "chosen", "rejected"}),
    ]
    for path, required_fields in checks:
        try:
            row = read_first_jsonl(path)
            missing = sorted(required_fields - set(row))
            if missing:
                fail(f"{path.relative_to(BASE_DIR)} missing fields in first row: {missing}")
                errors.append(str(path))
            else:
                ok(f"{path.relative_to(BASE_DIR)} first row has required fields")
        except Exception as exc:
            fail(f"{path.relative_to(BASE_DIR)} cannot be read as JSONL: {exc}")
            errors.append(str(path))
    return errors


def check_sqlite_databases():
    errors = []
    for path in REQUIRED_DATABASES:
        if not path.exists():
            fail(f"missing database {path.relative_to(BASE_DIR)}")
            errors.append(str(path))
            continue
        try:
            conn = sqlite3.connect(str(path))
            try:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            finally:
                conn.close()
            if tables:
                ok(f"{path.relative_to(BASE_DIR)} opens with {len(tables)} tables")
            else:
                fail(f"{path.relative_to(BASE_DIR)} opens but has no user tables")
                errors.append(str(path))
        except sqlite3.Error as exc:
            fail(f"{path.relative_to(BASE_DIR)} cannot be opened: {exc}")
            errors.append(str(path))
    return errors


def check_python_imports():
    errors = []
    modules = [
        BASE_DIR / "app.py",
        BASE_DIR / "train_lora.py",
        BASE_DIR / "train_dpo.py",
        BASE_DIR / "local_logging.py",
    ]
    for path in modules:
        module_name = f"smoke_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ok(f"imported {path.name}")
        except Exception as exc:
            fail(f"failed to import {path.name}: {exc}")
            errors.append(str(path))
    return errors


def check_model_outputs():
    paths = [
        BASE_DIR / "model",
        BASE_DIR / "output_lora" / "final_model",
        BASE_DIR / "output_dpo" / "final_model",
    ]
    for path in paths:
        if path.exists():
            ok(f"found {path.relative_to(BASE_DIR)}")
        else:
            warn(f"not found yet: {path.relative_to(BASE_DIR)}")
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="Run lightweight project smoke checks.")
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Skip importing Python entrypoints if dependencies are not installed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Project smoke test: {BASE_DIR}")
    errors = []
    errors.extend(check_files())
    errors.extend(check_jsonl_shapes())
    errors.extend(check_sqlite_databases())
    if not args.skip_imports:
        errors.extend(check_python_imports())
    check_model_outputs()

    if errors:
        print(f"\nSmoke test failed with {len(errors)} issue(s).")
        return 1
    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
