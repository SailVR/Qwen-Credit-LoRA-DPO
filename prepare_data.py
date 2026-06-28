import argparse
import os
import shutil
from pathlib import Path

from data_prep import generate_finance_dpo_data, generate_finance_sft_data, init_finance_sqlite


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIRS = [
    BASE_DIR / "cache" / "processed_finance_train_dataset",
    BASE_DIR / "cache" / "processed_finance_eval_dataset",
]


def remove_processed_cache():
    removed = []
    for path in CACHE_DIRS:
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(BASE_DIR)))
    return removed


def run_step(name, func):
    print(f"\n========== {name} ==========")
    func()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare finance SQLite DBs, SFT data and DPO data.")
    parser.add_argument("--skip-db", action="store_true", help="Skip SQLite database initialization.")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT dataset generation.")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip DPO dataset generation.")
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove tokenized dataset caches after regenerating data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(BASE_DIR)

    if not args.skip_db:
        run_step("初始化金融 SQLite 数据库", init_finance_sqlite.main)
    if not args.skip_sft:
        run_step("生成 6000 条金融 SFT 数据", generate_finance_sft_data.main)
    if not args.skip_dpo:
        run_step("生成金融 DPO 偏好数据", generate_finance_dpo_data.main)
    if args.clean_cache:
        removed = remove_processed_cache()
        if removed:
            print("\n已删除旧 tokenized 缓存：")
            for path in removed:
                print(f"- {path}")
        else:
            print("\n没有发现需要删除的 tokenized 缓存。")

    print("\n数据准备完成。")


if __name__ == "__main__":
    main()
