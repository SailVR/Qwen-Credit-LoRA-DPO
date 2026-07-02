import argparse
import os
import warnings

import pandas as pd
from datasets import Dataset
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from local_logging import LocalMetricCallback, plot_loss_curve, save_preference_samples, save_run_summary, setup_local_run

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "local_finance_dpo_dataset"

script_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_path, "data")
cache_dir = os.path.join(script_path, "model")

DPO_TRAIN_PATH = os.path.join(data_dir, "dpo_train.jsonl")
DPO_EVAL_PATH = os.path.join(data_dir, "dpo_eval.jsonl")
SFT_LORA_PATH = "./output_lora/final_model"
FINAL_MODEL_PATH = "./output_dpo/final_model"

DEFAULT_MAX_PROMPT_LENGTH = 2048
DEFAULT_MAX_LENGTH = 3072

RUN_CONFIG = {
    "model": MODEL_ID,
    "dataset": DATASET_ID,
    "training_type": "dpo_lora",
    "learning_rate": 5e-6,
    "batch_size": 1,
    "epochs": 1,
    "gradient_accumulation_steps": 4,
    "beta": 0.1,
    "max_prompt_length": DEFAULT_MAX_PROMPT_LENGTH,
    "max_length": DEFAULT_MAX_LENGTH,
    "sft_lora_path": SFT_LORA_PATH,
    "final_model_path": FINAL_MODEL_PATH,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen finance DPO LoRA model.")
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--sft-lora-path", default=SFT_LORA_PATH)
    parser.add_argument("--output-dir", default="./output_dpo/Qwen3-0.6B")
    parser.add_argument("--final-model-path", default=FINAL_MODEL_PATH)
    return parser.parse_args()


def build_run_config(args):
    config = dict(RUN_CONFIG)
    config.update(
        {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "beta": args.beta,
            "max_prompt_length": args.max_prompt_length,
            "max_length": args.max_length,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "sft_lora_path": args.sft_lora_path,
            "output_dir": args.output_dir,
            "final_model_path": args.final_model_path,
        }
    )
    return config


def load_model_and_tokenizer(sft_lora_path):
    print("---------------------")
    print("Downloading/loading base model...")
    model_dir = snapshot_download(MODEL_ID, cache_dir=cache_dir, revision="master")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    if not os.path.isdir(sft_lora_path):
        raise FileNotFoundError(
            f"SFT LoRA adapter not found: {sft_lora_path}. "
            "Please run `python train_lora.py` first."
        )

    print(f"Loading SFT LoRA adapter from: {sft_lora_path}")
    model = PeftModel.from_pretrained(model, sft_lora_path, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Model loaded on:", next(model.parameters()).device)
    return model, tokenizer


def load_dpo_dataset(path):
    df = pd.read_json(path, lines=True)
    required_columns = ["prompt", "chosen", "rejected"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return Dataset.from_pandas(df[required_columns], preserve_index=False)


def load_preference_samples(path, limit=5):
    df = pd.read_json(path, lines=True).head(limit)
    sample_columns = ["prompt", "chosen", "rejected"]
    optional_columns = [column for column in ["id", "task_type", "db_id", "source_input"] if column in df.columns]
    return df[optional_columns + sample_columns].to_dict(orient="records")


def build_trainer(model, tokenizer, train_dataset, eval_dataset, args, log_dir):
    common_kwargs = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": [LocalMetricCallback(os.path.join(log_dir, "metrics.jsonl"))],
    }

    try:
        return DPOTrainer(processing_class=tokenizer, **common_kwargs)
    except TypeError:
        return DPOTrainer(tokenizer=tokenizer, **common_kwargs)


def main():
    args_cli = parse_args()
    run_config = build_run_config(args_cli)
    log_dir = setup_local_run("dpo_lora", run_config)
    model, tokenizer = load_model_and_tokenizer(args_cli.sft_lora_path)
    train_dataset = load_dpo_dataset(DPO_TRAIN_PATH)
    eval_dataset = load_dpo_dataset(DPO_EVAL_PATH)

    print("Train rows:", train_dataset.num_rows)
    print("Eval rows:", eval_dataset.num_rows)

    args = DPOConfig(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=args_cli.eval_batch_size,
        gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
        num_train_epochs=args_cli.epochs,
        learning_rate=args_cli.learning_rate,
        logging_steps=args_cli.logging_steps,
        eval_strategy="steps",
        eval_steps=args_cli.eval_steps,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        beta=args_cli.beta,
        max_prompt_length=args_cli.max_prompt_length,
        max_length=args_cli.max_length,
        report_to="none",
        logging_dir=os.path.join(log_dir, "trainer"),
        remove_unused_columns=False,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        log_dir=log_dir,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    plot_loss_curve(log_dir, trainer.state.log_history)

    preference_samples = load_preference_samples(DPO_EVAL_PATH)
    save_preference_samples(log_dir, preference_samples)

    trainer.save_model(args_cli.final_model_path)
    tokenizer.save_pretrained(args_cli.final_model_path)
    save_run_summary(
        log_dir=log_dir,
        run_name="dpo_lora",
        trainer=trainer,
        final_model_path=args_cli.final_model_path,
        train_rows=train_dataset.num_rows,
        eval_rows=eval_dataset.num_rows,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        extra={
            "sft_lora_path": args_cli.sft_lora_path,
            "preference_sample_count": len(preference_samples),
        },
    )
    print(f"DPO LoRA model saved to: {args_cli.final_model_path}")


if __name__ == "__main__":
    main()
