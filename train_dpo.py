import os
import warnings

import pandas as pd
from datasets import Dataset
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from local_logging import LocalMetricCallback, plot_loss_curve, setup_local_run

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "local_finance_dpo_dataset"

DPO_TRAIN_PATH = "dpo_train.jsonl"
DPO_EVAL_PATH = "dpo_eval.jsonl"
SFT_LORA_PATH = "./output_lora/final_model"

MAX_PROMPT_LENGTH = 2048
MAX_LENGTH = 3072

RUN_CONFIG = {
    "model": MODEL_ID,
    "dataset": DATASET_ID,
    "training_type": "dpo_lora",
    "learning_rate": 5e-6,
    "batch_size": 1,
    "epochs": 1,
    "gradient_accumulation_steps": 4,
    "beta": 0.1,
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "max_length": MAX_LENGTH,
    "sft_lora_path": SFT_LORA_PATH,
}

script_path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_path, "model")


def load_model_and_tokenizer():
    print("---------------------")
    print("Downloading/loading base model...")
    model_dir = snapshot_download(MODEL_ID, cache_dir=cache_dir, revision="master")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    if not os.path.isdir(SFT_LORA_PATH):
        raise FileNotFoundError(
            f"SFT LoRA adapter not found: {SFT_LORA_PATH}. "
            "Please run `python train_lora.py` first."
        )

    print(f"Loading SFT LoRA adapter from: {SFT_LORA_PATH}")
    model = PeftModel.from_pretrained(model, SFT_LORA_PATH, is_trainable=True)
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
    log_dir = setup_local_run("dpo_lora", RUN_CONFIG)
    model, tokenizer = load_model_and_tokenizer()
    train_dataset = load_dpo_dataset(DPO_TRAIN_PATH)
    eval_dataset = load_dpo_dataset(DPO_EVAL_PATH)

    print("Train rows:", train_dataset.num_rows)
    print("Eval rows:", eval_dataset.num_rows)

    args = DPOConfig(
        output_dir="./output_dpo/Qwen3-0.6B",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        beta=0.1,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_length=MAX_LENGTH,
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

    trainer.train()
    plot_loss_curve(log_dir, trainer.state.log_history)
    trainer.save_model("./output_dpo/final_model")
    tokenizer.save_pretrained("./output_dpo/final_model")
    print("DPO LoRA model saved to: ./output_dpo/final_model")


if __name__ == "__main__":
    main()
