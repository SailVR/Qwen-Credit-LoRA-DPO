import argparse
import os

import pandas as pd
from datasets import Dataset, load_from_disk
from modelscope import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from local_logging import LocalMetricCallback, plot_loss_curve, save_predictions, save_run_summary, setup_local_run

DEFAULT_MAX_LENGTH = 4096
MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "local_finance_sft_dataset"

script_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_path, "data")
cache_dir = os.path.join(script_path, "model")
processed_cache_dir = os.path.join(script_path, "cache")

train_dataset_path = os.path.join(data_dir, "train_format.json")
eval_dataset_path = os.path.join(data_dir, "eval_format.json")
DEFAULT_FINAL_MODEL_PATH = "./output_lora/final_model"

RUN_CONFIG = {
    "model": MODEL_ID,
    "dataset": DATASET_ID,
    "data_max_length": DEFAULT_MAX_LENGTH,
    "learning_rate": 5e-5,
    "batch_size": 1,
    "epochs": 1,
    "gradient_accumulation_steps": 4,
    "training_type": "lora",
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "final_model_path": DEFAULT_FINAL_MODEL_PATH,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen finance LoRA SFT model.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="./output_lora/Qwen3-0.6B")
    parser.add_argument("--final-model-path", default=DEFAULT_FINAL_MODEL_PATH)
    return parser.parse_args()


def build_run_config(args):
    config = dict(RUN_CONFIG)
    config.update(
        {
            "data_max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "output_dir": args.output_dir,
            "final_model_path": args.final_model_path,
        }
    )
    return config


def get_process_paths(max_length):
    suffix = "" if max_length == DEFAULT_MAX_LENGTH else f"_len{max_length}"
    train_path = os.path.join(processed_cache_dir, f"processed_finance_train_dataset{suffix}")
    eval_path = os.path.join(processed_cache_dir, f"processed_finance_eval_dataset{suffix}")
    return train_path, eval_path

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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded on:", next(model.parameters()).device)
    return model, tokenizer


def build_process_fn(tokenizer, max_length):
    def process_fun(example):
        prompt = (
            f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_all = tokenizer(prompt, add_special_tokens=False)
        output_all = tokenizer(example["output"], add_special_tokens=False)

        input_ids = input_all["input_ids"] + output_all["input_ids"]
        attention_mask = input_all["attention_mask"] + output_all["attention_mask"]
        labels = [-100] * len(input_all["input_ids"]) + output_all["input_ids"]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return process_fun


def predict(message, model, tokenizer):
    text = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    generation_ids = model.generate(
        **input_ids,
        max_length=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
    )
    generation_ids = generation_ids[0]
    return tokenizer.decode(generation_ids[len(input_ids["input_ids"][0]) :], skip_special_tokens=True)


def get_processed_dataset(dataset_path, process_path, process_fun):
    print(f"Preparing dataset: {dataset_path}")
    if os.path.exists(process_path):
        print(f"Using cached tokenized dataset: {process_path}")
        dataset = load_from_disk(process_path)
    else:
        df = pd.read_json(dataset_path, lines=True)
        ds = Dataset.from_pandas(df)
        dataset = ds.map(process_fun, remove_columns=ds.column_names)
        dataset.save_to_disk(process_path)
        print(f"Tokenized dataset saved to: {process_path}")
    print("Rows:", dataset.num_rows)
    return dataset


def main():
    args_cli = parse_args()
    run_config = build_run_config(args_cli)
    log_dir = setup_local_run("lora_finetune", run_config)
    model, tokenizer = load_model_and_tokenizer()
    process_fun = build_process_fn(tokenizer, args_cli.max_length)

    train_process_path, eval_process_path = get_process_paths(args_cli.max_length)
    train_dataset = get_processed_dataset(train_dataset_path, train_process_path, process_fun)
    eval_dataset = get_processed_dataset(eval_dataset_path, eval_process_path, process_fun)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=args_cli.lora_r,
        lora_alpha=args_cli.lora_alpha,
        lora_dropout=args_cli.lora_dropout,
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=args_cli.eval_batch_size,
        gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args_cli.eval_steps,
        logging_steps=args_cli.logging_steps,
        num_train_epochs=args_cli.epochs,
        save_steps=args_cli.save_steps,
        learning_rate=args_cli.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        logging_dir=os.path.join(log_dir, "trainer"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[LocalMetricCallback(os.path.join(log_dir, "metrics.jsonl"))],
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    plot_loss_curve(log_dir, trainer.state.log_history)

    predictions = []
    test_df = pd.read_json(eval_dataset_path, lines=True).head(2)
    for _, row in test_df.iterrows():
        message = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        response = predict(message, model, tokenizer)
        text = f"\nQuestion: {row['input']}\nModel answer: {response}"
        predictions.append(text)
        print(text)

    save_predictions(log_dir, predictions)
    trainer.save_model(args_cli.final_model_path)
    save_run_summary(
        log_dir=log_dir,
        run_name="lora_finetune",
        trainer=trainer,
        final_model_path=args_cli.final_model_path,
        train_rows=train_dataset.num_rows,
        eval_rows=eval_dataset.num_rows,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        extra={"prediction_count": len(predictions)},
    )
    print(f"Model saved to: {args_cli.final_model_path}")


if __name__ == "__main__":
    main()
