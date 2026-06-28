import os
import warnings

import pandas as pd
from datasets import Dataset, load_from_disk
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from local_logging import LocalMetricCallback, plot_loss_curve, save_predictions, setup_local_run

warnings.filterwarnings("ignore")

MAX_LENGTH = 4096
MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "local_finance_sft_dataset"

script_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_path, "data")
cache_dir = os.path.join(script_path, "model")
processed_cache_dir = os.path.join(script_path, "cache")

train_dataset_path = os.path.join(data_dir, "train_format.json")
eval_dataset_path = os.path.join(data_dir, "eval_format.json")
train_process_path = os.path.join(processed_cache_dir, "processed_finance_train_dataset")
eval_process_path = os.path.join(processed_cache_dir, "processed_finance_eval_dataset")

RUN_CONFIG = {
    "model": MODEL_ID,
    "dataset": DATASET_ID,
    "data_max_length": MAX_LENGTH,
    "learning_rate": 5e-5,
    "batch_size": 1,
    "epochs": 1,
    "gradient_accumulation_steps": 4,
    "training_type": "full",
}

def load_model_and_tokenizer():
    print("---------------------")
    print("Downloading/loading base model...")
    model_dir = snapshot_download(MODEL_ID, cache_dir=cache_dir, revision="master")
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded on:", next(model.parameters()).device)
    return model, tokenizer


def build_process_fn(tokenizer):
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

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return process_fun


def predict(message, model, tokenizer):
    text = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    generation_ids = model.generate(**input_ids, max_length=2048, do_sample=True)
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
    log_dir = setup_local_run("full_finetune", RUN_CONFIG)
    model, tokenizer = load_model_and_tokenizer()
    process_fun = build_process_fn(tokenizer)

    train_dataset = get_processed_dataset(train_dataset_path, train_process_path, process_fun)
    eval_dataset = get_processed_dataset(eval_dataset_path, eval_process_path, process_fun)

    model.enable_input_require_grads()
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir="./output/Qwen3-0.6B",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        num_train_epochs=1,
        save_steps=400,
        learning_rate=5e-5,
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

    trainer.train()
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
    trainer.save_model("./output/final_model")
    print("Model saved to: ./output/final_model")


if __name__ == "__main__":
    main()
