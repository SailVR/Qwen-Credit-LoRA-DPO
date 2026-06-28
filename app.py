import gc
import json
import os
import sqlite3
import traceback
from collections import Counter

from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"
CACHE_DIR = os.path.join(BASE_DIR, "model")
SFT_LORA_PATH = os.path.join(BASE_DIR, "output_lora", "final_model")
DPO_LORA_PATH = os.path.join(BASE_DIR, "output_dpo", "final_model")

DATASET_PATH = os.path.join(BASE_DIR, "finance_sft_dataset.jsonl")
TRAIN_PATH = os.path.join(BASE_DIR, "train_format.json")
EVAL_PATH = os.path.join(BASE_DIR, "eval_format.json")
META_PATH = os.path.join(BASE_DIR, "finance_sft_dataset_meta.json")
DB_DIR = os.path.join(BASE_DIR, "finance_dbs")

DEFAULT_SYSTEM_PROMPT = (
    "你是银行信用卡与金融业务助手。回答要准确、简洁、合规；"
    "涉及客户隐私、规避风控、伪造材料、违法套现等请求时必须拒绝。"
)

MODEL_CONFIGS = {
    "base": {
        "name": "原模型",
        "description": "Qwen/Qwen3-0.6B，未加载本项目训练的 LoRA。",
        "adapter_path": None,
    },
    "sft": {
        "name": "LoRA SFT 后模型",
        "description": "加载 ./output_lora/final_model，用于观察监督微调后的效果。",
        "adapter_path": SFT_LORA_PATH,
    },
    "dpo": {
        "name": "DPO 后模型",
        "description": "加载 ./output_dpo/final_model，用于观察偏好优化后的效果。",
        "adapter_path": DPO_LORA_PATH,
    },
}

TASK_LABELS = {
    "intent_classification": "意图分类",
    "qa": "金融知识问答",
    "nl2sql": "NL2SQL",
    "compliance": "合规拒答",
}

MODEL_CACHE = {}
app = Flask(__name__)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def file_info(path):
    if not os.path.exists(path):
        return {"exists": False, "size_mb": 0}
    return {"exists": True, "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2)}


def load_dataset_summary():
    meta = read_json(META_PATH, default={}) or {}
    if meta:
        return {
            "total": meta.get("total", 0),
            "train": meta.get("train", 0),
            "eval": meta.get("eval", 0),
            "task_counts": dict(meta.get("task_counts", {})),
            "db_counts": dict(meta.get("nl2sql_db_counts", {})),
            "validated_sql_count": meta.get("validated_sql_count", 0),
            "random_seed": meta.get("random_seed", ""),
        }

    counts = Counter()
    db_counts = Counter()
    total = 0
    for row in iter_jsonl(DATASET_PATH):
        total += 1
        task_type = row.get("task_type", "unknown")
        counts[task_type] += 1
        if task_type == "nl2sql":
            db_counts[row.get("db_id", "unknown")] += 1
    return {
        "total": total,
        "train": sum(1 for _ in iter_jsonl(TRAIN_PATH)),
        "eval": sum(1 for _ in iter_jsonl(EVAL_PATH)),
        "task_counts": dict(counts),
        "db_counts": dict(db_counts),
        "validated_sql_count": 0,
        "random_seed": "",
    }


def load_samples(task_type=None, limit=12):
    samples = []
    for row in iter_jsonl(DATASET_PATH):
        if task_type and row.get("task_type") != task_type:
            continue
        samples.append(
            {
                "task_type": row.get("task_type", ""),
                "task_label": TASK_LABELS.get(row.get("task_type", ""), row.get("task_type", "")),
                "db_id": row.get("db_id", ""),
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
                "output": row.get("output", ""),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def get_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    tables = []
    try:
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table_name in table_names:
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            tables.append(
                {
                    "name": table_name,
                    "count": count,
                    "columns": [column[1] for column in columns],
                }
            )
    finally:
        conn.close()
    return tables


def load_database_summary():
    databases = []
    if not os.path.isdir(DB_DIR):
        return databases

    for name in sorted(os.listdir(DB_DIR)):
        if not name.endswith(".sqlite3"):
            continue
        path = os.path.join(DB_DIR, name)
        databases.append(
            {
                "db_id": name.replace(".sqlite3", ""),
                "file_name": name,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
                "tables": get_db_schema(path),
            }
        )
    return databases


def model_statuses():
    statuses = {}
    for key, config in MODEL_CONFIGS.items():
        adapter_path = config["adapter_path"]
        available = adapter_path is None or os.path.isdir(adapter_path)
        statuses[key] = {
            "key": key,
            "name": config["name"],
            "description": config["description"],
            "available": available,
            "loaded": key in MODEL_CACHE,
            "adapter_path": adapter_path,
        }
    return statuses


def get_torch_dtype(torch):
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(model_key):
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"未知模型类型: {model_key}")

    config = MODEL_CONFIGS[model_key]
    adapter_path = config["adapter_path"]
    if adapter_path and not os.path.isdir(adapter_path):
        rel_path = os.path.relpath(adapter_path, BASE_DIR)
        raise FileNotFoundError(f"未找到 {config['name']} 的权重目录: {rel_path}。请先完成对应训练。")

    import torch
    from modelscope import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR, revision="master")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=get_torch_dtype(torch),
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    MODEL_CACHE[model_key] = {"model": model, "tokenizer": tokenizer}
    return MODEL_CACHE[model_key]


def build_prompt(tokenizer, system_prompt, user_message):
    messages = [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def clean_answer(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def generate_answer(model_key, user_message, system_prompt, generation_config):
    import torch

    bundle = load_model(model_key)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    prompt = build_prompt(tokenizer, system_prompt, user_message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=True,
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return clean_answer(tokenizer.decode(answer_ids, skip_special_tokens=True))


def run_model(model_key, message, system_prompt, generation_config):
    config = MODEL_CONFIGS[model_key]
    try:
        return {
            "model_key": model_key,
            "model_name": config["name"],
            "ok": True,
            "answer": generate_answer(model_key, message, system_prompt, generation_config),
        }
    except Exception as exc:
        return {
            "model_key": model_key,
            "model_name": config["name"],
            "ok": False,
            "answer": "",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=MODEL_CONFIGS,
        statuses=model_statuses(),
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


@app.route("/dashboard")
def dashboard():
    task_type = request.args.get("task_type", "").strip()
    if task_type not in TASK_LABELS:
        task_type = ""

    files = {
        "finance_sft_dataset.jsonl": file_info(DATASET_PATH),
        "train_format.json": file_info(TRAIN_PATH),
        "eval_format.json": file_info(EVAL_PATH),
        "finance_sft_dataset_meta.json": file_info(META_PATH),
    }
    return render_template(
        "dashboard.html",
        task_labels=TASK_LABELS,
        active_task=task_type,
        summary=load_dataset_summary(),
        samples=load_samples(task_type=task_type or None),
        databases=load_database_summary(),
        files=files,
    )


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    model_key = (payload.get("model_key") or "base").strip()
    system_prompt = payload.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    if not message:
        return jsonify({"ok": False, "error": "请输入要提问的内容。"}), 400

    generation_config = {
        "temperature": float(payload.get("temperature") or 0.6),
        "top_p": float(payload.get("top_p") or 0.9),
        "max_new_tokens": int(payload.get("max_new_tokens") or 512),
    }

    selected_keys = list(MODEL_CONFIGS.keys()) if model_key == "all" else [model_key]
    if any(key not in MODEL_CONFIGS for key in selected_keys):
        return jsonify({"ok": False, "error": "未知模型选择。"}), 400

    results = [run_model(key, message, system_prompt, generation_config) for key in selected_keys]
    return jsonify({"ok": True, "results": results, "statuses": model_statuses()})


@app.route("/status")
def status():
    return jsonify(model_statuses())


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return jsonify({"ok": True, "statuses": model_statuses()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
