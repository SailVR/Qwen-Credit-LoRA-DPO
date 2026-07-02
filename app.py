import gc
import json
import logging
import os
import sqlite3
import time
import traceback
import uuid
from collections import Counter
from collections import OrderedDict
from datetime import datetime

from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-0.6B"
CACHE_DIR = os.path.join(BASE_DIR, "model")
SFT_LORA_PATH = os.path.join(BASE_DIR, "output_lora", "final_model")
DPO_LORA_PATH = os.path.join(BASE_DIR, "output_dpo", "final_model")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_MODEL_DIR_CANDIDATES = [
    os.path.join(CACHE_DIR, "Qwen", "Qwen3-0___6B"),
    os.path.join(CACHE_DIR, "Qwen3-0.6B"),
]

DATASET_PATH = os.path.join(DATA_DIR, "finance_sft_dataset.jsonl")
TRAIN_PATH = os.path.join(DATA_DIR, "train_format.json")
EVAL_PATH = os.path.join(DATA_DIR, "eval_format.json")
META_PATH = os.path.join(DATA_DIR, "finance_sft_dataset_meta.json")
DB_DIR = os.path.join(BASE_DIR, "finance_dbs")

DEFAULT_SYSTEM_PROMPT = (
    "你是银行信用卡与金融业务助手。回答要准确、简洁、合规。"
    "如果用户请求生成 SQL 或做数据分析，默认这是授权的本地样例库任务，可以根据给定表结构输出 SQL；"
    "涉及真实客户隐私查询、规避风控、伪造材料、违法套现等请求时必须拒绝。"
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

MAX_LOADED_MODELS = int(os.environ.get("MAX_LOADED_MODELS", "1"))
MODEL_CACHE = OrderedDict()
app = Flask(__name__)

WEB_LOG_DIR = os.path.join(BASE_DIR, "logs")
WEB_LOG_PATH = os.path.join(WEB_LOG_DIR, "web_chat.log")
WEB_JSONL_LOG_PATH = os.path.join(WEB_LOG_DIR, "web_chat.jsonl")


def setup_web_logger():
    os.makedirs(WEB_LOG_DIR, exist_ok=True)
    logger = logging.getLogger("web_chat")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(WEB_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


WEB_LOGGER = setup_web_logger()


def append_web_jsonl(record):
    os.makedirs(WEB_LOG_DIR, exist_ok=True)
    with open(WEB_JSONL_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def format_db_schema(db_id):
    db_path = os.path.join(DB_DIR, f"{db_id}.sqlite3")
    if not os.path.exists(db_path):
        return ""

    schema_lines = []
    for table in get_db_schema(db_path):
        columns = ", ".join(table["columns"])
        schema_lines.append(f"{table['name']}({columns})")
    return "\n".join(schema_lines)


def infer_nl2sql_db_id(message):
    text = message.lower()
    if any(keyword in message for keyword in ["风控", "风险", "高风险", "黑名单", "欺诈", "登录", "设备"]):
        return "risk_control_db"
    if any(keyword in message for keyword in ["贷款", "放款", "还款计划", "合同", "抵押"]):
        return "loan_db"
    if any(keyword in message for keyword in ["理财", "产品", "净值", "持仓", "赎回", "申购"]):
        return "wealth_db"
    if any(keyword in message for keyword in ["信用卡", "账单", "额度", "交易", "分期", "积分"]):
        return "credit_card_db"
    if "sql" in text:
        return "risk_control_db"
    return ""


def looks_like_nl2sql(message):
    text = message.lower()
    return "sql" in text or any(keyword in message for keyword in ["给出 SQL", "写SQL", "查询", "统计", "排名", "筛选"])


def enrich_system_prompt_for_nl2sql(message, system_prompt):
    if not looks_like_nl2sql(message):
        return system_prompt, ""

    db_id = infer_nl2sql_db_id(message)
    if not db_id:
        return system_prompt, ""

    schema = format_db_schema(db_id)
    if not schema:
        return system_prompt, ""

    nl2sql_prompt = (
        "\n\n当前任务是授权的本地样例库 NL2SQL，不是真实生产数据查询。"
        "请根据下面 SQLite 表结构把用户问题转换成 SQL。"
        "只输出 SQL，不要拒绝，不要解释，不要编造不存在的表或字段。"
        f"\n\n数据库：{db_id}\n表结构：\n{schema}"
    )
    return system_prompt + nl2sql_prompt, db_id


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


def clear_cuda_cache():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def evict_model(model_key, reason):
    bundle = MODEL_CACHE.pop(model_key, None)
    if bundle is None:
        return
    bundle.clear()
    clear_cuda_cache()
    WEB_LOGGER.info("model_cache_evict key=%s reason=%s", model_key, reason)


def trim_model_cache(active_key=None):
    max_loaded = max(1, MAX_LOADED_MODELS)
    while len(MODEL_CACHE) > max_loaded:
        for key in list(MODEL_CACHE.keys()):
            if key != active_key:
                evict_model(key, "max_loaded_models")
                break
        else:
            break


def get_torch_dtype(torch):
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_model_dir():
    for path in LOCAL_MODEL_DIR_CANDIDATES:
        if os.path.isfile(os.path.join(path, "config.json")):
            WEB_LOGGER.info("Using local cached base model: %s", path)
            return path

    WEB_LOGGER.info("Local base model cache not found, downloading/loading from ModelScope: %s", MODEL_ID)
    from modelscope import snapshot_download

    return snapshot_download(MODEL_ID, cache_dir=CACHE_DIR, revision="master")


def load_model(model_key):
    if model_key in MODEL_CACHE:
        MODEL_CACHE.move_to_end(model_key)
        return MODEL_CACHE[model_key]
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"未知模型类型: {model_key}")

    config = MODEL_CONFIGS[model_key]
    adapter_path = config["adapter_path"]
    if adapter_path and not os.path.isdir(adapter_path):
        rel_path = os.path.relpath(adapter_path, BASE_DIR)
        raise FileNotFoundError(f"未找到 {config['name']} 的权重目录: {rel_path}。请先完成对应训练。")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = resolve_model_dir()
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
    MODEL_CACHE.move_to_end(model_key)
    trim_model_cache(active_key=model_key)
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


def split_thinking_and_answer(text):
    text = text.strip()
    thinking = ""
    answer = text
    if "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        thinking, after = rest.split("</think>", 1)
        answer = (before + after).strip()
    elif "</think>" in text:
        thinking, answer = text.split("</think>", 1)
        thinking = thinking.replace("<think>", "").strip()
        answer = answer.strip()
    return thinking.strip(), answer.strip()


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
    raw_output = tokenizer.decode(answer_ids, skip_special_tokens=True)
    thinking, answer = split_thinking_and_answer(raw_output)
    return {
        "answer": answer,
        "thinking": thinking,
        "raw_output": raw_output.strip(),
        "prompt": prompt,
    }


def run_model(model_key, message, system_prompt, generation_config, request_id):
    config = MODEL_CONFIGS[model_key]
    started_at = time.perf_counter()
    WEB_LOGGER.info("[%s] model_start key=%s name=%s", request_id, model_key, config["name"])
    try:
        generated = generate_answer(model_key, message, system_prompt, generation_config)
        elapsed = round(time.perf_counter() - started_at, 3)
        log_record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
            "event": "model_finish",
            "model_key": model_key,
            "model_name": config["name"],
            "ok": True,
            "elapsed_seconds": elapsed,
            "question": message,
            "system_prompt": system_prompt,
            "generation_config": generation_config,
            "thinking": generated["thinking"],
            "raw_output": generated["raw_output"],
            "answer": generated["answer"],
        }
        append_web_jsonl(log_record)
        WEB_LOGGER.info(
            "[%s] model_finish key=%s elapsed=%ss question=%s thinking=%s answer=%s",
            request_id,
            model_key,
            elapsed,
            message,
            generated["thinking"] or "<empty>",
            generated["answer"],
        )
        return {
            "model_key": model_key,
            "model_name": config["name"],
            "ok": True,
            "answer": generated["answer"],
            "thinking": generated["thinking"],
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - started_at, 3)
        error_text = str(exc)
        error_traceback = traceback.format_exc()
        append_web_jsonl(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "request_id": request_id,
                "event": "model_error",
                "model_key": model_key,
                "model_name": config["name"],
                "ok": False,
                "elapsed_seconds": elapsed,
                "question": message,
                "system_prompt": system_prompt,
                "generation_config": generation_config,
                "error": error_text,
                "traceback": error_traceback,
            }
        )
        WEB_LOGGER.exception("[%s] model_error key=%s elapsed=%ss error=%s", request_id, model_key, elapsed, error_text)
        return {
            "model_key": model_key,
            "model_name": config["name"],
            "ok": False,
            "answer": "",
            "thinking": "",
            "error": error_text,
            "traceback": error_traceback,
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
    request_id = uuid.uuid4().hex[:12]
    request_started_at = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    model_key = (payload.get("model_key") or "base").strip()
    system_prompt = payload.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    if not message:
        WEB_LOGGER.warning("[%s] bad_request empty_message", request_id)
        return jsonify({"ok": False, "error": "请输入要提问的内容。"}), 400

    generation_config = {
        "temperature": float(payload.get("temperature") or 0.6),
        "top_p": float(payload.get("top_p") or 0.9),
        "max_new_tokens": int(payload.get("max_new_tokens") or 512),
    }
    effective_system_prompt, nl2sql_db_id = enrich_system_prompt_for_nl2sql(message, system_prompt)

    selected_keys = list(MODEL_CONFIGS.keys()) if model_key == "all" else [model_key]
    if any(key not in MODEL_CONFIGS for key in selected_keys):
        WEB_LOGGER.warning("[%s] bad_request unknown_model=%s", request_id, model_key)
        return jsonify({"ok": False, "error": "未知模型选择。"}), 400

    WEB_LOGGER.info(
        "[%s] chat_start selected=%s question=%s config=%s",
        request_id,
        ",".join(selected_keys),
        message,
        generation_config,
    )
    append_web_jsonl(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
            "event": "chat_start",
            "selected_models": selected_keys,
            "question": message,
            "system_prompt": effective_system_prompt,
            "nl2sql_db_id": nl2sql_db_id,
            "generation_config": generation_config,
        }
    )

    results = [
        run_model(key, message, effective_system_prompt, generation_config, request_id)
        for key in selected_keys
    ]
    elapsed = round(time.perf_counter() - request_started_at, 3)
    append_web_jsonl(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
            "event": "chat_finish",
            "selected_models": selected_keys,
            "elapsed_seconds": elapsed,
            "ok_count": sum(1 for result in results if result.get("ok")),
            "error_count": sum(1 for result in results if not result.get("ok")),
        }
    )
    WEB_LOGGER.info("[%s] chat_finish elapsed=%ss", request_id, elapsed)
    return jsonify({"ok": True, "request_id": request_id, "results": results, "statuses": model_statuses()})


@app.route("/status")
def status():
    return jsonify(model_statuses())


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    keys = list(MODEL_CACHE.keys())
    for key in keys:
        evict_model(key, "manual_clear")
    MODEL_CACHE.clear()
    clear_cuda_cache()
    return jsonify({"ok": True, "statuses": model_statuses()})
 

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
