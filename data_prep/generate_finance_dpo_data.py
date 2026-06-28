import json
import os
import random
from collections import Counter


RANDOM_SEED = 20260628
TRAIN_RATIO = 0.9

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SFT_PATH = os.path.join(DATA_DIR, "finance_sft_dataset.jsonl")
DPO_PATH = os.path.join(DATA_DIR, "finance_dpo_dataset.jsonl")
DPO_TRAIN_PATH = os.path.join(DATA_DIR, "dpo_train.jsonl")
DPO_EVAL_PATH = os.path.join(DATA_DIR, "dpo_eval.jsonl")
DPO_META_PATH = os.path.join(DATA_DIR, "finance_dpo_dataset_meta.json")

INTENT_LABELS = [
    "账单查询",
    "还款咨询",
    "额度调整",
    "分期办理",
    "积分权益",
    "卡片激活",
    "交易异议",
    "活动达标",
    "风险控制",
    "贷款咨询",
    "理财咨询",
    "人工客服",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(row):
    return (
        f"<|im_start|>system\n{row['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{row['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def wrong_intent(chosen):
    candidates = [label for label in INTENT_LABELS if label != chosen]
    return random.choice(candidates)


def rejected_qa(row):
    bad_answers = [
        "这个问题不用太担心，按自己的想法处理就行，具体规则每家银行都差不多。",
        "可以直接办理，一般不会有什么费用或风险，不需要再确认银行规则。",
        "这种情况一定不会影响征信，也不会产生额外费用，可以放心。",
        "建议多刷几笔提高活跃度，银行通常会自动给你更好的结果。",
    ]
    return random.choice(bad_answers)


def rejected_nl2sql(row):
    sql = row["output"].rstrip().rstrip(";")
    variants = [
        "SELECT * FROM customer;",
        "SELECT COUNT(*) FROM transaction_record;",
        "SELECT customer_id, name FROM customer WHERE city = '北京';",
        sql.replace("COUNT", "SUM", 1) + ";",
        sql.replace("WHERE", "WHERE 1 = 0 AND", 1) + ";" if "WHERE" in sql else sql + " WHERE 1 = 0;",
    ]
    bad = random.choice(variants)
    if bad == row["output"]:
        bad = "SELECT * FROM unknown_table;"
    return bad


def rejected_compliance(row):
    unsafe = [
        "可以通过拆分多笔交易、选择特定商户类型并避开整数金额来降低被风控识别的概率。",
        "你可以先临时包装收入流水，再提交贷款申请材料，这样审批看起来会更稳定。",
        "如果要查询他人账户，可以尝试使用对方的身份信息登录相关渠道。",
        "可以准备一份看起来合理的交易说明，把真实用途改成普通消费。",
        "活动奖励可以通过虚拟交易刷出来，之后再做退款处理。",
    ]
    return random.choice(unsafe)


def rejected_answer(row):
    task_type = row["task_type"]
    if task_type == "intent_classification":
        return wrong_intent(row["output"])
    if task_type == "qa":
        return rejected_qa(row)
    if task_type == "nl2sql":
        return rejected_nl2sql(row)
    if task_type == "compliance":
        return rejected_compliance(row)
    return "无法处理。"


def make_dpo_record(row, idx):
    rejected = rejected_answer(row)
    return {
        "id": f"finance_dpo_{idx:05d}",
        "task_type": row["task_type"],
        "db_id": row.get("db_id", ""),
        "prompt": build_prompt(row),
        "chosen": row["output"],
        "rejected": rejected,
        "source_input": row["input"],
    }


def main():
    random.seed(RANDOM_SEED)
    sft_rows = list(read_jsonl(SFT_PATH))
    dpo_rows = [make_dpo_record(row, idx) for idx, row in enumerate(sft_rows, start=1)]

    random.shuffle(dpo_rows)
    split_idx = int(len(dpo_rows) * TRAIN_RATIO)
    train_rows = dpo_rows[:split_idx]
    eval_rows = dpo_rows[split_idx:]

    write_jsonl(DPO_PATH, dpo_rows)
    write_jsonl(DPO_TRAIN_PATH, train_rows)
    write_jsonl(DPO_EVAL_PATH, eval_rows)

    meta = {
        "total": len(dpo_rows),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "task_counts": Counter(row["task_type"] for row in dpo_rows),
        "train_task_counts": Counter(row["task_type"] for row in train_rows),
        "eval_task_counts": Counter(row["task_type"] for row in eval_rows),
        "random_seed": RANDOM_SEED,
        "format": {
            "prompt": "chat template prefix ending with assistant header",
            "chosen": "preferred answer",
            "rejected": "inferior answer",
        },
        "files": {
            "all": os.path.basename(DPO_PATH),
            "train": os.path.basename(DPO_TRAIN_PATH),
            "eval": os.path.basename(DPO_EVAL_PATH),
        },
    }
    os.makedirs(os.path.dirname(DPO_META_PATH), exist_ok=True)
    with open(DPO_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
