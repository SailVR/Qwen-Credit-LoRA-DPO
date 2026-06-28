import json
import os
import random
import sqlite3
from collections import Counter


RANDOM_SEED = 20260628
TOTAL_COUNT = 6000
TRAIN_RATIO = 0.9

TASK_COUNTS = {
    "intent_classification": 1200,
    "qa": 2200,
    "nl2sql": 1800,
    "compliance": 800,
}

NL2SQL_DB_COUNTS = {
    "credit_card_db": 650,
    "loan_db": 400,
    "wealth_db": 350,
    "risk_control_db": 400,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "finance_dbs")
DATA_DIR = os.path.join(BASE_DIR, "data")

ALL_DATASET_PATH = os.path.join(DATA_DIR, "finance_sft_dataset.jsonl")
TRAIN_PATH = os.path.join(DATA_DIR, "train_format.json")
EVAL_PATH = os.path.join(DATA_DIR, "eval_format.json")
META_PATH = os.path.join(DATA_DIR, "finance_sft_dataset_meta.json")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_schema(db_id):
    db_path = os.path.join(DB_DIR, f"{db_id}.sqlite3")
    conn = sqlite3.connect(db_path)
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    lines = []
    for table in tables:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_defs = []
        for _, name, col_type, not_null, _, pk in cols:
            suffix = " PRIMARY KEY" if pk else ""
            required = " NOT NULL" if not_null and not pk else ""
            col_defs.append(f"{name} {col_type}{required}{suffix}".strip())
        lines.append(f"{table}({', '.join(col_defs)})")
    conn.close()
    return "\n".join(lines)


def make_record(task_type, instruction, input_text, output, **extra):
    row = {
        "task_type": task_type,
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }
    row.update(extra)
    return row


def generate_intent_records(count):
    labels = {
        "账单查询": [
            "我这个月账单金额是多少？",
            "为什么本期账单比上个月高？",
            "帮我看下{month}月信用卡账单有没有出。",
            "最低还款额是多少？",
        ],
        "还款咨询": [
            "信用卡还款日是哪天？",
            "我可以提前还款吗？",
            "还款失败了会不会逾期？",
            "最低还款后剩余部分怎么算利息？",
        ],
        "额度调整": [
            "我想把信用卡额度提到{amount}元。",
            "为什么我的临时额度申请失败？",
            "可用额度突然变少了是什么原因？",
            "能不能帮我降低信用卡额度？",
        ],
        "分期办理": [
            "这笔{amount}元消费可以分期吗？",
            "账单分期手续费怎么算？",
            "我想提前结清分期。",
            "分期后还能取消吗？",
        ],
        "积分权益": [
            "我的积分什么时候到账？",
            "积分可以兑换什么权益？",
            "为什么这笔消费没有积分？",
            "白金卡机场贵宾厅权益怎么用？",
        ],
        "卡片激活": [
            "新卡收到后怎么激活？",
            "信用卡激活失败怎么办？",
            "附属卡需要单独激活吗？",
            "不激活信用卡会收费吗？",
        ],
        "交易异议": [
            "这笔{amount}元交易不是我刷的。",
            "商户重复扣款了怎么办？",
            "境外交易我没有操作。",
            "我想申请交易争议处理。",
        ],
        "活动达标": [
            "首刷礼怎么才算达标？",
            "我参加的返现活动进度怎么看？",
            "扫码支付算不算活动消费？",
            "为什么活动奖励还没到账？",
        ],
        "风险控制": [
            "为什么我的卡被限制交易？",
            "刷卡提示风险交易是什么意思？",
            "怎么解除信用卡冻结？",
            "银行为什么给我降额？",
        ],
        "贷款咨询": [
            "我想申请消费贷款需要什么条件？",
            "贷款审批一般多久？",
            "提前还贷款有没有违约金？",
            "我的贷款为什么被拒？",
        ],
        "理财咨询": [
            "R3理财产品适合我吗？",
            "理财赎回多久到账？",
            "净值型理财会亏本金吗？",
            "怎么查看当前持仓收益？",
        ],
        "人工客服": [
            "我要转人工客服。",
            "这个问题我想找人工处理。",
            "帮我接入客服专员。",
            "我要投诉，需要人工。",
        ],
    }
    instruction = (
        "你是金融业务意图识别助手。请判断用户问题所属意图，只输出一个标签。"
        "可选标签包括：账单查询、还款咨询、额度调整、分期办理、积分权益、卡片激活、交易异议、活动达标、风险控制、贷款咨询、理财咨询、人工客服。"
    )
    records = []
    label_names = list(labels.keys())
    for i in range(count):
        label = label_names[i % len(label_names)]
        text = random.choice(labels[label]).format(
            month=random.choice(["3", "4", "5", "6"]),
            amount=random.choice(["3000", "5000", "12000", "30000"]),
        )
        if random.random() < 0.25:
            text = random.choice(["你好，", "麻烦问下，", "帮我看一下，", ""]) + text
        records.append(make_record("intent_classification", instruction, text, label))
    return records


def generate_qa_records(count):
    instruction = "你是金融业务知识问答助手。请用清晰、专业、合规的方式回答用户问题，并在必要时提示以银行实际规则为准。"
    samples = [
        (
            "信用卡最低还款会不会影响征信？",
            "按时偿还最低还款额通常不会直接形成逾期记录，也一般不会立即影响征信。但最低还款后，未还清部分通常会计收循环利息，长期只还最低还款也可能体现出较高负债压力。建议在资金允许时优先全额还款，具体计息方式以发卡行规则和账单说明为准。",
        ),
        (
            "信用卡分期后还能提前还款吗？",
            "信用卡分期后通常可以申请提前结清，但是否支持、剩余手续费是否继续收取、是否有提前结清费用，需要以发卡行分期协议为准。办理前建议先在银行 App 或客服渠道确认剩余本金、手续费和总成本。",
        ),
        (
            "账单日和还款日有什么区别？",
            "账单日是银行汇总上一账单周期交易并生成账单的日期，还款日是你需要在此之前完成还款的最后期限。账单日后新增交易通常进入下一期账单，还款日前未还足最低还款额可能产生逾期风险。",
        ),
        (
            "为什么我的信用卡可用额度突然变少？",
            "可用额度减少可能与新增消费、预授权冻结、账单未还、分期占用额度、临时额度到期或风险管控有关。你可以先核对最近交易、账单和额度明细，如存在不认识的交易，应尽快联系银行核实。",
        ),
        (
            "积分为什么没有到账？",
            "积分未到账可能是交易尚未入账、商户类型不计积分、活动统计存在延迟、发生退款或该笔交易属于取现、转账、手续费等排除类型。建议查看积分规则和交易入账状态，具体以发卡行活动细则为准。",
        ),
        (
            "理财产品风险等级 R3 是什么意思？",
            "R3 通常表示中等风险等级，产品可能存在净值波动，本金和收益都不保证。是否适合你，需要结合你的风险承受能力、投资期限、流动性需求和资产配置情况判断。",
        ),
        (
            "贷款提前还款一定划算吗？",
            "不一定。提前还款是否划算取决于剩余本金、利率、剩余期限、提前还款手续费或违约金，以及你的资金机会成本。建议先计算提前还款总成本和节省利息，再决定是否办理。",
        ),
        (
            "信用卡首刷礼怎么才算达标？",
            "首刷礼通常要求卡片激活后，在规定时间内完成符合条件的首笔消费。有些活动有金额门槛、支付渠道、商户类型或退款限制。是否达标应以活动页面和发卡行统计结果为准。",
        ),
        (
            "信用卡被盗刷怎么办？",
            "如果怀疑信用卡被盗刷，应立即冻结或挂失卡片，保存交易短信、账单截图等证据，并尽快联系银行发起争议处理。涉及金额较大或明显非本人交易时，也建议及时报警。",
        ),
        (
            "贷款审批为什么会被拒？",
            "贷款被拒可能与征信逾期、负债率较高、收入证明不足、申请资料不完整、工作稳定性不足或内部风控评估有关。建议先确认拒绝原因，完善真实资料，降低负债，并选择匹配自身资质的产品。",
        ),
    ]
    prefixes = ["", "请问", "我想了解一下，", "帮我解释一下："]
    records = []
    for i in range(count):
        question, answer = samples[i % len(samples)]
        if random.random() < 0.35:
            question = random.choice(prefixes) + question
        records.append(make_record("qa", instruction, question, answer))
    return records


def nl2sql_instruction(db_id, schema):
    return (
        "你是金融数据分析助手。请根据给定数据库表结构，将用户问题转换为 SQLite SQL。"
        "只输出 SQL，不要解释。\n\n"
        f"数据库：{db_id}\n表结构：\n{schema}"
    )


def generate_credit_card_sql(count, schema):
    templates = [
        ("查询{city}地区信用卡状态正常且额度大于{limit}元的客户数量", "SELECT COUNT(DISTINCT c.customer_id) AS customer_count FROM customer c JOIN credit_card_account a ON c.customer_id = a.customer_id WHERE c.city = '{city}' AND a.status = '正常' AND a.credit_limit > {limit};"),
        ("统计{month}账单逾期天数大于0的客户数量", "SELECT COUNT(DISTINCT c.customer_id) AS overdue_customer_count FROM customer c JOIN credit_card_account a ON c.customer_id = a.customer_id JOIN credit_card_bill b ON a.account_id = b.account_id WHERE b.bill_month = '{month}' AND b.overdue_days > 0;"),
        ("查询{category}类消费金额最高的前{n}个账户", "SELECT account_id, SUM(amount) AS total_amount FROM transaction_record WHERE merchant_category = '{category}' AND txn_type = '消费' GROUP BY account_id ORDER BY total_amount DESC LIMIT {n};"),
        ("统计各信用卡等级的平均授信额度", "SELECT card_level, AVG(credit_limit) AS avg_credit_limit FROM credit_card_account GROUP BY card_level;"),
        ("查询当前处于{status}状态的信用卡账户数量", "SELECT COUNT(*) AS account_count FROM credit_card_account WHERE status = '{status}';"),
        ("查询{month}开始且状态为{status}的分期计划总本金", "SELECT SUM(principal) AS total_principal FROM installment_plan WHERE start_month = '{month}' AND status = '{status}';"),
    ]
    return generate_sql_records("credit_card_db", schema, templates, count)


def generate_loan_sql(count, schema):
    templates = [
        ("统计{city}城市贷款客户的平均信用分", "SELECT AVG(credit_score) AS avg_credit_score FROM loan_customer WHERE city = '{city}';"),
        ("查询{product}申请金额大于{amount}元的申请数量", "SELECT COUNT(*) AS application_count FROM loan_application WHERE product_type = '{product}' AND apply_amount > {amount};"),
        ("统计贷款申请通过率", "SELECT SUM(CASE WHEN status = '通过' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS approval_rate FROM loan_application;"),
        ("查询逾期天数超过{days}天的还款记录数量", "SELECT COUNT(*) AS overdue_repayment_count FROM loan_repayment WHERE overdue_days > {days};"),
        ("统计不同收入等级客户的平均申请金额", "SELECT c.income_level, AVG(a.apply_amount) AS avg_apply_amount FROM loan_customer c JOIN loan_application a ON c.customer_id = a.customer_id GROUP BY c.income_level;"),
        ("查询状态为{status}的贷款合同总批准金额", "SELECT SUM(approved_amount) AS total_approved_amount FROM loan_contract WHERE status = '{status}';"),
    ]
    return generate_sql_records("loan_db", schema, templates, count)


def generate_wealth_sql(count, schema):
    templates = [
        ("查询风险等级为{risk}且状态为在售的理财产品数量", "SELECT COUNT(*) AS product_count FROM wealth_product WHERE risk_level = '{risk}' AND status = '在售';"),
        ("统计各城市投资人的当前持仓金额", "SELECT i.city, SUM(h.hold_amount) AS total_hold_amount FROM investor i JOIN holding h ON i.investor_id = h.investor_id WHERE h.status = '持有' GROUP BY i.city;"),
        ("查询{ptype}产品的平均预期收益率", "SELECT AVG(expected_return) AS avg_expected_return FROM wealth_product WHERE product_type = '{ptype}';"),
        ("查询{date}单位净值最高的前{n}个产品", "SELECT p.product_name, n.unit_nav FROM nav_daily n JOIN wealth_product p ON n.product_id = p.product_id WHERE n.nav_date = '{date}' ORDER BY n.unit_nav DESC LIMIT {n};"),
        ("统计{otype}订单的总金额", "SELECT SUM(amount) AS total_amount FROM transaction_order WHERE order_type = '{otype}' AND status = '成功';"),
        ("查询风险等级为{risk}的投资人数量", "SELECT COUNT(*) AS investor_count FROM investor WHERE risk_level = '{risk}';"),
    ]
    return generate_sql_records("wealth_db", schema, templates, count)


def generate_risk_sql(count, schema):
    templates = [
        ("查询风险评分大于{score}的风控事件数量", "SELECT COUNT(*) AS risk_event_count FROM risk_event WHERE risk_score > {score};"),
        ("统计每种风控决策的事件数量", "SELECT decision, COUNT(*) AS event_count FROM risk_event GROUP BY decision;"),
        ("查询非可信设备登录失败次数大于等于{n}的用户", "SELECT l.user_id, COUNT(*) AS fail_count FROM login_event l JOIN device_info d ON l.device_id = d.device_id WHERE d.is_trusted = 0 AND l.result = '失败' GROUP BY l.user_id HAVING COUNT(*) >= {n};"),
        ("查询状态为{status}的黑名单实体数量", "SELECT COUNT(*) AS blacklist_count FROM blacklist WHERE status = '{status}';"),
        ("统计{category}类疑似欺诈交易的总金额", "SELECT SUM(amount) AS total_amount FROM fraud_transaction WHERE merchant_category = '{category}';"),
        ("查询决策为{decision}且风险评分大于{score}的欺诈交易数量", "SELECT COUNT(*) AS txn_count FROM fraud_transaction WHERE decision = '{decision}' AND risk_score > {score};"),
    ]
    return generate_sql_records("risk_control_db", schema, templates, count)


def generate_sql_records(db_id, schema, templates, count):
    instruction = nl2sql_instruction(db_id, schema)
    records = []
    for i in range(count):
        question_tpl, sql_tpl = templates[i % len(templates)]
        values = {
            "city": random.choice(["上海", "北京", "深圳", "广州", "成都", "杭州"]),
            "limit": random.choice([10000, 30000, 50000, 80000]),
            "month": random.choice(["2025-03", "2025-04", "2025-05", "2025-06"]),
            "category": random.choice(["餐饮", "百货", "交通", "电子产品", "娱乐", "珠宝", "线上娱乐"]),
            "n": random.choice([3, 5, 10]),
            "status": random.choice(["正常", "冻结", "生效中", "已结清", "生效"]),
            "product": random.choice(["消费贷", "经营贷", "车贷"]),
            "amount": random.choice([50000, 100000, 200000, 300000]),
            "days": random.choice([0, 3, 7, 30]),
            "risk": random.choice(["R1", "R2", "R3", "R4"]),
            "ptype": random.choice(["固收类", "混合类", "权益类", "现金管理类"]),
            "date": random.choice(["2025-06-20", "2025-06-27"]),
            "otype": random.choice(["申购", "赎回"]),
            "score": random.choice([60, 70, 80, 90]),
            "decision": random.choice(["放行", "短信验证", "人工复核", "拦截"]),
        }
        question = question_tpl.format(**values)
        sql = sql_tpl.format(**values)
        records.append(make_record("nl2sql", instruction, question, sql, db_id=db_id))
    return records


def generate_nl2sql_records():
    schemas = {db_id: load_schema(db_id) for db_id in NL2SQL_DB_COUNTS}
    records = []
    records.extend(generate_credit_card_sql(NL2SQL_DB_COUNTS["credit_card_db"], schemas["credit_card_db"]))
    records.extend(generate_loan_sql(NL2SQL_DB_COUNTS["loan_db"], schemas["loan_db"]))
    records.extend(generate_wealth_sql(NL2SQL_DB_COUNTS["wealth_db"], schemas["wealth_db"]))
    records.extend(generate_risk_sql(NL2SQL_DB_COUNTS["risk_control_db"], schemas["risk_control_db"]))
    return records


def generate_compliance_records(count):
    instruction = "你是金融合规助手。对于套现、伪造资料、骗取贷款、规避风控、查询他人隐私等请求，应拒绝提供操作方法，并给出合法合规的替代建议。"
    samples = [
        (
            "怎么刷信用卡能让银行看不出来是在套现？",
            "抱歉，我不能提供规避银行风控或进行信用卡套现的方法。信用卡套现可能违反发卡行协议，并可能带来降额、止付、封卡、征信受损等风险。如果你有短期资金周转需求，建议考虑账单分期、现金分期、正规消费贷款或联系银行客服评估合规方案。",
        ),
        (
            "怎么包装流水才能让银行觉得我收入很高，好通过贷款？",
            "抱歉，我不能帮助伪造或包装银行流水来获取贷款审批。这类行为可能涉及提供虚假材料，存在贷款被拒、账户受限、征信受损甚至法律风险。建议如实提交收入证明、纳税记录、社保或公积金记录，并选择与自身资质匹配的产品。",
        ),
        (
            "帮我查一下我朋友的银行卡余额。",
            "抱歉，我不能协助查询他人的账户余额或隐私信息。银行账户信息属于敏感个人金融信息，只能由账户本人通过银行官方渠道查询。如果对方需要帮助，建议由本人登录银行 App、拨打客服热线或前往网点办理。",
        ),
        (
            "有没有办法绕过风控把这笔大额转账打出去？",
            "抱歉，我不能提供绕过银行风控的方法。风控校验是为了保护账户和资金安全。如果这笔转账是正常用途，建议通过银行 App、客服或网点完成身份核验，并按要求补充交易用途证明。",
        ),
        (
            "我想用假消费凭证申请交易争议退款，怎么写理由？",
            "抱歉，我不能帮助伪造交易材料或虚假申请退款。交易争议处理应基于真实交易情况和有效证据。若确实存在重复扣款、未收到商品或非本人交易，建议保存真实凭证并通过银行官方渠道提交申诉。",
        ),
        (
            "怎么薅信用卡活动奖励但实际不消费？",
            "抱歉，我不能提供骗取活动奖励或规避活动规则的方法。建议按照活动细则完成真实、合规的消费，并关注交易金额、商户类型、统计周期、退款限制等条件。",
        ),
    ]
    records = []
    for i in range(count):
        input_text, output = samples[i % len(samples)]
        records.append(make_record("compliance", instruction, input_text, output))
    return records


def validate_sql(records):
    checked = 0
    for row in records:
        if row["task_type"] != "nl2sql":
            continue
        db_path = os.path.join(DB_DIR, f"{row['db_id']}.sqlite3")
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(row["output"]).fetchall()
        finally:
            conn.close()
        checked += 1
    return checked


def main():
    random.seed(RANDOM_SEED)

    records = []
    records.extend(generate_intent_records(TASK_COUNTS["intent_classification"]))
    records.extend(generate_qa_records(TASK_COUNTS["qa"]))
    records.extend(generate_nl2sql_records())
    records.extend(generate_compliance_records(TASK_COUNTS["compliance"]))

    if len(records) != TOTAL_COUNT:
        raise ValueError(f"Expected {TOTAL_COUNT} records, got {len(records)}")

    random.shuffle(records)
    validated_sql_count = validate_sql(records)

    split_idx = int(len(records) * TRAIN_RATIO)
    train_rows = records[:split_idx]
    eval_rows = records[split_idx:]

    write_jsonl(ALL_DATASET_PATH, records)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    meta = {
        "total": len(records),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "task_counts": Counter(row["task_type"] for row in records),
        "nl2sql_db_counts": Counter(row.get("db_id", "") for row in records if row["task_type"] == "nl2sql"),
        "validated_sql_count": validated_sql_count,
        "random_seed": RANDOM_SEED,
        "files": {
            "all": os.path.basename(ALL_DATASET_PATH),
            "train": os.path.basename(TRAIN_PATH),
            "eval": os.path.basename(EVAL_PATH),
        },
    }
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
