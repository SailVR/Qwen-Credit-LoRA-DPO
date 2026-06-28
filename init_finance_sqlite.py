import os
import sqlite3
from datetime import datetime


DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finance_dbs")


def connect(db_name):
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(os.path.join(DB_DIR, db_name))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def execute_many(conn, sql_statements):
    for sql in sql_statements:
        conn.execute(sql)


def seed(conn, table, columns, rows):
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({column_sql}) VALUES ({placeholders})",
        rows,
    )


def init_credit_card_db():
    conn = connect("credit_card_db.sqlite3")
    execute_many(
        conn,
        [
            """
            CREATE TABLE IF NOT EXISTS customer (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                register_date TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS credit_card_account (
                account_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                card_level TEXT NOT NULL,
                credit_limit REAL NOT NULL,
                available_limit REAL NOT NULL,
                status TEXT NOT NULL,
                open_date TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS credit_card_bill (
                bill_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                bill_month TEXT NOT NULL,
                statement_amount REAL NOT NULL,
                minimum_payment REAL NOT NULL,
                due_date TEXT NOT NULL,
                paid_amount REAL NOT NULL,
                overdue_days INTEGER NOT NULL,
                FOREIGN KEY (account_id) REFERENCES credit_card_account(account_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS transaction_record (
                txn_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                txn_time TEXT NOT NULL,
                merchant_name TEXT NOT NULL,
                merchant_category TEXT NOT NULL,
                amount REAL NOT NULL,
                txn_type TEXT NOT NULL,
                city TEXT NOT NULL,
                FOREIGN KEY (account_id) REFERENCES credit_card_account(account_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS installment_plan (
                plan_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                principal REAL NOT NULL,
                periods INTEGER NOT NULL,
                fee_rate REAL NOT NULL,
                start_month TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (account_id) REFERENCES credit_card_account(account_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS points_record (
                record_id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                change_type TEXT NOT NULL,
                points INTEGER NOT NULL,
                source TEXT NOT NULL,
                change_time TEXT NOT NULL,
                FOREIGN KEY (account_id) REFERENCES credit_card_account(account_id)
            )
            """,
        ],
    )

    seed(
        conn,
        "customer",
        ["customer_id", "name", "city", "age", "gender", "risk_level", "register_date"],
        [
            (1, "张明", "上海", 34, "男", "低", "2023-01-12"),
            (2, "李娜", "北京", 29, "女", "中", "2023-03-18"),
            (3, "王强", "深圳", 42, "男", "中", "2022-11-05"),
            (4, "陈悦", "广州", 31, "女", "低", "2024-02-20"),
            (5, "赵磊", "上海", 45, "男", "高", "2022-06-09"),
        ],
    )
    seed(
        conn,
        "credit_card_account",
        ["account_id", "customer_id", "card_level", "credit_limit", "available_limit", "status", "open_date"],
        [
            (101, 1, "白金卡", 80000, 52000, "正常", "2023-01-20"),
            (102, 2, "金卡", 30000, 18000, "正常", "2023-04-01"),
            (103, 3, "白金卡", 60000, 15000, "正常", "2022-11-20"),
            (104, 4, "普卡", 15000, 9000, "正常", "2024-03-01"),
            (105, 5, "金卡", 50000, 2000, "冻结", "2022-07-01"),
        ],
    )
    seed(
        conn,
        "credit_card_bill",
        ["bill_id", "account_id", "bill_month", "statement_amount", "minimum_payment", "due_date", "paid_amount", "overdue_days"],
        [
            (1001, 101, "2025-05", 12680.50, 1268.05, "2025-06-10", 12680.50, 0),
            (1002, 102, "2025-05", 5300.00, 530.00, "2025-06-12", 530.00, 0),
            (1003, 103, "2025-05", 22100.00, 2210.00, "2025-06-15", 1200.00, 5),
            (1004, 104, "2025-05", 1680.00, 168.00, "2025-06-08", 1680.00, 0),
            (1005, 105, "2025-05", 48700.00, 4870.00, "2025-06-10", 0.00, 18),
        ],
    )
    seed(
        conn,
        "transaction_record",
        ["txn_id", "account_id", "txn_time", "merchant_name", "merchant_category", "amount", "txn_type", "city"],
        [
            (2001, 101, "2025-05-02 12:30:00", "海岸餐厅", "餐饮", 328.00, "消费", "上海"),
            (2002, 101, "2025-05-08 20:10:00", "星河商场", "百货", 2680.00, "消费", "上海"),
            (2003, 102, "2025-05-11 09:20:00", "云票出行", "交通", 86.00, "消费", "北京"),
            (2004, 103, "2025-05-15 21:45:00", "南山数码", "电子产品", 12999.00, "消费", "深圳"),
            (2005, 105, "2025-05-21 23:58:00", "境外线上娱乐", "娱乐", 18000.00, "消费", "澳门"),
        ],
    )
    seed(
        conn,
        "installment_plan",
        ["plan_id", "account_id", "principal", "periods", "fee_rate", "start_month", "status"],
        [
            (3001, 101, 12000.00, 12, 0.006, "2025-05", "生效中"),
            (3002, 103, 18000.00, 24, 0.005, "2025-04", "生效中"),
            (3003, 102, 5000.00, 6, 0.007, "2025-03", "已结清"),
        ],
    )
    seed(
        conn,
        "points_record",
        ["record_id", "account_id", "change_type", "points", "source", "change_time"],
        [
            (4001, 101, "增加", 328, "餐饮消费", "2025-05-02 12:31:00"),
            (4002, 101, "增加", 2680, "百货消费", "2025-05-08 20:12:00"),
            (4003, 102, "增加", 86, "交通消费", "2025-05-11 09:21:00"),
            (4004, 103, "兑换", -5000, "礼品兑换", "2025-05-20 10:00:00"),
        ],
    )

    conn.commit()
    conn.close()


def init_loan_db():
    conn = connect("loan_db.sqlite3")
    execute_many(
        conn,
        [
            """
            CREATE TABLE IF NOT EXISTS loan_customer (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                age INTEGER NOT NULL,
                income_level TEXT NOT NULL,
                credit_score INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS loan_application (
                application_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                product_type TEXT NOT NULL,
                apply_amount REAL NOT NULL,
                apply_date TEXT NOT NULL,
                status TEXT NOT NULL,
                reject_reason TEXT,
                FOREIGN KEY (customer_id) REFERENCES loan_customer(customer_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS loan_contract (
                contract_id INTEGER PRIMARY KEY,
                application_id INTEGER NOT NULL,
                approved_amount REAL NOT NULL,
                interest_rate REAL NOT NULL,
                term_months INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (application_id) REFERENCES loan_application(application_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS loan_repayment (
                repayment_id INTEGER PRIMARY KEY,
                contract_id INTEGER NOT NULL,
                due_date TEXT NOT NULL,
                repay_date TEXT,
                due_amount REAL NOT NULL,
                paid_amount REAL NOT NULL,
                overdue_days INTEGER NOT NULL,
                FOREIGN KEY (contract_id) REFERENCES loan_contract(contract_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS collateral (
                collateral_id INTEGER PRIMARY KEY,
                contract_id INTEGER NOT NULL,
                collateral_type TEXT NOT NULL,
                valuation_amount REAL NOT NULL,
                FOREIGN KEY (contract_id) REFERENCES loan_contract(contract_id)
            )
            """,
        ],
    )
    seed(
        conn,
        "loan_customer",
        ["customer_id", "name", "city", "age", "income_level", "credit_score"],
        [
            (1, "刘洋", "上海", 36, "高", 735),
            (2, "周敏", "杭州", 30, "中", 690),
            (3, "孙杰", "成都", 41, "中", 650),
            (4, "何静", "北京", 28, "低", 610),
        ],
    )
    seed(
        conn,
        "loan_application",
        ["application_id", "customer_id", "product_type", "apply_amount", "apply_date", "status", "reject_reason"],
        [
            (501, 1, "消费贷", 200000, "2025-01-15", "通过", None),
            (502, 2, "经营贷", 500000, "2025-02-20", "通过", None),
            (503, 3, "消费贷", 150000, "2025-03-12", "拒绝", "负债率过高"),
            (504, 4, "车贷", 180000, "2025-04-18", "审批中", None),
        ],
    )
    seed(
        conn,
        "loan_contract",
        ["contract_id", "application_id", "approved_amount", "interest_rate", "term_months", "start_date", "status"],
        [
            (601, 501, 180000, 0.045, 36, "2025-01-25", "正常"),
            (602, 502, 450000, 0.052, 24, "2025-03-01", "正常"),
        ],
    )
    seed(
        conn,
        "loan_repayment",
        ["repayment_id", "contract_id", "due_date", "repay_date", "due_amount", "paid_amount", "overdue_days"],
        [
            (701, 601, "2025-05-25", "2025-05-24", 5350.00, 5350.00, 0),
            (702, 601, "2025-06-25", None, 5350.00, 0.00, 3),
            (703, 602, "2025-05-30", "2025-06-10", 19800.00, 19800.00, 11),
            (704, 602, "2025-06-30", None, 19800.00, 0.00, 0),
        ],
    )
    seed(
        conn,
        "collateral",
        ["collateral_id", "contract_id", "collateral_type", "valuation_amount"],
        [
            (801, 602, "商铺", 980000),
            (802, 601, "车辆", 220000),
        ],
    )
    conn.commit()
    conn.close()


def init_wealth_db():
    conn = connect("wealth_db.sqlite3")
    execute_many(
        conn,
        [
            """
            CREATE TABLE IF NOT EXISTS investor (
                investor_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                open_date TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS wealth_product (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                product_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                expected_return REAL NOT NULL,
                status TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holding (
                holding_id INTEGER PRIMARY KEY,
                investor_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                hold_amount REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                redeem_date TEXT,
                status TEXT NOT NULL,
                FOREIGN KEY (investor_id) REFERENCES investor(investor_id),
                FOREIGN KEY (product_id) REFERENCES wealth_product(product_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS transaction_order (
                order_id INTEGER PRIMARY KEY,
                investor_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                amount REAL NOT NULL,
                order_time TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (investor_id) REFERENCES investor(investor_id),
                FOREIGN KEY (product_id) REFERENCES wealth_product(product_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nav_daily (
                product_id INTEGER NOT NULL,
                nav_date TEXT NOT NULL,
                unit_nav REAL NOT NULL,
                accumulated_nav REAL NOT NULL,
                PRIMARY KEY (product_id, nav_date),
                FOREIGN KEY (product_id) REFERENCES wealth_product(product_id)
            )
            """,
        ],
    )
    seed(
        conn,
        "investor",
        ["investor_id", "name", "city", "risk_level", "open_date"],
        [
            (1, "邓佳", "上海", "R3", "2023-05-01"),
            (2, "林峰", "深圳", "R4", "2022-09-12"),
            (3, "郭婷", "北京", "R2", "2024-01-08"),
            (4, "马超", "成都", "R3", "2023-12-20"),
        ],
    )
    seed(
        conn,
        "wealth_product",
        ["product_id", "product_name", "product_type", "risk_level", "expected_return", "status"],
        [
            (901, "稳健月月盈", "固收类", "R2", 0.032, "在售"),
            (902, "均衡优选一年", "混合类", "R3", 0.046, "在售"),
            (903, "科技成长精选", "权益类", "R4", 0.072, "停售"),
            (904, "现金添利", "现金管理类", "R1", 0.021, "在售"),
        ],
    )
    seed(
        conn,
        "holding",
        ["holding_id", "investor_id", "product_id", "hold_amount", "purchase_date", "redeem_date", "status"],
        [
            (10001, 1, 902, 120000, "2025-01-10", None, "持有"),
            (10002, 2, 903, 80000, "2024-12-01", None, "持有"),
            (10003, 3, 901, 50000, "2025-03-12", None, "持有"),
            (10004, 4, 904, 30000, "2025-05-20", None, "持有"),
        ],
    )
    seed(
        conn,
        "transaction_order",
        ["order_id", "investor_id", "product_id", "order_type", "amount", "order_time", "status"],
        [
            (11001, 1, 902, "申购", 120000, "2025-01-10 10:20:00", "成功"),
            (11002, 2, 903, "申购", 80000, "2024-12-01 11:00:00", "成功"),
            (11003, 3, 901, "申购", 50000, "2025-03-12 14:35:00", "成功"),
            (11004, 1, 902, "赎回", 20000, "2025-06-15 09:10:00", "处理中"),
        ],
    )
    seed(
        conn,
        "nav_daily",
        ["product_id", "nav_date", "unit_nav", "accumulated_nav"],
        [
            (901, "2025-06-20", 1.0120, 1.0420),
            (901, "2025-06-27", 1.0132, 1.0432),
            (902, "2025-06-20", 1.0560, 1.1260),
            (902, "2025-06-27", 1.0615, 1.1315),
            (903, "2025-06-20", 0.9820, 1.2100),
            (903, "2025-06-27", 1.0040, 1.2320),
            (904, "2025-06-20", 1.0008, 1.0108),
            (904, "2025-06-27", 1.0012, 1.0112),
        ],
    )
    conn.commit()
    conn.close()


def init_risk_control_db():
    conn = connect("risk_control_db.sqlite3")
    execute_many(
        conn,
        [
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id INTEGER PRIMARY KEY,
                city TEXT NOT NULL,
                age INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                register_date TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS device_info (
                device_id TEXT PRIMARY KEY,
                device_type TEXT NOT NULL,
                first_seen_time TEXT NOT NULL,
                is_trusted INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS login_event (
                event_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                login_time TEXT NOT NULL,
                device_id TEXT NOT NULL,
                ip_city TEXT NOT NULL,
                result TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profile(user_id),
                FOREIGN KEY (device_id) REFERENCES device_info(device_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_event (
                event_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                event_time TEXT NOT NULL,
                event_type TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                decision TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS blacklist (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_time TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS fraud_transaction (
                txn_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                txn_time TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant_category TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                decision TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
            )
            """,
        ],
    )
    seed(
        conn,
        "user_profile",
        ["user_id", "city", "age", "risk_level", "register_date"],
        [
            (1, "上海", 34, "低", "2023-01-12"),
            (2, "北京", 29, "中", "2023-03-18"),
            (3, "深圳", 42, "高", "2022-11-05"),
            (4, "广州", 31, "中", "2024-02-20"),
        ],
    )
    seed(
        conn,
        "device_info",
        ["device_id", "device_type", "first_seen_time", "is_trusted"],
        [
            ("D1001", "iPhone", "2024-01-10 08:00:00", 1),
            ("D1002", "Android", "2025-06-01 12:00:00", 0),
            ("D1003", "Web", "2025-05-20 10:30:00", 0),
            ("D1004", "iPad", "2024-09-18 20:20:00", 1),
        ],
    )
    seed(
        conn,
        "login_event",
        ["event_id", "user_id", "login_time", "device_id", "ip_city", "result"],
        [
            (12001, 1, "2025-06-27 09:00:00", "D1001", "上海", "成功"),
            (12002, 2, "2025-06-27 22:10:00", "D1002", "哈尔滨", "失败"),
            (12003, 2, "2025-06-27 22:12:00", "D1002", "哈尔滨", "失败"),
            (12004, 2, "2025-06-27 22:14:00", "D1002", "哈尔滨", "失败"),
            (12005, 3, "2025-06-27 23:50:00", "D1003", "澳门", "成功"),
        ],
    )
    seed(
        conn,
        "risk_event",
        ["event_id", "user_id", "event_time", "event_type", "risk_score", "decision"],
        [
            (13001, 1, "2025-06-27 09:01:00", "异地登录", 35, "放行"),
            (13002, 2, "2025-06-27 22:15:00", "连续登录失败", 82, "拦截"),
            (13003, 3, "2025-06-27 23:52:00", "高风险交易", 91, "人工复核"),
            (13004, 4, "2025-06-26 18:30:00", "设备变更", 60, "短信验证"),
        ],
    )
    seed(
        conn,
        "blacklist",
        ["entity_id", "entity_type", "reason", "created_time", "status"],
        [
            ("M9001", "商户", "疑似套现商户", "2025-05-01 10:00:00", "生效"),
            ("D1003", "设备", "多账户异常登录", "2025-06-20 11:30:00", "生效"),
            ("IP8888", "IP", "撞库攻击来源", "2025-06-25 08:20:00", "生效"),
        ],
    )
    seed(
        conn,
        "fraud_transaction",
        ["txn_id", "user_id", "txn_time", "amount", "merchant_category", "risk_score", "decision"],
        [
            (14001, 1, "2025-06-27 13:20:00", 680.00, "餐饮", 20, "放行"),
            (14002, 2, "2025-06-27 22:16:00", 9800.00, "电子产品", 76, "短信验证"),
            (14003, 3, "2025-06-27 23:55:00", 30000.00, "线上娱乐", 95, "拦截"),
            (14004, 4, "2025-06-26 19:10:00", 12000.00, "珠宝", 81, "人工复核"),
        ],
    )
    conn.commit()
    conn.close()


def main():
    init_credit_card_db()
    init_loan_db()
    init_wealth_db()
    init_risk_control_db()
    print(f"Initialized finance SQLite databases at: {DB_DIR}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
