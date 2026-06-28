# 金融领域 SFT / DPO 微调项目

本项目用于构建信用卡与金融业务场景数据，并基于 `Qwen/Qwen3-0.6B` 进行 LoRA SFT 与 DPO 偏好优化。训练日志、指标和 loss 曲线全部保存在本地 `logs/` 目录。

## 环境准备

```bash
conda create -n dpo python=3.10
conda activate dpo
pip install -r requirements.txt
```

## 项目结构

```text
cache/                          # tokenized 数据缓存
data/                           # SFT / DPO 的 json 和 jsonl 数据文件
data_prep/                      # 数据库初始化与数据生成模块
finance_dbs/                    # 四个金融业务 SQLite 数据库
logs/                           # 本地训练日志、metrics、loss 曲线
model/                          # ModelScope 模型缓存
output_lora/final_model/        # LoRA SFT 训练后的 adapter
output_dpo/final_model/         # DPO 训练后的 adapter，训练完成后生成
templates/index.html            # 三模型聊天对比页面
templates/dashboard.html        # 金融 SFT 数据看板页面
app.py                          # Flask Web 服务
prepare_data.py                 # 统一初始化数据库并生成 SFT / DPO 数据
data_prep/generate_finance_sft_data.py
data_prep/generate_finance_dpo_data.py
data_prep/init_finance_sqlite.py
local_logging.py                # 本地日志与 loss 曲线工具
train.py                        # 全参微调入口
train_lora.py                   # LoRA SFT 训练入口
train_dpo.py                    # DPO 训练入口
```

`health_data.db` 是旧医疗示例库，当前金融项目不需要；如果出现可以删除。

## 数据设计

SFT 数据共 6000 条，覆盖四类任务：

- 意图分类
- 金融知识问答
- NL2SQL
- 合规拒答

主要数据文件都保存在 `data/` 目录：

- `data/finance_sft_dataset.jsonl`：完整 SFT 数据集，6000 条
- `data/train_format.json`：SFT 训练集，5400 条
- `data/eval_format.json`：SFT 验证集，600 条
- `data/finance_sft_dataset_meta.json`：SFT 数据统计信息
- `data/finance_dpo_dataset.jsonl`：完整 DPO 偏好数据，6000 条
- `data/dpo_train.jsonl`：DPO 训练集，5400 条
- `data/dpo_eval.jsonl`：DPO 验证集，600 条
- `data/finance_dpo_dataset_meta.json`：DPO 数据统计信息

## SQLite 数据库

NL2SQL 场景使用四个本地 SQLite 数据库：

- `credit_card_db.sqlite3`：信用卡账户、账单、交易、还款等
- `loan_db.sqlite3`：贷款申请、放款、还款计划等
- `wealth_db.sqlite3`：理财产品、持仓、风险等级等
- `risk_control_db.sqlite3`：风控规则、风险事件、黑名单等

统一初始化数据库并生成训练数据：

```bash
python prepare_data.py
```

## 准备数据

推荐直接使用统一入口：

```bash
python prepare_data.py
```

执行顺序为：

1. 初始化四个 SQLite 数据库
2. 生成 6000 条金融 SFT 数据
3. 基于 SFT 数据生成 DPO 偏好数据

可选参数：

```bash
python prepare_data.py --skip-db
python prepare_data.py --skip-sft
python prepare_data.py --skip-dpo
python prepare_data.py --clean-cache
```

## 训练

LoRA SFT：

```bash
python train_lora.py
```

训练完成后保存到：

```text
output_lora/final_model/
```

DPO：

```bash
python train_dpo.py
```

`train_dpo.py` 默认加载 `./output_lora/final_model` 作为 SFT 后模型，再进行 DPO 偏好优化。训练完成后保存到：

```text
output_dpo/final_model/
```

## 本地日志

每次训练会在 `logs/` 下生成一个运行目录：

```text
logs/<timestamp>_<run_name>/
  config.json
  train.log
  metrics.jsonl
  loss_curve.png
  predictions.txt
```

文件说明：

- `config.json`：本次训练配置
- `train.log`：控制台输出日志
- `metrics.jsonl`：Trainer 记录的 loss、eval_loss 等指标
- `loss_curve.png`：训练 / 验证 loss 曲线
- `predictions.txt`：训练结束后的抽样输出

Web 调用日志固定写入：

```text
logs/web_chat.log
logs/web_chat.jsonl
```

- `web_chat.log`：控制台可读日志，会打印请求 id、调用模型、输入问题、思考内容、最终回答和耗时
- `web_chat.jsonl`：结构化日志，包含 system prompt、generation 参数、raw output、thinking、answer、error traceback 等字段

## Web 页面

启动本地 Web 服务：

```bash
python app.py
```

聊天对比首页：

```text
http://127.0.0.1:5000/
```

首页支持对比三个模型：

- 原模型：`Qwen/Qwen3-0.6B`
- LoRA SFT 后模型：`./output_lora/final_model`
- DPO 后模型：`./output_dpo/final_model`

页面功能：

- 单独选择一个模型聊天
- 一次请求同时对比三个模型
- 自定义系统提示词、temperature、top_p、最大生成长度
- 释放已加载模型缓存
- 右上角进入金融 SFT 数据看板

金融 SFT 数据看板：

```text
http://127.0.0.1:5000/dashboard
```

看板支持查看：

- SFT 数据集、训练集、验证集数量
- 四类任务分布
- 数据样本预览
- 四个 SQLite 业务库的表结构和行数

## 常用检查

```bash
python -m py_compile prepare_data.py data_prep/generate_finance_sft_data.py data_prep/generate_finance_dpo_data.py data_prep/init_finance_sqlite.py train.py train_lora.py train_dpo.py app.py
```

## 注意事项

- `train.py` 和 `train_lora.py` 直接读取 `data/train_format.json` / `data/eval_format.json`。
- 如果重新生成 SFT 数据，建议删除旧的 `cache/processed_finance_train_dataset` / `cache/processed_finance_eval_dataset` 后再训练；也可以运行 `python prepare_data.py --clean-cache`。
- Web 里的 DPO 模型对比需要 `output_dpo/final_model` 存在；如果只有 `output_dpo/Qwen3-0.6B`，说明训练还没有完成最终保存。
- NL2SQL 数据中的 SQL 已经过 SQLite 执行校验，但只代表样例库上的语法和可执行性，不代表生产库可直接使用。
- 合规拒答数据用于让模型拒绝套现、伪造材料、规避风控、查询他人隐私等请求。
- 本项目仅用于学习、研究和内部实验，不构成真实金融业务建议。
