# 金融领域 SFT / DPO 微调项目

本项目用于构建信用卡与金融业务场景数据，并基于 `Qwen/Qwen3-0.6B` 进行 LoRA SFT 与 DPO 偏好优化。训练日志、指标和 loss 曲线都保存在本地 `logs/` 目录。

## 环境准备

```bash
conda create -n dpo python=3.10
conda activate dpo
pip install -r requirements.txt
```

## 目录说明

```text
finance_dbs/                    # SQLite 业务数据库
logs/                           # 本地训练日志、metrics 和 loss 图
model/                          # ModelScope 模型缓存
output_lora/final_model/        # LoRA SFT 后的 adapter
output_dpo/final_model/         # DPO 后的 adapter
templates/                      # Web 聊天页面和数据看板模板
app.py                          # 原模型 / SFT LoRA / DPO LoRA 在线聊天对比与数据看板
generate_finance_sft_data.py    # 生成 6000 条金融 SFT 数据
generate_finance_dpo_data.py    # 生成 DPO 偏好样本
init_finance_sqlite.py          # 初始化四个 SQLite 数据库
local_logging.py                # 本地日志、metrics、loss 曲线工具
train.py                        # 全参微调入口
train_lora.py                   # LoRA SFT 训练入口
train_dpo.py                    # DPO 偏好优化训练入口
```

## 数据集

SFT 数据覆盖四类任务：

- 意图分类
- 金融知识问答
- NL2SQL
- 合规拒答

生成后主要文件：

- `finance_sft_dataset.jsonl`：完整数据集，6000 条
- `train_format.json`：训练集，5400 条
- `eval_format.json`：验证集，600 条
- `finance_sft_dataset_meta.json`：统计信息

DPO 数据文件：

- `finance_dpo_dataset.jsonl`：完整偏好数据，6000 条
- `dpo_train.jsonl`：DPO 训练集，5400 条
- `dpo_eval.jsonl`：DPO 验证集，600 条
- `finance_dpo_dataset_meta.json`：统计信息

## SQLite 数据库

NL2SQL 场景使用四个本地 SQLite 数据库：

- `credit_card_db.sqlite3`：信用卡账户、账单、交易、还款等
- `loan_db.sqlite3`：贷款申请、放款、还款计划等
- `wealth_db.sqlite3`：理财产品、持仓、风险等级等
- `risk_control_db.sqlite3`：风控规则、风险事件、黑名单等

初始化数据库：

```bash
python init_finance_sqlite.py
```

## 生成数据

生成 SFT 数据：

```bash
python generate_finance_sft_data.py
```

生成 DPO 偏好数据：

```bash
python generate_finance_dpo_data.py
```

## 训练

LoRA SFT：

```bash
python train_lora.py
```

训练完成后会保存到：

```text
output_lora/final_model/
```

DPO：

```bash
python train_dpo.py
```

`train_dpo.py` 默认加载 `./output_lora/final_model` 作为 SFT 基础 adapter，再进行 DPO 偏好优化。训练完成后保存到：

```text
output_dpo/final_model/
```

## 本地日志

训练过程会在 `logs/` 下生成本地运行目录：

```text
logs/<timestamp>_<run_name>/
  config.json
  train.log
  metrics.jsonl
  loss_curve.png
  predictions.txt
```

说明：

- `train.log`：控制台输出
- `metrics.jsonl`：Trainer 的 loss、eval_loss 等指标
- `loss_curve.png`：训练 / 验证 loss 曲线
- `predictions.txt`：训练结束后的抽样输出

## Web 聊天对比

`app.py` 提供一个本地 Flask 页面，首页用于在线对比三个模型在同一金融业务问题下的回答差异：

- 原模型：`Qwen/Qwen3-0.6B`
- LoRA SFT 后模型：`./output_lora/final_model`
- DPO 后模型：`./output_dpo/final_model`

启动：

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000/
```

页面支持：

- 选择单个模型聊天
- 一次请求同时对比三个模型
- 自定义系统提示词、temperature、top_p 和最大生成长度
- 释放已加载模型缓存
- 右上角进入金融 SFT 数据看板

说明：DPO 选项需要先运行 `python train_dpo.py`，生成 `./output_dpo/final_model` 后才能正常回答。

数据看板地址：

```text
http://127.0.0.1:5000/dashboard
```

看板支持查看 SFT 数据量、任务分布、样本预览和四个 SQLite 业务库结构。

## 常用检查

```bash
python -m py_compile generate_finance_sft_data.py generate_finance_dpo_data.py train.py train_lora.py train_dpo.py init_finance_sqlite.py app.py
```

## 注意事项

- `train.py` 和 `train_lora.py` 直接读取本地 `train_format.json` / `eval_format.json`。
- 如果重新生成数据，旧的 tokenized 缓存目录 `processed_finance_train_dataset` / `processed_finance_eval_dataset` 建议删除后再训练。
- NL2SQL 数据中的 SQL 已经过 SQLite 执行校验，但只代表样例库上的语法和可执行性，不代表生产库可直接使用。
- 合规拒答数据用于让模型拒绝套现、伪造材料、规避风控、查询他人隐私等请求。
- 本项目仅用于学习、研究和内部实验，不构成真实金融业务建议。
