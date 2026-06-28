# Qwen Credit LoRA DPO

金融信用卡业务场景的 SFT / DPO 微调示例项目。项目基于 `Qwen/Qwen3-0.6B`，覆盖金融意图分类、知识问答、NL2SQL 和合规拒答四类任务，并提供本地 Flask 页面用于对比原模型、LoRA SFT 模型和 DPO 模型的回答差异。

本仓库不包含模型权重、训练日志和本地缓存；这些文件会在本地运行时生成。

## 功能概览

- 构建约 6000 条金融领域 SFT 数据
- 构建约 6000 条 DPO 偏好数据
- 初始化 4 个金融业务 SQLite 样例库
- 支持 LoRA SFT 训练
- 支持基于 SFT LoRA 的 DPO 训练
- 训练过程本地输出日志、metrics 和 loss 曲线
- Web 页面支持三模型聊天对比
- Web 页面支持金融 SFT 数据看板
- Web 调用会记录请求、模型、问题、思考内容、原始输出和耗时

## 环境准备

```bash
conda create -n dpo python=3.10
conda activate dpo
pip install -r requirements.txt
```

说明：训练和推理需要 PyTorch。不同机器的 CUDA 版本不同，建议按本机环境安装合适的 `torch` 版本。

## 目录结构

仓库内主要文件：

```text
data/                           # SFT / DPO 数据文件
data_prep/                      # 数据库初始化与数据生成模块
finance_dbs/                    # SQLite 样例数据库
templates/                      # Web 页面模板
app.py                          # Flask Web 服务
prepare_data.py                 # 统一数据准备入口
local_logging.py                # 本地日志与 loss 曲线工具
train.py                        # 全参微调入口
train_lora.py                   # LoRA SFT 训练入口
train_dpo.py                    # DPO 训练入口
requirements.txt
```

本地运行后会生成但不会提交到 GitHub：

```text
cache/                          # tokenized 数据缓存
logs/                           # 训练日志、Web 调用日志、loss 曲线
model/                          # ModelScope 模型缓存
output/                         # 全参训练输出
output_lora/                    # LoRA SFT 输出
output_dpo/                     # DPO 输出
```

## 数据

SFT 数据覆盖四类任务：

- 意图分类
- 金融知识问答
- NL2SQL
- 合规拒答

主要数据文件：

```text
data/finance_sft_dataset.jsonl
data/train_format.json
data/eval_format.json
data/finance_sft_dataset_meta.json
data/finance_dpo_dataset.jsonl
data/dpo_train.jsonl
data/dpo_eval.jsonl
data/finance_dpo_dataset_meta.json
```

NL2SQL 使用 4 个本地 SQLite 样例库：

```text
finance_dbs/credit_card_db.sqlite3
finance_dbs/loan_db.sqlite3
finance_dbs/wealth_db.sqlite3
finance_dbs/risk_control_db.sqlite3
```

## 准备数据

推荐使用统一入口：

```bash
python prepare_data.py
```

执行顺序：

1. 初始化 4 个 SQLite 数据库
2. 生成 6000 条 SFT 数据
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

训练完成后输出：

```text
output_lora/final_model/
```

DPO：

```bash
python train_dpo.py
```

`train_dpo.py` 默认加载 `./output_lora/final_model`，再进行 DPO 偏好优化。训练完成后输出：

```text
output_dpo/final_model/
```

## 本地日志

训练日志会写入：

```text
logs/<timestamp>_<run_name>/
  config.json
  train.log
  metrics.jsonl
  loss_curve.png
  predictions.txt
```

Web 调用日志固定写入：

```text
logs/web_chat.log
logs/web_chat.jsonl
```

`web_chat.jsonl` 会记录：

- request id
- 调用模型
- 用户问题
- system prompt
- generation 参数
- raw output
- thinking
- answer
- error traceback
- 耗时

## Web 使用

启动：

```bash
python app.py
```

聊天对比页面：

```text
http://127.0.0.1:5000/
```

数据看板：

```text
http://127.0.0.1:5000/dashboard
```

聊天页支持：

- 选择原模型、LoRA SFT 后模型、DPO 后模型
- 单模型聊天或三模型同时对比
- 自定义 system prompt、temperature、top_p、最大生成长度
- 首次加载/生成时显示进度与等待秒数
- 对 NL2SQL 问题自动补充本地 SQLite schema，减少误拒答
- 释放已加载模型缓存

模型路径：

```text
原模型：Qwen/Qwen3-0.6B，本地缓存优先读取 model/Qwen/Qwen3-0___6B
SFT：output_lora/final_model/
DPO：output_dpo/final_model/
```

## 开源说明

`.gitignore` 已排除以下本地文件：

```text
cache/
logs/
model/
output/
output_lora/
output_dpo/
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
```

因此 GitHub 仓库只包含代码、样例数据和 SQLite 样例库，不包含训练后的模型权重。

## 常用检查

```bash
python -m py_compile prepare_data.py data_prep/generate_finance_sft_data.py data_prep/generate_finance_dpo_data.py data_prep/init_finance_sqlite.py train.py train_lora.py train_dpo.py app.py
```

## 注意事项

- `train.py` 和 `train_lora.py` 读取 `data/train_format.json` / `data/eval_format.json`。
- 重新生成 SFT 数据后，建议运行 `python prepare_data.py --clean-cache` 清理 tokenized 缓存。
- Web 对比 DPO 模型需要先生成 `output_dpo/final_model/`。
- NL2SQL 样例只代表本地 SQLite 样例库上的可执行性，不代表生产库可直接使用。
- 合规拒答数据用于让模型拒绝套现、伪造材料、规避风控、查询他人隐私等请求。
- 本项目仅用于学习、研究和内部实验，不构成真实金融业务建议。
