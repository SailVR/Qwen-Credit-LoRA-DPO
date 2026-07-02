# Qwen Credit LoRA DPO

金融信用卡业务场景的 SFT / DPO 微调示例项目。项目基于 `Qwen/Qwen3-0.6B`，覆盖金融意图分类、知识问答、NL2SQL 和合规拒答四类任务，并提供本地 Flask 页面用于对比原模型、LoRA SFT 模型和 DPO 模型的回答差异。

本仓库不包含模型权重、训练日志和本地缓存；这些文件会在本地运行时生成。

## 功能概览

- 构建约 6000 条金融领域 SFT 数据
- 构建约 6000 条 DPO 偏好数据
- 初始化 4 个金融业务 SQLite 样例库
- 支持 LoRA SFT 训练
- 支持基于 SFT LoRA 的 DPO 训练
- 支持训练参数命令行配置，便于对比不同学习率、epoch、LoRA rank 和 DPO beta
- 支持 SFT / DPO 数据质量检查和项目 smoke test
- 训练过程本地输出日志、metrics 和 loss 曲线
- 训练结束后沉淀 summary、最终评估指标、最终模型路径和样例输出
- Web 页面支持三模型聊天对比
- Web 推理默认只缓存最近使用的模型，切换模型时自动释放旧模型显存
- Web 页面支持金融 SFT 数据看板
- Web 调用会记录请求、模型、问题、思考内容、原始输出和耗时

## 环境准备

```bash
conda create -n dpo python=3.10
conda activate dpo
```

PyTorch 需要单独安装，版本必须与本机 CUDA 驱动匹配。下面是 CUDA 12.4 的示例命令：

```bash
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu124
```

然后安装项目依赖：

```bash
pip install -r requirements.txt
```

如果你的 CUDA 版本不是 12.4，请到 PyTorch 官网选择对应的安装命令。

## 目录结构

仓库内主要文件：

```text
data/                           # SFT / DPO 数据文件
checks/                         # 数据质量检查和项目 smoke test
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

## 数据检查

项目提供轻量级数据质量检查脚本：

```bash
python checks/check_sft_data.py
python checks/check_dpo_data.py
```

`check_sft_data.py` 会检查 JSONL 格式、必填字段、任务类型分布、重复样本、NL2SQL SQL 可执行性，以及 `DROP`、`DELETE`、`UPDATE` 等危险 SQL 关键词。

`check_dpo_data.py` 会检查 `prompt`、`chosen`、`rejected` 是否完整，`chosen` 和 `rejected` 是否完全相同，偏好对长度比例是否异常，以及任务类型、数据库和重复样本分布。

项目级快速检查：

```bash
python checks/smoke_test.py
```

`smoke_test.py` 会检查数据文件、JSONL 字段、SQLite 数据库、核心 Python 入口和模型输出目录状态。若依赖未安装，可跳过入口导入：

```bash
python checks/smoke_test.py --skip-imports
```

## 训练

LoRA SFT：

```bash
python train_lora.py
```

常用参数示例：

```bash
python train_lora.py --epochs 2 --learning-rate 3e-5 --lora-r 16 --max-length 4096
```

训练完成后输出：

```text
output_lora/final_model/
```

DPO：

```bash
python train_dpo.py
```

`train_dpo.py` 默认加载 `./output_lora/final_model`，再进行 DPO 偏好优化。

常用参数示例：

```bash
python train_dpo.py --epochs 1 --learning-rate 5e-6 --beta 0.05 --max-prompt-length 2048 --max-length 3072
```

训练完成后输出：

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
  summary.json
  predictions.txt
  preference_samples.jsonl
```

其中：

- `config.json` 记录本次训练配置和命令行参数
- `metrics.jsonl` 记录 step 级别训练 / 验证指标
- `loss_curve.png` 绘制 train loss / eval loss 曲线
- `summary.json` 记录最终模型路径、数据行数、global step、最新 / 最佳 loss、最终 train / eval metrics
- `predictions.txt` 记录 LoRA SFT 训练后的样例预测
- `preference_samples.jsonl` 记录 DPO eval 集偏好样例快照

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
- 默认最多缓存 1 个已加载模型，切换模型时自动释放旧模型显存
- 手动释放已加载模型缓存

模型路径：

```text
原模型：Qwen/Qwen3-0.6B，本地缓存优先读取 model/Qwen/Qwen3-0___6B
SFT：output_lora/final_model/
DPO：output_dpo/final_model/
```

模型缓存数量可通过环境变量调整：

Windows CMD：

```bat
set MAX_LOADED_MODELS=2
python app.py
```

PowerShell：

```powershell
$env:MAX_LOADED_MODELS=2
python app.py
```

单卡显存较紧时建议保持默认值 `1`，方便在原模型、SFT 和 DPO 模型之间切换对比时自动释放旧模型。

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
python -m py_compile prepare_data.py data_prep/generate_finance_sft_data.py data_prep/generate_finance_dpo_data.py data_prep/init_finance_sqlite.py train.py train_lora.py train_dpo.py app.py checks/check_sft_data.py checks/check_dpo_data.py checks/smoke_test.py
python checks/check_sft_data.py
python checks/check_dpo_data.py
python checks/smoke_test.py
```

## 注意事项

- `train.py` 和 `train_lora.py` 读取 `data/train_format.json` / `data/eval_format.json`。
- 重新生成 SFT 数据后，建议运行 `python prepare_data.py --clean-cache` 清理 tokenized 缓存。
- Web 对比 DPO 模型需要先生成 `output_dpo/final_model/`。
- NL2SQL 样例只代表本地 SQLite 样例库上的可执行性，不代表生产库可直接使用。
- 合规拒答数据用于让模型拒绝套现、伪造材料、规避风控、查询他人隐私等请求。
- 本项目仅用于学习、研究和内部实验，不构成真实金融业务建议。
