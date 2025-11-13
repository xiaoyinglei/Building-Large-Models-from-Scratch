# GPT Model Training from Scratch

一个从零开始构建和训练 GPT 模型的完整项目，代码高度模块化，易于理解和扩展。

## 项目架构

项目采用模块化设计，各部分职责分离，便于维护和扩展：

```
├── config.py              # 配置管理（加载、验证、默认配置）
├── gptmodel.py            # 模型架构（TransformerBlock、Attention、GPTModel）
├── data.py                # 数据加载和预处理
├── model_builder.py       # 模型构建工厂函数
├── training_utils.py      # 训练辅助工具（超参数、优化器、设备管理）
├── text_to_token_ids.py   # Token 编码/解码和损失计算
├── train.py               # 训练循环和评估
├── main.py                # 训练入口点（协调所有模块）
├── config.json            # 模型配置文件（可选）
└── the-verdict.txt        # 训练数据（自动下载）
```

## 模块说明

### 1. `config.py` - 配置管理
负责加载和管理 GPT 模型的配置。

**主要类和函数：**
- `GPTConfig`：配置数据类，包含模型超参数
- `load_config_from_file()`：从 JSON 文件加载配置
- `get_small_config()`：获取小规模测试配置
- `get_default_config()`：获取默认（124M）配置

**使用示例：**
```python
from config import load_config_from_file, get_small_config

# 从文件加载
cfg = load_config_from_file("config.json")

# 获取预定义配置
cfg = get_small_config()  # 快速测试
cfg = get_default_config()  # 完整模型
```

### 2. `gptmodel.py` - 模型架构
包含 GPT 模型的所有组件（Attention、FeedForward、TransformerBlock、GPTModel）。

**主要类和函数：**
- `MultiHeadAttention`：多头自注意力机制
- `LayerNorm`：层标准化
- `GELU`：激活函数
- `FeedForward`：前馈网络
- `TransformerBlock`：Transformer 块（包含 Attention 和 FF）
- `GPTModel`：完整的 GPT 模型
- `generate_text_simple()`：文本生成（贪心采样）

**使用示例：**
```python
from gptmodel import GPTModel
from config import get_small_config

cfg = get_small_config()
model = GPTModel(cfg)
model.eval()

# 生成文本
idx = torch.tensor([[1, 2, 3]])
output = generate_text_simple(model, idx, max_new_tokens=50, context_size=128)
```

### 3. `data.py` - 数据加载
处理数据集加载、预处理和 DataLoader 创建。

**主要类和函数：**
- `GPTDatasetV1`：滑动窗口数据集
- `create_dataloader_v1()`：创建 DataLoader
- `load_text_data()`：加载文本数据（本地或下载）
- `print_text_stats()`：打印数据统计信息
- `text_data`：全局文本数据（模块导入时自动加载）

**使用示例：**
```python
from data import create_dataloader_v1, text_data

train_loader = create_dataloader_v1(
    text_data,
    batch_size=32,
    max_length=256,
    stride=128,
    shuffle=True
)
```

### 4. `model_builder.py` - 模型构建工厂
封装模型构建和初始化逻辑。

**主要函数：**
- `build_model()`：从配置构建模型
- `build_model_from_file()`：从配置文件构建模型
- `build_small_model()`：构建测试用小模型
- `build_default_model()`：构建默认大模型
- `count_parameters()`：计算参数数量
- `print_model_info()`：打印模型信息

**使用示例：**
```python
from model_builder import build_small_model, print_model_info

model, cfg = build_small_model(device="mps")
print_model_info(model, cfg)
```

### 5. `training_utils.py` - 训练工具
提供训练超参数管理、优化器创建和设备管理。

**主要类和函数：**
- `TrainingConfig`：训练超参数配置（数据类）
- `TrainingConfig.get_quick_test_config()`：快速测试配置
- `TrainingConfig.get_full_training_config()`：完整训练配置
- `create_optimizer()`：创建 AdamW 优化器
- `get_device()`：获取最佳设备（mps > cuda > cpu）
- `print_device_info()`：打印设备信息

**使用示例：**
```python
from training_utils import TrainingConfig, create_optimizer, get_device

device = get_device()
train_config = TrainingConfig.get_quick_test_config()
optimizer = create_optimizer(model, learning_rate=3e-4)
```

### 6. `text_to_token_ids.py` - Token 处理
负责 Token 编码/解码和损失计算。

**主要函数：**
- `text_to_token_ids()`：文本转 Token ID
- `token_ids_to_text()`：Token ID 转文本
- `calc_loss_batch()`：计算单个 batch 的损失
- `calc_loss_loader()`：计算整个 DataLoader 的平均损失

**使用示例：**
```python
from text_to_token_ids import text_to_token_ids, calc_loss_batch

encoded = text_to_token_ids("Hello world", tokenizer)
loss = calc_loss_batch(input_batch, target_batch, model, device)
```

### 7. `train.py` - 训练循环
包含训练循环、评估和文本生成逻辑。

**主要函数：**
- `train_model_simple()`：主训练循环
- `evaluate_model()`：在 train/val 数据上评估
- `generate_and_print_sample()`：生成并打印样例文本

**使用示例：**
```python
from train import train_model_simple

train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=10,
    eval_freq=50,
    eval_iter=2,
    start_context="Every effort",
    tokenizer=tokenizer
)
```

### 8. `visualize.py` - 可视化模块
提供训练过程的损失可视化。

**主要函数：**
- `plot_losses()`：在同一张图中绘制训练损失、测试损失和 batch 损失
  - **左侧子图**：训练损失 vs 测试损失（逐评估步骤）
  - **右侧子图**：Batch 级别损失（子采样以保持清晰）

**使用示例：**
```python
from visualize import plot_losses

# 在训练后调用
plot_losses(train_losses, val_losses, batch_losses, out_path="training_losses.png")
```

**输出：**
- `training_losses.png`：包含两个子图的高分辨率可视化图像

### 9. `main.py` - 训练入口
协调所有模块，执行完整的训练流程。

**流程步骤：**
1. 初始化设备和随机种子
2. 初始化 Tokenizer
3. 加载模型配置
4. 打印数据统计
5. 构建模型
6. 准备 DataLoader
7. 创建优化器
8. 运行训练循环
9. 自动生成可视化图表
10. 报告结果

## 如何使用

### 安装依赖

```bash
pip install torch tiktoken requests matplotlib tqdm
```

- `torch`：深度学习框架
- `tiktoken`：GPT-2 分词器
- `requests`：下载文件
- `matplotlib`：损失可视化（可选，不安装时跳过绘图）
- `tqdm`：进度条显示（可选，不安装时不显示进度条）

### 快速测试（推荐首先运行）

使用小模型进行 1 个 epoch 的快速测试：

```bash
python main.py --quick
```

**预期输出：**
- 设备信息（MPS/CUDA/CPU）
- 模型配置和参数量
- 训练进度条（使用 tqdm，实时显示 batch 进度）
- 训练进度和损失
- 每个 epoch 的生成样例文本
- 最终训练总结
- **`training_losses.png`**：包含训练损失、测试损失和批次损失的可视化图表

### 命令行参数（推荐）

现在项目提供了一个命令行接口 `cli.py`，你可以通过命令行灵活选择配置、设备与训练模式。推荐使用命令行参数而不是修改 `main.py` 源码。

常用选项示例：

- 快速测试（默认）：

```bash
python main.py --quick
```

- 每 batch 打印一次损失（用于实时监控）：

```bash
python main.py --quick --log-every 1
```

- 每 10 个 batch 打印一次损失（减少输出量）：

```bash
python main.py --quick --log-every 10
```

- 使用本地 `config.json` 并指定设备（例如 MPS/CPU/CUDA）：

```bash
python main.py --config config.json --device mps
```

- 强制使用内置小模型（快速验证）：

```bash
python main.py --use-small
```

- 运行完整训练配置（较长）：

```bash
python main.py --full
```

- 覆盖 batch size：

```bash
python main.py --quick --batch-size 16
```

**所有可用参数：**

- `--config PATH`：指定配置文件路径（默认 `config.json`）。
- `--use-small`：使用内置的小模型配置（测试用）。
- `--use-default`：使用内置默认完整模型配置。
- `--device {cpu,cuda,mps}`：强制使用指定设备；否则脚本会自动选择（mps > cuda > cpu）。
- `--quick/--full`：选择快速测试或完整训练模式（默认 `--quick`）。
- `--batch-size N`：覆盖训练批次大小。
- `--seed N`：随机种子（默认 123）。
- `--log-every N`：每 N 个 batch 打印一次损失；0 表示不打印（默认 0）。

### 自定义配置

#### 方式 1：编辑 `config.json`

```json
{
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": false
}
```

然后通过命令行运行以从该文件加载：

```bash
python main.py --config config.json --full
```

#### 方式 2：修改 `training_utils.py` 中的 `TrainingConfig`

```python
from training_utils import TrainingConfig

custom_config = TrainingConfig(
    num_epochs=20,
    batch_size=16,
    learning_rate=5e-4,
    eval_freq=100,
    # ... 其他参数
)
```

#### 方式 3：在 `main.py` 中直接调用模块

```python
from config import get_default_config
from model_builder import build_model
from training_utils import TrainingConfig, create_optimizer

cfg = get_default_config()
model = build_model(cfg, device="mps")
train_config = TrainingConfig.get_full_training_config()
optimizer = create_optimizer(model)
# ... 手动调用训练
```

## 训练结果示例

快速测试运行输出示例：

```
======================================================================
GPT Model Training Pipeline
======================================================================

==================================================
Device: MPS
GPU: Apple Metal Performance Shaders
==================================================

✓ Tokenizer initialized (GPT-2)
✓ Using small model configuration (for testing)
Characters: 20479
Tokens: 5145

==================================================
Model Configuration
==================================================
Vocab size:      50257
Context length:  128
Embedding dim:   64
Num heads:       4
Num layers:      2
Dropout rate:    0.1
QKV bias:        False
Total params:    6,540,800
==================================================

Ep 1 (Step 000000): Train loss 11.009, Val loss 10.980
Every effort moves you long dismal underlying decorations...

Training Summary:
  Final train loss: 11.0085
  Final val loss:   10.9799
  Best val loss:    10.9799 (at step 0)

✓ Training pipeline finished successfully!
```

## 项目特点

✅ **模块化设计**：各部分职责清晰，易于维护和扩展  
✅ **清晰的数据流**：从配置加载 → 模型构建 → 数据准备 → 训练执行  
✅ **灵活的配置**：支持文件配置、预定义配置、自定义配置  
✅ **详细的文档**：每个模块和函数都有完整注释  
✅ **快速验证**：提供快速测试配置，可立即验证整个流程  
✅ **可扩展**：易于添加新功能（如学习率调度、模型检查点等）

## 常见问题

### Q: 如何改变模型大小？
A: 编辑 `config.json` 或在 `config.py` 中定义新的预设配置函数。

### Q: 如何使用不同的数据集？
A: 在 `data.py` 中修改 `load_text_data()` 函数的参数或使用 `create_dataloader_v1()` 的 `txt` 参数。

### Q: 如何保存和加载模型？
A: 在 `train.py` 中添加检查点保存逻辑（建议使用 `torch.save()` 和 `torch.load()`）。

### Q: 如何加速训练？
A: 
- 使用 GPU（CUDA）而不是 CPU
- 增加 `batch_size`
- 使用更小的 `context_length`
- 启用混合精度训练（fp16）

### Q: 生成的文本质量不好？
A: 这是正常的，因为模型训练时间短、数据集小。更长的训练和更大的模型会产生更好的结果。

## 文件列表

| 文件 | 大小 | 说明 |
|------|------|------|
| `config.py` | 2.3 KB | 配置管理 |
| `gptmodel.py` | 8.5 KB | 模型架构 |
| `data.py` | 4.2 KB | 数据加载 |
| `model_builder.py` | 3.1 KB | 模型构建工厂 |
| `training_utils.py` | 3.4 KB | 训练工具 |
| `text_to_token_ids.py` | 2.1 KB | Token 处理 |
| `train.py` | 3.6 KB | 训练循环 |
| `main.py` | 7.8 KB | 训练入口 |
| `config.json` | 0.2 KB | 模型配置 |
| `the-verdict.txt` | ~20 KB | 训练数据（自动下载） |

## 许可证

MIT License

## 作者

从零构建大模型项目

---

**开始训练：** `python main.py`
