# GPT Model Training from Scratch

一个从零开始构建和训练 GPT 模型的完整项目，代码高度模块化，易于理解和扩展。

**特点：** 模块化设计 | 灵活的配置系统 | 三种训练模式（快速/完整/自由） | 设备一致性检查 | 清晰的接口设计

## 快速开始

### 最简单的方式：直接运行

```bash
# 安装依赖
pip install torch tiktoken requests matplotlib tqdm

# 快速测试（1 epoch，小模型）
python main.py

# 查看输出
open training_losses.png  # macOS
# 或者 display training_losses.png (Linux)
```

**就这样！** 训练会自动读取 `config_run.py` 中的参数。

---

## 三种训练模式

### 模式 1: 快速测试（默认）

用于快速验证流程，**1 个 epoch，小模型，128 context**。

编辑 `config_run.py`：
```python
QUICK_MODE = True      # ← 启用快速模式
CUSTOM_MODE = False
```

运行：`python main.py`

---

### 模式 2: 完整训练

用于真正的模型训练，**10 个 epoch，大模型，256 context**。

编辑 `config_run.py`：
```python
QUICK_MODE = False     # ← 禁用快速模式
CUSTOM_MODE = False
```

运行：`python main.py`

---

### 模式 3: 自由训练（推荐）✨

**完全自定义所有训练参数** - 学习率、轮次、丢弃率、多头数量、上下文长度等。

编辑 `config_run.py`：
```python
QUICK_MODE = False     # 不使用
CUSTOM_MODE = True     # ← 启用自由模式

# 编辑 custom_params 字典中的任何参数
custom_params = {
    'num_epochs': 5,              # 训练轮次
    'batch_size': 16,             # 批大小
    'learning_rate': 1e-3,        # 学习率
    'weight_decay': 0.05,         # L2 正则
    'eval_freq': 25,              # 评估频率
    'max_length': 512,            # 上下文长度
    'stride': 256,                # 滑动窗口步长
    'num_workers': 0,             # DataLoader 线程数
    # 可选：模型架构覆盖（高级用户）
    'model_overrides': {
        'n_heads': 8,             # 注意力头数
        'n_layers': 6,            # Transformer 层数
        'emb_dim': 512,           # 嵌入维度
        'drop_rate': 0.1,         # Dropout 比率
    }
}
```

运行：`python main.py`

**优势：** 简洁的字典接口 | 所有参数一目了然 | 支持模型架构扩展 | 动态更新参数

---

## 项目架构

```
从零构建大模型/
├── main.py                # 训练入口点（协调所有模块）
├── environment.py         # 统一环境设置（设备、模型、数据加载器）
├── 
├── config.py              # 模型配置管理
├── model.py               # GPT 模型架构
├── data.py                # 数据加载和预处理
├── model_builder.py       # 模型构建工厂
├── 
├── training_utils.py      # 训练超参数和工具
├── train.py               # 训练循环和评估
├── generation.py          # 文本生成（greedy/top-k/top-p）
├── text_to_token_ids.py   # Token 编码/解码
├── visualize.py           # 训练曲线可视化
├── 
├── config_run.py          # 运行时参数配置（直接编辑此文件来改变训练）
├── config.json            # 模型配置（默认 124M GPT）
└── the-verdict.txt        # 训练数据（自动下载）
```

---

## 模块功能详解

### `config_run.py` - 参数配置（你的主要配置文件）

**这是你唯一需要编辑的文件来控制训练！**

```python
DEVICE = None              # None=自动选择，或指定 "cpu"/"cuda"/"mps"
USE_SMALL = True           # 使用小模型（测试用）
USE_DEFAULT = True         # 使用默认 124M 模型

QUICK_MODE = True          # 模式 1: 快速测试
CUSTOM_MODE = False        # 模式 3: 自由训练

custom_params = { ... }    # 自由模式下的所有参数

SEED = 123                 # 随机种子
LOG_EVERY = 0              # 每 N 步打印 batch 损失
```

---

### `environment.py` - 统一环境设置

**负责初始化整个训练环境。** 返回一个 `RuntimeEnv` 对象，包含：

- `device`：计算设备（CPU/GPU/MPS）
- `model`：构建好的 GPT 模型
- `train_loader` / `val_loader`：数据加载器
- `optimizer`：优化器
- `train_config`：训练配置
- `tokenizer`：GPT-2 分词器

```python
from environment import prepare_environment

# 一行代码初始化所有环境
env = prepare_environment()

# 直接使用
model = env.model
device = env.device
optimizer = env.optimizer
# ...
```

**特性：** 
- ✅ 设备一致性检查（确保模型和数据在同一设备）
- ✅ 自动选择最佳设备（MPS > CUDA > CPU）
- ✅ 支持三种训练模式的无缝切换

---

### `training_utils.py` - 训练超参数

#### `TrainingConfig` - 预设配置

```python
from training_utils import TrainingConfig

# 快速测试配置
cfg = TrainingConfig.get_quick_test_config()

# 完整训练配置
cfg = TrainingConfig.get_full_training_config()
```

#### `CustomTrainingConfig` - 自由配置（新增）✨

```python
from training_utils import CustomTrainingConfig

# 从字典创建
custom_cfg = CustomTrainingConfig.from_dict({
    'num_epochs': 5,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 0.05,
    'max_length': 512,
    'model_overrides': {'n_heads': 8, 'n_layers': 6}
})

# 动态更新参数
custom_cfg.update(learning_rate=1e-4, num_epochs=10)

# 转换为字典（便于序列化）
params_dict = custom_cfg.to_dict()
```

---

### 其他模块简要说明

| 模块 | 功能 |
|------|------|
| `config.py` | 加载和管理模型配置（词表大小、层数、嵌入维度等） |
| `model.py` | GPT 模型架构（Transformer、Attention、Embedding） |
| `data.py` | 数据加载、预处理、创建 DataLoader |
| `model_builder.py` | 根据配置构建模型的工厂函数 |
| `train.py` | 训练循环、评估、批量损失跟踪 |
| `generation.py` | 文本生成（贪心 / top-k / top-p 采样） |
| `text_to_token_ids.py` | Token 编码/解码、损失计算 |
| `visualize.py` | 绘制训练曲线（保存为 PNG） |
| `main.py` | 训练入口，协调所有模块 |

---

## 使用示例

### 示例 1: 快速验证（默认配置）

```bash
python main.py
```

输出示例：
```
✓ Text loaded from the-verdict.txt
Device: MPS
✓ Tokenizer initialized (GPT-2)
✓ Using small model configuration (for testing)
...
Epoch 1: 100%|███████████████████████████| 9/9 [00:01<00:00, 5.23it/s]
Ep 1 (End Epoch): Train loss 10.896, Val loss 10.881
✓ Saved training loss plot to training_losses.png
✓ Training pipeline finished successfully!
```

---

### 示例 2: 自由训练 - 自定义学习率和轮次

编辑 `config_run.py`：
```python
CUSTOM_MODE = True

custom_params = {
    'num_epochs': 20,           # 20 轮
    'batch_size': 32,
    'learning_rate': 5e-4,      # 降低学习率
    'weight_decay': 0.1,
    'eval_freq': 100,
    'max_length': 512,
    'stride': 256,
}
```

运行：
```bash
python main.py
```

---

### 示例 3: 自由训练 - 自定义模型架构

编辑 `config_run.py`：
```python
CUSTOM_MODE = True

custom_params = {
    'num_epochs': 5,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'max_length': 256,
    # 扩展：自定义模型架构
    'model_overrides': {
        'n_heads': 6,           # 6 个注意力头
        'n_layers': 8,          # 8 层 Transformer
        'emb_dim': 384,         # 384 维嵌入
        'drop_rate': 0.2,       # 20% dropout
    }
}
```

运行：
```bash
python main.py
```

---

### 示例 4: 完整训练 - 真实场景

编辑 `config_run.py`：
```python
DEVICE = "cuda"             # 指定 GPU
QUICK_MODE = False          # 完整配置
CUSTOM_MODE = False
SEED = 42
LOG_EVERY = 10              # 每 10 步打印损失
```

运行：
```bash
python main.py
```

---

## 输出说明

### 训练输出

```
======================================================================
GPT Model Training Pipeline
======================================================================

Device: MPS
GPU: Apple Metal Performance Shaders

✓ Tokenizer initialized (GPT-2)
✓ Using small model configuration (for testing)

Model Configuration
Vocab size:      50257
Context length:  128
Embedding dim:   64
Num heads:       4
Num layers:      2
Total params:    6,540,800

✓ Using quick training config (1 epoch, for testing)
✓ Data loaders ready
✓ Device consistency verified (using mps)

======================================================================
TRAINING START
======================================================================

Epoch 1: 100%|███████████████████████████| 9/9 [00:01<00:00, 5.23it/s]
Ep 1 (End Epoch): Train loss 10.896, Val loss 10.881
  ✓ Loss plot saved: training_losses.png
✓ Saved training loss plot to training_losses.png

======================================================================
TRAINING COMPLETE
======================================================================

Training Summary:
  Final train loss: 10.8964
  Final val loss:   10.8813
  Best val loss:    10.8813 (at step 0)

✓ Training pipeline finished successfully!
```

### 生成的文件

- **`training_losses.png`**：训练曲线图
  - 左图：训练损失 vs 验证损失（逐评估步骤）
  - 右图：批次级别损失（子采样显示）

---

## 自由训练模式 - 深入指南

### 自由模式参数详解

| 参数 | 类型 | 说明 | 典型值 |
|------|------|------|--------|
| `num_epochs` | int | 训练轮数 | 5, 10, 20 |
| `batch_size` | int | 批大小 | 8, 16, 32, 64 |
| `learning_rate` | float | 学习率 | 1e-4, 3e-4, 1e-3 |
| `weight_decay` | float | L2 正则化强度 | 0.0, 0.05, 0.1 |
| `eval_freq` | int | 每多少步进行一次评估 | 25, 50, 100 |
| `eval_iter` | int | 评估时使用的 batch 数 | 2, 5, 10 |
| `max_length` | int | 上下文窗口大小 | 128, 256, 512, 1024 |
| `stride` | int | 滑动窗口步长（通常是 max_length/2） | 64, 128, 256 |
| `start_context` | str | 生成文本的起始词 | "Every effort moves you" |
| `num_workers` | int | DataLoader 线程数 | 0 (CPU), 2-4 (GPU) |
| `model_overrides` | dict | 可选的模型架构覆盖 | 见下表 |

### 模型架构覆盖 - 高级用法

在 `custom_params` 中添加 `model_overrides` 字典来自定义模型架构：

```python
custom_params = {
    'num_epochs': 5,
    'batch_size': 16,
    # ... 其他训练参数 ...
    'model_overrides': {
        'emb_dim': 512,       # 嵌入维度（需要整除 n_heads）
        'n_heads': 8,         # 注意力头数
        'n_layers': 6,        # Transformer 层数
        'drop_rate': 0.15,    # Dropout 比率 (0.0-0.5)
        'qkv_bias': False,    # Q/K/V 是否有偏置
    }
}
```

**约束条件：**
- `emb_dim` 必须能整除 `n_heads`（通常 emb_dim % n_heads == 0）
- `n_heads` 建议值：4, 6, 8, 12
- `n_layers` 建议值：2-12
- `drop_rate` 范围：0.0-0.5

---

## 灵活扩展 - 编程接口

### 直接使用 `CustomTrainingConfig`

```python
from training_utils import CustomTrainingConfig

# 方式 1: 从字典创建
config = CustomTrainingConfig.from_dict({
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'max_length': 256,
})

# 方式 2: 直接构造
config = CustomTrainingConfig(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=0.1,
)

# 方式 3: 动态更新
config.update(
    learning_rate=5e-4,
    num_epochs=20,
    max_length=512
)

# 方式 4: 序列化为字典
params = config.to_dict()
```

### 集成到自己的脚本

```python
from environment import prepare_environment
from train import train_model_simple
from training_utils import CustomTrainingConfig
import config_run

# 修改 config_run
config_run.CUSTOM_MODE = True
config_run.custom_params = {
    'num_epochs': 5,
    'batch_size': 16,
    'learning_rate': 1e-3,
    # ... 其他参数
}

# 初始化环境
env = prepare_environment()

# 运行训练
train_losses, val_losses, _ = train_model_simple(
    model=env.model,
    train_loader=env.train_loader,
    val_loader=env.val_loader,
    optimizer=env.optimizer,
    device=env.device,
    num_epochs=env.train_config.num_epochs,
    eval_freq=env.train_config.eval_freq,
    eval_iter=env.train_config.eval_iter,
    start_context=env.train_config.start_context,
    tokenizer=env.tokenizer,
    log_every=config_run.LOG_EVERY,
)

# 处理结果
print(f"Final train loss: {train_losses[-1]:.4f}")
print(f"Final val loss: {val_losses[-1]:.4f}")
```

---

## 常见问题（FAQ）

### Q: 我应该如何选择学习率？

**A:** 
- **小模型快速测试**：`3e-4` ~ `1e-3`
- **大模型训练**：`1e-4` ~ `5e-4`
- **微调**：`1e-4` ~ `3e-4`

一般遵循规则：学习率越小，收敛越慢但越稳定；学习率越大，训练越快但容易波动。

### Q: batch_size 怎么选择？

**A:** 
- **内存充足**：32 或 64（更稳定的梯度）
- **内存有限**：8 或 16（更频繁的更新）
- **规律**：batch_size 越大，同 epoch 的迭代数越少

### Q: max_length（上下文）设置多少合适？

**A:**
- **快速测试**：128（最快）
- **平衡**：256（推荐）
- **更长依赖**：512, 1024（需要更多显存）

### Q: 模型一直不收敛怎么办？

**尝试以下方案：**
1. 降低学习率（1e-4 ~ 5e-5）
2. 增加 weight_decay（0.1 ~ 0.2）
3. 增加 dropout 比率（0.2 ~ 0.3）
4. 检查数据是否有问题
5. 尝试更小的 batch_size

### Q: 如何保存训练好的模型？

**A:** 在 `train.py` 中添加以下代码：

```python
import torch
from pathlib import Path

# 在训练完成后
model_path = Path("checkpoints") / f"gpt_epoch_{epoch}.pt"
model_path.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"✓ Model saved to {model_path}")
```

### Q: 如何使用不同的数据集？

**A:** 编辑 `config_run.py` 的 `custom_params`，然后在 `environment.py` 的 `prepare_dataloaders()` 中修改数据源：

```python
# 在 environment.py 中
from data import create_dataloader_v1

# 改为你的数据
my_text = open("my_data.txt").read()

train_loader = create_dataloader_v1(
    my_text,  # 改这里
    batch_size=train_config.batch_size,
    # ... 其他参数
)
```

### Q: 生成的文本质量很差？

**A:** 这是**正常的**！原因：
- 训练数据很小（~20KB）
- 训练时间很短（通常 1-10 epoch）
- 模型很小（124M GPT）

**改进方案：**
- 增加 `num_epochs` 到 20-100
- 使用更大的数据集（MB 级别）
- 增加模型大小（`n_layers`, `emb_dim`）
- 使用更好的采样策略（见 `generation.py`）

---

## 项目特点

✅ **三种训练模式**：快速测试 / 完整训练 / 自由训练  
✅ **简洁配置**：一个 `config_run.py` 文件，无需命令行参数  
✅ **灵活参数**：支持完全自定义所有训练和模型参数  
✅ **模块化设计**：清晰的模块分工，易于扩展和复用  
✅ **设备一致性**：自动检查模型和数据在同一设备  
✅ **可视化输出**：自动生成训练曲线图  
✅ **进度跟踪**：tqdm 进度条 + 详细日志  
✅ **易用接口**：`CustomTrainingConfig` 支持编程和序列化

---

## 依赖列表

| 包 | 版本 | 用途 |
|----|----|------|
| torch | ≥ 2.0 | 深度学习框架 |
| tiktoken | latest | GPT-2 分词器 |
| requests | latest | 下载文件 |
| matplotlib | latest | 可视化（可选） |
| tqdm | latest | 进度条（可选） |

**最小安装（仅必需）：**
```bash
pip install torch tiktoken requests
```

**完整安装（推荐）：**
```bash
pip install torch tiktoken requests matplotlib tqdm
```

---

## 快速命令参考

| 需求 | 命令 |
|------|------|
| 快速测试 | `python main.py` （QUICK_MODE=True） |
| 完整训练 | 编辑 `config_run.py`，改 `QUICK_MODE=False` |
| 自由训练 | 编辑 `config_run.py`，改 `CUSTOM_MODE=True` |
| 自定义学习率 | 编辑 `custom_params['learning_rate']` |
| 自定义轮次 | 编辑 `custom_params['num_epochs']` |
| 使用 GPU | 编辑 `config_run.py`，改 `DEVICE="cuda"` |
| 查看训练曲线 | 打开 `training_losses.png` |

---

## 许可证

MIT License

---

**需要帮助？** 查看上面的"常见问题"部分或编辑 `config_run.py` 中的参数尝试不同的配置。

**最后更新：** 2025年11月13日  
**维护者：** GitHub Copilot  

