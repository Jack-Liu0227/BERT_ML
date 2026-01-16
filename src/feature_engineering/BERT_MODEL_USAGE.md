# BERT 模型加载系统使用指南

## 概述

本项目实现了一个统一的 BERT 模型加载系统，支持从本地路径加载不同的 BERT 模型（SteelBERT、MatSciBERT 等），避免每次从 Hugging Face 在线加载。

## 目录结构

```
./models/
├── steelbert/          # SteelBERT 模型
│   ├── config.json
│   ├── pytorch_model.bin (或 model.safetensors)
│   └── tokenizer files...
└── matscibert/         # MatSciBERT 模型
    ├── config.json
    ├── pytorch_model.bin (或 model.safetensors)
    └── tokenizer files...
```

## 快速开始

### 1. 准备模型文件

#### 方法 A: 下载 MatSciBERT 模型（推荐）

使用提供的下载脚本：

```bash
python scripts/download_matscibert.py
```

可选参数：
- `--save_path`: 自定义保存路径（默认: `./models/matscibert`）
- `--force`: 强制重新下载

#### 方法 B: 手动下载模型

```python
from src.feature_engineering.bert_model_loader import BERTModelLoader

# 下载 MatSciBERT
BERTModelLoader.download_model_from_huggingface(
    model_id="m3rg-iitd/matscibert",
    save_path="./models/matscibert"
)
```

#### 方法 C: 迁移现有 SteelBERT 模型

如果你已经有 `./SteelBERTmodel` 目录，建议将其移动到新的统一结构：

```bash
# Windows PowerShell
Move-Item -Path "./SteelBERTmodel" -Destination "./models/steelbert"

# Linux/Mac
mv ./SteelBERTmodel ./models/steelbert
```

### 2. 使用模型

#### 在命令行中使用

**使用模型名称（推荐）：**

```bash
python src/feature_engineering/example.py \
    --model_name matscibert \
    --data_path "datasets/your_data.csv" \
    --feature_dir "Features/output" \
    ...
```

**使用模型路径（兼容旧版本）：**

```bash
python src/feature_engineering/example.py \
    --model_path "./models/steelbert" \
    --data_path "datasets/your_data.csv" \
    --feature_dir "Features/output" \
    ...
```

#### 在代码中使用

**方法 1: 使用 FeatureProcessor（推荐）**

```python
from src.feature_engineering.feature_processor import FeatureProcessor

# 使用模型名称
processor = FeatureProcessor(
    data_path="datasets/your_data.csv",
    model_name="matscibert",  # 或 "steelbert"
    use_process_embedding=True,
    use_element_embedding=True,
    ...
)

# 或使用模型路径（兼容旧版本）
processor = FeatureProcessor(
    data_path="datasets/your_data.csv",
    model_path="./models/steelbert",
    use_process_embedding=True,
    use_element_embedding=True,
    ...
)
```

**方法 2: 直接使用 BERTModelLoader**

```python
from src.feature_engineering.bert_model_loader import BERTModelLoader

# 初始化加载器
loader = BERTModelLoader()

# 加载模型
tokenizer, model = loader.load_model("matscibert")

# 使用模型
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
```

## 支持的模型

当前预配置的模型：

| 模型名称 | 本地路径 | 描述 |
|---------|---------|------|
| `steelbert` | `./models/steelbert` | 专门针对钢铁材料训练的 BERT 模型 |
| `matscibert` | `./models/matscibert` | 材料科学领域的 BERT 模型 |

## 添加自定义模型

### 方法 1: 修改配置（推荐）

编辑 `src/feature_engineering/bert_model_loader.py`，在 `MODEL_CONFIGS` 字典中添加新模型：

```python
MODEL_CONFIGS = {
    "steelbert": {
        "path": "./models/steelbert",
        "description": "SteelBERT - 专门针对钢铁材料训练的 BERT 模型"
    },
    "matscibert": {
        "path": "./models/matscibert",
        "description": "MatSciBERT - 材料科学领域的 BERT 模型"
    },
    "your_model": {  # 添加新模型
        "path": "./models/your_model",
        "description": "你的自定义模型描述"
    }
}
```

### 方法 2: 直接使用路径

无需修改配置，直接使用模型路径：

```python
processor = FeatureProcessor(
    data_path="datasets/your_data.csv",
    model_path="./path/to/your/custom/model",
    ...
)
```

## 故障排查

### 问题 1: 模型路径不存在

**错误信息：** `模型路径不存在: ...`

**解决方案：**
1. 检查模型是否已下载
2. 确认路径是否正确
3. 使用下载脚本重新下载模型

### 问题 2: 缺少模型文件

**错误信息：** `缺少 config.json 文件` 或 `缺少模型权重文件`

**解决方案：**
1. 确认模型下载完整
2. 检查目录中是否包含必要文件：
   - `config.json`（必需）
   - `pytorch_model.bin` 或 `model.safetensors`（至少一个）
   - tokenizer 相关文件

### 问题 3: 网络连接问题

**错误信息：** 下载模型时超时或连接失败

**解决方案：**
1. 检查网络连接
2. 确认可以访问 https://huggingface.co
3. 如果使用代理，配置环境变量：
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```

## 技术细节

### BERTModelLoader 类

核心类，负责模型的加载和管理。

**主要方法：**
- `load_model(model_name)`: 加载指定模型
- `get_model_path(model_name)`: 获取模型路径
- `list_available_models()`: 列出可用模型
- `download_model_from_huggingface(model_id, save_path)`: 下载模型

### 兼容性

- 完全兼容旧版本代码（使用 `model_path` 参数）
- 自动映射旧路径 `./SteelBERTmodel` 到新配置
- 支持相对路径和绝对路径

## 最佳实践

1. **使用模型名称而非路径**：更清晰、更易维护
2. **统一模型存储位置**：所有模型放在 `./models/` 目录下
3. **提前下载模型**：避免运行时下载导致延迟
4. **验证模型完整性**：下载后检查所有必要文件是否存在

