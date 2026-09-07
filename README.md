# Arsenal RAG

一个面向阿森纳足球知识的本地 RAG（检索增强生成）问答项目。项目使用 Markdown 作为知识源，FAISS 做向量检索，Transformers 模型生成回答。

## 快速开始

### 1. 安装依赖

需要 Python 3.10 或更高版本。建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

### 2. 构建索引

首次运行会从 Hugging Face 下载公开模型，请确保网络可访问：

```bash
python build_index.py
```

默认读取 `data/documents/` 下所有层级的 Markdown 文件，并生成：

- `vector_db/index.faiss`
- `vector_db/metadata.pkl`

### 3. 启动问答

```bash
python main.py
```

输入问题即可开始问答，输入 `exit` 或 `quit` 退出。

示例：

```text
User: 阿森纳有哪些英超冠军赛季？
```

模型较大或硬件有限时，可以先通过环境变量指定本地模型目录：

```bash
export ARSENAL_EMBEDDING_MODEL=/path/to/embedding-model
export ARSENAL_GENERATOR_MODEL=/path/to/generator-model
python build_index.py
python main.py
```

## 特点

- 知识源、切块、向量索引、检索和生成模块职责清晰
- 支持从任意工作目录启动，路径可通过环境变量或 CLI 覆盖
- 生成索引时会校验向量和元数据数量，避免静默错位

## 环境

- Python 3.10 或更高版本
- 可用的 CPU 或 CUDA 环境
- 首次运行需要从 Hugging Face 下载模型；也可以通过环境变量指向本地模型目录

## 配置模型

默认模型是公开的 Hugging Face 模型：

- Embedding：`Qwen/Qwen3-Embedding-0.6B`
- Generator：`Qwen/Qwen2.5-1.5B-Instruct`

模型较大时可以使用本地路径，在 shell 中设置变量：

```bash
export ARSENAL_EMBEDDING_MODEL=/path/to/embedding-model
export ARSENAL_GENERATOR_MODEL=/path/to/generator-model
```

程序不会自动读取 `.env` 文件；使用 shell、容器编排或 CI 的环境变量注入即可。

## 使用

知识文档放在 `data/documents/` 下，支持 `.md` 文件。数据目录由项目维护者自行整理，本仓库不修改其中的内容。

先构建索引：

```bash
python build_index.py
```

也可以指定文档目录、输出位置或 Embedding 模型：

```bash
python build_index.py \
	--document-dir /path/to/documents \
	--index-path /path/to/index.faiss \
	--metadata-path /path/to/metadata.pkl \
	--model-path /path/to/embedding-model
```

启动交互式问答：

```bash
python main.py
```

常用参数：

```bash
python main.py --top-k 5 --max-new-tokens 256 --log-level INFO
```

`test.py` 是人工评测脚本，用来比较 RAG 与无检索 baseline：

```bash
python test.py "阿森纳的门将有哪些？"
```

## 目录说明

```text
data/documents/   原始 Markdown 知识源，由用户自行维护
data/chunks/      预留目录，当前构建流程不直接使用
vector_db/        生成的 FAISS 索引和 metadata
src/              加载、切块、嵌入、检索、提示词和生成逻辑
build_index.py    索引构建入口
main.py           交互式问答入口
test.py           RAG 与 baseline 的人工对比脚本
```

## 数据与索引注意事项

`vector_db/index.faiss` 和 `vector_db/metadata.pkl` 必须成对使用。修改文档、切块逻辑或 Embedding 模型后，请重新运行 `build_index.py`。

metadata 当前使用 pickle，只应加载可信来源生成的文件。发布到 GitHub 前，请根据数据和模型的来源确认相应许可证，并避免提交私有模型、密钥或未经授权的数据。
