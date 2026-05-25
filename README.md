---
domain: multi-modal
tags:
  - rag
  - customer-service
  - multimodal
  - langgraph
datasets:
  evaluation:
  test:
  train:
models:
license: Apache License 2.0
---

# RAG 智能客服系统

基于多模态 RAG（检索增强生成）的智能客服系统，支持对 20+ 种消费电子产品（空调、相机、洗碗机、电钻、健身追踪器、VR 设备等）进行智能问答。用户可上传设备图片，系统通过视觉理解辅助诊断，实现图文多模态交互。

本项目为阿里巴巴 ModelScope 竞赛参赛作品。

## 功能特性

- **混合检索**：语义向量搜索（BAAI/bge-small-zh-v1.5）+ BM25 关键词搜索，使用 RRF 融合排序
- **多模态输入**：支持文本 + 图片提问，视觉模型自动分析设备图片
- **幻觉防控**：生成答案后进行事实核查验证，降低幻觉风险
- **通用客服模板**：退换货、发票、投诉等通用问题走模板快速响应
- **批量评测**：支持 400 题批量评测，断点续跑，输出竞赛提交文件
- **暗色主题 UI**：内置暗紫色风格聊天界面

## 系统架构

```
用户请求 → 查询分解 → 混合检索 → LLM 生成 → 事实验证 → 返回答案
```

| 模块 | 文件 | 说明 |
|------|------|------|
| 知识预处理 | `preprocessing.py` | 解析产品手册，结构化分块，输出知识库 |
| 向量存储 | `vector_store.py` | 混合检索索引（语义 + BM25），RRF 融合 |
| RAG 引擎 | `graph.py` | LangGraph 状态图：分解 → 检索 → 生成 → 验证 |
| API 服务 | `app.py` | FastAPI 应用，含聊天接口和内置前端 |
| 批量评测 | `evaluate.py` | 多线程批量评测，断点续跑 |

## 技术栈

**模型**
- Embedding：`BAAI/bge-small-zh-v1.5`（本地 512 维）
- 对话/推理：可配置，默认 Ollama `deepseek-r1:8b`；云端可选 `mimo-v2.5-pro` / `mimo-v2-omni`
- 视觉：Ollama `qwen2.5vl:3b` 或云端 `mimo-v2-omni`
- 重排序：阿里 DashScope `gte-rerank-v2`

**框架**：LangGraph · FastAPI · Sentence-Transformers · rank-bm25 · jieba

## 快速开始

### 环境要求

- Python 3.10+
- 本地模式：需安装 Ollama 并拉取模型
- 云端模式：需配置阿里 DashScope API Key

### 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（默认端口 8000）
python app.py
```

访问 `http://localhost:8000/` 即可使用聊天界面。

### Docker 部署

```bash
docker-compose up --build
```

### 模式切换

在 `graph.py` 中通过 `USE_LOCAL_MODEL` 切换：
- `True`（默认）：使用本地 Ollama 模型
- `False`：使用云端 API

### 批量评测

```bash
python run_evaluate.py        # 评测 400 题，生成 submission.csv
python clean_submission.py    # 后处理（去重、格式化）
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 服务端口 | `8000` |
| `KAFU_API_TOKEN` | API 鉴权 Token | `sk_customer_20260304` |
| `DASHSCOPE_API_KEY` | 阿里 DashScope API Key | - |

## Clone with HTTP

```bash
git clone https://www.modelscope.cn/studios/chayedan123/rag-customer-service.git
```

## 许可证

Apache License 2.0
