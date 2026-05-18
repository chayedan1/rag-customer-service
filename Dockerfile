# 使用官方轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 安装必要的系统底层依赖（例如 Pillow 图像处理可能需要的依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户 (Hugging Face Spaces 默认以 UID 1000 运行)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 设置工作目录在用户目录下
WORKDIR $HOME/app

# 先复制依赖文件并安装，利用 Docker 缓存层优化构建速度
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 复制整个工程的所有源码和数据，并赋予用户所有权
COPY --chown=user . .

# 设置默认运行端口（Hugging Face / ModelScope 均使用 7860 端口）
ENV PORT=7860
EXPOSE 7860

# 运行 FastAPI 接口
CMD ["python", "app.py"]

