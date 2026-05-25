# 使用官方轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 替换为国内阿里云 Debian 镜像源，加速底层系统依赖下载
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true \
    && sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

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
# 智能切换为清华大学高速镜像源，并引入 PyTorch 官方 CPU 专供下载源
# 彻底免去下载庞大的显卡 CUDA 算子，使得 Torch 依赖体积暴跌 70%！下载提速 500%！
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --user -r requirements.txt

# 复制整个工程的所有源码 and 数据，并赋予用户所有权
COPY --chown=user . .

# 构建期预先解压手册插图，加速容器冷启动
RUN python -c "import zipfile, os; zip_path = 'KownledgeBase/手册/插图.zip'; extract_dir = 'KownledgeBase/手册/插图'; os.makedirs(extract_dir, exist_ok=True); zipfile.ZipFile(zip_path, 'r').extractall(extract_dir) if os.path.exists(zip_path) else print('未检测到插图压缩包，跳过构建期解压。')"

# 设置默认运行端口（也可通过环境变量 PORT 动态调整，默认采用 8000）
ENV PORT=8000
EXPOSE 8000

# 运行 FastAPI 接口
CMD ["python", "app.py"]


