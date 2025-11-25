# 超解像動画処理ツール Dockerfile
#
# ビルド方法:
#   docker build -t super-resolution-movies -f docker/Dockerfile .
#
# CPU版ビルド:
#   docker build -t super-resolution-movies:cpu --build-arg USE_GPU=false -f docker/Dockerfile .

# =============================================================================
# ベースイメージ
# =============================================================================
FROM python:3.11-slim AS base

# ビルド引数
ARG USE_GPU=true
ARG DEBIAN_FRONTEND=noninteractive

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 作業ディレクトリの設定
WORKDIR /workspace

# =============================================================================
# 依存パッケージのインストール
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ffmpegと関連ライブラリ
    ffmpeg \
    # その他必要なツール
    wget \
    unzip \
    ca-certificates \
    git \
    # 画像処理ライブラリ
    libgl1 \
    libglib2.0-0 \
    # Vulkanサポート（GPU利用時）
    libvulkan1 \
    mesa-vulkan-drivers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Real-ESRGAN NCNN Vulkanのダウンロード（オプション）
# =============================================================================
# 注意: 実際の使用時にはライセンスを確認してください
# Real-ESRGANは別途ダウンロードする必要があります
#
# 以下はReal-ESRGAN NCNN Vulkanを手動でダウンロードする場合の例:
# RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
#     && unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d /opt/realesrgan \
#     && chmod +x /opt/realesrgan/realesrgan-ncnn-vulkan \
#     && rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip
#
# ENV PATH="/opt/realesrgan:${PATH}"

# =============================================================================
# Pythonパッケージのインストール
# =============================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# アプリケーションコードのコピー
# =============================================================================
COPY app/ ./app/
COPY config/ ./config/

# =============================================================================
# エントリポイント
# =============================================================================
# デフォルトのヘルプを表示
# ENTRYPOINT ["python", "-m", "app.main"]
# CMD ["--help"]

# =============================================================================
# ヘルスチェック
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import app; print('OK')" || exit 1

# =============================================================================
# メタデータ
# =============================================================================
LABEL maintainer="your-email@example.com" \
      description="動画超解像処理ツール - ffmpeg + Deep Learning" \
      version="0.1.0"
