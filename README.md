# 超解像動画処理ツール (Super Resolution Movies)

Real-ESRGANモデルを使用して動画を高解像度化するツールです。NVIDIA GPUによるCUDA加速に対応し、効率的な処理が可能です。

## 特徴

- **Real-ESRGAN対応**: 実写映像に最適化されたAI超解像モデル
- **GPU加速**: NVIDIA GPU (CUDA) による高速処理
- **メモリ効率**: 1フレームずつ処理してメモリ使用量を最小化
- **再開機能**: 中断しても処理済みフレームをスキップして再開可能
- **Docker対応**: 依存関係を含む再現可能な環境

## 動作環境

### 必須要件
- Docker Desktop (WSL2バックエンド推奨)
- NVIDIA GPU + ドライバー 450.80.02以降
- NVIDIA Container Toolkit

### 推奨スペック
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM) 以上
- **RAM**: 64GB以上
- **ストレージ**: 高速SSD (フレーム保存用)

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/milky1210/super_resolution_movies.git
cd super_resolution_movies
```

### 2. Dockerイメージのビルド

```powershell
# GPU版（推奨）
docker compose -f docker/docker-compose.yml --profile gpu build app-gpu

# CPU版（GPUがない場合）
docker compose -f docker/docker-compose.yml build app
```

### 3. コンテナの起動

```powershell
# GPU版コンテナを起動
docker compose -f docker/docker-compose.yml --profile gpu up -d

# 起動確認
docker ps
```

## 使用方法

### 基本的な使い方

```powershell
# 1. 入力動画を data/ ディレクトリに配置
cp your_video.mp4 data/

# 2. 超解像処理を実行（2倍拡大）
docker exec sr-movies-gpu python -m app.main `
    -i /data/your_video.mp4 `
    -o /data/your_video_2x.mp4 `
    --scale 2

# 3. 出力動画を確認
ls data/your_video_2x.mp4
```

### コマンドラインオプション

```
usage: python -m app.main [-h] -i INPUT -o OUTPUT [--scale {2,3,4}]
                          [--model MODEL] [--fps FPS] [--tmp-dir TMP_DIR]
                          [--no-cleanup] [--crf CRF] [--preset PRESET]
                          [--list-models] [--log-level {DEBUG,INFO,WARNING,ERROR}]

オプション:
  -i, --input INPUT     入力動画のパス
  -o, --output OUTPUT   出力動画のパス
  --scale {2,3,4}       スケール倍率（デフォルト: 2）
  --model MODEL         使用する超解像モデル
  --fps FPS             出力動画のFPS（未指定時は元動画を引き継ぐ）
  --tmp-dir TMP_DIR     中間ファイルを保存する一時ディレクトリ
  --no-cleanup          処理後に一時ファイルを削除しない
  --crf CRF             出力品質（0-51、低いほど高品質、デフォルト: 18）
  --preset PRESET       エンコード速度（ultrafast～veryslow、デフォルト: medium）
  --list-models         利用可能なモデルの一覧を表示
  --log-level LEVEL     ログレベル（デフォルト: INFO）
```

### 処理の中断と再開

処理を中断しても、`--tmp-dir` を指定していれば再開可能です：

```powershell
# 処理開始（一時ディレクトリを指定）
docker exec sr-movies-gpu python -m app.main `
    -i /data/video.mp4 `
    -o /data/video_2x.mp4 `
    --scale 2 `
    --tmp-dir /data/tmp_video `
    --no-cleanup

# 中断後、同じコマンドで再開（処理済みフレームはスキップされる）
docker exec sr-movies-gpu python -m app.main `
    -i /data/video.mp4 `
    -o /data/video_2x.mp4 `
    --scale 2 `
    --tmp-dir /data/tmp_video `
    --no-cleanup
```

### 一時ファイルのクリーンアップ

```powershell
# 特定の一時ディレクトリを削除
docker exec sr-movies-gpu python -m app.main --clean /data/tmp_video

# または手動で削除
docker exec sr-movies-gpu rm -rf /data/tmp_video
```

## 処理パイプライン

1. **フレーム抽出**: 入力動画をPNGフレームに分解
2. **超解像処理**: Real-ESRGANモデルで各フレームを高解像度化
3. **動画再構成**: 超解像フレームを動画に結合（元音声を保持）

```
入力動画 (1080p) → フレーム抽出 → Real-ESRGAN → 動画再構成 → 出力動画 (2160p)
```

## 利用可能なモデル

| モデル名 | 説明 | 推奨用途 |
|---------|------|----------|
| RealESRGAN_x4plus | 高品質汎用モデル（デフォルト） | 実写映像全般 |
| RealESRNet_x4plus | シャープな出力 | ノイズの少ない映像 |
| realesr-general-x4v3 | 最新汎用モデル | 幅広い映像 |

## パフォーマンス

### 処理速度の目安（RTX 3090）

| 解像度 | スケール | 速度 |
|--------|----------|------|
| 1080x608 | 2x | 約0.4フレーム/秒 |
| 1920x1080 | 2x | 約0.2フレーム/秒 |

### メモリ使用量

- **VRAM**: 約8-12GB（モデルと解像度による）
- **システムRAM**: 約600MB（メモリ効率化済み）

## トラブルシューティング

### GPUが認識されない

```powershell
# NVIDIAドライバーの確認
nvidia-smi

# Dockerコンテナ内でGPUを確認
docker exec sr-movies-gpu python -c "import torch; print(torch.cuda.is_available())"
```

### メモリ不足エラー

処理はメモリ効率的に1フレームずつ行われますが、大きな解像度の場合はバッチサイズを調整してください。

### 処理が遅い

- `--preset fast` を使用してエンコード速度を上げる
- 一時ディレクトリにSSDを使用する

## 開発

### ローカル開発環境

```powershell
# 仮想環境の作成
python -m venv .venv
.venv\Scripts\Activate.ps1

# 依存関係のインストール
pip install -r requirements.txt
```

### コンテナの再ビルド

コードを変更した場合：

```powershell
# コンテナを停止
docker compose -f docker/docker-compose.yml --profile gpu down

# イメージを再ビルド
docker compose -f docker/docker-compose.yml --profile gpu build app-gpu

# コンテナを再起動
docker compose -f docker/docker-compose.yml --profile gpu up -d
```

**注意**: ソースコードはボリュームマウントされているため、Python コードの変更は再ビルド不要で即座に反映されます。再ビルドが必要なのは `Dockerfile` や `requirements.txt` を変更した場合のみです。

## ディレクトリ構造

```
super_resolution_movies/
├── app/                          # アプリケーションコード
│   ├── main.py                   # CLIエントリポイント
│   └── video_pipeline/           # 処理パイプライン
│       ├── config.py             # 設定管理
│       ├── extract_frames.py     # フレーム抽出
│       ├── upscale_frames.py     # 超解像処理
│       ├── realesrgan_cuda.py    # Real-ESRGAN CUDA実装
│       ├── rebuild_video.py      # 動画再構成
│       └── gpu_utils.py          # GPU検出ユーティリティ
├── config/                       # 設定ファイル
│   └── settings.example.yaml     # 設定例
├── data/                         # 入出力データ
├── docker/                       # Docker関連
│   ├── Dockerfile                # イメージ定義
│   └── docker-compose.yml        # Compose設定
├── requirements.txt              # Python依存関係
└── README.md                     # このファイル
```

## ライセンス

MIT License

## 謝辞

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 超解像モデル
- [FFmpeg](https://ffmpeg.org/) - 動画処理
