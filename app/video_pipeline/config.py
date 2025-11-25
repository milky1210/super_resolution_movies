"""
設定ファイルの読み込みと管理を行うモジュール

YAML形式の設定ファイルを読み込み、アプリケーション全体で
使用する設定値を提供する。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    """超解像モデルの設定"""
    name: str
    executable: str
    args_template: str
    supported_scales: list[int] = field(default_factory=lambda: [2, 3, 4])
    description: str = ""


@dataclass
class Settings:
    """アプリケーション全体の設定"""
    # 一時ディレクトリのデフォルトパス
    default_tmp_dir: str = "/tmp/super_resolution"
    
    # ffmpeg関連の設定
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    
    # デフォルトの出力設定
    default_scale: int = 2
    default_output_format: str = "mp4"
    default_video_codec: str = "libx264"
    default_audio_codec: str = "aac"
    default_crf: int = 18  # 品質設定（低いほど高品質）
    
    # フレーム抽出設定
    frame_format: str = "png"
    frame_pattern: str = "%06d.png"
    
    # 超解像モデルの設定
    models: dict[str, ModelConfig] = field(default_factory=dict)
    default_model: str = "realesrgan"
    
    # GPU設定
    use_gpu: bool = True
    gpu_id: int = 0
    
    # ログ設定
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # クリーンアップ設定
    cleanup_tmp: bool = True


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    設定ファイルを読み込んでSettingsオブジェクトを返す
    
    Args:
        config_path: 設定ファイルのパス。Noneの場合はデフォルト設定を返す
        
    Returns:
        Settings: 読み込んだ設定オブジェクト
    """
    settings = Settings()
    
    # デフォルトのモデル設定を追加
    settings.models = {
        "realesrgan": ModelConfig(
            name="Real-ESRGAN",
            executable="realesrgan-ncnn-vulkan",
            args_template="-i {input} -o {output} -s {scale} -n {model_name}",
            supported_scales=[2, 3, 4],
            description="Real-ESRGAN NCNN Vulkan版（GPU/CPU両対応）"
        ),
        "realesrgan_anime": ModelConfig(
            name="Real-ESRGAN Anime",
            executable="realesrgan-ncnn-vulkan",
            args_template="-i {input} -o {output} -s {scale} -n realesrgan-x4plus-anime",
            supported_scales=[4],
            description="Real-ESRGAN アニメ特化モデル"
        ),
    }
    
    if config_path is None:
        # 環境変数からデフォルトパスを取得
        config_path = os.environ.get("SR_CONFIG_PATH")
    
    if config_path is None:
        # デフォルトの設定ファイルパスを探す
        default_paths = [
            Path("config/settings.yaml"),
            Path("config/settings.yml"),
            Path("/workspace/config/settings.yaml"),
        ]
        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path and Path(config_path).exists():
        settings = _load_from_yaml(config_path, settings)
    
    return settings


def _load_from_yaml(config_path: str, defaults: Settings) -> Settings:
    """
    YAMLファイルから設定を読み込む
    
    Args:
        config_path: YAMLファイルのパス
        defaults: デフォルト設定
        
    Returns:
        Settings: 読み込んだ設定オブジェクト
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # 基本設定の更新
    settings_dict = {
        "default_tmp_dir": data.get("default_tmp_dir", defaults.default_tmp_dir),
        "ffmpeg_path": data.get("ffmpeg_path", defaults.ffmpeg_path),
        "ffprobe_path": data.get("ffprobe_path", defaults.ffprobe_path),
        "default_scale": data.get("default_scale", defaults.default_scale),
        "default_output_format": data.get("default_output_format", defaults.default_output_format),
        "default_video_codec": data.get("default_video_codec", defaults.default_video_codec),
        "default_audio_codec": data.get("default_audio_codec", defaults.default_audio_codec),
        "default_crf": data.get("default_crf", defaults.default_crf),
        "frame_format": data.get("frame_format", defaults.frame_format),
        "frame_pattern": data.get("frame_pattern", defaults.frame_pattern),
        "default_model": data.get("default_model", defaults.default_model),
        "use_gpu": data.get("use_gpu", defaults.use_gpu),
        "gpu_id": data.get("gpu_id", defaults.gpu_id),
        "log_level": data.get("log_level", defaults.log_level),
        "log_file": data.get("log_file", defaults.log_file),
        "cleanup_tmp": data.get("cleanup_tmp", defaults.cleanup_tmp),
    }
    
    settings = Settings(**settings_dict)
    
    # モデル設定の更新
    settings.models = defaults.models.copy()
    if "models" in data:
        for model_key, model_data in data["models"].items():
            settings.models[model_key] = ModelConfig(
                name=model_data.get("name", model_key),
                executable=model_data.get("executable", ""),
                args_template=model_data.get("args_template", ""),
                supported_scales=model_data.get("supported_scales", [2, 3, 4]),
                description=model_data.get("description", ""),
            )
    
    return settings


def get_video_info(video_path: str, ffprobe_path: str = "ffprobe") -> dict[str, Any]:
    """
    ffprobeを使用して動画の情報を取得する
    
    Args:
        video_path: 動画ファイルのパス
        ffprobe_path: ffprobeのパス
        
    Returns:
        dict: 動画の情報（fps, width, height, duration等）
    """
    import json
    import subprocess
    
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    video_info: dict[str, Any] = {
        "duration": None,
        "fps": None,
        "width": None,
        "height": None,
        "has_audio": False,
    }
    
    # フォーマット情報から継続時間を取得
    if "format" in data:
        video_info["duration"] = float(data["format"].get("duration", 0))
    
    # ストリーム情報からビデオ・オーディオ情報を取得
    for stream in data.get("streams", []):
        if stream["codec_type"] == "video":
            video_info["width"] = stream.get("width")
            video_info["height"] = stream.get("height")
            # FPSの計算
            if "r_frame_rate" in stream:
                fps_parts = stream["r_frame_rate"].split("/")
                if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                    video_info["fps"] = int(fps_parts[0]) / int(fps_parts[1])
        elif stream["codec_type"] == "audio":
            video_info["has_audio"] = True
    
    return video_info
