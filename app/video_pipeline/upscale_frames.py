"""
超解像モデルを呼び出すラッパーモジュール

Real-ESRGANなどの外部CLIツールを呼び出し、
フレーム画像の超解像処理を行う。
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import ModelConfig, Settings

logger = logging.getLogger(__name__)


def upscale_frames(
    input_dir: str,
    output_dir: str,
    settings: Settings,
    model_key: Optional[str] = None,
    scale: int = 2,
    model_name: Optional[str] = None,
) -> int:
    """
    ディレクトリ内のフレーム画像を一括で超解像処理する
    
    Args:
        input_dir: 入力フレームのディレクトリ
        output_dir: 出力フレームのディレクトリ
        settings: アプリケーション設定
        model_key: 使用するモデルのキー（settings内のキー）
        scale: スケール倍率（2, 3, 4など）
        model_name: モデル内部名（オプション）
        
    Returns:
        int: 処理したフレーム数
        
    Raises:
        FileNotFoundError: 入力ディレクトリが存在しない場合
        ValueError: 指定されたモデルが設定に存在しない場合
        RuntimeError: 超解像処理に失敗した場合
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
    
    # モデル設定の取得
    model_key = model_key or settings.default_model
    if model_key not in settings.models:
        raise ValueError(f"モデル '{model_key}' が設定に存在しません。利用可能: {list(settings.models.keys())}")
    
    model_config = settings.models[model_key]
    
    # スケールのバリデーション
    if scale not in model_config.supported_scales:
        logger.warning(
            f"スケール {scale} はモデル '{model_key}' でサポートされていません。"
            f"サポート: {model_config.supported_scales}"
        )
    
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 入力フレームの取得
    frame_files = sorted(input_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(input_dir.glob("*.jpg"))
    
    if not frame_files:
        logger.warning("処理するフレームが見つかりません")
        return 0
    
    logger.info(f"超解像処理開始: {len(frame_files)}フレーム, スケール: {scale}x")
    
    # モデル実行ファイルの存在確認
    executable = model_config.executable
    executable_path = shutil.which(executable)
    
    if executable_path is None:
        logger.warning(f"超解像モデル '{executable}' が見つかりません。フォールバック処理を実行します。")
        return _fallback_upscale(frame_files, output_dir, scale)
    
    # バッチ処理かフレームごとの処理かを判定
    # Real-ESRGANはディレクトリ単位の処理をサポート
    if _supports_batch_processing(model_config):
        return _batch_upscale(
            input_dir, output_dir, model_config, scale, model_name, settings
        )
    else:
        return _sequential_upscale(
            frame_files, output_dir, model_config, scale, model_name, settings
        )


def _supports_batch_processing(model_config: ModelConfig) -> bool:
    """モデルがバッチ処理をサポートするか判定"""
    # Real-ESRGANはディレクトリ指定でバッチ処理可能
    return "realesrgan" in model_config.executable.lower()


def _batch_upscale(
    input_dir: Path,
    output_dir: Path,
    model_config: ModelConfig,
    scale: int,
    model_name: Optional[str],
    settings: Settings,
) -> int:
    """
    ディレクトリ単位でバッチ処理を行う（Real-ESRGAN向け）
    """
    # デフォルトのモデル名を設定
    if model_name is None:
        if scale == 4:
            model_name = "realesrgan-x4plus"
        elif scale == 2:
            model_name = "realesrgan-x4plus"  # 2xでも4xモデルを使用し、出力サイズを調整
        else:
            model_name = "realesrgan-x4plus"
    
    # コマンドの構築
    cmd = [
        model_config.executable,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-s", str(scale),
        "-n", model_name,
    ]
    
    # GPU設定
    if settings.use_gpu:
        cmd.extend(["-g", str(settings.gpu_id)])
    
    # フォーマット指定
    cmd.extend(["-f", settings.frame_format])
    
    logger.info(f"バッチ処理実行: {' '.join(cmd)}")
    
    try:
        # 入力フレーム数をカウント（進捗表示用）
        input_frames = list(input_dir.glob("*.png"))
        total_frames = len(input_frames)
        
        # プロセス実行
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 進捗バーを表示しながら出力を監視
        with tqdm(total=total_frames, desc="超解像処理", unit="frames") as pbar:
            processed = 0
            while True:
                # 出力ディレクトリのファイル数をチェック
                current_files = len(list(output_dir.glob("*.png")))
                if current_files > processed:
                    pbar.update(current_files - processed)
                    processed = current_files
                
                # プロセスが終了したかチェック
                if process.poll() is not None:
                    # 最終的なファイル数で更新
                    final_files = len(list(output_dir.glob("*.png")))
                    if final_files > processed:
                        pbar.update(final_files - processed)
                    break
        
        if process.returncode != 0:
            _, stderr = process.communicate()
            raise RuntimeError(f"超解像処理に失敗しました: {stderr}")
        
        output_frames = len(list(output_dir.glob("*.png")))
        logger.info(f"バッチ処理完了: {output_frames}フレーム")
        return output_frames
        
    except FileNotFoundError:
        logger.error(f"実行ファイルが見つかりません: {model_config.executable}")
        raise


def _sequential_upscale(
    frame_files: list[Path],
    output_dir: Path,
    model_config: ModelConfig,
    scale: int,
    model_name: Optional[str],
    settings: Settings,
) -> int:
    """
    フレームを1枚ずつ順次処理する
    """
    processed = 0
    
    for frame_path in tqdm(frame_files, desc="超解像処理", unit="frames"):
        output_path = output_dir / frame_path.name
        
        # コマンドテンプレートを展開
        cmd_str = model_config.args_template.format(
            input=str(frame_path),
            output=str(output_path),
            scale=scale,
            model_name=model_name or "default",
        )
        
        cmd = [model_config.executable] + cmd_str.split()
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            logger.error(f"フレーム処理に失敗: {frame_path}, エラー: {e.stderr}")
            continue
    
    logger.info(f"順次処理完了: {processed}/{len(frame_files)}フレーム")
    return processed


def _fallback_upscale(
    frame_files: list[Path],
    output_dir: Path,
    scale: int,
) -> int:
    """
    超解像モデルが利用できない場合のフォールバック処理
    
    PILを使用した単純なリサイズを行う（品質は劣る）
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillowがインストールされていません。pip install Pillow を実行してください。")
        raise RuntimeError("フォールバック処理に必要なPillowがインストールされていません")
    
    logger.warning("フォールバックモード: 単純なバイキュービックリサイズを使用します（品質は超解像モデルより劣ります）")
    
    processed = 0
    for frame_path in tqdm(frame_files, desc="リサイズ処理（フォールバック）", unit="frames"):
        try:
            img = Image.open(frame_path)
            new_size = (img.width * scale, img.height * scale)
            resized = img.resize(new_size, Image.Resampling.BICUBIC)
            
            output_path = output_dir / frame_path.name
            resized.save(output_path, "PNG")
            processed += 1
        except Exception as e:
            logger.error(f"フレームのリサイズに失敗: {frame_path}, エラー: {e}")
            continue
    
    logger.info(f"フォールバック処理完了: {processed}/{len(frame_files)}フレーム")
    return processed


def list_available_models(settings: Settings) -> dict[str, dict]:
    """
    利用可能なモデルの一覧を取得する
    
    Args:
        settings: アプリケーション設定
        
    Returns:
        dict: モデルキーをキーとする情報辞書
    """
    result = {}
    for key, config in settings.models.items():
        executable_path = shutil.which(config.executable)
        result[key] = {
            "name": config.name,
            "executable": config.executable,
            "available": executable_path is not None,
            "supported_scales": config.supported_scales,
            "description": config.description,
        }
    return result
