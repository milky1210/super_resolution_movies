"""
超解像動画処理ツールのCLIエントリポイント

動画を ffmpeg + Deep Learning ベースの超解像モデルで
エンドツーエンド処理する。

使用方法:
    python -m app.main -i input.mp4 -o output.mp4 --scale 2
    python app/main.py -i input.mp4 -o output.mp4 --scale 4 --model realesrgan
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .video_pipeline import (
    Settings,
    extract_audio,
    extract_frames,
    load_config,
    rebuild_video,
    upscale_frames,
)
from .video_pipeline.config import get_video_info
from .video_pipeline.upscale_frames import list_available_models
from .video_pipeline.gpu_utils import log_gpu_info

# ロガーの設定
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    ロギングの設定を行う
    
    Args:
        level: ログレベル（DEBUG, INFO, WARNING, ERROR）
        log_file: ログファイルのパス（Noneの場合は標準出力のみ）
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパースする
    
    Returns:
        argparse.Namespace: パースされた引数
    """
    parser = argparse.ArgumentParser(
        description="動画超解像処理ツール - ffmpeg + Deep Learningベースの超解像モデルを使用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  python -m app.main -i input.mp4 -o output.mp4

  # スケールとモデルを指定
  python -m app.main -i input.mp4 -o output.mp4 --scale 4 --model realesrgan

  # FPSと一時ディレクトリを指定
  python -m app.main -i input.mp4 -o output.mp4 --fps 30 --tmp-dir /tmp/sr_work

  # 利用可能なモデルを表示
  python -m app.main --list-models
        """,
    )
    
    # 入出力オプション
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="入力動画のパス",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="出力動画のパス",
    )
    
    # 処理オプション
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="スケール倍率（デフォルト: 2）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="使用する超解像モデル（設定ファイル内のキー）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="出力動画のフレームレート（未指定時は元動画を引き継ぐ）",
    )
    
    # ディレクトリオプション
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="中間フレームを保存する一時ディレクトリ",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="処理後に一時ファイルを削除しない",
    )
    
    # 設定オプション
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="設定ファイル（YAML）のパス",
    )
    
    # エンコードオプション
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="出力動画の品質（0-51、低いほど高品質、デフォルト: 18）",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", 
                 "medium", "slow", "slower", "veryslow"],
        help="エンコード速度プリセット（デフォルト: medium）",
    )
    
    # その他のオプション
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="利用可能な超解像モデルの一覧を表示",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル（デフォルト: INFO）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="ログファイルのパス",
    )
    
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[bool, str]:
    """
    入力パラメータのバリデーション
    
    Args:
        args: コマンドライン引数
        
    Returns:
        tuple[bool, str]: (成功フラグ, エラーメッセージ)
    """
    if not args.input:
        return False, "入力動画パス（-i/--input）を指定してください"
    
    input_path = Path(args.input)
    if not input_path.exists():
        return False, f"入力動画が見つかりません: {input_path}"
    
    if not args.output:
        return False, "出力動画パス（-o/--output）を指定してください"
    
    output_path = Path(args.output)
    if output_path.exists():
        logger.warning(f"出力ファイルが既に存在します。上書きされます: {output_path}")
    
    # 出力ディレクトリの書き込み権限確認
    output_dir = output_path.parent
    if output_dir.exists() and not output_dir.is_dir():
        return False, f"出力パスの親が有効なディレクトリではありません: {output_dir}"
    
    return True, ""


def process_video(
    input_path: str,
    output_path: str,
    settings: Settings,
    scale: int = 2,
    model_key: Optional[str] = None,
    fps: Optional[float] = None,
    tmp_dir: Optional[str] = None,
    cleanup: bool = True,
    crf: Optional[int] = None,
    preset: str = "medium",
) -> bool:
    """
    動画の超解像処理を実行する
    
    Args:
        input_path: 入力動画のパス
        output_path: 出力動画のパス
        settings: アプリケーション設定
        scale: スケール倍率
        model_key: 使用するモデルのキー
        fps: 出力FPS（Noneの場合は元動画を使用）
        tmp_dir: 一時ディレクトリ（Noneの場合は自動作成）
        cleanup: 処理後に一時ファイルを削除するか
        crf: 出力品質
        preset: エンコードプリセット
        
    Returns:
        bool: 成功した場合True
    """
    # 一時ディレクトリの設定
    if tmp_dir:
        work_dir = Path(tmp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="super_resolution_"))
        use_temp = True
    
    # サブディレクトリの作成
    frames_original_dir = work_dir / "frames_original"
    frames_upscaled_dir = work_dir / "frames_upscaled"
    audio_file = work_dir / "audio.aac"
    
    logger.info(f"作業ディレクトリ: {work_dir}")
    
    try:
        # 1. 動画情報の取得
        logger.info("=" * 50)
        logger.info("ステップ 1/4: 動画情報の取得")
        logger.info("=" * 50)
        
        video_info = get_video_info(input_path, settings.ffprobe_path)
        logger.info(f"解像度: {video_info['width']}x{video_info['height']}")
        logger.info(f"FPS: {video_info['fps']}")
        logger.info(f"長さ: {video_info['duration']:.2f}秒")
        logger.info(f"音声: {'あり' if video_info['has_audio'] else 'なし'}")
        
        target_fps = fps if fps is not None else video_info["fps"]
        
        # 2. フレーム抽出
        logger.info("=" * 50)
        logger.info("ステップ 2/4: フレーム抽出")
        logger.info("=" * 50)
        
        frame_count, actual_fps = extract_frames(
            video_path=input_path,
            output_dir=str(frames_original_dir),
            ffmpeg_path=settings.ffmpeg_path,
            ffprobe_path=settings.ffprobe_path,
            frame_pattern=settings.frame_pattern,
            fps=target_fps,
        )
        logger.info(f"抽出完了: {frame_count}フレーム")
        
        # 音声抽出
        has_audio = False
        if video_info["has_audio"]:
            has_audio = extract_audio(
                video_path=input_path,
                output_path=str(audio_file),
                ffmpeg_path=settings.ffmpeg_path,
                ffprobe_path=settings.ffprobe_path,
            )
        
        # 3. 超解像処理
        logger.info("=" * 50)
        logger.info("ステップ 3/4: 超解像処理")
        logger.info("=" * 50)
        
        upscaled_count = upscale_frames(
            input_dir=str(frames_original_dir),
            output_dir=str(frames_upscaled_dir),
            settings=settings,
            model_key=model_key,
            scale=scale,
        )
        logger.info(f"超解像処理完了: {upscaled_count}フレーム")
        
        # 4. 動画再構成
        logger.info("=" * 50)
        logger.info("ステップ 4/4: 動画再構成")
        logger.info("=" * 50)
        
        success = rebuild_video(
            frames_dir=str(frames_upscaled_dir),
            output_path=output_path,
            fps=actual_fps,
            ffmpeg_path=settings.ffmpeg_path,
            frame_pattern=settings.frame_pattern,
            audio_path=str(audio_file) if has_audio else None,
            video_codec=settings.default_video_codec,
            audio_codec=settings.default_audio_codec,
            crf=crf if crf is not None else settings.default_crf,
            preset=preset,
        )
        
        if success:
            logger.info("=" * 50)
            logger.info("処理完了！")
            logger.info(f"出力ファイル: {output_path}")
            target_resolution = (
                video_info["width"] * scale,
                video_info["height"] * scale
            )
            logger.info(f"出力解像度: {target_resolution[0]}x{target_resolution[1]}")
            logger.info("=" * 50)
        
        return success
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        raise
        
    finally:
        # クリーンアップ
        if cleanup and use_temp:
            logger.info("一時ファイルを削除中...")
            shutil.rmtree(work_dir, ignore_errors=True)
        elif not cleanup:
            logger.info(f"一時ファイルを保持: {work_dir}")


def main() -> int:
    """
    メインエントリポイント
    
    Returns:
        int: 終了コード（0: 成功, 1: エラー）
    """
    args = parse_args()
    
    # ロギングの設定
    setup_logging(args.log_level, args.log_file)
    
    # 設定の読み込み
    settings = load_config(args.config)
    
    # モデル一覧の表示
    if args.list_models:
        models = list_available_models(settings)
        print("\n利用可能な超解像モデル:")
        print("-" * 60)
        for key, info in models.items():
            status = "✓" if info["available"] else "✗"
            print(f"  [{status}] {key}")
            print(f"      名前: {info['name']}")
            print(f"      実行ファイル: {info['executable']}")
            print(f"      対応スケール: {info['supported_scales']}")
            print(f"      説明: {info['description']}")
            print()
        return 0
    
    # GPU利用可能性チェック
    gpu_available = log_gpu_info()
    if not gpu_available:
        logger.warning("GPUが利用できません。CPUモードで処理します（処理が遅くなります）")
    
    # 入力検証
    valid, error_msg = validate_inputs(args)
    if not valid:
        logger.error(error_msg)
        return 1
    
    try:
        # 動画処理の実行
        success = process_video(
            input_path=args.input,
            output_path=args.output,
            settings=settings,
            scale=args.scale,
            model_key=args.model,
            fps=args.fps,
            tmp_dir=args.tmp_dir,
            cleanup=not args.no_cleanup,
            crf=args.crf,
            preset=args.preset,
        )
        
        return 0 if success else 1
        
    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"処理エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("処理が中断されました")
        return 130


if __name__ == "__main__":
    sys.exit(main())
