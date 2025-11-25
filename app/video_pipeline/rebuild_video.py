"""
超解像フレームと音声から動画を再構成するモジュール

ffmpegを使用して、超解像処理されたフレーム画像と
元の音声ストリームから出力動画を生成する。
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


def rebuild_video(
    frames_dir: str,
    output_path: str,
    fps: float,
    ffmpeg_path: str = "ffmpeg",
    frame_pattern: str = "%06d.png",
    audio_path: Optional[str] = None,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 18,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
) -> bool:
    """
    フレーム画像から動画を再構成する
    
    Args:
        frames_dir: 超解像フレームのディレクトリ
        output_path: 出力動画のパス
        fps: 出力動画のフレームレート
        ffmpeg_path: ffmpegのパス
        frame_pattern: フレームファイル名のパターン
        audio_path: 音声ファイルのパス（Noneの場合は音声なし）
        video_codec: ビデオコーデック
        audio_codec: オーディオコーデック
        crf: 品質設定（低いほど高品質、0-51の範囲）
        preset: エンコード速度プリセット
        pixel_format: ピクセルフォーマット
        
    Returns:
        bool: 成功した場合True
        
    Raises:
        FileNotFoundError: フレームディレクトリが存在しない場合
        subprocess.CalledProcessError: ffmpegの実行に失敗した場合
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    if not frames_dir.exists():
        raise FileNotFoundError(f"フレームディレクトリが見つかりません: {frames_dir}")
    
    # フレームファイルの存在確認
    frame_files = list(frames_dir.glob("*.png"))
    if not frame_files:
        frame_files = list(frames_dir.glob("*.jpg"))
    
    if not frame_files:
        logger.error("フレームファイルが見つかりません")
        return False
    
    total_frames = len(frame_files)
    logger.info(f"動画再構成開始: {total_frames}フレーム, FPS: {fps}")
    
    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ffmpegコマンドの構築
    input_pattern = frames_dir / frame_pattern
    
    cmd = [
        ffmpeg_path,
        "-y",  # 上書き確認なし
        "-framerate", str(fps),
        "-i", str(input_pattern),
    ]
    
    # 音声がある場合は追加
    if audio_path and Path(audio_path).exists():
        cmd.extend(["-i", str(audio_path)])
        cmd.extend(["-c:a", audio_codec])
        cmd.extend(["-shortest"])  # 短い方に合わせる
        logger.info(f"音声を追加: {audio_path}")
    
    # ビデオエンコード設定
    cmd.extend([
        "-c:v", video_codec,
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", pixel_format,
    ])
    
    # 出力ファイル
    cmd.append(str(output_path))
    
    logger.info(f"動画エンコード開始: {' '.join(cmd)}")
    
    # ffmpegの実行（進捗表示付き）
    process = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    with tqdm(total=total_frames, desc="動画エンコード", unit="frames") as pbar:
        frame_count = 0
        if process.stderr:
            for line in process.stderr:
                # ffmpegの出力からフレーム数を抽出
                if "frame=" in line:
                    try:
                        parts = line.split("frame=")
                        if len(parts) > 1:
                            frame_str = parts[1].split()[0].strip()
                            new_count = int(frame_str)
                            pbar.update(new_count - frame_count)
                            frame_count = new_count
                    except (ValueError, IndexError):
                        pass
        
        process.wait()
    
    if process.returncode != 0:
        logger.error("動画エンコードに失敗しました")
        raise subprocess.CalledProcessError(process.returncode, cmd)
    
    # 出力ファイルの確認
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"動画再構成完了: {output_path} ({file_size:.2f} MB)")
        return True
    else:
        logger.error("出力ファイルが作成されませんでした")
        return False


def get_recommended_settings(
    source_resolution: tuple[int, int],
    target_scale: int,
    content_type: str = "general",
) -> dict:
    """
    ソース解像度とスケールに基づいて推奨エンコード設定を返す
    
    Args:
        source_resolution: ソース解像度 (width, height)
        target_scale: スケール倍率
        content_type: コンテンツタイプ ("general", "anime", "film")
        
    Returns:
        dict: 推奨設定
    """
    target_height = source_resolution[1] * target_scale
    
    # 解像度に基づくCRF設定
    if target_height >= 2160:  # 4K
        base_crf = 18
        preset = "slow"
    elif target_height >= 1080:  # Full HD
        base_crf = 20
        preset = "medium"
    else:
        base_crf = 22
        preset = "medium"
    
    # コンテンツタイプによる調整
    if content_type == "anime":
        base_crf -= 2  # アニメは低CRFで高品質
        preset = "slow"
    elif content_type == "film":
        base_crf -= 1
        preset = "slower"
    
    return {
        "crf": max(0, min(51, base_crf)),
        "preset": preset,
        "video_codec": "libx264",
        "audio_codec": "aac",
        "pixel_format": "yuv420p",
    }
