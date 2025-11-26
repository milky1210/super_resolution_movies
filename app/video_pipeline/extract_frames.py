"""
ffmpegを使用して動画からフレームと音声を抽出するモジュール

動画ファイルを受け取り、指定されたディレクトリに
連番PNG画像と音声ファイルを出力する。
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import get_video_info

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    frame_pattern: str = "%06d.png",
    fps: Optional[float] = None,
    scale_factor: Optional[float] = None,
) -> tuple[int, float]:
    """
    動画からフレームを抽出する
    
    Args:
        video_path: 入力動画のパス
        output_dir: フレーム出力ディレクトリ
        ffmpeg_path: ffmpegのパス
        ffprobe_path: ffprobeのパス
        frame_pattern: 出力ファイル名のパターン（例：%06d.png）
        fps: 抽出するFPS（Noneの場合は元動画のFPSを使用）
        scale_factor: 抽出時のリサイズ倍率（0.5で半分、Noneでそのまま）
        
    Returns:
        tuple[int, float]: (抽出したフレーム数, 使用したFPS)
        
    Raises:
        FileNotFoundError: 入力動画が存在しない場合
        subprocess.CalledProcessError: ffmpegの実行に失敗した場合
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"入力動画が見つかりません: {video_path}")
    
    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 動画情報の取得
    logger.info(f"動画情報を取得中: {video_path}")
    video_info = get_video_info(str(video_path), ffprobe_path)
    
    original_fps = video_info.get("fps", 30.0)
    target_fps = fps if fps is not None else original_fps
    duration = video_info.get("duration", 0)
    
    logger.info(f"元動画のFPS: {original_fps}, 使用FPS: {target_fps}, 長さ: {duration:.2f}秒")
    
    # フィルタチェーンの構築
    vf_filters = [f"fps={target_fps}"]
    
    if scale_factor is not None and scale_factor != 1.0:
        # 縮小リサイズ（偶数にする）
        orig_w = video_info.get("width", 1920)
        orig_h = video_info.get("height", 1080)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        # 偶数に丸める
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        vf_filters.append(f"scale={new_w}:{new_h}")
        logger.info(f"リサイズ: {orig_w}x{orig_h} -> {new_w}x{new_h} (x{scale_factor})")
    
    # ffmpegコマンドの構築
    output_path = output_dir / frame_pattern
    cmd = [
        ffmpeg_path,
        "-y",  # 上書き確認なし
        "-i", str(video_path),
        "-vf", ",".join(vf_filters),
        "-q:v", "2",  # 高品質
        str(output_path)
    ]
    
    logger.info(f"フレーム抽出開始: {' '.join(cmd)}")
    
    # ffmpegの実行（進捗表示付き）
    process = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # 予想フレーム数を計算
    expected_frames = int(duration * target_fps) if duration else 0
    
    with tqdm(total=expected_frames, desc="フレーム抽出", unit="frames") as pbar:
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
        raise subprocess.CalledProcessError(process.returncode, cmd)
    
    # 実際に抽出されたフレーム数をカウント
    actual_frames = len(list(output_dir.glob("*.png")))
    logger.info(f"フレーム抽出完了: {actual_frames}フレーム")
    
    return actual_frames, target_fps


def extract_audio(
    video_path: str,
    output_path: str,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
    audio_codec: str = "copy",
) -> bool:
    """
    動画から音声を抽出する
    
    Args:
        video_path: 入力動画のパス
        output_path: 音声ファイルの出力パス
        ffmpeg_path: ffmpegのパス
        ffprobe_path: ffprobeのパス
        audio_codec: 音声コーデック（"copy"で無変換コピー）
        
    Returns:
        bool: 音声が存在して抽出できた場合True
        
    Raises:
        FileNotFoundError: 入力動画が存在しない場合
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"入力動画が見つかりません: {video_path}")
    
    # 動画に音声があるか確認
    video_info = get_video_info(str(video_path), ffprobe_path)
    if not video_info.get("has_audio", False):
        logger.info("この動画には音声ストリームがありません")
        return False
    
    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ffmpegコマンドの構築
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(video_path),
        "-vn",  # ビデオ無効
        "-acodec", audio_codec,
        str(output_path)
    ]
    
    logger.info(f"音声抽出開始: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"音声抽出完了: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"音声抽出に失敗しました: {e.stderr}")
        return False
