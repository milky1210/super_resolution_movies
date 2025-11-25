"""
Real-ESRGAN CUDA版を使った超解像処理

学習済みReal-ESRGANモデルを使用してGPU（CUDA）で高速処理を行う
basicsrのバージョン互換問題を回避するため、直接モデルをロード
"""

import logging
from pathlib import Path
import urllib.request
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _download_model(model_url: str, model_path: Path) -> Path:
    """モデルファイルをダウンロード"""
    if model_path.exists():
        logger.info(f"モデルが存在します: {model_path}")
        return model_path
    
    logger.info(f"モデルをダウンロード中: {model_url}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, model_path)
    logger.info(f"ダウンロード完了: {model_path}")
    return model_path


def _create_realesrgan_model(scale: int = 4):
    """Real-ESRGANモデルを作成（RRDBNetアーキテクチャ）"""
    import torch
    import torch.nn as nn
    
    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            
        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x
    
    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
            
        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x
    
    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            
        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out
    
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)


def upscale_with_cuda(
    input_dir: Path,
    output_dir: Path,
    scale: int = 2,
    model_name: str = 'RealESRGAN_x4plus',
) -> int:
    """
    CUDAを使用してフレームを超解像処理
    
    Args:
        input_dir: 入力フレームディレクトリ
        output_dir: 出力フレームディレクトリ
        scale: スケール倍率
        model_name: モデル名
        
    Returns:
        int: 処理したフレーム数
    """
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import numpy as np
    except ImportError as e:
        logger.error(f"必要なライブラリがインストールされていません: {e}")
        logger.error("pip install torch pillow numpy を実行してください")
        raise
    
    # CUDAが利用可能かチェック
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        logger.warning("CUDAが利用できません。CPUで処理します（遅くなります）")
    else:
        logger.info(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
    
    # モデルのダウンロードとロード（実写向け高品質モデル）
    model_dir = Path('/tmp/realesrgan_models')
    
    # 実写向けモデルの選択（品質順）
    # 1. realesr-general-x4v3: 最新の実写向け汎用モデル（推奨）
    # 2. RealESRNet_x4plus: ノイズ除去なし、シャープな結果
    # 3. RealESRGAN_x4plus: バランス型
    
    if model_name == 'realesr-general-x4v3':
        model_path = model_dir / 'realesr-general-x4v3.pth'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    elif model_name == 'RealESRNet_x4plus':
        model_path = model_dir / 'RealESRNet_x4plus.pth'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
    else:
        model_path = model_dir / 'RealESRGAN_x4plus.pth'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    try:
        _download_model(model_url, model_path)
        model = _create_realesrgan_model(scale=4)
        state_dict = torch.load(model_path, map_location=device)
        
        # state_dictのキーを調整（params_emaまたはparamsから取得）
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(device)
        logger.info(f"Real-ESRGANモデルをロード: {model_name}")
        use_model = True
    except Exception as e:
        logger.warning(f"モデルのロードに失敗: {e}")
        logger.info("bicubicアップスケーリングにフォールバック")
        use_model = False
    
    # 入力フレームを取得
    frame_files = sorted(input_dir.glob("*.png"))
    if not frame_files:
        logger.warning("処理するフレームが見つかりません")
        return 0
    
    logger.info(f"GPU超解像処理開始: {len(frame_files)}フレーム, デバイス: {device}")
    
    # メモリ効率重視: 1フレームずつ処理（VRAM 24GB, RAM 64GBを効率的に使用）
    # バッチ処理はCPUメモリを大量消費するため、単一フレーム処理に変更
    processed = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 処理済みフレームをスキップ（再開機能）
    existing_frames = set(f.name for f in output_dir.glob("*.png"))
    frames_to_process = [f for f in frame_files if f.name not in existing_frames]
    
    if existing_frames:
        logger.info(f"処理済みフレーム: {len(existing_frames)}件をスキップ")
        processed = len(existing_frames)
    
    logger.info(f"処理対象: {len(frames_to_process)}フレーム")
    
    # 定期的なメモリクリーンアップ間隔
    gc_interval = 50
    
    for i, frame_path in enumerate(tqdm(frames_to_process, desc="GPU超解像処理", unit="frame")):
        try:
            # 画像読み込み（メモリ効率的に1枚ずつ）
            img = Image.open(frame_path).convert('RGB')
            w, h = img.size
            
            # PyTorchテンソルに変換
            img_np = np.array(img).astype(np.float32) / 255.0
            del img  # PIL画像を即座に解放
            
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
            del img_np  # numpy配列を即座に解放
            
            with torch.no_grad():
                if use_model:
                    # Real-ESRGANモデルで処理（4倍出力）
                    output_tensor = model(img_tensor)
                    del img_tensor  # 入力テンソルを即座に解放
                    
                    # scale=2の場合は4倍出力を2倍にリサイズ
                    if scale == 2:
                        output_tensor = F.interpolate(
                            output_tensor,
                            size=(h * 2, w * 2),  # PyTorchは(height, width)の順
                            mode='bicubic',
                            align_corners=False
                        )
                else:
                    # Bicubicフォールバック
                    output_tensor = F.interpolate(
                        img_tensor,
                        size=(h * scale, w * scale),
                        mode='bicubic',
                        align_corners=False
                    )
                    del img_tensor
            
            # 出力を保存（GPU→CPU転送を最小化）
            output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            del output_tensor  # GPUテンソルを即座に解放
            
            output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
            output_img = Image.fromarray(output_np, 'RGB')
            del output_np  # numpy配列を即座に解放
            
            output_path = output_dir / frame_path.name
            output_img.save(output_path, 'PNG', compress_level=1)
            del output_img  # PIL画像を即座に解放
            
            processed += 1
            
            # 定期的なメモリクリーンアップ
            if (i + 1) % gc_interval == 0:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            
        except Exception as e:
            logger.error(f"フレーム処理エラー: {frame_path}, {e}")
            # エラー時もメモリをクリーンアップ
            if device == 'cuda':
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            continue
    
    # 最終クリーンアップ
    if device == 'cuda':
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    logger.info(f"GPU超解像処理完了: {processed}/{len(frame_files)}フレーム")
    return processed


def is_cuda_available() -> bool:
    """CUDAが利用可能かチェック"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
