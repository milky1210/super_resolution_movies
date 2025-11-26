"""
超解像アルゴリズム比較テスト

古典アルゴリズム vs ディープラーニングの速度・品質比較
"""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# 利用可能なアルゴリズム
ALGORITHMS = {
    # 古典アルゴリズム（OpenCV/PIL）
    "bicubic": "古典: Bicubic補間",
    "lanczos": "古典: Lanczos補間", 
    "nearest": "古典: 最近傍補間",
    # 古典 + 強調処理
    "lanczos_sharpen": "古典: Lanczos + シャープ",
    "lanczos_unsharp": "古典: Lanczos + Unsharp Mask",
    "lanczos_edge": "古典: Lanczos + エッジ強調",
    # 強調のみ（リサイズなし）
    "enhance_sharpen": "強調のみ: シャープ",
    "enhance_unsharp": "強調のみ: Unsharp Mask",
    "enhance_unsharp_strong": "強調のみ: Unsharp 強",
    "enhance_detail": "強調のみ: Detail強調",
    "enhance_contrast": "強調のみ: コントラスト強調",
    "enhance_combined": "強調のみ: 複合(Unsharp+コントラスト)",
    # ディープラーニング
    "realesrgan_x4plus": "Deep: RealESRGAN x4plus",
    "realesrnet_x4plus": "Deep: RealESRNet x4plus (シャープ)",
    "realesr_general_v3": "Deep: RealESR General v3 (汎用)",
}


def upscale_classical(
    input_path: Path,
    output_path: Path,
    scale: int,
    method: str = "bicubic"
) -> float:
    """
    古典アルゴリズムで超解像
    
    Returns:
        処理時間（秒）
    """
    from PIL import Image, ImageFilter, ImageEnhance
    
    start = time.perf_counter()
    
    img = Image.open(input_path).convert('RGB')
    w, h = img.size
    
    # 強調のみモード（リサイズなし）
    if method.startswith("enhance_"):
        upscaled = img
    else:
        new_size = (w * scale, h * scale)
        
        # ベースのリサンプリング方法を決定
        base_method = method.split("_")[0] if "_" in method else method
        
        if base_method == "bicubic":
            resample = Image.BICUBIC
        elif base_method == "lanczos":
            resample = Image.LANCZOS
        elif base_method == "nearest":
            resample = Image.NEAREST
        else:
            resample = Image.BICUBIC
        
        upscaled = img.resize(new_size, resample=resample)
    
    # 強調処理を適用
    if "sharpen" in method:
        # シャープフィルタ
        upscaled = upscaled.filter(ImageFilter.SHARPEN)
    elif "unsharp_strong" in method:
        # 強いUnsharp Mask
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=2))
    elif "unsharp" in method:
        # Unsharp Mask（より強いシャープ効果）
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    elif "edge" in method:
        # エッジ強調
        upscaled = upscaled.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif "detail" in method:
        # ディテール強調
        upscaled = upscaled.filter(ImageFilter.DETAIL)
    elif "contrast" in method:
        # コントラスト強調
        enhancer = ImageEnhance.Contrast(upscaled)
        upscaled = enhancer.enhance(1.2)
    elif "combined" in method:
        # 複合処理: Unsharp + コントラスト
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
        enhancer = ImageEnhance.Contrast(upscaled)
        upscaled = enhancer.enhance(1.15)
    
    upscaled.save(output_path, 'PNG')
    
    elapsed = time.perf_counter() - start
    return elapsed


def upscale_deep(
    input_path: Path,
    output_path: Path,
    scale: int,
    model_name: str = "RealESRGAN_x4plus"
) -> float:
    """
    ディープラーニングで超解像
    
    Returns:
        処理時間（秒）
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    
    # モデルをロード（キャッシュから）
    model, device = _get_cached_model(model_name)
    
    start = time.perf_counter()
    
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if model is not None:
            output_tensor = model(img_tensor)
        else:
            # フォールバック: bicubic
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            output_tensor = F.interpolate(
                img_tensor, size=(h * 4, w * 4), 
                mode='bicubic', align_corners=False
            )
    
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    
    output_img = Image.fromarray(output_np, 'RGB')
    output_img.save(output_path, 'PNG')
    
    elapsed = time.perf_counter() - start
    
    # メモリクリア
    del img_tensor, output_tensor
    torch.cuda.empty_cache()
    
    return elapsed


# モデルキャッシュ
_model_cache: Dict[str, Tuple] = {}


def _get_cached_model(model_name: str):
    """モデルをキャッシュからロード"""
    global _model_cache
    
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # RRDBNetモデル定義
    model = _create_realesrgan_model()
    
    # モデルURLマッピング
    model_urls = {
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    }
    
    model_dir = Path('/tmp/realesrgan_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        import urllib.request
        url = model_urls.get(model_name, model_urls["RealESRGAN_x4plus"])
        logger.info(f"モデルダウンロード中: {model_name}")
        urllib.request.urlretrieve(url, model_path)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(device)
        logger.info(f"モデルロード完了: {model_name}")
    except Exception as e:
        logger.warning(f"モデルロード失敗: {e}")
        model = None
    
    _model_cache[model_name] = (model, device)
    return model, device


def _create_realesrgan_model():
    """RRDBNetモデルを作成"""
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
    
    return RRDBNet()


def run_comparison(
    input_dir: Path,
    output_base_dir: Path,
    pre_shrink: float = 1.0,
    algorithms: List[str] = None,
    output_size: Tuple[int, int] = None,  # 出力サイズを指定（None=自動）
) -> Dict[str, Dict]:
    """
    アルゴリズム比較テストを実行
    
    Args:
        input_dir: 入力フレームディレクトリ
        output_base_dir: 出力ベースディレクトリ
        pre_shrink: 前処理縮小率（1.0=なし, 0.5=半分）
        algorithms: テストするアルゴリズム一覧
        output_size: 最終出力サイズ (width, height)。Noneの場合は元画像サイズを維持
    
    Returns:
        各アルゴリズムの結果（処理時間、出力パス等）
    """
    from PIL import Image
    
    if algorithms is None:
        algorithms = ["bicubic", "lanczos", "realesrgan_x4plus"]
    
    frame_files = sorted(input_dir.glob("*.png"))
    if not frame_files:
        logger.error("入力フレームが見つかりません")
        return {}
    
    # 元画像サイズを取得
    first_img = Image.open(frame_files[0])
    original_size = first_img.size  # (width, height)
    first_img.close()
    
    # 出力サイズを決定
    if output_size is None:
        output_size = original_size
    
    results = {}
    
    for algo in algorithms:
        algo_dir = output_base_dir / f"{algo}_shrink{pre_shrink}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        
        times = []
        
        for frame_path in frame_files:
            start_total = time.perf_counter()
            
            # 前処理縮小
            if pre_shrink < 1.0:
                img = Image.open(frame_path).convert('RGB')
                w, h = img.size
                new_w, new_h = int(w * pre_shrink), int(h * pre_shrink)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                
                # 一時ファイルに保存
                temp_path = algo_dir / f"temp_{frame_path.name}"
                img.save(temp_path, 'PNG')
                process_input = temp_path
            else:
                process_input = frame_path
            
            temp_output_path = algo_dir / f"temp_out_{frame_path.name}"
            output_path = algo_dir / frame_path.name
            
            # アルゴリズム実行
            if algo in ["bicubic", "lanczos", "nearest"] or algo.startswith("lanczos_") or algo.startswith("enhance_"):
                upscale_classical(process_input, temp_output_path, scale=4, method=algo)
            else:
                model_map = {
                    "realesrgan_x4plus": "RealESRGAN_x4plus",
                    "realesrnet_x4plus": "RealESRNet_x4plus",
                    "realesr_general_v3": "realesr-general-x4v3",
                }
                model_name = model_map.get(algo, "RealESRGAN_x4plus")
                upscale_deep(process_input, temp_output_path, scale=4, model_name=model_name)
            
            # 最終出力サイズにリサイズ
            result_img = Image.open(temp_output_path).convert('RGB')
            if result_img.size != output_size:
                result_img = result_img.resize(output_size, Image.LANCZOS)
            result_img.save(output_path, 'PNG')
            result_img.close()
            
            # 一時ファイル削除
            temp_output_path.unlink(missing_ok=True)
            if pre_shrink < 1.0:
                temp_path.unlink(missing_ok=True)
            
            elapsed = time.perf_counter() - start_total
            times.append(elapsed)
        
        avg_time = sum(times) / len(times) if times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        
        results[algo] = {
            "algorithm": ALGORITHMS.get(algo, algo),
            "pre_shrink": pre_shrink,
            "output_size": f"{output_size[0]}x{output_size[1]}",
            "avg_time_sec": avg_time,
            "fps": fps,
            "output_dir": str(algo_dir),
            "frame_count": len(times),
        }
        
        print(f"{ALGORITHMS.get(algo, algo)}: {avg_time:.3f}秒/frame ({fps:.2f} fps)")
    
    return results


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    input_dir = Path("/data/test_frames/input")
    output_dir = Path("/data/test_frames/comparison")
    
    # 全て1080x608で出力
    TARGET_SIZE = (1080, 608)
    
    # 1. 元解像度のまま強調処理のみ（最速）
    print("\n=== 元解像度 強調のみ (1080x608 → 1080x608) ===")
    run_comparison(
        input_dir, output_dir,
        pre_shrink=1.0,
        algorithms=["enhance_sharpen", "enhance_unsharp", "enhance_unsharp_strong", "enhance_detail", "enhance_contrast", "enhance_combined"],
        output_size=TARGET_SIZE
    )
    
    # 2. 0.5x縮小 → 4x超解像 → 1080x608にリサイズ
    print("\n=== 0.5x縮小→4x SR→1080p (540x304 → 2160x1216 → 1080x608) ===")
    run_comparison(
        input_dir, output_dir,
        pre_shrink=0.5,
        algorithms=["lanczos", "lanczos_unsharp", "realesrgan_x4plus", "realesrnet_x4plus"],
        output_size=TARGET_SIZE
    )
    
    # 3. 0.25x縮小 → 4x超解像 → 1080x608（高速設定）
    print("\n=== 0.25x縮小→4x SR→1080p (270x152 → 1080x608) ===")
    run_comparison(
        input_dir, output_dir,
        pre_shrink=0.25,
        algorithms=["lanczos", "lanczos_unsharp", "realesrgan_x4plus"],
        output_size=TARGET_SIZE
    )
