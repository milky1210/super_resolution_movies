"""
GPU利用可能性チェックユーティリティ

GPUが正しくマウントされているか、利用可能かを確認する。
"""

import logging
import subprocess
from typing import Dict, Any

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, Any]:
    """
    GPUの利用可能性をチェックする
    
    Returns:
        Dict[str, Any]: GPU情報
            - available: GPUが利用可能かどうか
            - device_count: デバイス数
            - devices: デバイス情報のリスト
            - error: エラーメッセージ（ある場合）
    """
    result = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    
    try:
        # nvidia-smiコマンドで確認
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", 
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        
        devices = []
        for line in output.strip().split("\n"):
            if line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    devices.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "driver_version": parts[2],
                        "memory_mb": int(parts[3]),
                    })
        
        result["available"] = len(devices) > 0
        result["device_count"] = len(devices)
        result["devices"] = devices
        
    except FileNotFoundError:
        result["error"] = "nvidia-smiが見つかりません。NVIDIAドライバがインストールされていない可能性があります。"
    except subprocess.CalledProcessError as e:
        result["error"] = f"nvidia-smiの実行に失敗しました: {e.output}"
    except subprocess.TimeoutExpired:
        result["error"] = "nvidia-smiの実行がタイムアウトしました。"
    except Exception as e:
        result["error"] = f"予期しないエラー: {str(e)}"
    
    return result


def check_vulkan_support() -> Dict[str, Any]:
    """
    Vulkanサポートをチェックする
    
    Returns:
        Dict[str, Any]: Vulkan情報
            - available: Vulkanが利用可能かどうか
            - error: エラーメッセージ（ある場合）
    """
    result = {
        "available": False,
        "error": None,
    }
    
    try:
        # vulkaninfo コマンドで確認（あれば）
        subprocess.check_output(
            ["vulkaninfo", "--summary"],
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        result["available"] = True
    except FileNotFoundError:
        # vulkaninfoがない場合でもlibvulkan1があれば利用可能
        try:
            import ctypes
            ctypes.CDLL("libvulkan.so.1")
            result["available"] = True
        except Exception:
            result["error"] = "Vulkanライブラリが見つかりません。"
    except subprocess.CalledProcessError as e:
        result["error"] = f"vulkaninfoの実行に失敗しました"
    except subprocess.TimeoutExpired:
        result["error"] = "vulkaninfoの実行がタイムアウトしました。"
    except Exception as e:
        result["error"] = f"予期しないエラー: {str(e)}"
    
    return result


def log_gpu_info() -> bool:
    """
    GPU情報をログに出力し、利用可能かどうかを返す
    
    Returns:
        bool: GPUが利用可能な場合True
    """
    logger.info("=" * 60)
    logger.info("GPU利用可能性チェック")
    logger.info("=" * 60)
    
    gpu_info = check_gpu_availability()
    
    if gpu_info["available"]:
        logger.info(f"✓ NVIDIA GPU: 利用可能")
        logger.info(f"  デバイス数: {gpu_info['device_count']}")
        for device in gpu_info["devices"]:
            logger.info(f"  - GPU {device['index']}: {device['name']}")
            logger.info(f"    ドライバ: {device['driver_version']}")
            logger.info(f"    メモリ: {device['memory_mb']} MB")
    else:
        logger.warning(f"✗ NVIDIA GPU: 利用不可")
        if gpu_info["error"]:
            logger.warning(f"  理由: {gpu_info['error']}")
    
    # Vulkanチェック
    vulkan_info = check_vulkan_support()
    if vulkan_info["available"]:
        logger.info(f"✓ Vulkan: 利用可能")
    else:
        logger.warning(f"✗ Vulkan: 利用不可")
        if vulkan_info["error"]:
            logger.warning(f"  理由: {vulkan_info['error']}")
    
    logger.info("=" * 60)
    
    # GPUまたはVulkanが利用可能であればTrue
    return gpu_info["available"] or vulkan_info["available"]


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    is_available = log_gpu_info()
    print(f"\nGPU/Vulkan利用可能: {is_available}")
