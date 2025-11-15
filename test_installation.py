"""
Test script to verify installation and system setup.
Run this to check if everything is installed correctly.
"""

import sys
from pathlib import Path


def check_imports():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("Checking Package Imports")
    print("=" * 60)
    
    packages = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "PIL": "pillow",
        "torch": "torch",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics",
        "plyer": "plyer",
        "dotenv": "python-dotenv"
    }
    
    all_ok = True
    
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name:20s} - OK")
        except ImportError as e:
            print(f"✗ {package_name:20s} - MISSING")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check if all required directories exist."""
    print("\n" + "=" * 60)
    print("Checking Directory Structure")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    directories = [
        "config",
        "src",
        "models",
        "data",
        "data/images",
        "data/videos",
        "data/outputs",
        "logs"
    ]
    
    all_ok = True
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name:20s} - EXISTS")
        else:
            print(f"✗ {dir_name:20s} - MISSING")
            all_ok = False
    
    return all_ok


def check_python_version():
    """Check Python version."""
    print("\n" + "=" * 60)
    print("Checking Python Version")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 12:
        print("✓ Python version OK (3.12+)")
        return True
    else:
        print("✗ Python version should be 3.12 or higher")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("\n" + "=" * 60)
    print("Checking CUDA/GPU Support")
    print("=" * 60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available (CPU mode only)")
            print("  GPU acceleration disabled - detection will be slower")
            print("  To enable GPU: Install CUDA and PyTorch with CUDA support")
        
        return True
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_camera():
    """Check if camera is accessible."""
    print("\n" + "=" * 60)
    print("Checking Camera Access")
    print("=" * 60)
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            print("✓ Camera device 0 accessible")
            
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            
            cap.release()
            return True
        else:
            print("✗ Camera not accessible")
            print("  Check camera permissions in Windows Settings")
            return False
    except Exception as e:
        print(f"✗ Error checking camera: {e}")
        return False


def check_model():
    """Check if YOLO model exists."""
    print("\n" + "=" * 60)
    print("Checking YOLO Model")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    
    # Check for any .pt files
    model_files = list(models_dir.glob("*.pt"))
    
    if model_files:
        print(f"✓ Found {len(model_files)} model file(s):")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")
    else:
        print("⚠ No model files found in models/ directory")
        print("  Model will be auto-downloaded on first run")
        print("  For weapon detection, download a custom model (see README)")
    
    return True


def test_basic_import():
    """Test basic module imports."""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    try:
        from config.settings import Settings
        print("✓ config.settings imported")
        
        from src.utils import setup_logger
        print("✓ src.utils imported")
        
        from src.alert_service import AlertService
        print("✓ src.alert_service imported")
        
        from src.image_detector import detect_from_image
        print("✓ src.image_detector imported")
        
        from src.video_detector import detect_from_video
        print("✓ src.video_detector imported")
        
        from src.camera_detector import detect_from_camera
        print("✓ src.camera_detector imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Run all checks."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║        Weapon Detection System - Installation Check            ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    results = {
        "Python Version": check_python_version(),
        "Package Imports": check_imports(),
        "Directory Structure": check_directories(),
        "Module Imports": test_basic_import(),
        "CUDA/GPU": check_cuda(),
        "Camera": check_camera(),
        "Model Files": check_model()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check}")
    
    print()
    print(f"Passed: {passed}/{total} checks")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Download a weapon detection model (see README.md)")
        print("  2. Run: uv run python main.py camera")
    else:
        print("\n⚠️ Some checks failed. Review the output above.")
        print("   Run 'uv sync' to reinstall dependencies")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
