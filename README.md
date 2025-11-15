# Weapon Detection System 🔍

A comprehensive Python-based weapon detection system using **YOLOv8** and **OpenCV** that detects weapons from images, videos, and live camera feeds with real-time alerts.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- 🖼️ **Image Detection** - Detect weapons in static images
- 🎥 **Video Detection** - Process video files frame-by-frame
- 📹 **Live Camera Detection** - Real-time weapon detection from webcam
- ⚠️ **Multi-Channel Alerts** - Console, sound, desktop notifications
- 💾 **Automatic Logging** - Save detection logs and annotated frames
- 🎯 **High Accuracy** - Powered by YOLOv8 deep learning model
- ⚙️ **Configurable** - Adjust confidence thresholds, frame skip, alerts
- 🖥️ **Windows Optimized** - DirectShow backend for camera access

## 📋 Requirements

- Python 3.12+
- Windows OS (optimized for Windows)
- Webcam (for live detection)
- CUDA-capable GPU (optional, for faster inference)

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies using UV:

```bash
# Clone the repository
git clone https://github.com/yourusername/detect-weapon-from-video.git
cd detect-weapon-from-video

# Install dependencies
uv sync
```

### 2. Download YOLO Model

The system will automatically download the YOLOv8 model on first run. However, for weapon detection, you'll need a custom-trained model since standard COCO models don't include weapon classes.

**Option A: Use Pre-trained Weapon Detection Model (Recommended)**

Download a pre-trained weapon detection model from:

- [Roboflow Universe - Weapon Detection](https://universe.roboflow.com/search?q=weapon%20detection)
- [Ultralytics Hub](https://hub.ultralytics.com/)
- [Kaggle Datasets](https://www.kaggle.com/search?q=weapon+detection)

Place the model file in the `models/` directory and update the `MODEL_NAME` in `config/settings.py`.

**Option B: Use Default YOLOv8 (For Testing)**

The default YOLOv8 model will be downloaded automatically but won't detect weapons specifically. Useful for testing the system with general object detection.

### 3. Run Detection

#### Detect from Image

```bash
uv run python main.py image data/images/photo.jpg
```

#### Detect from Video

```bash
uv run python main.py video data/videos/video.mp4
```

#### Detect from Live Camera

```bash
uv run python main.py camera
```

## 📖 Usage Examples

### Basic Detection

```bash
# Detect weapons in an image
python main.py image test.jpg

# Detect weapons in a video
python main.py video security_footage.mp4

# Start live camera detection
python main.py camera
```

### Advanced Options

```bash
# Custom confidence threshold (0.0 to 1.0)
python main.py image photo.jpg --confidence 0.7

# Show display window while processing
python main.py image photo.jpg --show

# Use specific camera device
python main.py camera --camera-id 1

# Set maximum runtime for camera (in seconds)
python main.py camera --max-runtime 60
```

### Programmatic Usage

You can also use the detection functions directly in Python:

```python
from src.image_detector import detect_from_image
from src.video_detector import detect_from_video
from src.camera_detector import detect_from_camera

# Detect from image
result = detect_from_image("photo.jpg", confidence=0.5, show=True)
if result['weapon_detected']:
    print(f"Found {result['num_detections']} weapons!")

# Detect from video
result = detect_from_video("video.mp4")
print(f"Detected weapons in {result['frames_with_detections']} frames")

# Detect from camera
result = detect_from_camera(camera_id=0, max_runtime=30)
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
# Detection parameters
CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence (0.0-1.0)
MODEL_NAME = "yolov8n.pt"    # Model file name

# Camera settings
CAMERA_ID = 0                # Default camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_SKIP = 1               # Process every Nth frame

# Alert settings
ALERT_ENABLED = True
ALERT_COOLDOWN = 5.0         # Seconds between alerts
ALERT_SOUND_ENABLED = True
ALERT_NOTIFICATION_ENABLED = True
ALERT_SAVE_FRAMES = True     # Save frames with detections

# Output settings
SAVE_ANNOTATED_IMAGES = True
SAVE_ANNOTATED_VIDEOS = True
```

## 📁 Project Structure

```
detect-weapon-from-video/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration settings
├── src/
│   ├── __init__.py
│   ├── image_detector.py    # Image detection module
│   ├── video_detector.py    # Video detection module
│   ├── camera_detector.py   # Camera detection module
│   ├── alert_service.py     # Alert system
│   └── utils.py             # Helper functions
├── models/                  # YOLO model weights
├── data/
│   ├── images/              # Sample images
│   ├── videos/              # Sample videos
│   └── outputs/             # Detection results
├── logs/                    # Detection logs
├── main.py                  # CLI interface
├── pyproject.toml           # Project dependencies
└── README.md
```

## 🎯 How It Works

1. **Model Loading** - YOLOv8 model is loaded for object detection
2. **Input Processing** - Images/video frames are processed by the model
3. **Detection** - Model identifies weapons with confidence scores
4. **Annotation** - Bounding boxes and labels are drawn on detections
5. **Alerts** - Multi-channel alerts triggered when weapons detected
6. **Output** - Annotated results saved to disk with logs

## 🔔 Alert System

When a weapon is detected, the system triggers:

- **⚠️ Console Alert** - Detailed detection information in terminal
- **🔊 Sound Alert** - Windows system beep (1000 Hz, 500ms)
- **📬 Desktop Notification** - Windows toast notification
- **📝 File Logging** - Timestamped logs in `logs/detections.log`
- **💾 Frame Saving** - Annotated frames saved to `data/outputs/`

Alerts are rate-limited (default: 5 seconds cooldown) to prevent spam.

## 🐛 Troubleshooting

### Camera Issues

**Problem**: "Failed to open camera"

**Solutions**:

1. Check camera permissions: `Settings > Privacy > Camera`
2. Ensure camera not in use by another application
3. Try different camera ID: `--camera-id 1`
4. Verify camera connection with Windows Camera app

### Import Errors

**Problem**: "Import could not be resolved"

**Solutions**:

1. Install dependencies: `uv sync`
2. Activate environment: `uv run python main.py`
3. Check Python version: `python --version` (needs 3.12+)

### Model Not Found

**Problem**: "Model file not found"

**Solutions**:

1. Place YOLO weights in `models/` directory
2. Update `MODEL_NAME` in `config/settings.py`
3. Let system auto-download default model on first run

### Low FPS

**Problem**: Slow detection speed

**Solutions**:

1. Increase `FRAME_SKIP` in settings (process fewer frames)
2. Use smaller model: `yolov8n.pt` instead of `yolov8l.pt`
3. Lower camera resolution in settings
4. Use CUDA-enabled PyTorch with GPU

### GPU/CUDA Setup

To use GPU acceleration on Windows:

```bash
# Install PyTorch with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:

```python
import torch
print(torch.cuda.is_available())  # Should print True
```

## 📊 Performance

- **YOLOv8n** (Nano): ~30-60 FPS on CPU, ~120-200 FPS on GPU
- **YOLOv8s** (Small): ~20-40 FPS on CPU, ~80-120 FPS on GPU
- **YOLOv8m** (Medium): ~10-20 FPS on CPU, ~50-80 FPS on GPU

Performance varies based on hardware, image resolution, and model size.

## 🔒 Security & Ethics

This system is intended for **security and safety applications only**:

- Monitor restricted areas for dangerous objects
- Enhance security screening processes
- Educational and research purposes

**Important Considerations**:

- Respect privacy laws and regulations
- Obtain proper consent for surveillance
- Be aware of false positives - human verification recommended
- Follow ethical guidelines for AI deployment
- Comply with local laws regarding monitoring systems

## 📚 Dependencies

Core libraries:

- `ultralytics` - YOLOv8 framework
- `opencv-python` - Computer vision operations
- `torch` - PyTorch deep learning framework
- `numpy` - Array operations
- `pillow` - Image processing
- `plyer` - Cross-platform notifications

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Supreeth741**

- GitHub: [@Supreeth TP](https://github.com/Supreeth741)

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision tools
- Weapon detection datasets from Roboflow and Kaggle communities

## 📞 Support

For issues and questions:

- Open an [issue](https://github.com/Supreeth741/Detect-Weapon-From-Video/issues)
- Check existing documentation
- Review troubleshooting section

---

**⚠️ Note**: Standard YOLO models trained on COCO dataset do NOT include weapon classes. You'll need a custom-trained weapon detection model for accurate results. See the "Download YOLO Model" section above.

**Happy Detecting! 🎯**
