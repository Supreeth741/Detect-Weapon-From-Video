# Quick Start Guide 🚀

## Installation Complete! ✓

All dependencies have been installed. Here's how to get started:

## Step 1: Get a Weapon Detection Model

⚠️ **Important**: Standard YOLOv8 models don't detect weapons!

### Option A: Download Pre-trained Weapon Detection Model (Recommended)

Visit one of these sources:

1. **Roboflow Universe**: https://universe.roboflow.com/search?q=weapon%20detection
2. **Ultralytics Hub**: https://hub.ultralytics.com/
3. **Kaggle**: https://www.kaggle.com/search?q=weapon+detection

Download a `.pt` model file and place it in the `models/` directory.

Then update `config/settings.py`:

```python
MODEL_NAME = "your-weapon-model.pt"  # Change this to your model filename
```

### Option B: Test with Default YOLOv8 (General Object Detection)

The system will auto-download YOLOv8n on first run. Good for testing the system, but won't detect weapons specifically.

## Step 2: Add Test Media (Optional)

Add test images to `data/images/` or videos to `data/videos/` for testing.

## Step 3: Run Your First Detection

### Test Camera (Easiest)

```powershell
uv run python main.py camera
```

Press 'q' to quit, 'p' to pause, 's' for screenshot

### Test Image

```powershell
# Add an image to data/images/ first
uv run python main.py image data/images/test.jpg --show
```

### Test Video

```powershell
# Add a video to data/videos/ first
uv run python main.py video data/videos/test.mp4
```

## Usage Examples

### Basic Commands

```powershell
# Image detection with display
uv run python main.py image photo.jpg --show

# Video detection with custom confidence
uv run python main.py video footage.mp4 --confidence 0.7

# Camera with 60 second limit
uv run python main.py camera --max-runtime 60

# Use different camera
uv run python main.py camera --camera-id 1
```

### Help

```powershell
uv run python main.py --help
uv run python main.py image --help
uv run python main.py video --help
uv run python main.py camera --help
```

## Expected Output

When a weapon is detected:

- ⚠️ Console alert with details
- 🔊 System beep sound
- 📬 Windows notification
- 💾 Annotated image saved to `data/outputs/`
- 📝 Log entry in `logs/detections.log`

## Configuration

Edit `config/settings.py` to customize:

- Confidence threshold (default: 0.50)
- Alert settings (sound, notifications, cooldown)
- Camera resolution and FPS
- Frame skip for performance
- Output formats

## Troubleshooting

### Camera won't open?

1. Check Windows Settings > Privacy > Camera
2. Close other apps using camera
3. Try: `uv run python main.py camera --camera-id 1`

### Import errors?

```powershell
uv sync
```

### Slow performance?

- Increase `FRAME_SKIP` in `config/settings.py`
- Use smaller model (yolov8n.pt instead of yolov8m.pt)
- Lower camera resolution

### Need GPU acceleration?

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

1. ✅ Install dependencies (Done!)
2. ⬜ Download weapon detection model
3. ⬜ Test with camera: `uv run python main.py camera`
4. ⬜ Review results in `data/outputs/`
5. ⬜ Customize settings in `config/settings.py`

## Need Help?

- Check `README.md` for full documentation
- Review examples in `example.py`
- See troubleshooting section in README

---

**Ready to detect weapons! 🎯**

Run: `uv run python main.py camera` to start!
