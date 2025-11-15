# Model Directory

Place your YOLO model weights here.

## Recommended Models

### For Testing

- `yolov8n.pt` - Will be auto-downloaded on first run (general object detection)

### For Weapon Detection

Download a custom-trained weapon detection model from:

- [Roboflow Universe](https://universe.roboflow.com/search?q=weapon%20detection)
- [Ultralytics Hub](https://hub.ultralytics.com/)
- [Kaggle Datasets](https://www.kaggle.com/search?q=weapon+detection)

After downloading, place the `.pt` file here and update `MODEL_NAME` in `config/settings.py`.

## Note

Standard COCO-trained models (yolov8n.pt, yolov8s.pt, etc.) do NOT include weapon classes like "knife", "gun", "rifle". You need a custom-trained model for actual weapon detection.
