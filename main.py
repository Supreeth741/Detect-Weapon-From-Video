"""
Weapon Detection System - Main CLI Interface
Detect weapons from images, videos, or live camera feeds.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.image_detector import detect_from_image
from src.video_detector import detect_from_video
from src.camera_detector import detect_from_camera
from config.settings import Settings


def print_banner():
    """Print application banner."""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║          Weapon Detection System - YOLOv8 + OpenCV            ║
║                  Detect weapons from any source                ║
╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_results(results: dict, mode: str):
    """
    Print detection results in a formatted way.
    
    Args:
        results: Detection results dictionary
        mode: Detection mode (image/video/camera)
    """
    print("\n" + "=" * 60)
    print(f"DETECTION RESULTS - {mode.upper()}")
    print("=" * 60)
    
    if mode == "image":
        print(f"Image: {results['image_path']}")
        print(f"Weapon Detected: {'YES ⚠️' if results['weapon_detected'] else 'NO ✓'}")
        print(f"Number of Detections: {results['num_detections']}")
        
        if results['detections']:
            print("\nDetected Weapons:")
            for idx, detection in enumerate(results['detections'], 1):
                print(f"  {idx}. {detection['class_name']} "
                      f"(confidence: {detection['confidence']:.2%})")
        
        if results['output_path']:
            print(f"\nOutput saved: {results['output_path']}")
    
    elif mode == "video":
        print(f"Video: {results['video_path']}")
        print(f"Weapon Detected: {'YES ⚠️' if results['weapon_detected'] else 'NO ✓'}")
        print(f"Total Frames: {results['total_frames']}")
        print(f"Processed Frames: {results['processed_frames']}")
        print(f"Frames with Detections: {results['frames_with_detections']}")
        print(f"Total Detections: {results['total_detections']}")
        print(f"Processing Time: {results['processing_time']:.1f}s")
        print(f"Average FPS: {results['avg_fps']:.1f}")
        
        if results['output_path']:
            print(f"\nOutput saved: {results['output_path']}")
    
    elif mode == "camera":
        print(f"Camera ID: {results['camera_id']}")
        print(f"Weapon Detected: {'YES ⚠️' if results['weapon_detected'] else 'NO ✓'}")
        print(f"Total Frames: {results['total_frames']}")
        print(f"Processed Frames: {results['processed_frames']}")
        print(f"Frames with Detections: {results['frames_with_detections']}")
        print(f"Total Detections: {results['total_detections']}")
        print(f"Session Duration: {results['processing_time']:.1f}s")
        print(f"Average FPS: {results['avg_fps']:.1f}")
    
    print("=" * 60 + "\n")


def handle_image_mode(args):
    """Handle image detection mode."""
    image_path = args.input
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\n🔍 Detecting weapons in image: {image_path}")
    
    try:
        results = detect_from_image(
            image_path,
            confidence=args.confidence,
            show=args.show
        )
        print_results(results, "image")
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        sys.exit(1)


def handle_video_mode(args):
    """Handle video detection mode."""
    video_path = args.input
    
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        sys.exit(1)
    
    print(f"\n🎥 Detecting weapons in video: {video_path}")
    print("Processing... This may take a while.\n")
    
    try:
        results = detect_from_video(
            video_path,
            confidence=args.confidence,
            show=args.show
        )
        print_results(results, "video")
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        sys.exit(1)


def handle_camera_mode(args):
    """Handle camera detection mode."""
    camera_id = args.camera_id
    
    print(f"\n📹 Starting live camera detection (device {camera_id})")
    print("Press 'q' to quit, 'p' to pause, 's' to screenshot\n")
    
    try:
        results = detect_from_camera(
            camera_id=camera_id,
            confidence=args.confidence,
            max_runtime=args.max_runtime
        )
        print_results(results, "camera")
    except Exception as e:
        print(f"❌ Error with camera: {e}")
        print("\nTroubleshooting tips:")
        print("  - Check camera is connected and not in use by another app")
        print("  - Check Windows camera permissions (Settings > Privacy > Camera)")
        print("  - Try a different camera ID (--camera-id 1, 2, etc.)")
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    print_banner()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Detect weapons from images, videos, or live camera feeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect from image
  python main.py image path/to/photo.jpg
  
  # Detect from video
  python main.py video path/to/video.mp4
  
  # Detect from camera
  python main.py camera
  
  # Custom confidence threshold
  python main.py image photo.jpg --confidence 0.7
  
  # Show display window
  python main.py image photo.jpg --show
        """
    )
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Detection mode")
    subparsers.required = True
    
    # Image mode
    image_parser = subparsers.add_parser("image", help="Detect weapons in an image")
    image_parser.add_argument("input", type=str, help="Path to input image")
    image_parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=None,
        help=f"Confidence threshold (0.0-1.0, default: {Settings.CONFIDENCE_THRESHOLD})"
    )
    image_parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the image"
    )
    
    # Video mode
    video_parser = subparsers.add_parser("video", help="Detect weapons in a video")
    video_parser.add_argument("input", type=str, help="Path to input video")
    video_parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=None,
        help=f"Confidence threshold (0.0-1.0, default: {Settings.CONFIDENCE_THRESHOLD})"
    )
    video_parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the video while processing"
    )
    
    # Camera mode
    camera_parser = subparsers.add_parser("camera", help="Detect weapons from live camera")
    camera_parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    camera_parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=None,
        help=f"Confidence threshold (0.0-1.0, default: {Settings.CONFIDENCE_THRESHOLD})"
    )
    camera_parser.add_argument(
        "--max-runtime",
        type=int,
        default=None,
        help="Maximum runtime in seconds (default: unlimited)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate confidence if provided
    if hasattr(args, 'confidence') and args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            print("❌ Error: Confidence must be between 0.0 and 1.0")
            sys.exit(1)
    
    # Route to appropriate handler
    try:
        if args.mode == "image":
            handle_image_mode(args)
        elif args.mode == "video":
            handle_video_mode(args)
        elif args.mode == "camera":
            handle_camera_mode(args)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
