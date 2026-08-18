import sys

print("Checking environment...")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError as e:
    print(f"Error importing ultralytics: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"Error importing cv2: {e}")
