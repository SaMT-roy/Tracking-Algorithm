"""
Configuration parameters for video analytics.
"""
import os
import numpy as np

DISPLAY_VIDEO = True
WRITE_VIDEO = False

# Paths
INPUT_VIDEO = os.environ.get('INPUT_VIDEO', "rtsp://admin:Core@1986@182.48.203.80:554/Streaming/Channels/601")
OUTPUT_VIDEO = os.environ.get('OUTPUT_VIDEO', f'{INPUT_VIDEO} dwell time & interaction time.mp4')
MODEL_PATH = os.environ.get('MODEL_PATH', 'yolo12n.onnx')

# Model configuration
MODEL_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Detection parameters
CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.5

# Tracking parameters
IOU_THRESHOLD = 0.2
EXPIRE_AFTER = 120  # frames
FRAME_SKIP = 1

# ROI definitions
ROIS = [
    [(600, 530), (1200, 530), (1200, 970), (550, 970)],  # Counter area
    [(550, 970), (180, 970), (350, 470), (600, 470)],    # Customer area
]

# Analytics configuration
INOCCUPANCY_THRESHOLD = 120  # frames to trigger staff absence alert
INTERACTION_MIN_DURATION = 5.0  # seconds to report an interaction

# Visualization
COLORS = {
    'track_box': (0, 255, 0),    # Green
    'roi': (0, 255, 0),          # Green
    'interaction': (255, 0, 0),  # Blue
    'alert': (0, 0, 255),        # Red
    'label_text': (0, 255, 255), # Yellow
    'label_bg': (0, 0, 0)        # Black
}
