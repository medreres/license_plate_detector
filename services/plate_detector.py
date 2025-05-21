from ultralytics import YOLO
import cv2
from config import LICENSE_PLATE_MODEL_PATH, CONFIDENCE_THRESHOLD
from util import read_license_plate


class PlateDetector:
    """Handles license plate detection and recognition"""

    def __init__(self):
        self.model = YOLO(LICENSE_PLATE_MODEL_PATH)

    def detect_and_read_plate(self, frame):
        plates = self._detect_plate_using_yolo(frame)
        if plates:
            for plate_coords in plates:
                text, confidence = self._read_plate_text(frame, plate_coords)
                if text and confidence > CONFIDENCE_THRESHOLD:
                    return text
        return None

    def _detect_plate_using_yolo(self, frame):
        """Detect license plates using YOLO model"""
        detections = self.model(frame)[0]
        return detections.boxes.data.tolist()

    def _read_plate_text(self, frame, plate_coords):
        """Try to read text from detected plate area"""
        x1, y1, x2, y2 = [int(coord) for coord in plate_coords[:4]]
        plate_crop = frame[y1:y2, x1:x2, :]
        return read_license_plate(plate_crop)
