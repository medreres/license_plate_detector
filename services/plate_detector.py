from ultralytics import YOLO
from config import LICENSE_PLATE_MODEL_PATH, CONFIDENCE_THRESHOLD
from util import read_license_plate
import cv2
import numpy as np
import imutils


class PlateDetector:
    """Handles license plate detection and recognition"""

    def __init__(self):
        self.model = YOLO(LICENSE_PLATE_MODEL_PATH)

    def detect_and_read_plate(self, frame):
        # First try edge detection
        edge_plates = self._detect_plate_using_edges(frame)
        if edge_plates:
            for plate_coords in edge_plates:
                result = self._read_plate_text(frame, plate_coords)
                if result and result["confidence"] > CONFIDENCE_THRESHOLD:
                    return result["text"]

        # If edge detection fails, try YOLO
        yolo_plates = self._detect_plate_using_yolo(frame)
        if yolo_plates:
            for plate_coords in yolo_plates:
                result = self._read_plate_text(frame, plate_coords)
                if result and result["confidence"] > CONFIDENCE_THRESHOLD:
                    return result["text"]
        return None

    def _detect_plate_using_edges(self, frame):
        """Detect license plates using traditional edge detection method"""
        # Convert and find edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        # Find contours
        keypoints = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        detected_plates = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:  # Looking for rectangular shapes
                # Create mask and get coordinates
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [approx], 0, 255, -1)
                new_image = cv2.bitwise_and(frame, frame, mask=mask)

                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))

                detected_plates.append(
                    [float(topy), float(topx), float(bottomy), float(bottomx), 1.0, 0]
                )

        return detected_plates

    def _detect_plate_using_yolo(self, frame):
        """Detect license plates using YOLO model"""
        detections = self.model(frame)[0]
        return detections.boxes.data.tolist()

    def _read_plate_text(self, frame, plate_coords):
        """Try to read text from detected plate area using multiple preprocessing methods"""
        x1, y1, x2, y2 = [int(coord) for coord in plate_coords[:4]]
        plate_crop = frame[y1:y2, x1:x2, :]

        # Try different preprocessing methods
        methods = ["adaptive_threshold", "otsu_threshold", "inverse_threshold"]
        best_result = None
        best_confidence = 0

        for method in methods:
            try:
                # Convert to grayscale and apply threshold
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                if method == "adaptive_threshold":
                    processed = cv2.adaptiveThreshold(
                        gray,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11,
                        2,
                    )
                else:
                    threshold_type = (
                        cv2.THRESH_BINARY_INV
                        if method == "inverse_threshold"
                        else cv2.THRESH_BINARY
                    )
                    processed = cv2.threshold(
                        gray, 0, 255, threshold_type + cv2.THRESH_OTSU
                    )[1]

                text, confidence = read_license_plate(processed)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = {
                        "text": text,
                        "confidence": confidence,
                        "method": method,
                    }

            except Exception as e:
                continue

        return best_result if best_result else None
