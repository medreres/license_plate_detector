from ultralytics import YOLO
import cv2
import numpy as np
from util import read_license_plate
import imutils
import os

license_plate_detector = YOLO("./models/license_plate.pt")

CONFIDENCE_THRESHOLD = 0.4
sample_dir = "/Users/medreres/Desktop/university/8_sem/diploma/assets"


def detect_plate_using_edge_detection(frame):
    """Detect license plates using traditional edge detection method"""
    # Convert and find edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


def detect_plate_using_yolo(frame, model):
    """Detect license plates using YOLO model"""
    detections = model(frame)[0]
    return detections.boxes.data.tolist()


def read_plate_text(frame, plate_coords):
    """Try to read text from detected plate area"""
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
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
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
                best_result = {"text": text, "confidence": confidence, "method": method}

        except Exception as e:
            continue

    return (
        best_result
        if best_result and best_result["confidence"] > CONFIDENCE_THRESHOLD
        else None
    )


def draw_detection(frame, coords, text):
    """Draw bounding box and text on frame"""
    x1, y1, x2, y2 = [int(coord) for coord in coords[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )


def main():
    # Process images from directory
    for image_file in os.listdir(sample_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg")):
            continue

        # Read image
        image_path = os.path.join(sample_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_file}")
            continue

        print(f"\nProcessing image: {image_file}")

        # Try edge detection first
        print("Attempting Edge Detection method...")
        edge_plates = detect_plate_using_edge_detection(frame)

        # Try to read plates detected by edge detection
        edge_detection_success = False
        for plate_coords in edge_plates:
            result = read_plate_text(frame, plate_coords)
            if result:
                edge_detection_success = True
                print("Successfully read plate using Edge Detection")
                print(f"License Plate Detected: {result['text']}")
                print(f"Method: {result['method']}")
                print(f"Confidence: {result['confidence']:.2f}")
                draw_detection(frame, plate_coords, result["text"])

        # If edge detection couldn't read any plates, try YOLO
        if not edge_detection_success:
            print("Edge Detection couldn't read plates. Trying YOLO detection...")
            yolo_plates = detect_plate_using_yolo(frame, license_plate_detector)

            for plate_coords in yolo_plates:
                result = read_plate_text(frame, plate_coords)
                if result:
                    print("Successfully read plate using YOLO")
                    print(f"License Plate Detected: {result['text']}")
                    print(f"Method: {result['method']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    draw_detection(frame, plate_coords, result["text"])

        # Display results
        cv2.imshow("License Plate Detection", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
