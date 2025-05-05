from ultralytics import YOLO
import cv2
import numpy as np
from util import read_license_plate
import imutils
import os

# Load the license plate detector model
license_plate_detector = YOLO("./models/license_plate.pt")
vehicle_detector = YOLO("yolov8n.pt")

# Directory containing the sample images
sample_dir = "/Users/medreres/Desktop/university/8_sem/diploma/assets"

CONFIDENCE_THRESHOLD = 0.4

# Process all jpg files in the directory
for image_file in os.listdir(sample_dir):
    if not image_file.lower().endswith((".jpg", ".jpeg")):
        continue

    # Read the image
    image_path = os.path.join(sample_dir, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not read image {image_file}")
        continue

    print(f"\nProcessing image: {image_file}")

    # Try edge detection first
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    edge_detected_plates = []

    # Find potential license plate contours
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            # Create mask for the plate
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [approx], 0, 255, -1)
            new_image = cv2.bitwise_and(frame, frame, mask=mask)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            # Add the coordinates to detected plates
            edge_detected_plates.append(
                [
                    float(topy),
                    float(topx),
                    float(bottomy),
                    float(bottomx),
                    1.0,  # confidence score
                    0,  # class id
                ]
            )

    # Process edge-detected plates first
    plates_to_process = edge_detected_plates
    successful_read = False

    # Try processing edge-detected plates first
    for license_plate_detection in plates_to_process:
        # Extract coordinates and score
        (
            plate_top_left_x,
            plate_top_left_y,
            plate_bottom_right_x,
            plate_bottom_right_y,
            _score,
            _class_id,
        ) = license_plate_detection

        # Crop the license plate region
        license_plate_crop = frame[
            int(plate_top_left_y) : int(plate_bottom_right_y),
            int(plate_top_left_x) : int(plate_bottom_right_x),
            :,
        ]

        # Create a list to store OCR results
        ocr_results = []

        # Try multiple preprocessing techniques
        preprocessing_methods = [
            {
                "name": "adaptive_threshold",
                "process": lambda img: cv2.adaptiveThreshold(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                ),
            },
            {
                "name": "otsu_threshold",
                "process": lambda img: cv2.threshold(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )[1],
            },
            {
                "name": "inverse_threshold",
                "process": lambda img: cv2.threshold(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    0,
                    255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                )[1],
            },
        ]

        for method in preprocessing_methods:
            try:
                # Apply preprocessing
                processed_img = method["process"](license_plate_crop)

                # Perform OCR
                text, confidence = read_license_plate(processed_img)

                if confidence > CONFIDENCE_THRESHOLD:  # Adjust threshold as needed
                    ocr_results.append(
                        {
                            "text": text,
                            "confidence": confidence,
                            "method": method["name"],
                            "processed_img": processed_img,
                        }
                    )

                # Display preprocessed image
                cv2.imshow(f"Preprocessed ({method['name']})", processed_img)

            except Exception as e:
                print(f"Error with {method['name']} preprocessing: {str(e)}")

        # Select the best result
        if ocr_results:
            successful_read = True
            best_result = max(ocr_results, key=lambda x: x["confidence"])
            license_plate_text = best_result["text"]
            license_plate_text_score = best_result["confidence"]

            # Only draw rectangle and text if we successfully detected a plate
            cv2.rectangle(
                frame,
                (int(plate_top_left_x), int(plate_top_left_y)),
                (int(plate_bottom_right_x), int(plate_bottom_right_y)),
                (0, 255, 0),  # Green color
                2,  # Thickness
            )

            # Add text above the rectangle
            cv2.putText(
                frame,
                license_plate_text,
                (int(plate_top_left_x), int(plate_top_left_y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            print(
                f"Detected License Plate: {license_plate_text} "
                f"(Confidence: {license_plate_text_score:.2f}, "
                f"Method: {best_result['method']})"
            )
        else:
            print("No license plate text detected from edge detection")

    # If no successful reads from edge detection, try YOLO
    if not successful_read:
        print(
            "No readable plates found through edge detection, trying YOLO detector..."
        )
        license_plates_detections = license_plate_detector(frame)[0]
        plates_to_process = license_plates_detections.boxes.data.tolist()

        # Process YOLO detections
        for license_plate_detection in plates_to_process:
            # Extract coordinates and score
            (
                plate_top_left_x,
                plate_top_left_y,
                plate_bottom_right_x,
                plate_bottom_right_y,
                _score,
                _class_id,
            ) = license_plate_detection

            # Crop the license plate region
            license_plate_crop = frame[
                int(plate_top_left_y) : int(plate_bottom_right_y),
                int(plate_top_left_x) : int(plate_bottom_right_x),
                :,
            ]

            # Create a list to store OCR results
            ocr_results = []

            # Try multiple preprocessing techniques
            preprocessing_methods = [
                {
                    "name": "adaptive_threshold",
                    "process": lambda img: cv2.adaptiveThreshold(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11,
                        2,
                    ),
                },
                {
                    "name": "otsu_threshold",
                    "process": lambda img: cv2.threshold(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        0,
                        255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )[1],
                },
                {
                    "name": "inverse_threshold",
                    "process": lambda img: cv2.threshold(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        0,
                        255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                    )[1],
                },
            ]

            for method in preprocessing_methods:
                try:
                    # Apply preprocessing
                    processed_img = method["process"](license_plate_crop)

                    # Perform OCR
                    text, confidence = read_license_plate(processed_img)

                    if confidence > CONFIDENCE_THRESHOLD:  # Adjust threshold as needed
                        ocr_results.append(
                            {
                                "text": text,
                                "confidence": confidence,
                                "method": method["name"],
                                "processed_img": processed_img,
                            }
                        )

                    # Display preprocessed image
                    cv2.imshow(f"Preprocessed ({method['name']})", processed_img)

                except Exception as e:
                    print(f"Error with {method['name']} preprocessing: {str(e)}")

            # Select the best result
            if ocr_results:
                best_result = max(ocr_results, key=lambda x: x["confidence"])
                license_plate_text = best_result["text"]
                license_plate_text_score = best_result["confidence"]

                # Only draw rectangle and text if we successfully detected a plate
                cv2.rectangle(
                    frame,
                    (int(plate_top_left_x), int(plate_top_left_y)),
                    (int(plate_bottom_right_x), int(plate_bottom_right_y)),
                    (0, 255, 0),  # Green color
                    2,  # Thickness
                )

                # Add text above the rectangle
                cv2.putText(
                    frame,
                    license_plate_text,
                    (int(plate_top_left_x), int(plate_top_left_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                print(
                    f"Detected License Plate: {license_plate_text} "
                    f"(Confidence: {license_plate_text_score:.2f}, "
                    f"Method: {best_result['method']})"
                )
            else:
                print("No license plate text detected from YOLO")

    # Show the frame
    cv2.imshow("License Plate Detection", frame)
    cv2.waitKey(0)  # Wait until a key is pressed

# Clean up
cv2.destroyAllWindows()
