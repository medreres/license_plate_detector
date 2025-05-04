from ultralytics import YOLO
import cv2
import numpy as np
from util import read_license_plate
import os

# Load the license plate detector model
license_plate_detector = YOLO("./models/license_plate.pt")

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

    # Detect license plates
    license_plates_detections = license_plate_detector(frame)[0]

    plates_detections_list = license_plates_detections.boxes.data.tolist()

    if len(plates_detections_list) == 0:
        plates_detections_list = [[0, 0, frame.shape[1], frame.shape[0], 1.0, 0]]

    # Process each detected license plate
    for license_plate_detection in plates_detections_list:
        # Extract coordinates and score
        (
            plate_top_left_x,
            plate_top_left_y,
            plate_bottom_right_x,
            plate_bottom_right_y,
            score,
            class_id,
        ) = license_plate_detection

        # Crop the license plate region
        license_plate_crop = frame[
            int(plate_top_left_y) : int(plate_bottom_right_y),
            int(plate_top_left_x) : int(plate_bottom_right_x),
            :,
        ]

        # Create a list to store OCR results
        ocr_results = []

        # TODO try using detecting boundaries instead of using pretrained model
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
            # inverse threshold helped detecting
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
            print("No license plate text detected")

        # Draw rectangle around license plate
        cv2.rectangle(
            frame,
            (int(plate_top_left_x), int(plate_top_left_y)),
            (int(plate_bottom_right_x), int(plate_bottom_right_y)),
            (0, 255, 0),  # Green color
            2,  # Thickness
        )

    # Show the frame
    cv2.imshow(f"License Plate Detection - {image_file}", frame)
    cv2.waitKey(0)  # Wait until a key is pressed

# Clean up
cv2.destroyAllWindows()
