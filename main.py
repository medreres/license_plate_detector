from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate

coco_detector = YOLO("yolov8n.pt")
license_plate_detector = YOLO("./models/license_plate.pt")

capture = cv2.VideoCapture("./assets/traffic_sample.mp4")

vehicles_class_ids = [2, 3, 5, 7]

vehicle_tracker = Sort()

while True:
    ret, frame = capture.read()
    if not ret:
        break

    detections = coco_detector(frame)[0]
    vehicle_detections_for_current_frame = []
    for detection in detections.boxes.data.tolist():
        top_left_x, top_left_y, bottom_right_x, bottom_right_y, score, class_id = (
            detection
        )

        if int(class_id) in vehicles_class_ids:
            vehicle_detections_for_current_frame.append(
                [
                    top_left_x,
                    top_left_y,
                    bottom_right_x,
                    bottom_right_y,
                    score,
                ]
            )

    track_ids = vehicle_tracker.update(np.asarray(vehicle_detections_for_current_frame))

    license_plates_detections = license_plate_detector(frame)[0]
    for license_plate_detection in license_plates_detections.boxes.data.tolist():
        (
            plate_top_left_x,
            plate_top_left_y,
            plate_bottom_right_x,
            plate_bottom_right_y,
            score,
            class_id,
        ) = license_plate_detection

        (
            car_top_left_x,
            car_top_left_y,
            car_bottom_right_x,
            car_bottom_right_y,
            car_id,
        ) = get_car(license_plate_detection, track_ids)

        license_plate_crop = frame[
            int(plate_top_left_y) : int(plate_bottom_right_y),
            int(plate_top_left_x) : int(plate_bottom_right_x),
            :,
        ]

        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_gray_threshold = cv2.threshold(
            license_plate_crop_gray,
            64,
            255,
            cv2.THRESH_BINARY_INV,
        )

        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_gray
        )

        if license_plate_text:
            print(license_plate_text, license_plate_text_score)

        cv2.imshow("license_plate_crop", license_plate_crop)
        cv2.imshow("threshold", license_plate_crop_gray_threshold)
        cv2.waitKey(1)
