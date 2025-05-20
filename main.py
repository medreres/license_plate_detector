from ultralytics import YOLO
import cv2
import numpy as np
from util import read_license_plate
import imutils
import os
from datetime import datetime, timezone
import json
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import psycopg2
from psycopg2.extras import DictCursor

# Load both models
license_plate_detector = YOLO("./models/license_plate.pt")
vehicle_detector = YOLO(
    "./models/yolov8n.pt"
)  # Using YOLOv8 nano for vehicle detection

CONFIDENCE_THRESHOLD = 0.4
VEHICLE_CONFIDENCE_THRESHOLD = 0.3
VEHICLE_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
]  # Common vehicle classes in COCO
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


def detect_vehicles(frame, model):
    """Detect vehicles using YOLO model"""
    detections = model(frame)[0]
    vehicles = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = detection
        class_name = model.names[int(class_id)]

        if class_name in VEHICLE_CLASSES and confidence > VEHICLE_CONFIDENCE_THRESHOLD:
            vehicles.append(
                {
                    "coords": [x1, y1, x2, y2],
                }
            )

    return vehicles


def draw_vehicle_detection(frame, vehicle):
    """Draw vehicle bounding box and class on frame"""
    x1, y1, x2, y2 = [int(coord) for coord in vehicle["coords"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    label = f"{vehicle['class']} ({vehicle['confidence']:.2f})"
    cv2.putText(
        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
    )


# TODO separate to repository, service etc.


class ParkingSystem:
    def __init__(self):
        # Replace JSON file handling with database connection
        self.db_params = {
            "dbname": "parking_system",
            "user": "parking_user",
            "password": "parking_password",
            "host": "localhost",
            "port": "5432",
        }
        self.ensure_db_connection()
        self.hourly_rate = 20  # UAH per hour
        self.window = None
        self.current_frame = None
        self.operator_available = True
        self.drop_target = None
        self.photo = None
        self.default_text = "Перетягніть сюди зображення"

    def ensure_db_connection(self):
        """Ensure database connection and create tables if needed"""
        try:
            with psycopg2.connect(**self.db_params) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Read and execute the init.sql file
                    with open("init.sql", "r") as sql_file:
                        cur.execute(sql_file.read())
        except psycopg2.OperationalError as e:
            messagebox.showerror(
                "Database Error", f"Could not connect to database: {e}"
            )
            raise

    def get_db_connection(self):
        """Get a database connection with dictionary cursor"""
        return psycopg2.connect(**self.db_params)

    def vehicle_entry(self, frame):
        plate_text = None
        plates = detect_plate_using_yolo(frame, license_plate_detector)

        if plates:
            for plate_coords in plates:
                result = read_plate_text(frame, plate_coords)
                if result:
                    plate_text = result["text"]
                    break

        entry_time = datetime.now()
        ticket_number = None

        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get next ticket number
                cur.execute("SELECT COUNT(*) FROM parking_records")
                count = cur.fetchone()[0]
                ticket_number = f"T{count + 1:04d}"

                if not plate_text:
                    plate_text = f"UNKNOWN_{ticket_number}"

                # Insert new record with simplified schema
                cur.execute(
                    """
                    INSERT INTO parking_records 
                    (plate_number, ticket_number, entry_time)
                    VALUES (%s, %s, %s)
                    """,
                    (plate_text, ticket_number, entry_time),
                )
                conn.commit()

        if plate_text.startswith("UNKNOWN_"):
            messagebox.showinfo(
                "Увага",
                "Номер авто не розпізнано.\n"
                "Талон видано без прив'язки до номера.\n"
                "Будь ласка, зверніться до оператора для внесення номера.",
            )

        messagebox.showinfo(
            "Вʼїзд",
            f"{'Номер авто: ' + plate_text if not plate_text.startswith('UNKNOWN_') else 'Номер авто: Не розпізнано'}\n"
            f"Номер талону: {ticket_number}\n"
            f"Час вʼїзду: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}",
        )
        self.raise_barrier()
        return True

    def vehicle_exit(self, frame):
        plates = detect_plate_using_yolo(frame, license_plate_detector)
        plate_text = None
        vehicle_data = None

        if plates:
            for plate_coords in plates:
                result = read_plate_text(frame, plate_coords)
                if result:
                    plate_text = result["text"]
                    break

        with self.get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                if plate_text:
                    cur.execute(
                        """
                        SELECT id, plate_number, ticket_number, entry_time
                        FROM parking_records 
                        WHERE plate_number = %s
                        ORDER BY entry_time DESC
                        LIMIT 1
                        """,
                        (plate_text,),
                    )
                    vehicle_data = cur.fetchone()

                if not vehicle_data:
                    ticket_number = simpledialog.askstring(
                        "Введіть талон", "Введіть номер талону:"
                    )
                    if not ticket_number:
                        return False

                    cur.execute(
                        """
                        SELECT id, plate_number, ticket_number, entry_time
                        FROM parking_records 
                        WHERE ticket_number = %s
                        """,
                        (ticket_number,),
                    )
                    vehicle_data = cur.fetchone()

                if not vehicle_data:
                    messagebox.showerror(
                        "Помилка", "Талон не знайдено. Зверніться до оператора."
                    )
                    return False

                cost = self.calculate_cost(vehicle_data["entry_time"])
                response = messagebox.askyesno(
                    "Оплата",
                    f"{'Номер авто: ' + vehicle_data['plate_number'] if not vehicle_data['plate_number'].startswith('UNKNOWN_') else 'Номер авто: Не зареєстровано'}\n"
                    f"Номер талону: {vehicle_data['ticket_number']}\n"
                    f"Час перебування: {self.calculate_duration(vehicle_data['entry_time'])}\n"
                    f"До сплати: {cost} UAH\n\n"
                    "Бажаєте оплатити?",
                )

                if response:
                    self.process_payment(cost)
                    self.raise_barrier()
                    return True

        return False

    def calculate_cost(self, entry_time):
        current_time = datetime.now()
        time_diff = current_time - entry_time
        hours = time_diff.total_seconds() / 3600
        return max(self.hourly_rate, round(hours * self.hourly_rate))

    def calculate_duration(self, entry_time):
        current_time = datetime.now()
        time_diff = current_time - entry_time
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)
        return f"{hours} год {minutes} хв"

    def raise_barrier(self):
        messagebox.showinfo("Шлагбаум", "Шлагбаум піднято. Проїзд дозволено.")
        # Тут може бути код для управління реальним шлагбаумом

    def process_payment(self, amount):
        messagebox.showinfo("Термінал оплати", f"Очікується оплата: {amount} UAH")
        # Тут може бути інтеграція з реальним платіжним терміналом
        messagebox.showinfo("Оплата", "Оплату успішно здійснено")

    def contact_operator(self):
        if self.operator_available:
            messagebox.showinfo("Оператор", "Оператор підключений. Очікуйте допомоги.")
        else:
            messagebox.showwarning(
                "Оператор", "Оператор тимчасово недоступний. Спробуйте пізніше."
            )

    def create_gui(self):
        self.window = TkinterDnD.Tk()
        self.window.title("Система паркування")
        self.window.geometry("800x800")

        # Create drop target with larger size
        self.drop_target = tk.Label(
            self.window,
            text=self.default_text,
            width=400,  # Increased width
            height=400,  # Increased height
            relief="solid",
        )
        self.drop_target.pack(pady=20)

        # Configure drag and drop
        self.drop_target.drop_target_register(DND_FILES)
        self.drop_target.dnd_bind("<<Drop>>", self.handle_drop)

        # Create buttons
        tk.Button(self.window, text="Вʼїзд", command=self.handle_entry).pack(pady=10)
        tk.Button(self.window, text="Виїзд", command=self.handle_exit).pack(pady=10)
        tk.Button(
            self.window, text="Звʼязатись з оператором", command=self.contact_operator
        ).pack(pady=10)

        self.window.mainloop()

    def handle_drop(self, event):
        file_path = event.data
        file_path = file_path.strip("{}")

        if not os.path.isfile(file_path):
            messagebox.showerror("Помилка", "Невірний файл")
            return

        # Read the image with OpenCV
        self.current_frame = cv2.imread(file_path)

        if self.current_frame is None:
            messagebox.showerror("Помилка", "Не вдалося завантажити зображення")
            return

        # Convert OpenCV image to PIL format for display
        image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Resize image while maintaining aspect ratio
        display_size = (400, 400)
        image_pil.thumbnail(display_size, Image.Resampling.LANCZOS)

        # Convert to PhotoImage for Tkinter
        self.photo = ImageTk.PhotoImage(image_pil)

        # Update drop target with image
        self.drop_target.config(
            image=self.photo, text=""
        )  # Clear text when showing image

    def clear_image(self):
        """Reset the drop target to its initial state"""
        self.current_frame = None
        self.photo = None
        self.drop_target.config(image="", text=self.default_text)

    def handle_entry(self):
        if self.current_frame is None:
            messagebox.showwarning("Попередження", "Спочатку перетягніть зображення")
            return
        result = self.vehicle_entry(self.current_frame)
        if result:  # Only clear if entry was successful
            self.clear_image()

    def handle_exit(self):
        if self.current_frame is None:
            messagebox.showwarning("Попередження", "Спочатку перетягніть зображення")
            return
        result = self.vehicle_exit(self.current_frame)
        if result:  # Only clear if exit was successful
            self.clear_image()


def main():
    parking_system = ParkingSystem()
    parking_system.create_gui()


if __name__ == "__main__":
    main()
