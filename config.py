# Database configuration
DB_CONFIG = {
    "dbname": "parking_system",
    "user": "parking_user",
    "password": "parking_password",
    "host": "localhost",
    "port": "5432",
}

# Pricing configuration
PRICE_FIRST_HOUR = 70  # UAH
PRICE_PER_DAY = 150  # UAH

# Detection configuration
CONFIDENCE_THRESHOLD = 0.4
VEHICLE_CONFIDENCE_THRESHOLD = 0.3
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# Model paths
LICENSE_PLATE_MODEL_PATH = "./models/license_plate.pt"
VEHICLE_MODEL_PATH = "./models/yolov8n.pt"

# UI configuration
WINDOW_SIZE = "800x800"
DROP_TARGET_SIZE = {
    "width": 400,
    "height": 400,
}
