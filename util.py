import easyocr

ALLOWED_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # English uppercase
    "АБВГДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"  # Ukrainian uppercase
    "0123456789"
)

# Initialize the OCR reader with allowlist including Ukrainian letters
reader = easyocr.Reader(["uk", "en"], gpu=True)

# Expand the character mapping dictionaries
dict_char_to_int = {
    "O": "0",
    "I": "1",
    "Z": "7",
    "J": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "Q": "0",
    "D": "0",
}

dict_int_to_char = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "J",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "Z",
    "8": "B",
}


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    First 2 and last 2 characters must be letters, middle characters must be numbers.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 8:
        return False

    return True


uk_license_plate_format = {
    0: dict_int_to_char,  # letter
    1: dict_int_to_char,  # letter
    2: dict_char_to_int,  # number
    3: dict_char_to_int,  # number
    4: dict_char_to_int,  # number
    5: dict_char_to_int,  # number
    6: dict_int_to_char,  # letter
    7: dict_int_to_char,  # letter
}


def format_license(text: str) -> str:
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    First 2 and last 2 characters should be letters, middle characters should be numbers.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate: str = ""
    for index, character in enumerate(text):
        license_plate += uk_license_plate_format[index].get(character, character)
    return license_plate


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    # Try multiple confidence thresholds
    confidence_thresholds = [0.5, 0.3, 0.2]

    for threshold in confidence_thresholds:
        detections = reader.readtext(
            license_plate_crop,
            allowlist=ALLOWED_CHARS,
            min_size=10,
            paragraph=False,
            contrast_ths=0.3,
            adjust_contrast=0.5,
            width_ths=0.5,
            batch_size=4,
        )

        for detection in detections:
            bbox, text, score = detection

            if score < threshold:
                continue

            text = text.upper().replace(" ", "")
            print(f"Detected text: {text} with confidence: {score}")

            if license_complies_format(text):
                return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
