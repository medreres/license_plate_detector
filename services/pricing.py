from datetime import datetime
import math
from config import PRICE_FIRST_HOUR, PRICE_PER_DAY


class ParkingPricing:
    """Handles all pricing related calculations"""

    @staticmethod
    def calculate_cost(entry_time):
        current_time = datetime.now()
        time_diff = current_time - entry_time
        minutes = time_diff.total_seconds() / 60
        hours = time_diff.total_seconds() / 3600

        # Free parking for less than 15 minutes
        if minutes < 15:
            return 0

        if hours <= 2:
            return math.ceil(hours) * PRICE_FIRST_HOUR  # Round up to nearest hour
        else:
            days = math.ceil(hours / 24)  # Round up to nearest day
            return days * PRICE_PER_DAY

    @staticmethod
    def calculate_duration(entry_time):
        current_time = datetime.now()
        time_diff = current_time - entry_time
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)
        return f"{hours} год {minutes} хв"
