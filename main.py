from datetime import datetime
from tkinter import messagebox, simpledialog
from services.plate_detector import PlateDetector
from services.pricing import ParkingPricing
from database.db_manager import DatabaseManager
from ui.parking_ui import ParkingUI


class ParkingSystem:
    """Main parking system class that coordinates all operations"""

    def __init__(self):
        self.db = DatabaseManager()
        self.plate_detector = PlateDetector()
        self.pricing = ParkingPricing()
        self.operator_available = True
        self.ui = ParkingUI(self)

    def vehicle_entry(self, frame):
        plate_text = self.plate_detector.detect_and_read_plate(frame)
        entry_time = datetime.now()
        ticket_number = self.db.record_entry(plate_text, entry_time)

        info_message = f"Номер талону: {ticket_number}\n"
        if plate_text and not plate_text.startswith("UNKNOWN_"):
            info_message = f"Номер авто: {plate_text}\n" + info_message
        info_message += f"Час вʼїзду: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}"

        messagebox.showinfo("Вʼїзд", info_message)
        self.raise_barrier()
        return True

    def vehicle_exit(self, frame):
        plate_text = self.plate_detector.detect_and_read_plate(frame)
        vehicle_data = None

        if plate_text:
            vehicle_data = self.db.get_vehicle_data(plate_text=plate_text)

        if not vehicle_data:
            ticket_number = simpledialog.askstring(
                "Введіть талон", "Введіть номер талону:"
            )
            if ticket_number:
                vehicle_data = self.db.get_vehicle_data(ticket_number=ticket_number)

        if not vehicle_data:
            messagebox.showerror(
                "Помилка", "Талон не знайдено. Зверніться до оператора."
            )
            return False

        cost = self.pricing.calculate_cost(vehicle_data["entry_time"])
        duration = self.pricing.calculate_duration(vehicle_data["entry_time"])

        if messagebox.askyesno(
            "Оплата",
            f"{'Номер авто: ' + vehicle_data['plate_number'] if not vehicle_data['plate_number'].startswith('UNKNOWN_') else 'Номер авто: Не зареєстровано'}\n"
            f"Номер талону: {vehicle_data['ticket_number']}\n"
            f"Час перебування: {duration}\n"
            f"До сплати: {cost} UAH\n\n"
            "Бажаєте оплатити?",
        ):
            self.process_payment(cost)
            self.raise_barrier()
            return True

        return False

    def raise_barrier(self):
        messagebox.showinfo("Шлагбаум", "Шлагбаум піднято. Проїзд дозволено.")

    def process_payment(self, amount):
        messagebox.showinfo("Термінал оплати", f"Очікується оплата: {amount} UAH")
        messagebox.showinfo("Оплата", "Оплату успішно здійснено")

    def contact_operator(self):
        if self.operator_available:
            messagebox.showinfo("Оператор", "Оператор підключений. Очікуйте допомоги.")
        else:
            messagebox.showwarning(
                "Оператор", "Оператор тимчасово недоступний. Спробуйте пізніше."
            )

    def run(self):
        self.ui.create_gui()


def main():
    parking_system = ParkingSystem()
    parking_system.run()


if __name__ == "__main__":
    main()
