import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import cv2
import os
from config import WINDOW_SIZE, DROP_TARGET_SIZE


class ParkingUI:
    """Handles all UI related operations"""

    def __init__(self, parking_system):
        self.parking_system = parking_system
        self.window = None
        self.drop_target = None
        self.photo = None
        self.current_frame = None
        self.default_text = "Перетягніть сюди зображення"

    def create_gui(self):
        self.window = TkinterDnD.Tk()
        self.window.title("Система паркування")
        self.window.geometry(WINDOW_SIZE)

        self.drop_target = tk.Label(
            self.window,
            text=self.default_text,
            width=DROP_TARGET_SIZE["width"],
            height=DROP_TARGET_SIZE["height"],
            relief="solid",
        )
        self.drop_target.pack(pady=20)

        self.drop_target.drop_target_register(DND_FILES)
        self.drop_target.dnd_bind("<<Drop>>", self.handle_drop)

        tk.Button(self.window, text="Вʼїзд", command=self.handle_entry).pack(pady=10)
        tk.Button(self.window, text="Виїзд", command=self.handle_exit).pack(pady=10)
        tk.Button(
            self.window,
            text="Звʼязатись з оператором",
            command=self.parking_system.contact_operator,
        ).pack(pady=10)

        self.window.mainloop()

    def handle_drop(self, event):
        file_path = event.data.strip("{}")
        if not os.path.isfile(file_path):
            messagebox.showerror("Помилка", "Невірний файл")
            return

        self.current_frame = cv2.imread(file_path)
        if self.current_frame is None:
            messagebox.showerror("Помилка", "Не вдалося завантажити зображення")
            return

        self.update_image_display()

    def update_image_display(self):
        image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image_pil)
        self.drop_target.config(image=self.photo, text="")

    def clear_image(self):
        self.current_frame = None
        self.photo = None
        self.drop_target.config(image="", text=self.default_text)

    def handle_entry(self):
        if self.current_frame is None:
            messagebox.showwarning("Попередження", "Спочатку перетягніть зображення")
            return
        if self.parking_system.vehicle_entry(self.current_frame):
            self.clear_image()

    def handle_exit(self):
        if self.current_frame is None:
            messagebox.showwarning("Попередження", "Спочатку перетягніть зображення")
            return
        if self.parking_system.vehicle_exit(self.current_frame):
            self.clear_image()
