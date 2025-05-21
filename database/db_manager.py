import psycopg2
from psycopg2.extras import DictCursor
from tkinter import messagebox
from config import DB_CONFIG


class DatabaseManager:
    """Handles all database operations"""

    def __init__(self):
        self.db_params = DB_CONFIG
        self.ensure_db_connection()

    def ensure_db_connection(self):
        try:
            with psycopg2.connect(**self.db_params) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    with open("init.sql", "r") as sql_file:
                        cur.execute(sql_file.read())
        except psycopg2.OperationalError as e:
            messagebox.showerror(
                "Database Error", f"Could not connect to database: {e}"
            )
            raise

    def get_connection(self):
        return psycopg2.connect(**self.db_params)

    def record_entry(self, plate_text, entry_time):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM parking_records")
                count = cur.fetchone()[0]
                ticket_number = f"T{count + 1:04d}"

                if not plate_text:
                    plate_text = f"UNKNOWN_{ticket_number}"

                cur.execute(
                    """
                    INSERT INTO parking_records 
                    (plate_number, ticket_number, entry_time)
                    VALUES (%s, %s, %s)
                    RETURNING ticket_number
                    """,
                    (plate_text, ticket_number, entry_time),
                )
                conn.commit()
                return ticket_number

    def get_vehicle_data(self, plate_text=None, ticket_number=None):
        with self.get_connection() as conn:
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
                elif ticket_number:
                    cur.execute(
                        """
                        SELECT id, plate_number, ticket_number, entry_time
                        FROM parking_records 
                        WHERE ticket_number = %s
                        """,
                        (ticket_number,),
                    )
                return cur.fetchone()
