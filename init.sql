CREATE TABLE IF NOT EXISTS parking_records (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL,
    ticket_number VARCHAR(10) NOT NULL UNIQUE,
    entry_time TIMESTAMP NOT NULL
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_plate_number ON parking_records(plate_number);
CREATE INDEX IF NOT EXISTS idx_ticket_number ON parking_records(ticket_number);  