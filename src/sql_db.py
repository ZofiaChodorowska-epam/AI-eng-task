import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/parking.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Dynamic pricing and working hours could be tables, but let's keep it simple.
    # We need spots availability.
    c.execute('''CREATE TABLE IF NOT EXISTS spots (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 spot_number TEXT UNIQUE,
                 is_occupied INTEGER DEFAULT 0,
                 price_per_hour REAL DEFAULT 5.0
                 )''')
    
    # Drop table to force schema update for Stage 2 (development only)
    c.execute('DROP TABLE IF EXISTS reservations')
    
    c.execute('''CREATE TABLE IF NOT EXISTS reservations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT,
                 car_number TEXT,
                 start_time TEXT,
                 end_time TEXT,
                 status TEXT DEFAULT 'pending'
                 )''')
    
    # Populate with some mock spots if empty
    c.execute('SELECT count(*) FROM spots')
    if c.fetchone()[0] == 0:
        spots = [(f"A{i}", 0, 5.0) for i in range(1, 11)] # 10 spots
        c.executemany('INSERT INTO spots (spot_number, is_occupied, price_per_hour) VALUES (?, ?, ?)', spots)
        
    conn.commit()
    conn.close()

def get_available_spots():
    conn = get_db_connection()
    rows = conn.execute('SELECT spot_number, price_per_hour FROM spots WHERE is_occupied = 0').fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_working_hours():
    # Mocking dynamic retrieval, maybe from a config table or just function
    return "06:00 AM - 12:00 AM daily"

def check_availability(start_time, end_time):
    # Simplified check: just return total free spots count for now
    spots = get_available_spots()
    return len(spots)

def create_reservation(name, car_number, start_time, end_time):
    conn = get_db_connection()
    c = conn.cursor()
    # Default status is 'pending'
    c.execute('INSERT INTO reservations (name, car_number, start_time, end_time, status) VALUES (?, ?, ?, ?, ?)',
              (name, car_number, start_time, end_time, 'pending'))
    conn.commit()
    conn.close()
    return "Reservation request received. Waiting for admin approval..."

def update_reservation_status(reservation_id, status):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('UPDATE reservations SET status = ? WHERE id = ?', (status, reservation_id))
    conn.commit()
    conn.close()
    return True

def get_pending_reservations():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM reservations WHERE status = 'pending'").fetchall()
    conn.close()
    return [dict(row) for row in rows]
    
def get_reservation_status(name, car_number):
    conn = get_db_connection()
    # Get latest reservation for user
    row = conn.execute("SELECT status FROM reservations WHERE name = ? AND car_number = ? ORDER BY id DESC LIMIT 1", 
                       (name, car_number)).fetchone()
    conn.close()
    if row:
        return row["status"]
    return None
