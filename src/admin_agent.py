import time
import sys
import os

# Ensure src is in path or run from root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.sql_db import get_pending_reservations, update_reservation_status

def admin_loop():
    print("--- Parking Bot Admin Agent ---")
    print("Waiting for pending reservations...")
    
    while True:
        try:
            pending = get_pending_reservations()
            
            if not pending:
                time.sleep(2) # Poll every 2 seconds
                continue
                
            print(f"\n[ALERT] Found {len(pending)} pending reservation(s).")
            
            for res in pending:
                print(f"--- Reservation ID: {res['id']} ---")
                print(f"User: {res['name']}")
                print(f"Car:  {res['car_number']}")
                print(f"Time: {res['start_time']} to {res['end_time']}")
                
                while True:
                    choice = input("Approve this reservation? (y/n/skip): ").strip().lower()
                    if choice == 'y':
                        update_reservation_status(res['id'], 'confirmed')
                        print(f"Reservation {res['id']} CONFIRMED.")
                        break
                    elif choice == 'n':
                        update_reservation_status(res['id'], 'rejected')
                        print(f"Reservation {res['id']} REJECTED.")
                        break
                    elif choice == 'skip':
                        print("Skipping for now...")
                        break
                    else:
                        print("Invalid input. Please enter 'y', 'n', or 'skip'.")
            
            print("All pending processed. Waiting for new requests...")
            
        except KeyboardInterrupt:
            print("\nAdmin Agent stopping.")
            break
        except Exception as e:
            print(f"Error in admin loop: {e}")
            time.sleep(2)

if __name__ == "__main__":
    admin_loop()
