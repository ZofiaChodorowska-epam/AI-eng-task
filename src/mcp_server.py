from mcp.server.fastmcp import FastMCP
import datetime
import os

# Initialize FastMCP server
mcp = FastMCP("Parking Logger")

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
LOG_FILE = os.path.join(DATA_DIR, "confirmed_reservations.txt")
os.makedirs(DATA_DIR, exist_ok=True)

@mcp.tool()
def log_reservation(name: str, car_number: str, duration: str = "N/A", approved_at: str = None) ->str:
    """
    Log a confirmed reservation to the persistent storage file.
    
    Args:
        name: Name of the user
        car_number: License plate
        duration: Reservation period/duration info
        approved_at: ISO timestamp of approval (defaults to now)
    """
    if not approved_at:
        approved_at = datetime.datetime.now().isoformat()
        
    entry = f"{name} | {car_number} | {duration} | {approved_at}\n"
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(entry)
        return f"Successfully logged reservation for {name}."
    except Exception as e:
        return f"Error logging reservation: {str(e)}"

if __name__ == "__main__":
    # fastmcp likely handles uvicorn execution via .run() or CLI
    # We will try to run with uvicorn pointing to 'mcp_server:mcp.app' if exposed, 
    # OR rely on mcp.run() if it exists. 
    # Given the docs are unseen, let's assume .run() is a safe bet for a "script" usage
    mcp.run()
