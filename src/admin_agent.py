import asyncio
import sys
import os
import contextlib

# Ensure src is in path or run from root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.sql_db import get_pending_reservations, update_reservation_status

# MCP Imports
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

async def log_reservation_via_mcp(reservation):
    """
    Connect to local MCP server and log the reservation.
    """
    # Adjust path to mcp_server.py
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    
    print(f"[Debug] MCP Server Script: {server_script}")
    print(f"[Debug] Using Python Executable: {sys.executable}")
    
    server_params = StdioServerParameters(
        command=sys.executable, # Use the current venv python
        args=[server_script],
        env=os.environ.copy()
    )
    
    print("Connecting to MCP Server to log reservation...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List tools to verify (optional debug)
                # tools = await session.list_tools()
                
                # Call the tool
                result = await session.call_tool(
                    "log_reservation",
                    arguments={
                        "name": reservation["name"],
                        "car_number": reservation["car_number"],
                        "duration": f"{reservation['start_time']} (End: {reservation['end_time']})" if reservation['end_time'] else f"{reservation['start_time']}"
                    }
                )
                
                # Inspect result
                if result and result.content:
                    print(f"[MCP] {result.content[0].text}")
                else:
                    print("[MCP] Tool called but no content returned.")
                    
    except Exception as e:
        print(f"[MCP Error] Failed to log: {e}")

async def admin_loop():
    print("--- Parking Bot Admin Agent (Async) ---")
    print("Waiting for pending reservations...")
    
    while True:
        try:
            # Synchronous DB call in async loop is blocking but okay for CLI tool
            pending = get_pending_reservations()
            
            if not pending:
                await asyncio.sleep(2) # Async sleep
                continue
                
            print(f"\n[ALERT] Found {len(pending)} pending reservation(s).")
            
            for res in pending:
                print(f"--- Reservation ID: {res['id']} ---")
                print(f"User: {res['name']}")
                print(f"Car:  {res['car_number']}")
                print(f"Time: {res['start_time']} to {res['end_time']}")
                
                # We need to run input() in a separate thread to not block the loop, 
                # OR just block since it's a CLI tool where user attention is required anyway.
                # Blocking is fine here.
                while True:
                    # Use asyncio.to_thread for input if we wanted non-blocking, but blocking is fine.
                    choice = await asyncio.to_thread(input, "Approve this reservation? (y/n/skip): ")
                    choice = choice.strip().lower()
                    
                    if choice == 'y':
                        update_reservation_status(res['id'], 'confirmed')
                        print(f"Reservation {res['id']} CONFIRMED.")
                        # Call MCP Tool
                        await log_reservation_via_mcp(res)
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
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(admin_loop())
    except KeyboardInterrupt:
        pass
