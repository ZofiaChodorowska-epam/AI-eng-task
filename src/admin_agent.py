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
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def log_reservation_via_mcp(reservation):
    """
    Connect to local MCP server and log the reservation.
    """
    # Adjust path to mcp_server.py
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    
    logger.debug(f"MCP Server Script: {server_script}")
    logger.debug(f"Using Python Executable: {sys.executable}")
    
    server_params = StdioServerParameters(
        command=sys.executable, # Use the current venv python
        args=[server_script],
        env=os.environ.copy()
    )
    
    logger.info("Connecting to MCP Server to log reservation...")
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
                    logger.info(f"[MCP] {result.content[0].text}")
                else:
                    logger.info("[MCP] Tool called but no content returned.")
                    
    except Exception as e:
        logger.error(f"[MCP Error] Failed to log: {e}")

async def process_admin_action(reservation: dict, action: str) -> str:
    """
    Shared abstraction for processing an admin action ('approve', 'reject', 'skip').
    Updates DB status and logs via MCP if approved.
    """
    res_id = reservation['id']
    name = reservation['name']
    
    if action == 'approve':
        update_reservation_status(res_id, 'confirmed')
        msg = f"Reservation {res_id} CONFIRMED for {name}."
        logger.info(msg)
        await log_reservation_via_mcp(reservation)
        return msg
    elif action == 'reject':
        update_reservation_status(res_id, 'rejected')
        msg = f"Reservation {res_id} REJECTED for {name}."
        logger.info(msg)
        return msg
    elif action == 'skip':
        msg = f"Skipping reservation {res_id} for {name}."
        logger.info(msg)
        return msg
    else:
        msg = f"Invalid action: {action}"
        logger.warning(msg)
        return msg

async def admin_loop():
    logger.info("--- Parking Bot Admin Agent (Async) ---")
    logger.info("Waiting for pending reservations...")
    
    while True:
        try:
            # Synchronous DB call in async loop is blocking but okay for CLI tool
            pending = get_pending_reservations()
            
            if not pending:
                await asyncio.sleep(2) # Async sleep
                continue
                
            logger.info(f"\n[ALERT] Found {len(pending)} pending reservation(s).")
            
            for res in pending:
                logger.info(f"--- Reservation ID: {res['id']} ---")
                logger.info(f"User: {res['name']}")
                logger.info(f"Car:  {res['car_number']}")
                logger.info(f"Time: {res['start_time']} to {res['end_time']}")
                
                # We need to run input() in a separate thread to not block the loop, 
                # OR just block since it's a CLI tool where user attention is required anyway.
                # Blocking is fine here.
                while True:
                    # Use asyncio.to_thread for input if we wanted non-blocking, but blocking is fine.
                    choice = await asyncio.to_thread(input, "Approve this reservation? (y/n/skip): ")
                    choice = choice.strip().lower()
                    
                    if choice == 'y':
                        await process_admin_action(res, 'approve')
                        break
                    elif choice == 'n':
                        await process_admin_action(res, 'reject')
                        break
                    elif choice == 'skip':
                        await process_admin_action(res, 'skip')
                        break
                    else:
                        print("Invalid input. Please enter 'y', 'n', or 'skip'.")
            
            logger.info("All pending processed. Waiting for new requests...")
            
        except KeyboardInterrupt:
            logger.info("\nAdmin Agent stopping.")
            break
        except Exception as e:
            logger.error(f"Error in admin loop: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(admin_loop())
    except KeyboardInterrupt:
        pass
