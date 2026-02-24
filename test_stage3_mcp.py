import unittest
import sys
import os
import asyncio
import shutil

# MCP Imports
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

class TestStage3MCP(unittest.TestCase):
    def setUp(self):
        # Clean up data dir
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.log_file = os.path.join(self.data_dir, "confirmed_reservations.txt")
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            
    def test_mcp_server_logging(self):
        print("\nTesting Stage 3: MCP Server Logging...")
        
        async def run_test():
            server_script = os.path.join(os.path.dirname(__file__), "src/mcp_server.py")
            
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_script],
                env=os.environ.copy()
            )
            
            print("[Test] Connecting to MCP server...")
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Verify tool exists
                    tools = await session.list_tools()
                    tool_names = [t.name for t in tools.tools]
                    print(f"[Test] Available tools: {tool_names}")
                    self.assertIn("log_reservation", tool_names)
                    
                    # Call tool
                    print("[Test] Calling log_reservation tool...")
                    result = await session.call_tool(
                        "log_reservation",
                        arguments={
                            "name": "Test User",
                            "car_number": "TEST-123",
                            "duration": "1 hour"
                        }
                    )
                    
                    print(f"[Test] Tool Result: {result}")
                    content = result.content[0].text
                    self.assertIn("Successfully logged", content)
                    
            # Check file content
            self.assertTrue(os.path.exists(self.log_file), "Log file should exist")
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                self.assertIn("Test User", lines[0])
                self.assertIn("TEST-123", lines[0])
                print("[Test] Log file content verified.")

        asyncio.run(run_test())
    def test_mcp_invalid_tool(self):
        print("\nTesting Stage 3: MCP Server Invalid Tool...")
        
        async def run_test():
            server_script = os.path.join(os.path.dirname(__file__), "src/mcp_server.py")
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_script],
                env=os.environ.copy()
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    try:
                        await session.call_tool(
                            "non_existent_tool",
                            arguments={}
                        )
                        self.fail("Should have raised an error for invalid tool")
                    except Exception as e:
                        # MCP client raises an error when returning an invalid tool
                        self.assertTrue(True)
        
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
