from mcp.server.fastmcp import FastMCP      # MCP 서버를 만들기 위한 import

# Create an MCP server
mcp = FastMCP("Gwai-study")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport='stdio')