import random 
from fastmcp import FastMCP

mcp = FastMCP(name="Demo Server")

@mcp.tool
def roll_dice(n_dice=1):
    return [random.randint(1,6) for _ in range(n_dice)]

@mcp.tool
def add_numbers(a,b):
    return a+b

if __name__ == "__main__":
    mcp.run()
