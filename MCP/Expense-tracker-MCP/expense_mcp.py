import json
import os
from datetime import datetime
from typing import List, Dict
from fastmcp import FastMCP

mcp = FastMCP(name="Expense Tracker MCP")

DATA_FILE = "expenses.json"


# ---------------------------
# Utility functions
# ---------------------------

def load_data() -> List[Dict]:
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: List[Dict]):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# MCP TOOLS
# ---------------------------

@mcp.tool
def add_expense(amount: float, category: str, note: str = "") -> Dict:
    """Add a new expense entry"""

    data = load_data()

    expense = {
        "amount": amount,
        "category": category.lower(),
        "note": note,
        "time": datetime.now().isoformat()
    }

    data.append(expense)
    save_data(data)

    return {
        "status": "success",
        "message": "Expense added",
        "expense": expense
    }


@mcp.tool
def list_expenses() -> List[Dict]:
    """List all expenses"""
    return load_data()


@mcp.tool
def total_expense() -> Dict:
    """Get total expense amount"""

    data = load_data()
    total = sum(item["amount"] for item in data)

    return {
        "total": total,
        "count": len(data)
    }


@mcp.tool
def category_summary() -> Dict:
    """Get category-wise expense summary"""

    data = load_data()
    summary = {}

    for item in data:
        cat = item["category"]
        summary[cat] = summary.get(cat, 0) + item["amount"]

    return summary


@mcp.tool
def clear_expenses() -> Dict:
    """Delete all expenses"""

    save_data([])
    return {"status": "cleared", "message": "All expenses removed"}


# ---------------------------
# Run MCP server
# ---------------------------

if __name__ == "__main__":
    mcp.run()
