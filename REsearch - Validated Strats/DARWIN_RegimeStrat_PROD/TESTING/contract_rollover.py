#!/usr/bin/env python3
"""
Contract rollover reminder for NQ futures.
Shows current contract and next rollover date.
"""
from datetime import datetime, date

def get_contract_schedule() -> dict:
    """Get quarterly contract schedule for current year."""
    now = datetime.now()
    year = now.year
    
    return {
        "NQ_H": {"month": 3, "name": "March", "roll_by": date(year, 3, 10)},
        "NQ_M": {"month": 6, "name": "June", "roll_by": date(year, 6, 9)},
        "NQ_U": {"month": 9, "name": "September", "roll_by": date(year, 9, 8)},
        "NQ_Z": {"month": 12, "name": "December", "roll_by": date(year, 12, 8)},
    }

def get_current_contract() -> str:
    """Get current front month contract."""
    now = datetime.now()
    month = now.month
    
    if month <= 3:
        return "NQ_H"
    elif month <= 6:
        return "NQ_M"
    elif month <= 9:
        return "NQ_U"
    else:
        return "NQ_Z"

def get_next_contract(current: str) -> str:
    """Get next contract in cycle."""
    cycle = ["NQ_H", "NQ_M", "NQ_U", "NQ_Z"]
    idx = cycle.index(current)
    return cycle[(idx + 1) % 4]

def main():
    """Show contract rollover information."""
    schedule = get_contract_schedule()
    current = get_current_contract()
    next_contract = get_next_contract(current)
    
    today = date.today()
    roll_by = schedule[current]["roll_by"]
    days_to_roll = (roll_by - today).days
    
    print("=" * 50)
    print(f"NQ CONTRACT ROLLOVER STATUS")
    print("=" * 50)
    print(f"Current Date: {today.strftime('%Y-%m-%d')}")
    print(f"Current Contract: {current} ({schedule[current]['name']} {today.year})")
    print(f"Next Contract: {next_contract} ({schedule[next_contract]['name']} {today.year})")
    print(f"Roll By Date: {roll_by.strftime('%Y-%m-%d')}")
    print(f"Days to Rollover: {days_to_roll}")
    
    if days_to_roll <= 5:
        print("\n⚠️  URGENT: Rollover due soon!")
    elif days_to_roll <= 14:
        print("\n📅 Rollover coming up - prepare to update symbol")
    else:
        print(f"\n✅ No rollover needed for {days_to_roll} days")
    
    print("\nDarwinex Symbol Format:")
    for symbol, info in schedule.items():
        print(f"  {symbol}: {info['name']} contract")
    
    print("\nTo update symbol in production:")
    print("1. Edit config.env: MT5_NQ_SYMBOL=<NEW_SYMBOL>")
    print("2. Or wait for auto-detection (if implemented)")

if __name__ == "__main__":
    main()
