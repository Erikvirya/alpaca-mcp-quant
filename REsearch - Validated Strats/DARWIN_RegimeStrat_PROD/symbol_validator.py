#!/usr/bin/env python3
"""
Symbol validation utility for NQ contracts on Darwinex.
Checks which NQ contracts are available and validates rollover readiness.
"""
import logging
from datetime import datetime, date
from zoneinfo import ZoneInfo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def validate_symbol_exists(symbol: str) -> bool:
    """Check if symbol exists on Darwinex MT5."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        
        # Search for symbol
        symbol_info = mt5.symbol_info(symbol)
        mt5.shutdown()
        
        if symbol_info is None:
            logger.warning(f"❌ Symbol not found: {symbol}")
            return False
        
        logger.info(f"✅ Symbol validated: {symbol} ({symbol_info.description})")
        return True
        
    except Exception as e:
        logger.error(f"Symbol validation failed for {symbol}: {e}")
        return False

def find_available_nq_symbols() -> list:
    """Find all available NQ symbols on Darwinex."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return []
        
        # Get all symbols
        all_symbols = mt5.symbols_get()
        mt5.shutdown()
        
        # Filter for NQ contracts (NQ_ format)
        nq_symbols = []
        for s in all_symbols:
            if s.name.startswith("NQ_") and len(s.name) == 4:  # NQ_H, NQ_M, etc.
                nq_symbols.append(s.name)
        
        return sorted(nq_symbols)
        
    except Exception as e:
        logger.error(f"Symbol search failed: {e}")
        return []

def get_target_symbol() -> str:
    """Get target symbol based on current date."""
    now = datetime.now()
    month = now.month
    
    if month <= 3:
        return "NQ_H"  # March contract
    elif month <= 6:
        return "NQ_M"  # June contract  
    elif month <= 9:
        return "NQ_U"  # September contract
    else:
        return "NQ_Z"  # December contract

def check_rollover_readiness() -> dict:
    """Check if next quarter's contract is available for rollover."""
    target = get_target_symbol()
    available = find_available_nq_symbols()
    
    result = {
        "current_date": datetime.now().strftime('%Y-%m-%d'),
        "target_symbol": target,
        "available_symbols": available,
        "target_available": target in available,
        "rollover_ready": False
    }
    
    if target in available:
        result["rollover_ready"] = True
        logger.info(f"✅ Rollover ready: {target} is available")
    else:
        logger.warning(f"⚠️ Rollover not ready: {target} not yet available")
        if available:
            logger.info(f"Currently available: {', '.join(available)}")
    
    return result

def main():
    """Run symbol validation checks."""
    logger.info("=" * 60)
    logger.info("NQ SYMBOL VALIDATION - DARWINEX")
    logger.info("=" * 60)
    
    # Check current target
    target = get_target_symbol()
    logger.info(f"Target symbol for current date: {target}")
    
    # Find all available NQ symbols
    available = find_available_nq_symbols()
    logger.info(f"Available NQ symbols: {available}")
    
    # Validate target exists
    if validate_symbol_exists(target):
        logger.info("✅ Target symbol is ready for trading")
    else:
        logger.warning("⚠️ Target symbol not available - will use fallback")
    
    # Check rollover readiness
    logger.info("\n--- ROLLOVER READINESS CHECK ---")
    result = check_rollover_readiness()
    
    # Show quarterly schedule
    logger.info("\n--- QUARTERLY SCHEDULE ---")
    schedule = [
        ("NQ_H", "March", "Q1"),
        ("NQ_M", "June", "Q2"), 
        ("NQ_U", "September", "Q3"),
        ("NQ_Z", "December", "Q4")
    ]
    
    for symbol, month, quarter in schedule:
        status = "✅ AVAILABLE" if symbol in available else "❌ NOT AVAILABLE"
        logger.info(f"{symbol}: {month} contract ({quarter}) - {status}")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
