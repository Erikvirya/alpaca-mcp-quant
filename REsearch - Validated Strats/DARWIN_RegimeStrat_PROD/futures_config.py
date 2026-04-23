"""
Futures contract specifications for Darwinex Zero MT5 production.
Subset of the full futures_config — only NQ needed for production.
"""

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class FuturesContract:
    """Specification for a single futures product."""
    symbol: str
    name: str
    point_value: float
    tick_size: float
    margin: float
    exchange: str
    roll_months: tuple = (3, 6, 9, 12)

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value

    def notional(self, price: float) -> float:
        return price * self.point_value


NQ = FuturesContract(
    symbol="NQ=F",
    name="E-mini Nasdaq-100",
    point_value=20.0,
    tick_size=0.25,
    margin=34900.0,   # Darwinex Zero confirmed
    exchange="CME",
)

COMMISSION_PER_SIDE = 4.00  # Darwinex Zero confirmed ($4/contract/side)
