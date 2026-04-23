#!/usr/bin/env python3
"""
Sandbox for smart limit execution logic and guardrails.
Simulates the exact algorithm from mt5_data.place_smart_order without
live MT5, reproducing every branch:

  Smart limit algorithm:
    Step 1 — Spread check (retry up to 6x / 30s)
    Step 2 — Record initial_mid (slippage reference)
    Step 3-5 — Place limit at ask+cushion (BUY) / bid-cushion (SELL)
              If no fill: walkback (re-read, re-price, retry up to 5x)

  Guardrails tested:
    - Delta == 0           : no trade
    - NaN allocation       : abort
    - Insufficient margin  : abort
    - Spread stays wide    : RuntimeError -> risk_reducing fallback / abort
    - Walkback exhausted   : RuntimeError -> risk_reducing fallback / abort
    - Crisis regime        : skip smart, go straight to emergency
    - Risk-off fill        : smart first, emergency fallback on failure
    - Risk-on fill         : smart only, abort on failure
"""
import csv
import logging
import os
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from logging_config import setup_logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXEC_SANDBOX_LOG = os.path.join(SCRIPT_DIR, "execution_sandbox_log.csv")

# ── Mirror constants from mt5_data.py ────────────────────────────────────────
MAX_SPREAD_PTS    = 0.75
SPREAD_WAIT_S     = 5
SPREAD_RETRIES    = 6
PASSIVE_WAIT_S    = 3.0
WALKBACK_STEP_PTS = 0.25
WALKBACK_WAIT_S   = 1.0
WALKBACK_RETRIES  = 8
NQ_MARGIN         = 16500
NQ_POINT_VALUE    = 20


# ── Simulated market tick sequence ───────────────────────────────────────────

@dataclass
class Tick:
    bid: float
    ask: float
    @property
    def spread(self): return round(self.ask - self.bid, 4)
    @property
    def mid(self):    return (self.bid + self.ask) / 2.0


@dataclass
class FillOutcome:
    """
    passive_fill_attempt: which attempt the market ticks down to our resting limit (0=never).
    Immediate fills (limit>=ask for BUY, limit<=bid for SELL) are auto-detected.
    """
    passive_fill_attempt: int = 0


class SimulatedMarket:
    """Feed of tick sequences + fill decisions for one scenario."""
    def __init__(self, ticks: List[Tick], fill: FillOutcome):
        self.ticks    = ticks
        self.fill     = fill
        self._idx     = 0
        self._attempt = 0

    def get_tick(self) -> Tick:
        t = self.ticks[min(self._idx, len(self.ticks) - 1)]
        self._idx += 1
        return t

    def try_fill(self, limit_price: float, direction: str,
                 tick: 'Tick') -> Optional[float]:
        """
        GTC BUY_LIMIT / SELL_LIMIT fill model:
          immediate: limit >= ask (BUY) or limit <= bid (SELL) -> fill at ask/bid
          passive:   market ticks to us on passive_fill_attempt -> fill at limit
        """
        self._attempt += 1
        if direction == "BUY":
            if limit_price >= tick.ask:
                return tick.ask
            if self._attempt == self.fill.passive_fill_attempt:
                return limit_price
        else:
            if limit_price <= tick.bid:
                return tick.bid
            if self._attempt == self.fill.passive_fill_attempt:
                return limit_price
        return None


# ── Simulated execution engine (mirrors mt5_data logic exactly) ──────────────

def sim_place_smart_order(market: SimulatedMarket, symbol: str,
                           volume: int, direction: str) -> dict:
    """
    Mirrors new mt5_data.place_smart_order: passive resting limit at mid,
    step 0.25pt toward aggressor each retry until immediate fill or passive fill.
    """
    # Step 1: spread gate
    tick = None
    for i in range(SPREAD_RETRIES):
        tick = market.get_tick()
        if tick.spread <= MAX_SPREAD_PTS:
            break
        logger.warning(f"spread_wide spread={tick.spread:.2f} attempt={i+1}/{SPREAD_RETRIES}")
    else:
        raise RuntimeError(
            f"Spread too wide after {SPREAD_RETRIES} checks: spread={tick.spread:.2f}"
        )

    # Step 2: record initial_mid
    initial_mid = tick.mid
    sign = 1 if direction == "BUY" else -1

    # Steps 3+: passive resting limit, walking toward aggressor
    for attempt in range(1, WALKBACK_RETRIES + 1):
        tick = market.get_tick()
        mid = tick.mid
        step = (attempt - 1) * WALKBACK_STEP_PTS
        limit_price = round(mid + sign * step, 2)
        wait_s = PASSIVE_WAIT_S if attempt == 1 else WALKBACK_WAIT_S

        logger.info(
            f"smart_order attempt={attempt}/{WALKBACK_RETRIES} dir={direction} "
            f"limit={limit_price:.2f} mid={mid:.2f} bid={tick.bid:.2f} "
            f"ask={tick.ask:.2f} step={step:+.2f}pt wait={wait_s}s "
            f"initial_mid={initial_mid:.2f}"
        )

        fill_price = market.try_fill(limit_price, direction, tick)
        if fill_price is not None:
            immediate = (direction == "BUY" and limit_price >= tick.ask) or \
                        (direction == "SELL" and limit_price <= tick.bid)
            fill_type = "immediate" if immediate else "passive"
            logger.info(
                f"smart_fill_{fill_type} fill={fill_price:.2f} limit={limit_price:.2f} "
                f"initial_mid={initial_mid:.2f} attempt={attempt}"
            )
            return {
                "fill_price": fill_price, "volume": volume,
                "initial_mid": initial_mid, "spread_at_order": tick.spread,
                "limit_price": limit_price, "attempts": attempt,
                "order_type": "SMART",
            }

        logger.warning(
            f"no_fill attempt={attempt}/{WALKBACK_RETRIES} "
            f"limit={limit_price:.2f} stepping_toward_aggressor"
        )

    raise RuntimeError(f"Smart order failed after {WALKBACK_RETRIES} walkback attempts")


def sim_place_emergency_order(market: SimulatedMarket, symbol: str,
                               volume: int, direction: str) -> dict:
    """Mirror of mt5_data.place_emergency_order."""
    tick = market.get_tick()
    initial_mid = tick.mid
    price = tick.ask if direction == "BUY" else tick.bid
    # Emergency always fills (deviation=50 pts)
    fill_price = round(price + (0.25 if direction == "BUY" else -0.25), 2)
    logger.warning(
        f"EMERGENCY_ORDER dir={direction} symbol={symbol} vol={volume} "
        f"price={price:.2f} fill={fill_price:.2f} spread={tick.spread:.2f} deviation=50"
    )
    return {
        "fill_price": fill_price, "volume": volume,
        "initial_mid": initial_mid, "spread_at_order": tick.spread,
        "limit_price": price, "attempts": 1,
        "order_type": "EMERGENCY",
    }


def sim_place_market_order(market: SimulatedMarket, symbol: str, volume: int,
                            direction: str, risk_reducing: bool,
                            crisis_mode: bool) -> dict:
    """Mirror of mt5_data.place_market_order routing logic."""
    if crisis_mode:
        logger.warning("crisis_mode=True skipping_smart -> emergency")
        return sim_place_emergency_order(market, symbol, volume, direction)

    try:
        return sim_place_smart_order(market, symbol, volume, direction)
    except RuntimeError as e:
        if risk_reducing:
            logger.warning(f"smart_failed falling_back_to_emergency error={e}")
            return sim_place_emergency_order(market, symbol, volume, direction)
        raise


# ── Scenario runner ──────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    direction: str       # BUY or SELL
    volume: int
    regime: str          # TREND / CRISIS / DEFAULT
    risk_reducing: bool
    target_cts: int
    actual_cts: int
    equity: float
    last_close: float
    alloc: float          # NaN triggers NaN guard
    market: SimulatedMarket
    expected_result: str  # FILLED / ABORT / EMERGENCY_FILLED / FAIL
    expected_order_type: str  # SMART / EMERGENCY / NONE


def run_scenario(s: Scenario) -> dict:
    result = {
        "scenario": s.name,
        "direction": s.direction,
        "volume": s.volume,
        "regime": s.regime,
        "risk_reducing": s.risk_reducing,
        "equity": s.equity,
        "alloc": s.alloc,
        "target_cts": s.target_cts,
        "actual_cts": s.actual_cts,
        "last_close": s.last_close,
        "fill_price": "N/A",
        "initial_mid": "N/A",
        "spread_at_fill": "N/A",
        "slip_total_pts": "N/A",
        "slip_spread_pts": "N/A",
        "slip_walkback_pts": "N/A",
        "attempts": "N/A",
        "order_type": "N/A",
        "result": "N/A",
        "expected": s.expected_result,
        "pass": False,
        "notes": "",
    }

    logger.info(f"\n{'='*60}\nSCENARIO: {s.name}\n{'='*60}")

    # ── Guardrail 1: NaN allocation ──────────────────────────────────────────
    if math.isnan(s.alloc):
        logger.error("nan_allocation guardrail triggered -> abort")
        result.update({"result": "ABORT", "order_type": "NONE",
                       "notes": "NaN allocation"})
        result["pass"] = (s.expected_result == "ABORT")
        return result

    # ── Guardrail 2: delta == 0 ───────────────────────────────────────────────
    delta = s.target_cts - s.actual_cts
    if delta == 0:
        logger.info("delta=0 no_trade guardrail")
        result.update({"result": "NO_TRADE", "order_type": "NONE",
                       "notes": "delta=0"})
        result["pass"] = (s.expected_result == "NO_TRADE")
        return result

    # ── Guardrail 3: margin check (only when increasing exposure) ────────────
    increasing = abs(s.target_cts) > abs(s.actual_cts)
    if delta != 0 and increasing:
        additional = abs(s.target_cts) - abs(s.actual_cts)
        margin_needed = additional * NQ_MARGIN
        margin_free = s.equity * 0.5  # simulate 50% margin available
        if margin_free < margin_needed:
            logger.warning(
                f"margin_insufficient free={margin_free:.0f} needed={margin_needed:.0f}"
            )
            result.update({"result": "ABORT", "order_type": "NONE",
                           "notes": f"Margin: need ${margin_needed:,.0f}, free ${margin_free:,.0f}"})
            result["pass"] = (s.expected_result == "ABORT")
            return result

    # ── Order routing ─────────────────────────────────────────────────────────
    crisis_mode = (s.regime == "CRISIS")
    try:
        fill = sim_place_market_order(
            s.market, "NQ_H", s.volume, s.direction,
            s.risk_reducing, crisis_mode
        )

        # Slippage decomposition
        sign = 1 if s.direction == "BUY" else -1
        slip_total  = (fill["fill_price"] - s.last_close) * sign
        slip_spread = (fill["fill_price"] - fill["initial_mid"]) * sign
        slip_wb     = (fill["initial_mid"] - s.last_close) * sign

        logger.info(
            f"fill_price={fill['fill_price']:.2f} initial_mid={fill['initial_mid']:.2f} "
            f"signal_close={s.last_close:.2f} slip_total={slip_total:+.2f} "
            f"slip_spread={slip_spread:+.2f} slip_walkback={slip_wb:+.2f} "
            f"attempts={fill['attempts']} order_type={fill['order_type']}"
        )

        filled = "EMERGENCY_FILLED" if fill["order_type"] == "EMERGENCY" else "FILLED"
        result.update({
            "fill_price": f"{fill['fill_price']:.2f}",
            "initial_mid": f"{fill['initial_mid']:.2f}",
            "spread_at_fill": f"{fill['spread_at_order']:.2f}",
            "slip_total_pts": f"{slip_total:+.2f}",
            "slip_spread_pts": f"{slip_spread:+.2f}",
            "slip_walkback_pts": f"{slip_wb:+.2f}",
            "attempts": fill["attempts"],
            "order_type": fill["order_type"],
            "result": filled,
        })

    except RuntimeError as e:
        logger.error(f"order_failed error={e}")
        result.update({"result": "FAIL", "order_type": "NONE", "notes": str(e)})

    result["pass"] = (result["result"] == s.expected_result and
                      result["order_type"] == s.expected_order_type)
    return result


# ── Build scenarios ───────────────────────────────────────────────────────────

def build_scenarios() -> List[Scenario]:
    BASE_BID   = 21000.00
    BASE_ASK   = 21000.25   # 0.25pt spread (normal)
    WIDE_BID   = 21000.00
    WIDE_ASK   = 21002.00   # 2.0pt spread (too wide)
    CLOSE      = 21000.00
    EQUITY     = 1_000_000.0

    def ok_tick(drift=0.0):
        return Tick(bid=BASE_BID + drift, ask=BASE_ASK + drift)

    def wide_tick():
        return Tick(bid=WIDE_BID, ask=WIDE_ASK)

    return [

        # ── Guardrail scenarios ────────────────────────────────────────────
        Scenario(
            name="G1: delta=0 (no trade)",
            direction="BUY", volume=0, regime="TREND",
            risk_reducing=False, target_cts=5, actual_cts=5,
            equity=EQUITY, last_close=CLOSE, alloc=0.5,
            market=SimulatedMarket([ok_tick()], FillOutcome()),
            expected_result="NO_TRADE", expected_order_type="NONE",
        ),
        Scenario(
            name="G2: NaN allocation",
            direction="BUY", volume=2, regime="TREND",
            risk_reducing=False, target_cts=2, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=float("nan"),
            market=SimulatedMarket([ok_tick()], FillOutcome()),
            expected_result="ABORT", expected_order_type="NONE",
        ),
        Scenario(
            name="G3: Insufficient margin (need 10 contracts)",
            direction="BUY", volume=10, regime="TREND",
            risk_reducing=False, target_cts=10, actual_cts=0,
            equity=100_000.0,
            last_close=CLOSE, alloc=0.99,
            market=SimulatedMarket([ok_tick()], FillOutcome()),
            expected_result="ABORT", expected_order_type="NONE",
        ),

        # ── Passive fill (market ticks to our resting limit) ─────────────
        # S1/S2: Best case — fill at mid, saves half the spread vs IOC-at-ask
        Scenario(
            name="S1: BUY passive fill at mid (market ticks to us, attempt 1)",
            direction="BUY", volume=3, regime="TREND",
            risk_reducing=False, target_cts=3, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=0.5,
            # passive_fill_attempt=1: ask doesn't drop to limit but market ticks to mid
            market=SimulatedMarket(
                [ok_tick()] + [ok_tick()] * WALKBACK_RETRIES,
                FillOutcome(passive_fill_attempt=1),
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),
        Scenario(
            name="S2: SELL passive fill at mid (market ticks to us, attempt 1)",
            direction="SELL", volume=2, regime="TREND",
            risk_reducing=False, target_cts=3, actual_cts=5,
            equity=EQUITY, last_close=CLOSE, alloc=0.3,
            market=SimulatedMarket(
                [ok_tick()] + [ok_tick()] * WALKBACK_RETRIES,
                FillOutcome(passive_fill_attempt=1),
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),

        # ── Immediate fill (limit steps to ask level) ─────────────────────
        # S3/S4: No passive fill → attempt 2 limit = mid+0.25 = ask → fills immediately
        Scenario(
            name="S3: BUY no passive -> immediate fill at ask, attempt 2",
            direction="BUY", volume=2, regime="TREND",
            risk_reducing=False, target_cts=2, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=0.4,
            market=SimulatedMarket(
                [ok_tick()] + [ok_tick()] * WALKBACK_RETRIES,
                FillOutcome(passive_fill_attempt=0),  # no passive; attempt 2 = ask = immediate
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),
        Scenario(
            name="S4: SELL no passive -> immediate fill at bid, attempt 2",
            direction="SELL", volume=2, regime="TREND",
            risk_reducing=False, target_cts=1, actual_cts=3,
            equity=EQUITY, last_close=CLOSE, alloc=0.2,
            market=SimulatedMarket(
                [ok_tick()] + [ok_tick()] * WALKBACK_RETRIES,
                FillOutcome(passive_fill_attempt=0),
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),

        # ── Wide but acceptable spread (0.75pt) ──────────────────────────
        # S5: spread=0.75pt → need 3 steps to reach ask; immediate fill at attempt 3
        Scenario(
            name="S5: Wide spread (0.75pt) -> immediate fill at attempt 3",
            direction="BUY", volume=2, regime="TREND",
            risk_reducing=False, target_cts=2, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=0.4,
            # bid=21000, ask=21000.75, mid=21000.375
            # att1: limit=21000.375 < 21000.75 → no immediate
            # att2: limit=21000.625 < 21000.75 → no immediate
            # att3: limit=21000.875 >= 21000.75 → IMMEDIATE FILL
            market=SimulatedMarket(
                [Tick(bid=21000.00, ask=21000.75)] * (WALKBACK_RETRIES + 2),
                FillOutcome(passive_fill_attempt=0),
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),

        # ── Spread gate failures ───────────────────────────────────────────
        Scenario(
            name="S6: Spread stays wide (2pt) -> FAIL (risk-on, abort)",
            direction="BUY", volume=2, regime="TREND",
            risk_reducing=False, target_cts=2, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=0.4,
            market=SimulatedMarket(
                [wide_tick()] * SPREAD_RETRIES,
                FillOutcome(passive_fill_attempt=0),
            ),
            expected_result="FAIL", expected_order_type="NONE",
        ),
        Scenario(
            name="S7: Spread stays wide (2pt) -> emergency fallback (risk-off)",
            direction="SELL", volume=3, regime="TREND",
            risk_reducing=True, target_cts=2, actual_cts=5,
            equity=EQUITY, last_close=CLOSE, alloc=0.3,
            market=SimulatedMarket(
                [wide_tick()] * (SPREAD_RETRIES + 4),
                FillOutcome(passive_fill_attempt=0),
            ),
            expected_result="EMERGENCY_FILLED", expected_order_type="EMERGENCY",
        ),
        Scenario(
            name="S8: Wide spread, narrows after 2 retries, then passive fill",
            direction="BUY", volume=2, regime="TREND",
            risk_reducing=False, target_cts=2, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=0.4,
            market=SimulatedMarket(
                [wide_tick(), wide_tick(), ok_tick()] + [ok_tick()] * WALKBACK_RETRIES,
                FillOutcome(passive_fill_attempt=1),
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),

        # ── Crisis / emergency ────────────────────────────────────────────
        Scenario(
            name="S9: Crisis mode -> skip smart, direct emergency",
            direction="SELL", volume=5, regime="CRISIS",
            risk_reducing=True, target_cts=0, actual_cts=5,
            equity=EQUITY, last_close=CLOSE, alloc=0.0,
            market=SimulatedMarket(
                [ok_tick()] * (SPREAD_RETRIES + WALKBACK_RETRIES),
                FillOutcome(passive_fill_attempt=0),
            ),
            expected_result="EMERGENCY_FILLED", expected_order_type="EMERGENCY",
        ),
        Scenario(
            name="S10: Recovery regime -> risk-on, passive fill at mid",
            direction="BUY", volume=3, regime="RECOVERY",
            risk_reducing=False, target_cts=3, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=0.6,
            market=SimulatedMarket(
                [ok_tick()] + [ok_tick()] * WALKBACK_RETRIES,
                FillOutcome(passive_fill_attempt=1),
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),

        # ── Drifting market ───────────────────────────────────────────────
        # S11: market drifts up 0.25pt per tick; limit tracks mid+step, immediate at att2
        Scenario(
            name="S11: Market drifts up 0.25pt/tick -> immediate at att2 (higher price)",
            direction="BUY", volume=2, regime="TREND",
            risk_reducing=False, target_cts=2, actual_cts=0,
            equity=EQUITY, last_close=CLOSE, alloc=0.4,
            # Each tick: ask rises 0.25pt. att1 limit=new_mid < new_ask (passive, no fill)
            # att2: limit=new_mid+0.25 >= new_ask -> immediate fill at drifted ask
            market=SimulatedMarket(
                [ok_tick()] + [ok_tick(drift=i * 0.25) for i in range(WALKBACK_RETRIES)],
                FillOutcome(passive_fill_attempt=0),
            ),
            expected_result="FILLED", expected_order_type="SMART",
        ),
    ]


# ── Logging + summary ─────────────────────────────────────────────────────────

def log_result(row: dict):
    fields = [
        "scenario", "direction", "volume", "regime", "risk_reducing",
        "equity", "alloc", "target_cts", "actual_cts", "last_close",
        "fill_price", "initial_mid", "spread_at_fill",
        "slip_total_pts", "slip_spread_pts", "slip_walkback_pts",
        "attempts", "order_type", "result", "expected", "pass", "notes",
    ]
    exists = os.path.exists(EXEC_SANDBOX_LOG)
    with open(EXEC_SANDBOX_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def run_all():
    setup_logging("exec_sandbox")
    logger.info("=" * 60)
    logger.info("SMART LIMIT EXECUTION SANDBOX")
    logger.info("=" * 60)

    scenarios = build_scenarios()
    results = []

    for s in scenarios:
        r = run_scenario(s)
        log_result(r)
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["pass"])
    total  = len(results)

    print(f"\n{'Scenario':<52} {'Expected':<20} {'Got':<20} {'Type':<12} "
          f"{'Fill':>9} {'Slip':>7} {'Att':>4} {'PASS'}")
    print("-" * 135)
    for r in results:
        slip  = r["slip_total_pts"] if r["slip_total_pts"] != "N/A" else "  N/A"
        fill  = r["fill_price"]     if r["fill_price"]     != "N/A" else "    N/A"
        att   = str(r["attempts"])  if r["attempts"]       != "N/A" else "  -"
        ok    = "PASS" if r["pass"] else "FAIL"
        print(f"{r['scenario']:<52} {r['expected']:<20} {r['result']:<20} "
              f"{r['order_type']:<12} {fill:>9} {slip:>7} {att:>4}  {ok}")
        if r["notes"] and not r["pass"]:
            print(f"  {'':52} > {r['notes']}")

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} scenarios passed")
    if passed == total:
        print("ALL PASS - execution logic validated")
    else:
        print(f"FAILURES: {total - passed}")
    print(f"Log: {EXEC_SANDBOX_LOG}")
    logger.info(f"execution_sandbox_complete passed={passed}/{total}")


if __name__ == "__main__":
    if os.path.exists(EXEC_SANDBOX_LOG):
        os.remove(EXEC_SANDBOX_LOG)
    run_all()
