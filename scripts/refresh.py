#!/usr/bin/env python3
"""
Refresh paper-trading data and dashboard.
Fetches current prices, tracks performance, simulates strategies, and generates dashboard.
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DASHBOARD_DIR = BASE_DIR / "dashboard"
ALERTS_CSV = DATA_DIR / "alerts.csv"
CALLS_CSV = DATA_DIR / "calls.csv"

DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex/tokens/"
GECKOTERMINAL_BASE = "https://api.geckoterminal.com/api/v2/networks/solana/tokens/"
PUMPFUN_BASE = "https://frontend-api.pump.fun/coins/"

RATE_LIMIT_DELAY = 0.5  # seconds between API calls
REQUEST_TIMEOUT = 10
SLIPPAGE_PCT = 0.02  # 2% slippage per leg


def fetch_with_retry(url: str, max_retries: int = 3) -> Optional[dict]:
    """Fetch JSON from URL with retries and error handling."""
    for attempt in range(max_retries):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"  HTTP {resp.status_code} for {url}")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_price_dexscreener(ca: str) -> Optional[Tuple[float, float, float]]:
    """Fetch from DexScreener. Returns (price_usd, mc_usd, liq_usd) or None."""
    data = fetch_with_retry(f"{DEXSCREENER_BASE}{ca}")
    if not data or "pairs" not in data or not data["pairs"]:
        return None
    
    pair = data["pairs"][0]
    try:
        price = float(pair.get("priceUsd", 0))
        mc = float(pair.get("marketCap", 0) or pair.get("fdv", 0))
        liq = float(pair.get("liquidity", {}).get("usd", 0))
        return (price, mc, liq)
    except (ValueError, TypeError, KeyError):
        return None


def fetch_price_geckoterminal(ca: str) -> Optional[Tuple[float, float, float]]:
    """Fetch from GeckoTerminal. Returns (price_usd, mc_usd, liq_usd) or None."""
    data = fetch_with_retry(f"{GECKOTERMINAL_BASE}{ca}")
    if not data or "data" not in data:
        return None
    
    try:
        attrs = data["data"]["attributes"]
        price = float(attrs.get("price_usd", 0))
        mc = float(attrs.get("market_cap_usd", 0) or attrs.get("fdv_usd", 0))
        liq = float(attrs.get("reserve_in_usd", 0))
        return (price, mc, liq)
    except (ValueError, TypeError, KeyError):
        return None


def fetch_price_pumpfun(ca: str) -> Optional[Tuple[float, float, float, bool]]:
    """
    Fetch from pump.fun. Returns (price_usd, mc_usd, liq_usd, on_bonding_curve) or None.
    on_bonding_curve=True means token is still on pump.fun and hasn't migrated yet.
    """
    data = fetch_with_retry(f"{PUMPFUN_BASE}{ca}")
    if not data:
        return None
    
    try:
        price = float(data.get("usd_market_cap", 0)) / float(data.get("total_supply", 1))
        mc = float(data.get("usd_market_cap", 0))
        liq = float(data.get("virtual_sol_reserves", 0)) * 180  # rough SOL price estimate
        # Check if still on bonding curve (not migrated to Raydium)
        on_bonding_curve = not data.get("complete", False)
        return (price, mc, liq, on_bonding_curve)
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return None


def fetch_ohlcv_geckoterminal(ca: str, since_ts: int) -> Optional[Tuple[float, float]]:
    """
    Fetch OHLCV from GeckoTerminal to get ATH and min since alert.
    Returns (ath, min) or None.
    """
    url = f"{GECKOTERMINAL_BASE}{ca}/ohlcv/minute?before_timestamp={int(time.time())}&limit=1000"
    data = fetch_with_retry(url)
    if not data or "data" not in data:
        return None
    
    try:
        ohlcv = data["data"]["attributes"]["ohlcv_list"]
        ath = 0.0
        min_price = float('inf')
        for candle in ohlcv:
            ts = candle[0]
            if ts >= since_ts:
                high = float(candle[2])
                low = float(candle[3])
                ath = max(ath, high)
                min_price = min(min_price, low)
        return (ath, min_price) if ath > 0 else None
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def fetch_current_price(ca: str, alert_time: str) -> Dict:
    """
    Fetch current price from multiple sources.
    Returns dict with price, mc, liq, status.
    """
    result = {
        "ca": ca,
        "current_price_usd": 0.0,
        "current_mc_usd": 0.0,
        "current_liq_usd": 0.0,
        "status": "dead",
        "on_bonding_curve": False,
    }
    
    print(f"Fetching {ca}...")
    
    # Try DexScreener first
    data = fetch_price_dexscreener(ca)
    if data:
        result["current_price_usd"], result["current_mc_usd"], result["current_liq_usd"] = data
    
    # Fallback to GeckoTerminal
    if result["current_price_usd"] == 0:
        data = fetch_price_geckoterminal(ca)
        if data:
            result["current_price_usd"], result["current_mc_usd"], result["current_liq_usd"] = data
    
    # Fallback to pump.fun
    if result["current_price_usd"] == 0:
        data = fetch_price_pumpfun(ca)
        if data:
            result["current_price_usd"], result["current_mc_usd"], result["current_liq_usd"], result["on_bonding_curve"] = data
    else:
        # Even if we got price from DexScreener/GeckoTerminal, check pump.fun for bonding curve status
        pumpfun_data = fetch_price_pumpfun(ca)
        if pumpfun_data:
            # Use pump.fun bonding curve status if available
            result["on_bonding_curve"] = pumpfun_data[3]
            # If pump.fun has a price and we're on bonding curve, prefer pump.fun data
            if result["on_bonding_curve"] and pumpfun_data[0] > 0:
                result["current_price_usd"] = pumpfun_data[0]
                result["current_mc_usd"] = pumpfun_data[1]
                result["current_liq_usd"] = pumpfun_data[2]
    
    # Determine status with proper bonding curve handling
    if result["current_price_usd"] == 0:
        # No price data from any source - definitely dead
        result["status"] = "dead"
    elif result["on_bonding_curve"]:
        # Still on pump.fun bonding curve - treat as active regardless of liquidity
        result["status"] = "active"
    elif result["current_liq_usd"] == 0 and result["current_price_usd"] > 0:
        # Has price but no liquidity - likely still on bonding curve (pump.fun API may have failed)
        # Treat as active until proven otherwise
        result["status"] = "active"
    elif result["current_liq_usd"] >= 1000:
        # Migrated pool with sufficient liquidity
        result["status"] = "active"
    else:
        # Migrated but liquidity dried up (liq > 0 but < $1k)
        result["status"] = "dead"
    
    return result


def calculate_strategy_a(alert_price: float, ath: float, min_price: float, current_price: float, status: str) -> Dict:
    """
    Strategy A: Sell ALL at 2x with stop-loss.
    $10 stake, 0.45% fee + 2% slippage per leg.
    - Stop loss at -50% from entry
    - Sell all at 2x
    - Conservative: assume stop hit before target if order unknown
    """
    FEE_PCT = 0.0045
    SLIPPAGE_PCT = 0.02
    TOTAL_COST_PCT = FEE_PCT + SLIPPAGE_PCT
    
    initial_stake = 10.0
    
    # Account for entry costs (fee + slippage)
    entry_cost = initial_stake * TOTAL_COST_PCT
    net_stake = initial_stake - entry_cost
    tokens_bought = net_stake / alert_price
    
    target_price = alert_price * 2.0
    stop_price = alert_price * 0.5
    
    realized_pnl = 0.0
    remaining_tokens = tokens_bought
    
    # Conservative: if we hit both stop and target, assume stop hit first
    hit_stop = min_price <= stop_price
    hit_target = ath >= target_price
    
    if hit_stop:
        # Stopped out completely at -50%
        exit_value = remaining_tokens * stop_price
        exit_cost = exit_value * TOTAL_COST_PCT
        realized_pnl = exit_value - exit_cost - initial_stake
        remaining_tokens = 0
    elif hit_target:
        # Sold all at 2x
        exit_value = remaining_tokens * target_price
        exit_cost = exit_value * TOTAL_COST_PCT
        realized_pnl = exit_value - exit_cost - initial_stake
        remaining_tokens = 0
    
    # Mark remaining to market
    if remaining_tokens > 0:
        mark_price = 0 if status == "dead" else current_price
        current_value = remaining_tokens * mark_price
        unrealized_pnl = current_value - initial_stake
    else:
        unrealized_pnl = 0.0
    
    total_pnl = realized_pnl + unrealized_pnl
    multiple = (initial_stake + total_pnl) / initial_stake if initial_stake > 0 else 0
    
    return {
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "multiple": multiple,
        "hit_target": hit_target,
        "hit_stop": hit_stop,
    }


def calculate_strategy_b(alert_price: float, ath: float, min_price: float, current_price: float, status: str) -> Dict:
    """
    Strategy B: Sell ALL at +50% (1.5x) with stop-loss.
    $10 stake, 0.45% fee + 2% slippage per leg.
    - Stop loss at -50% from entry
    - Sell all at 1.5x
    - Conservative: assume stop hit before target if order unknown
    """
    FEE_PCT = 0.0045
    SLIPPAGE_PCT = 0.02
    TOTAL_COST_PCT = FEE_PCT + SLIPPAGE_PCT
    
    initial_stake = 10.0
    
    # Account for entry costs
    entry_cost = initial_stake * TOTAL_COST_PCT
    net_stake = initial_stake - entry_cost
    tokens_bought = net_stake / alert_price
    
    target_price = alert_price * 1.5
    stop_price = alert_price * 0.5
    
    realized_pnl = 0.0
    remaining_tokens = tokens_bought
    
    hit_stop = min_price <= stop_price
    hit_target = ath >= target_price
    
    if hit_stop:
        exit_value = remaining_tokens * stop_price
        exit_cost = exit_value * TOTAL_COST_PCT
        realized_pnl = exit_value - exit_cost - initial_stake
        remaining_tokens = 0
    elif hit_target:
        exit_value = remaining_tokens * target_price
        exit_cost = exit_value * TOTAL_COST_PCT
        realized_pnl = exit_value - exit_cost - initial_stake
        remaining_tokens = 0
    
    # Mark remaining to market
    if remaining_tokens > 0:
        mark_price = 0 if status == "dead" else current_price
        current_value = remaining_tokens * mark_price
        unrealized_pnl = current_value - initial_stake
    else:
        unrealized_pnl = 0.0
    
    total_pnl = realized_pnl + unrealized_pnl
    multiple = (initial_stake + total_pnl) / initial_stake if initial_stake > 0 else 0
    
    return {
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "multiple": multiple,
        "hit_target": hit_target,
        "hit_stop": hit_stop,
    }


def calculate_strategy_c(alert_price: float, ath: float, min_price: float, current_price: float, status: str) -> Dict:
    """
    Strategy C: Sell HALF at 2x, hold rest with stop-loss.
    $10 stake, 0.45% fee + 2% slippage per leg.
    - Sell half at 2x
    - Stop loss at -50% from entry
    - Mark remainder to market
    - Conservative: assume stop hit before target if order unknown
    """
    FEE_PCT = 0.0045
    SLIPPAGE_PCT = 0.02
    TOTAL_COST_PCT = FEE_PCT + SLIPPAGE_PCT
    
    initial_stake = 10.0
    
    # Account for entry costs
    entry_cost = initial_stake * TOTAL_COST_PCT
    net_stake = initial_stake - entry_cost
    tokens_bought = net_stake / alert_price
    
    target_price = alert_price * 2.0
    stop_price = alert_price * 0.5
    
    realized_pnl = 0.0
    remaining_tokens = tokens_bought
    
    hit_stop = min_price <= stop_price
    hit_target = ath >= target_price
    
    if hit_stop:
        # Stopped out completely at -50%
        exit_value = remaining_tokens * stop_price
        exit_cost = exit_value * TOTAL_COST_PCT
        realized_pnl = exit_value - exit_cost - initial_stake
        remaining_tokens = 0
    elif hit_target:
        # Sold half at 2x
        half_tokens = remaining_tokens / 2
        exit_value = half_tokens * target_price
        exit_cost = exit_value * TOTAL_COST_PCT
        realized_pnl = exit_value - exit_cost - (initial_stake / 2)
        remaining_tokens = half_tokens
    
    # Mark remaining to market
    if remaining_tokens > 0:
        mark_price = 0 if status == "dead" else current_price
        
        # Calculate cost basis for remaining tokens
        if hit_target:
            remaining_cost_basis = initial_stake / 2
        else:
            remaining_cost_basis = initial_stake
        
        current_value = remaining_tokens * mark_price
        unrealized_pnl = current_value - remaining_cost_basis
    else:
        unrealized_pnl = 0.0
    
    total_pnl = realized_pnl + unrealized_pnl
    multiple = (initial_stake + total_pnl) / initial_stake if initial_stake > 0 else 0
    
    return {
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "multiple": multiple,
        "hit_target": hit_target,
        "hit_stop": hit_stop,
    }


def calculate_strategy_d(alert_price: float, current_price: float, status: str, alert_time: str) -> Dict:
    """
    Strategy D: Hold 24h baseline.
    $10 stake, 0.45% fee + 2% slippage per leg.
    - Buy at alert
    - Hold 24 hours
    - Sell at current price
    """
    FEE_PCT = 0.0045
    SLIPPAGE_PCT = 0.02
    TOTAL_COST_PCT = FEE_PCT + SLIPPAGE_PCT
    
    initial_stake = 10.0
    
    # Account for entry costs
    entry_cost = initial_stake * TOTAL_COST_PCT
    net_stake = initial_stake - entry_cost
    tokens_bought = net_stake / alert_price
    
    # Check if 24h passed
    alert_dt = datetime.fromisoformat(alert_time.replace('Z', '+00:00'))
    now_dt = datetime.now(timezone.utc)
    hours_elapsed = (now_dt - alert_dt).total_seconds() / 3600
    
    if hours_elapsed >= 24:
        # Sell at current or 0 if dead
        exit_price = current_price if status == "active" else 0
        exit_value = tokens_bought * exit_price
        exit_cost = exit_value * TOTAL_COST_PCT
        pnl = exit_value - exit_cost - initial_stake
        return {
            "pnl": pnl,
            "exited": True,
        }
    else:
        # Still holding, mark to market
        mark_price = current_price if status == "active" else 0
        unrealized = tokens_bought * mark_price - net_stake
        return {
            "pnl": unrealized,
            "exited": False,
        }




def load_existing_calls() -> Dict[str, Dict]:
    """Load existing calls.csv to preserve ATH/min tracking."""
    if not CALLS_CSV.exists():
        return {}
    
    calls = {}
    with open(CALLS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            calls[row['ca']] = row
    return calls


def save_calls(calls: List[Dict]):
    """Save updated calls to calls.csv."""
    if not calls:
        return
    
    fieldnames = [
        'ca', 'symbol', 'alert_time_utc', 'alert_price_usd', 
        'current_price_usd', 'current_mc_usd', 'current_liq_usd',
        'ath_since_alert', 'min_since_alert', 'max_multiple',
        'status', 'last_updated_utc'
    ]
    
    with open(CALLS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calls)


def refresh_data():
    """Main refresh function: fetch prices and update calls.csv."""
    if not ALERTS_CSV.exists():
        print(f"No alerts file found at {ALERTS_CSV}")
        return
    
    print("Loading alerts...")
    alerts = []
    with open(ALERTS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alerts.append(row)
    
    print(f"Found {len(alerts)} alerts")
    
    # Load existing calls for ATH/min tracking
    existing_calls = load_existing_calls()
    
    updated_calls = []
    for alert in alerts:
        ca = alert['ca']
        alert_time = alert['alert_time_utc']
        alert_price = float(alert['price_usd'])
        symbol = alert['symbol']
        
        # Fetch current data
        current = fetch_current_price(ca, alert_time)
        
        # Try to get OHLCV for better ATH/min
        alert_ts = int(datetime.fromisoformat(alert_time.replace('Z', '+00:00')).timestamp())
        ohlcv = fetch_ohlcv_geckoterminal(ca, alert_ts)
        
        # Calculate ATH and min
        existing = existing_calls.get(ca, {})
        prev_ath = float(existing.get('ath_since_alert', 0))
        prev_min = float(existing.get('min_since_alert', alert_price))
        
        if ohlcv:
            ath_ohlcv, min_ohlcv = ohlcv
            ath = max(prev_ath, ath_ohlcv, current['current_price_usd'])
            min_price = min(prev_min, min_ohlcv) if min_ohlcv < float('inf') else prev_min
        else:
            ath = max(prev_ath, current['current_price_usd'])
            min_price = prev_min
            if current['current_price_usd'] > 0:
                min_price = min(min_price, current['current_price_usd'])
        
        max_multiple = ath / alert_price if alert_price > 0 else 0
        
        updated_calls.append({
            'ca': ca,
            'symbol': symbol,
            'alert_time_utc': alert_time,
            'alert_price_usd': alert_price,
            'current_price_usd': current['current_price_usd'],
            'current_mc_usd': current['current_mc_usd'],
            'current_liq_usd': current['current_liq_usd'],
            'ath_since_alert': ath,
            'min_since_alert': min_price,
            'max_multiple': max_multiple,
            'status': current['status'],
            'last_updated_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        })
    
    print(f"Saving {len(updated_calls)} calls...")
    save_calls(updated_calls)
    print("Data refresh complete!")


def generate_dashboard():
    """Generate dashboard: charts, HTML, and README."""
    if not CALLS_CSV.exists() or not ALERTS_CSV.exists():
        print("Missing data files, skipping dashboard generation")
        return
    
    print("Generating dashboard...")
    
    # Load data
    alerts_df = pd.read_csv(ALERTS_CSV)
    calls_df = pd.read_csv(CALLS_CSV)
    
    # Merge
    df = alerts_df.merge(calls_df, on=['ca'], suffixes=('_alert', '_call'))
    
    # Separate missed calls from real alerts
    df_real = df[df['alert_type'] != 'missed'].copy()
    df_missed = df[df['alert_type'] == 'missed'].copy()
    
    # Identify EXC (exception) calls in real alerts
    df_real['is_exc'] = df_real['reason'].str.startswith('EXC narr', na=False)
    
    # Calculate all 4 strategies for each REAL alert
    results_a = []  # Sell ALL at 2x
    results_b = []  # Sell ALL at +50%
    results_c = []  # Sell HALF at 2x
    results_d = []  # Hold 24h
    
    for _, row in df_real.iterrows():
        # Strategy A: Sell ALL at 2x
        strat_a = calculate_strategy_a(
            row['price_usd'],
            row['ath_since_alert'],
            row['min_since_alert'],
            row['current_price_usd'],
            row['status']
        )
        results_a.append({
            'ca': row['ca'],
            'symbol': row['symbol_alert'],
            'alert_type': row['alert_type'],
            'alert_time_utc': row['alert_time_utc_alert'],
            'max_multiple': row['max_multiple'],
            **strat_a
        })
        
        # Strategy B: Sell ALL at +50%
        strat_b = calculate_strategy_b(
            row['price_usd'],
            row['ath_since_alert'],
            row['min_since_alert'],
            row['current_price_usd'],
            row['status']
        )
        results_b.append({
            'ca': row['ca'],
            'symbol': row['symbol_alert'],
            'alert_type': row['alert_type'],
            'alert_time_utc': row['alert_time_utc_alert'],
            'max_multiple': row['max_multiple'],
            **strat_b
        })
        
        # Strategy C: Sell HALF at 2x
        strat_c = calculate_strategy_c(
            row['price_usd'],
            row['ath_since_alert'],
            row['min_since_alert'],
            row['current_price_usd'],
            row['status']
        )
        results_c.append({
            'ca': row['ca'],
            'symbol': row['symbol_alert'],
            'alert_type': row['alert_type'],
            'alert_time_utc': row['alert_time_utc_alert'],
            'max_multiple': row['max_multiple'],
            **strat_c
        })
        
        # Strategy D: Hold 24h
        strat_d = calculate_strategy_d(
            row['price_usd'],
            row['current_price_usd'],
            row['status'],
            row['alert_time_utc_alert']
        )
        results_d.append({
            'ca': row['ca'],
            'symbol': row['symbol_alert'],
            'alert_type': row['alert_type'],
            'alert_time_utc': row['alert_time_utc_alert'],
            'max_multiple': row['max_multiple'],
            **strat_d
        })
    
    results_a_df = pd.DataFrame(results_a)
    results_b_df = pd.DataFrame(results_b)
    results_c_df = pd.DataFrame(results_c)
    results_d_df = pd.DataFrame(results_d)
    
    # Calculate Strategy A for MISSED calls (for comparison only)
    missed_results_a = []
    for _, row in df_missed.iterrows():
        strat_a = calculate_strategy_a(
            row['price_usd'],
            row['ath_since_alert'],
            row['min_since_alert'],
            row['current_price_usd'],
            row['status']
        )
        # Extract skip reason (first part of reason field)
        reason_parts = row['reason'].split(',', 1)
        skip_reason = reason_parts[0].replace('skip ', '').strip() if reason_parts else 'unknown'
        
        missed_results_a.append({
            'ca': row['ca'],
            'symbol': row['symbol_alert'],
            'alert_time_utc': row['alert_time_utc_alert'],
            'max_multiple': row['max_multiple'],
            'skip_reason': skip_reason,
            **strat_a
        })
    
    missed_a_df = pd.DataFrame(missed_results_a) if missed_results_a else pd.DataFrame()
    
    # Create dashboard directory
    DASHBOARD_DIR.mkdir(exist_ok=True)
    
    # Generate charts
    generate_charts(results_a_df, results_b_df, results_c_df, results_d_df)
    
    # Generate README
    generate_readme(results_a_df, results_b_df, results_c_df, results_d_df, df_real, df_missed, missed_a_df)
    
    # Generate HTML dashboard
    generate_html(results_a_df, results_b_df, results_c_df, results_d_df, df_real, df_missed, missed_a_df)
    
    print("Dashboard generation complete!")


def generate_charts(results_a_df: pd.DataFrame, results_b_df: pd.DataFrame, results_c_df: pd.DataFrame, results_d_df: pd.DataFrame):
    """Generate matplotlib charts."""
    
    # Chart 1: Cumulative P&L comparison for all strategies
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort all by time
    a_sorted = results_a_df.sort_values('alert_time_utc')
    b_sorted = results_b_df.sort_values('alert_time_utc')
    c_sorted = results_c_df.sort_values('alert_time_utc')
    d_sorted = results_d_df.sort_values('alert_time_utc')
    
    ax.plot(range(len(a_sorted)), a_sorted['total_pnl'].cumsum(), marker='o', label='A: Sell ALL at 2x', linewidth=2)
    ax.plot(range(len(b_sorted)), b_sorted['total_pnl'].cumsum(), marker='s', label='B: Sell ALL at +50%', linewidth=2)
    ax.plot(range(len(c_sorted)), c_sorted['total_pnl'].cumsum(), marker='^', label='C: Sell HALF at 2x', linewidth=2)
    ax.plot(range(len(d_sorted)), d_sorted['pnl'].cumsum(), marker='d', label='D: Hold 24h', linewidth=2)
    
    ax.set_xlabel('Alert Number')
    ax.set_ylabel('Cumulative P&L (USD)')
    ax.set_title('Strategy Comparison: Cumulative P&L')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / 'cumulative_pnl_comparison.png', dpi=100)
    plt.close()
    
    # Chart 2: Max Multiple Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_multiples = results_a_df['max_multiple']
    ax.hist(all_multiples, bins=20, alpha=0.7, color='purple', edgecolor='black')
    
    ax.set_xlabel('Max Multiple')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Max Multiples Since Alert')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / 'max_multiple_dist.png', dpi=100)
    plt.close()
    
    # Chart 3: Strategy Total P&L Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['A: Sell ALL\nat 2x', 'B: Sell ALL\nat +50%', 'C: Sell HALF\nat 2x', 'D: Hold\n24h']
    totals = [
        results_a_df['total_pnl'].sum(),
        results_b_df['total_pnl'].sum(),
        results_c_df['total_pnl'].sum(),
        results_d_df['pnl'].sum()
    ]
    colors = ['green' if x > 0 else 'red' for x in totals]
    
    bars = ax.bar(strategies, totals, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${total:.2f}',
                ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    ax.set_ylabel('Total P&L (USD)')
    ax.set_title('Strategy Comparison: Total P&L')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / 'strategy_comparison.png', dpi=100)
    plt.close()


def format_zurich_time(utc_str: str) -> str:
    """Convert UTC timestamp to Europe/Zurich time (CET/CEST with DST)."""
    try:
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            # Fallback for Python < 3.9
            from backports.zoneinfo import ZoneInfo
        
        dt = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
        zurich_tz = ZoneInfo('Europe/Zurich')
        zurich_time = dt.astimezone(zurich_tz)
        # Determine if DST is active
        tz_name = 'CEST' if zurich_time.dst() else 'CET'
        return zurich_time.strftime(f'%Y-%m-%d %H:%M {tz_name}')
    except:
        return utc_str


def generate_readme(results_a_df: pd.DataFrame, results_b_df: pd.DataFrame, results_c_df: pd.DataFrame, results_d_df: pd.DataFrame, full_df: pd.DataFrame, df_missed: pd.DataFrame, missed_a_df: pd.DataFrame):
    """Generate README.md with summary tables."""
    
    # Calculate overall stats for all strategies (REAL alerts only)
    total_alerts = len(results_a_df)
    
    strategies = {
        'A: Sell ALL at 2x': {
            'df': results_a_df,
            'pnl_col': 'total_pnl',
            'desc': 'Sell all at 2x, -50% stop'
        },
        'B: Sell ALL at +50%': {
            'df': results_b_df,
            'pnl_col': 'total_pnl',
            'desc': 'Sell all at 1.5x, -50% stop'
        },
        'C: Sell HALF at 2x': {
            'df': results_c_df,
            'pnl_col': 'total_pnl',
            'desc': 'Sell half at 2x, hold rest, -50% stop'
        },
        'D: Hold 24h': {
            'df': results_d_df,
            'pnl_col': 'pnl',
            'desc': 'Hold 24h then sell'
        }
    }
    
    # Find best strategy
    best_strategy = None
    best_pnl = float('-inf')
    for name, data in strategies.items():
        total_pnl = data['df'][data['pnl_col']].sum()
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_strategy = name
    
    readme = f"""# PAPER TRADING - NO REAL MONEY

## Memecoin Paper-Trading Dashboard

**📊 [View Live Dashboard](https://andreolf.github.io/memecoin-paper-tracker/)**

This dashboard tracks theoretical P&L from Solana memecoin alerts. **No actual trades are executed.**

**Last Updated:** {format_zurich_time(datetime.now(timezone.utc).isoformat())}

**Tracked:** {total_alerts} real alerts{f" + {len(df_missed)} missed (filtered out)" if len(df_missed) > 0 else ""}

---

## 📊 Strategy Comparison

All strategies use:
- **$10 stake per alert**
- **0.45% trading fee + 2% slippage per leg**
- **-50% stop-loss** (where applicable)
- **Conservative ordering**: If both stop and target hit, assume stop hit first

**🏆 Current Best Performer: {best_strategy}** (Total P&L: ${best_pnl:.2f})

### Overall Performance

| Strategy | Description | Total Alerts | Total P&L | Win Rate | Avg P&L | Best Trade | Worst Trade |
|----------|-------------|--------------|-----------|----------|---------|------------|-------------|
"""
    
    for name, data in strategies.items():
        df = data['df']
        pnl_col = data['pnl_col']
        desc = data['desc']
        
        total_pnl = df[pnl_col].sum()
        win_rate = (df[pnl_col] > 0).mean() * 100
        avg_pnl = df[pnl_col].mean()
        best_trade = df[pnl_col].max()
        worst_trade = df[pnl_col].min()
        
        star = "⭐ " if name == best_strategy else ""
        readme += f"| {star}{name} | {desc} | {len(df)} | **${total_pnl:.2f}** | {win_rate:.1f}% | ${avg_pnl:.2f} | ${best_trade:.2f} | ${worst_trade:.2f} |\n"
    
    readme += f"""
---

## 📈 Performance by Alert Type

### Strategy A: Sell ALL at 2x

| Type | Count | Total P&L | Realized | Unrealized | Win Rate | Avg Max Multiple |
|------|-------|-----------|----------|------------|----------|------------------|
"""
    
    for alert_type in results_a_df['alert_type'].unique():
        subset = results_a_df[results_a_df['alert_type'] == alert_type]
        readme += f"| {alert_type} | {len(subset)} | ${subset['total_pnl'].sum():.2f} | ${subset['realized_pnl'].sum():.2f} | ${subset['unrealized_pnl'].sum():.2f} | {(subset['total_pnl'] > 0).mean() * 100:.1f}% | {subset['max_multiple'].mean():.2f}x |\n"
    
    readme += f"""
### Strategy B: Sell ALL at +50%

| Type | Count | Total P&L | Realized | Unrealized | Win Rate | Avg Max Multiple |
|------|-------|-----------|----------|------------|----------|------------------|
"""
    
    for alert_type in results_b_df['alert_type'].unique():
        subset = results_b_df[results_b_df['alert_type'] == alert_type]
        readme += f"| {alert_type} | {len(subset)} | ${subset['total_pnl'].sum():.2f} | ${subset['realized_pnl'].sum():.2f} | ${subset['unrealized_pnl'].sum():.2f} | {(subset['total_pnl'] > 0).mean() * 100:.1f}% | {subset['max_multiple'].mean():.2f}x |\n"
    
    readme += f"""
### Strategy C: Sell HALF at 2x

| Type | Count | Total P&L | Realized | Unrealized | Win Rate | Avg Max Multiple |
|------|-------|-----------|----------|------------|----------|------------------|
"""
    
    for alert_type in results_c_df['alert_type'].unique():
        subset = results_c_df[results_c_df['alert_type'] == alert_type]
        readme += f"| {alert_type} | {len(subset)} | ${subset['total_pnl'].sum():.2f} | ${subset['realized_pnl'].sum():.2f} | ${subset['unrealized_pnl'].sum():.2f} | {(subset['total_pnl'] > 0).mean() * 100:.1f}% | {subset['max_multiple'].mean():.2f}x |\n"
    
    readme += f"""
### Strategy D: Hold 24h

| Type | Count | Total P&L | Win Rate | Avg Max Multiple |
|------|-------|-----------|----------|------------------|
"""
    
    for alert_type in results_d_df['alert_type'].unique():
        subset = results_d_df[results_d_df['alert_type'] == alert_type]
        readme += f"| {alert_type} | {len(subset)} | ${subset['pnl'].sum():.2f} | {(subset['pnl'] > 0).mean() * 100:.1f}% | {subset['max_multiple'].mean():.2f}x |\n"
    
    readme += """
---

## 📉 Charts

### Cumulative P&L Comparison
![Cumulative P&L](dashboard/cumulative_pnl_comparison.png)

### Max Multiple Distribution
![Max Multiple Distribution](dashboard/max_multiple_dist.png)

### Strategy Total P&L
![Strategy Comparison](dashboard/strategy_comparison.png)

---

## 🔗 Interactive Dashboard

View the full interactive dashboard: [dashboard/index.html](dashboard/index.html)

Or visit: **https://andreolf.github.io/memecoin-paper-tracker/**

---

## ⚠️ Disclaimer

This is a **paper-trading tracker only**. No real trades are executed. All P&L figures are theoretical and based on simulated strategies with assumed execution at specific price levels plus 2% slippage. Actual trading results would differ significantly due to additional slippage, timing, liquidity constraints, MEV, and market conditions.

**Assumptions:**
- 0.45% trading fee per leg
- 2% slippage per leg
- Perfect execution at target/stop prices (unrealistic)
- No MEV or sandwich attacks
- Conservative ordering when order unknown

---

*Auto-generated by scripts/refresh.py*
"""
    
    with open(BASE_DIR / 'README.md', 'w') as f:
        f.write(readme)


"""
New elegant HTML generation for the dashboard
"""

def format_money(value: float) -> str:
    """Format money with proper +/− signs."""
    if value > 0:
        return f"+${value:.2f}"
    elif value < 0:
        return f"−${abs(value):.2f}"
    else:
        return f"${value:.2f}"


def generate_html(results_a_df, results_b_df, results_c_df, results_d_df, full_df, df_missed, missed_a_df):
    """Generate elegant standalone HTML dashboard with modern design."""
    
    # Calculate KPIs for Strategy A (primary strategy) - REAL alerts only
    total_calls = len(results_a_df)
    total_pnl_a = results_a_df['total_pnl'].sum()
    win_rate_a = (results_a_df['total_pnl'] > 0).mean() * 100 if total_calls > 0 else 0
    roi_a = (total_pnl_a / (total_calls * 10) * 100) if total_calls > 0 else 0
    
    # Best call: find the one with highest max_multiple
    if total_calls > 0:
        best_idx = results_a_df['max_multiple'].idxmax()
        best_call_symbol = results_a_df.loc[best_idx, 'symbol']
        best_call_multiple = results_a_df.loc[best_idx, 'max_multiple']
    else:
        best_call_symbol = "N/A"
        best_call_multiple = 0
    
    # Active count: only truly open positions (not hit stop, not hit target, not dead)
    active_count = 0
    for _, row in results_a_df.iterrows():
        if not row['hit_stop'] and not row['hit_target'] and full_df[full_df['ca'] == row['ca']]['status'].iloc[0] == 'active':
            active_count += 1
    
    # Calculate EXC calls subtotal
    exc_mask = full_df['is_exc'] if 'is_exc' in full_df.columns else pd.Series([False] * len(full_df))
    exc_count = exc_mask.sum() if len(full_df) > 0 else 0
    exc_pnl = results_a_df[exc_mask]['total_pnl'].sum() if exc_count > 0 and len(results_a_df) > 0 else 0
    
    # Calculate missed calls summary
    missed_count = len(df_missed)
    missed_total_pnl = missed_a_df['total_pnl'].sum() if len(missed_a_df) > 0 else 0
    filters_saved = -missed_total_pnl  # Positive if we avoided losses
    
    # Pre-compute class names and plurals for Python 3.11 compatibility (avoid nested f-string quotes)
    exc_pnl_class = "positive" if exc_pnl > 0 else "negative" if exc_pnl < 0 else "neutral"
    exc_plural = "s" if exc_count != 1 else ""
    filters_saved_class = "positive" if filters_saved > 0 else "negative" if filters_saved < 0 else "neutral"
    filters_saved_label = "Saved" if filters_saved > 0 else "Missed gains" if filters_saved < 0 else "Neutral"
    missed_plural = "s" if missed_count != 1 else ""
    
    # Prepare chart data (cumulative P&L for Strategy A)
    sorted_a = results_a_df.sort_values('alert_time_utc')
    cumulative_pnl = sorted_a['total_pnl'].cumsum().tolist() if total_calls > 0 else []
    # Prepend 0 to start from origin
    cumulative_pnl_with_origin = [0] + cumulative_pnl if cumulative_pnl else [0]
    chart_labels = ["Start"] + [f"Call {i+1}" for i in range(len(cumulative_pnl))] if cumulative_pnl else ["Start"]
    
    # Per-call P&L bars
    per_call_pnl = sorted_a['total_pnl'].tolist() if total_calls > 0 else []
    per_call_labels = [f"Call {i+1}" for i in range(len(per_call_pnl))] if per_call_pnl else []
    
    # Build calls table rows
    calls_rows = []
    for _, row in full_df.iterrows():
        ca = row['ca']
        ca_short = f"{ca[:4]}...{ca[-4:]}"
        
        # Check if this is an EXC call
        is_exc_call = row.get('is_exc', False)
        exc_badge = '<span class="exc-badge" title="Exception call">EXC</span>' if is_exc_call else ''
        
        # Find corresponding Strategy A result
        a_row = results_a_df[results_a_df['ca'] == ca].iloc[0] if len(results_a_df[results_a_df['ca'] == ca]) > 0 else None
        
        # Determine status pill based on strategy outcome
        if a_row is not None:
            if a_row['hit_stop']:
                status_pill = '<span class="status-pill status-stopped">Stopped −50%</span>'
            elif a_row['hit_target']:
                status_pill = '<span class="status-pill status-hit-target">Hit 2x</span>'
            elif row['status'] == 'dead':
                status_pill = '<span class="status-pill status-dead">Dead</span>'
            else:
                status_pill = '<span class="status-pill status-active">Active</span>'
        else:
            # Fallback if no strategy A data
            if row['status'] == 'dead':
                status_pill = '<span class="status-pill status-dead">Dead</span>'
            else:
                status_pill = '<span class="status-pill status-active">Active</span>'
        
        pnl_a = a_row['total_pnl'] if a_row is not None else 0
        pnl_class = 'positive' if pnl_a > 0 else 'negative' if pnl_a < 0 else 'neutral'
        
        alert_mc_formatted = f"${row['mc_usd']:,.0f}" if 'mc_usd' in row else "N/A"
        current_mc_formatted = f"${row['current_mc_usd']:,.0f}"
        
        calls_rows.append(f"""
        <tr>
            <td>
                <div class="token-cell">
                    <strong>{row['symbol_alert'] if 'symbol_alert' in row else row.get('symbol', 'N/A')}</strong> {exc_badge}
                    <div class="ca-container">
                        <code class="ca-code" id="ca-{ca[:8]}">{ca_short}</code>
                        <button class="copy-btn" onclick="copyCA('{ca}', 'ca-{ca[:8]}')" title="Copy full CA">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                        </button>
                        <a href="https://dexscreener.com/solana/{ca}" target="_blank" class="external-link" title="DexScreener">DEX</a>
                        <a href="https://pump.fun/{ca}" target="_blank" class="external-link" title="pump.fun">PF</a>
                        <a href="https://gmgn.ai/sol/token/{ca}" target="_blank" class="external-link" title="GMGN">GM</a>
                    </div>
                </div>
            </td>
            <td>{format_zurich_time(row['alert_time_utc_alert'] if 'alert_time_utc_alert' in row else row['alert_time_utc'])}</td>
            <td class="number">{alert_mc_formatted}</td>
            <td class="number">{current_mc_formatted}</td>
            <td class="number">{row['max_multiple']:.2f}x</td>
            <td>{status_pill}</td>
            <td class="number pnl-{pnl_class}">{format_money(pnl_a)}</td>
        </tr>
        """)
    
    calls_table_html = ''.join(calls_rows) if calls_rows else '<tr><td colspan="7" class="empty-state">No calls yet. Data will appear as alerts are tracked.</td></tr>'
    
    # Strategy comparison data with bars
    strategies_comparison = [
        {'name': 'A: Sell ALL at 2x', 'pnl': results_a_df['total_pnl'].sum(), 'active': True},
        {'name': 'B: Sell ALL at +50%', 'pnl': results_b_df['total_pnl'].sum(), 'active': False},
        {'name': 'C: Sell HALF at 2x', 'pnl': results_c_df['total_pnl'].sum(), 'active': False},
        {'name': 'D: Hold 24h', 'pnl': results_d_df['pnl'].sum(), 'active': False},
    ]
    
    # Calculate max absolute value for bar scaling
    max_abs_pnl = max([abs(s['pnl']) for s in strategies_comparison]) if strategies_comparison else 1
    if max_abs_pnl == 0:
        max_abs_pnl = 1  # Avoid division by zero
    
    strategy_rows = []
    for s in strategies_comparison:
        active_badge = '<span class="active-badge">Active Rule</span>' if s['active'] else ''
        pnl_class = 'positive' if s['pnl'] > 0 else 'negative' if s['pnl'] < 0 else 'neutral'
        
        # Calculate bar width (percentage)
        bar_width = abs(s['pnl']) / max_abs_pnl * 100
        bar_color = '#10b981' if s['pnl'] > 0 else '#ef4444' if s['pnl'] < 0 else '#71717a'
        
        strategy_rows.append(f"""
        <tr class="{'strategy-active' if s['active'] else ''}">
            <td>
                {s['name']} {active_badge}
                <div class="strategy-bar-container">
                    <div class="strategy-bar" style="width: {bar_width}%; background: {bar_color};"></div>
                </div>
            </td>
            <td class="number pnl-{pnl_class}">{format_money(s['pnl'])}</td>
        </tr>
        """)
    
    strategy_table_html = ''.join(strategy_rows)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memecoin Paper Trading Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0b0f;
            --bg-secondary: #14151a;
            --bg-tertiary: #1c1d24;
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --border: #27272a;
            --accent: #6366f1;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }}
        
        @media (prefers-color-scheme: light) {{
            :root {{
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-tertiary: #f3f4f6;
                --text-primary: #18181b;
                --text-secondary: #52525b;
                --text-muted: #71717a;
                --border: #e4e4e7;
            }}
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header */
        header {{
            margin-bottom: 40px;
        }}
        
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 16px;
        }}
        
        h1 {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        
        .github-link {{
            color: var(--text-secondary);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            transition: color 0.2s;
        }}
        
        .github-link:hover {{
            color: var(--text-primary);
        }}
        
        .header-meta {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
        }}
        
        .badge-warning {{
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }}
        
        .badge-info {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }}
        
        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        
        .kpi-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        
        .kpi-label {{
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        
        .kpi-value {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        
        .kpi-value.positive {{
            color: var(--success);
        }}
        
        .kpi-value.negative {{
            color: var(--danger);
        }}
        
        .kpi-value.neutral {{
            color: var(--text-primary);
        }}
        
        .kpi-sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        
        /* Section */
        .section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            letter-spacing: -0.01em;
        }}
        
        /* Strategy Comparison Table */
        .strategy-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .strategy-table tr {{
            border-bottom: 1px solid var(--border);
        }}
        
        .strategy-table tr:last-child {{
            border-bottom: none;
        }}
        
        .strategy-table td {{
            padding: 16px 0;
            font-size: 14px;
        }}
        
        .strategy-table tr.strategy-active {{
            background: rgba(99, 102, 241, 0.05);
        }}
        
        .strategy-table tr.strategy-active td:first-child {{
            font-weight: 600;
        }}
        
        .active-badge {{
            display: inline-block;
            padding: 2px 8px;
            background: rgba(99, 102, 241, 0.1);
            color: var(--accent);
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }}
        
        /* Calls Table */
        .calls-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        
        .calls-table thead {{
            background: var(--bg-tertiary);
        }}
        
        .calls-table th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }}
        
        .calls-table td {{
            padding: 16px 12px;
            border-bottom: 1px solid var(--border);
        }}
        
        .calls-table tbody tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        .calls-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        
        .token-cell {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        
        .ca-container {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .ca-code {{
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
            font-size: 12px;
            color: var(--text-muted);
            padding: 2px 6px;
            background: var(--bg-tertiary);
            border-radius: 4px;
        }}
        
        .copy-btn {{
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px;
            display: flex;
            align-items: center;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        
        .copy-btn:hover {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        
        .copy-btn.copied {{
            color: var(--success);
        }}
        
        .external-link {{
            color: var(--accent);
            text-decoration: none;
            font-size: 11px;
            font-weight: 500;
            transition: opacity 0.2s;
        }}
        
        .external-link:hover {{
            opacity: 0.7;
        }}
        
        .status-pill {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .status-active {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
        }}
        
        .status-hit-target {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            font-weight: 600;
        }}
        
        .status-stopped {{
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
        }}
        
        .status-dead {{
            background: rgba(113, 113, 122, 0.1);
            color: var(--text-muted);
        }}
        
        .strategy-bar-container {{
            margin-top: 6px;
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            overflow: hidden;
        }}
        
        .strategy-bar {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }}
        
        .number {{
            font-variant-numeric: tabular-nums;
        }}
        
        .pnl-positive {{
            color: var(--success);
            font-weight: 600;
        }}
        
        .pnl-negative {{
            color: var(--danger);
            font-weight: 600;
        }}
        
        .pnl-neutral {{
            color: var(--text-secondary);
        }}
        
        .exc-badge {{
            display: inline-block;
            padding: 2px 6px;
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning);
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            margin-left: 6px;
            vertical-align: middle;
        }}
        
        .empty-state {{
            text-align: center;
            color: var(--text-muted);
            padding: 40px 20px !important;
        }}
        
        /* Charts */
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        /* Footer */
        footer {{
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
        }}
        
        /* Mobile */
        @media (max-width: 768px) {{
            body {{
                padding: 16px;
            }}
            
            h1 {{
                font-size: 24px;
            }}
            
            .header-top {{
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .kpi-grid {{
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            }}
            
            .kpi-value {{
                font-size: 24px;
            }}
            
            .section {{
                padding: 16px;
            }}
            
            .calls-table {{
                font-size: 12px;
            }}
            
            .calls-table th,
            .calls-table td {{
                padding: 8px;
            }}
            
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-top">
                <h1>📊 Memecoin Paper Trading</h1>
                <a href="https://github.com/andreolf/memecoin-paper-tracker" target="_blank" class="github-link">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    View on GitHub
                </a>
            </div>
            <div class="header-meta">
                <span class="badge badge-warning">📝 PAPER TRADING — no real money</span>
                <span class="badge badge-info">Last updated: {format_zurich_time(datetime.now(timezone.utc).isoformat())}</span>
            </div>
        </header>
        
        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Calls</div>
                <div class="kpi-value neutral">{total_calls}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Win Rate</div>
                <div class="kpi-value {'positive' if win_rate_a > 50 else 'negative' if win_rate_a < 50 else 'neutral'}">{win_rate_a:.1f}%</div>
                <div class="kpi-sub">Strategy A</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total P&L</div>
                <div class="kpi-value {'positive' if total_pnl_a > 0 else 'negative' if total_pnl_a < 0 else 'neutral'}">{format_money(total_pnl_a)}</div>
                <div class="kpi-sub">ROI: {format_money(roi_a) if roi_a != 0 else '0.0'}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Best Call</div>
                <div class="kpi-value neutral" style="font-size: 1.5em;">{best_call_symbol}</div>
                <div class="kpi-sub">{best_call_multiple:.2f}x multiple</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Active</div>
                <div class="kpi-value neutral">{active_count}</div>
                <div class="kpi-sub">Open positions</div>
            </div>
        </div>
        
        <!-- Strategy Comparison -->
        <div class="section">
            <h2 class="section-title">Strategy Comparison</h2>
            <table class="strategy-table">
                <tbody>
                    {strategy_table_html}
                </tbody>
            </table>
            <p style="margin-top: 16px; font-size: 13px; color: var(--text-muted);">
                All strategies: $10 stake, 0.45% fee + 2% slippage per leg, -50% stop (where applicable)
            </p>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="section">
                <h2 class="section-title">Cumulative P&L</h2>
                <div class="chart-container">
                    <canvas id="cumulativeChart"></canvas>
                </div>
            </div>
            <div class="section">
                <h2 class="section-title">Per-Call P&L</h2>
                <div class="chart-container">
                    <canvas id="perCallChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Calls Table -->
        <div class="section">
            <h2 class="section-title">All Calls</h2>
            <div style="overflow-x: auto;">
                <table class="calls-table">
                    <thead>
                        <tr>
                            <th>Token</th>
                            <th>Alert Time</th>
                            <th>Alert MC</th>
                            <th>Current MC</th>
                            <th>Max Multiple</th>
                            <th>Status</th>
                            <th>P&L (A)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {calls_table_html}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- EXC Calls Summary (if any) -->
        {f'''
        <div class="section">
            <h2 class="section-title">Exception Calls (EXC)</h2>
            <p style="margin-bottom: 16px; color: var(--text-secondary); font-size: 14px;">
                Strong-narrative exception calls taken despite edge-case signals.
            </p>
            <div class="kpi-card">
                <div class="kpi-label">EXC Calls Subtotal</div>
                <div class="kpi-value {exc_pnl_class}">{format_money(exc_pnl)}</div>
                <div class="kpi-sub">{exc_count} call{exc_plural}</div>
            </div>
        </div>
        ''' if exc_count > 0 else ''}
        
        <!-- Missed Calls (Filtered Out) -->
        {f'''
        <div class="section">
            <h2 class="section-title">Missed Calls (Filtered Out)</h2>
            <p style="margin-bottom: 16px; color: var(--text-secondary); font-size: 14px;">
                Coins deliberately skipped by scanner filters. Shows what they would have returned under Strategy A.
            </p>
            
            <div class="kpi-card" style="margin-bottom: 20px;">
                <div class="kpi-label">Filters Impact</div>
                <div class="kpi-value {filters_saved_class}">
                    {format_money(filters_saved)}
                </div>
                <div class="kpi-sub">{filters_saved_label} • {missed_count} call{missed_plural}</div>
            </div>
            
            {"""
            <div style="overflow-x: auto;">
                <table class="calls-table">
                    <thead>
                        <tr>
                            <th>Token</th>
                            <th>Skip Reason</th>
                            <th>Max Multiple</th>
                            <th>Would-be P&L</th>
                        </tr>
                    </thead>
                    <tbody>
            """ + "".join([f'''
                        <tr>
                            <td><strong>{row["symbol"]}</strong></td>
                            <td><code style="font-size: 11px;">{row["skip_reason"]}</code></td>
                            <td class="number">{row["max_multiple"]:.2f}x</td>
                            <td class="number pnl-{'positive' if row["total_pnl"] > 0 else 'negative' if row["total_pnl"] < 0 else 'neutral'}">{format_money(row["total_pnl"])}</td>
                        </tr>
            ''' for _, row in missed_a_df.iterrows()]) + """
                    </tbody>
                </table>
            </div>
            """ if len(missed_a_df) > 0 else '<p style="color: var(--text-muted); text-align: center; padding: 20px;">No missed calls yet.</p>'}
        </div>
        ''' if missed_count > 0 else ''}
        
        <footer>
            <p>Paper-trading simulation only • No real trades executed • Data auto-refreshed hourly</p>
            <p style="margin-top: 8px;">Strategy A: Sell ALL at 2x, -50% stop, 0.45% fee + 2% slippage</p>
            {f'<p style="margin-top: 8px; color: var(--text-muted);">{exc_count} EXC calls • {missed_count} missed calls tracked</p>' if exc_count > 0 or missed_count > 0 else ''}
        </footer>
    </div>
    
    <script>
        // Copy CA functionality
        function copyCA(ca, elementId) {{
            navigator.clipboard.writeText(ca).then(() => {{
                const btn = event.target.closest('.copy-btn');
                btn.classList.add('copied');
                setTimeout(() => btn.classList.remove('copied'), 2000);
            }});
        }}
        
        // Chart.js setup
        const chartColors = {{
            primary: getComputedStyle(document.documentElement).getPropertyValue('--accent'),
            success: getComputedStyle(document.documentElement).getPropertyValue('--success'),
            danger: getComputedStyle(document.documentElement).getPropertyValue('--danger'),
            text: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary'),
            grid: getComputedStyle(document.documentElement).getPropertyValue('--border')
        }};
        
        Chart.defaults.color = chartColors.text;
        Chart.defaults.borderColor = chartColors.grid;
        
        // Cumulative P&L Chart
        const cumulativeCtx = document.getElementById('cumulativeChart');
        if (cumulativeCtx) {{
            new Chart(cumulativeCtx, {{
                type: 'line',
                data: {{
                    labels: {chart_labels},
                    datasets: [{{
                        label: 'Cumulative P&L',
                        data: {cumulative_pnl_with_origin},
                        borderColor: chartColors.primary,
                        backgroundColor: chartColors.primary + '20',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointBorderWidth: 2,
                        pointBackgroundColor: chartColors.primary,
                        pointBorderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{
                                color: chartColors.grid
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Per-Call P&L Chart
        const perCallCtx = document.getElementById('perCallChart');
        if (perCallCtx) {{
            new Chart(perCallCtx, {{
                type: 'bar',
                data: {{
                    labels: {per_call_labels},
                    datasets: [{{
                        label: 'P&L',
                        data: {per_call_pnl},
                        backgroundColor: {per_call_pnl}.map(v => v >= 0 ? chartColors.success : chartColors.danger),
                        borderRadius: 6,
                        borderSkipped: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{
                                color: chartColors.grid
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>'''
    
    with open(DASHBOARD_DIR / 'index.html', 'w') as f:
        f.write(html)



if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    DASHBOARD_DIR.mkdir(exist_ok=True)
    
    print("=== Memecoin Paper-Trading Refresh ===\n")
    
    refresh_data()
    generate_dashboard()
    
    print("\n=== Complete ===")
