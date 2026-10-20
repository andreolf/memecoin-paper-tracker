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


def fetch_price_pumpfun(ca: str) -> Optional[Tuple[float, float, float]]:
    """Fetch from pump.fun. Returns (price_usd, mc_usd, liq_usd) or None."""
    data = fetch_with_retry(f"{PUMPFUN_BASE}{ca}")
    if not data:
        return None
    
    try:
        price = float(data.get("usd_market_cap", 0)) / float(data.get("total_supply", 1))
        mc = float(data.get("usd_market_cap", 0))
        liq = float(data.get("virtual_sol_reserves", 0)) * 180  # rough SOL price estimate
        return (price, mc, liq)
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
            result["current_price_usd"], result["current_mc_usd"], result["current_liq_usd"] = data
    
    # Determine status
    if result["current_liq_usd"] >= 1000 and result["current_price_usd"] > 0:
        result["status"] = "active"
    else:
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
    
    # Calculate all 4 strategies for each alert
    results_a = []  # Sell ALL at 2x
    results_b = []  # Sell ALL at +50%
    results_c = []  # Sell HALF at 2x
    results_d = []  # Hold 24h
    
    for _, row in df.iterrows():
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
    
    # Create dashboard directory
    DASHBOARD_DIR.mkdir(exist_ok=True)
    
    # Generate charts
    generate_charts(results_a_df, results_b_df, results_c_df, results_d_df)
    
    # Generate README
    generate_readme(results_a_df, results_b_df, results_c_df, results_d_df, df)
    
    # Generate HTML dashboard
    generate_html(results_a_df, results_b_df, results_c_df, results_d_df, df)
    
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


def generate_readme(results_a_df: pd.DataFrame, results_b_df: pd.DataFrame, results_c_df: pd.DataFrame, results_d_df: pd.DataFrame, full_df: pd.DataFrame):
    """Generate README.md with summary tables."""
    
    # Calculate overall stats for all strategies
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


def generate_html(results_a_df: pd.DataFrame, results_b_df: pd.DataFrame, results_c_df: pd.DataFrame, results_d_df: pd.DataFrame, full_df: pd.DataFrame):
    """Generate standalone HTML dashboard."""
    
    # Calculate best strategy
    strategies_data = {
        'A: Sell ALL at 2x': results_a_df['total_pnl'].sum(),
        'B: Sell ALL at +50%': results_b_df['total_pnl'].sum(),
        'C: Sell HALF at 2x': results_c_df['total_pnl'].sum(),
        'D: Hold 24h': results_d_df['pnl'].sum()
    }
    best_strategy = max(strategies_data, key=strategies_data.get)
    best_pnl = strategies_data[best_strategy]
    
    # Build table rows for Strategy C (most detailed)
    rows = []
    for _, row in results_c_df.iterrows():
        ca = row['ca']
        pumpfun_link = f"https://pump.fun/{ca}"
        dex_link = f"https://dexscreener.com/solana/{ca}"
        
        rows.append(f"""
        <tr>
            <td>{format_zurich_time(row['alert_time_utc'])}</td>
            <td>
                <a href="{pumpfun_link}" target="_blank">{row['symbol']}</a> /
                <a href="{dex_link}" target="_blank">DEX</a>
            </td>
            <td>{row['alert_type']}</td>
            <td>{row['max_multiple']:.2f}x</td>
            <td class="{'profit' if row['total_pnl'] > 0 else 'loss'}">${row['total_pnl']:.2f}</td>
            <td>${row['realized_pnl']:.2f}</td>
            <td>${row['unrealized_pnl']:.2f}</td>
            <td>{'✓' if row['hit_target'] else ''}</td>
            <td>{'✓' if row['hit_stop'] else ''}</td>
        </tr>
        """)
    
    table_html = ''.join(rows)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PAPER TRADING Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .warning {{
            background: #ff6b6b;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
            font-weight: bold;
            text-align: center;
        }}
        .best-strategy {{
            background: #51cf66;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-weight: bold;
            text-align: center;
            font-size: 1.2em;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #1a1f3a;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #2d3561;
        }}
        .stat-card.best {{
            border: 2px solid #51cf66;
            box-shadow: 0 0 20px rgba(81, 207, 102, 0.3);
        }}
        .stat-card h3 {{
            color: #8b9dc3;
            font-size: 0.85em;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .stat-card .value.profit {{ color: #51cf66; }}
        .stat-card .value.loss {{ color: #ff6b6b; }}
        .stat-card .details {{
            font-size: 0.85em;
            color: #8b9dc3;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1a1f3a;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        thead {{
            background: #2d3561;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }}
        th:hover {{
            background: #3d4571;
        }}
        tbody tr {{
            border-bottom: 1px solid #2d3561;
        }}
        tbody tr:hover {{
            background: #232947;
        }}
        .profit {{
            color: #51cf66;
        }}
        .loss {{
            color: #ff6b6b;
        }}
        a {{
            color: #667eea;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart {{
            background: #1a1f3a;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #2d3561;
        }}
        .chart img {{
            width: 100%;
            border-radius: 5px;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            color: #8b9dc3;
            font-size: 0.9em;
        }}
        .info-box {{
            background: #1a1f3a;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #2d3561;
            margin: 20px 0;
        }}
        .info-box h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        .info-box ul {{
            margin-left: 20px;
            color: #8b9dc3;
        }}
        .info-box ul li {{
            margin: 5px 0;
        }}
    </style>
    <script>
        function sortTable(n) {{
            const table = document.getElementById("dataTable");
            let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            switching = true;
            dir = "asc";
            while (switching) {{
                switching = false;
                rows = table.rows;
                for (i = 1; i < (rows.length - 1); i++) {{
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[n];
                    y = rows[i + 1].getElementsByTagName("TD")[n];
                    let xContent = x.textContent || x.innerText;
                    let yContent = y.textContent || y.innerText;
                    
                    let xNum = parseFloat(xContent.replace(/[^0-9.-]/g, ''));
                    let yNum = parseFloat(yContent.replace(/[^0-9.-]/g, ''));
                    
                    if (!isNaN(xNum) && !isNaN(yNum)) {{
                        if (dir == "asc") {{
                            if (xNum > yNum) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xNum < yNum) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }} else {{
                        if (dir == "asc") {{
                            if (xContent.toLowerCase() > yContent.toLowerCase()) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xContent.toLowerCase() < yContent.toLowerCase()) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }}
                }}
                if (shouldSwitch) {{
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount++;
                }} else {{
                    if (switchcount == 0 && dir == "asc") {{
                        dir = "desc";
                        switching = true;
                    }}
                }}
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Memecoin Paper-Trading Dashboard</h1>
            <p>Last Updated: {format_zurich_time(datetime.now(timezone.utc).isoformat())}</p>
        </header>
        
        <div class="warning">
            ⚠️ PAPER TRADING - NO REAL MONEY - ALL P&L IS THEORETICAL
        </div>
        
        <div class="best-strategy">
            🏆 Current Best Performer: {best_strategy} (${best_pnl:.2f} Total P&L)
        </div>
        
        <div class="info-box">
            <h3>All Strategies Use:</h3>
            <ul>
                <li>$10 stake per alert</li>
                <li>0.45% trading fee + 2% slippage per leg</li>
                <li>-50% stop-loss (where applicable)</li>
                <li>Conservative ordering: stop before target when uncertain</li>
            </ul>
        </div>
        
        <div class="stats">
            <div class="stat-card {'best' if best_strategy == 'A: Sell ALL at 2x' else ''}">
                <h3>Strategy A: Sell ALL at 2x</h3>
                <div class="value {'profit' if results_a_df['total_pnl'].sum() > 0 else 'loss'}">
                    ${results_a_df['total_pnl'].sum():.2f}
                </div>
                <div class="details">
                    Win Rate: {(results_a_df['total_pnl'] > 0).mean() * 100:.1f}%<br>
                    Avg: ${results_a_df['total_pnl'].mean():.2f}
                </div>
            </div>
            <div class="stat-card {'best' if best_strategy == 'B: Sell ALL at +50%' else ''}">
                <h3>Strategy B: Sell ALL at +50%</h3>
                <div class="value {'profit' if results_b_df['total_pnl'].sum() > 0 else 'loss'}">
                    ${results_b_df['total_pnl'].sum():.2f}
                </div>
                <div class="details">
                    Win Rate: {(results_b_df['total_pnl'] > 0).mean() * 100:.1f}%<br>
                    Avg: ${results_b_df['total_pnl'].mean():.2f}
                </div>
            </div>
            <div class="stat-card {'best' if best_strategy == 'C: Sell HALF at 2x' else ''}">
                <h3>Strategy C: Sell HALF at 2x</h3>
                <div class="value {'profit' if results_c_df['total_pnl'].sum() > 0 else 'loss'}">
                    ${results_c_df['total_pnl'].sum():.2f}
                </div>
                <div class="details">
                    Win Rate: {(results_c_df['total_pnl'] > 0).mean() * 100:.1f}%<br>
                    Avg: ${results_c_df['total_pnl'].mean():.2f}
                </div>
            </div>
            <div class="stat-card {'best' if best_strategy == 'D: Hold 24h' else ''}">
                <h3>Strategy D: Hold 24h</h3>
                <div class="value {'profit' if results_d_df['pnl'].sum() > 0 else 'loss'}">
                    ${results_d_df['pnl'].sum():.2f}
                </div>
                <div class="details">
                    Win Rate: {(results_d_df['pnl'] > 0).mean() * 100:.1f}%<br>
                    Avg: ${results_d_df['pnl'].mean():.2f}
                </div>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart">
                <img src="cumulative_pnl_comparison.png" alt="Cumulative P&L">
            </div>
            <div class="chart">
                <img src="max_multiple_dist.png" alt="Max Multiple Distribution">
            </div>
            <div class="chart">
                <img src="strategy_comparison.png" alt="Strategy Comparison">
            </div>
        </div>
        
        <h2 style="margin-bottom: 20px;">Strategy C: Sell HALF at 2x (Click headers to sort)</h2>
        <table id="dataTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Alert Time (CET/CEST)</th>
                    <th onclick="sortTable(1)">Token</th>
                    <th onclick="sortTable(2)">Type</th>
                    <th onclick="sortTable(3)">Max Multiple</th>
                    <th onclick="sortTable(4)">Total P&L</th>
                    <th onclick="sortTable(5)">Realized</th>
                    <th onclick="sortTable(6)">Unrealized</th>
                    <th onclick="sortTable(7)">Hit 2x</th>
                    <th onclick="sortTable(8)">Hit Stop</th>
                </tr>
            </thead>
            <tbody>
                {table_html}
            </tbody>
        </table>
        
        <footer>
            <p>Auto-generated by scripts/refresh.py | Paper-trading only, no real trades executed</p>
            <p style="margin-top: 10px; font-size: 0.85em;">Includes 2% slippage + 0.45% fees per leg</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(DASHBOARD_DIR / 'index.html', 'w') as f:
        f.write(html)


if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    DASHBOARD_DIR.mkdir(exist_ok=True)
    
    print("=== Memecoin Paper-Trading Refresh ===\n")
    
    refresh_data()
    generate_dashboard()
    
    print("\n=== Complete ===")
