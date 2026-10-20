"""
Tests for paper-trading strategy calculations and data handling.
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add parent directory to path to import refresh module
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from refresh import calculate_strategy_a, calculate_strategy_b, calculate_strategy_c, calculate_strategy_d, format_zurich_time


class TestStrategyA:
    """Test Strategy A: Sell ALL at 2x with stop-loss."""
    
    def test_hit_2x_target_only(self):
        """Test when 2x target is hit but not stop."""
        result = calculate_strategy_a(
            alert_price=1.0,
            ath=2.5,  # Above 2x
            min_price=0.6,  # Above stop
            current_price=2.2,
            status='active'
        )
        
        assert result['hit_target']
        assert not result['hit_stop']
        
        # Entry: $10 - 2.45% (fee+slippage) = $9.755 in tokens at $1 = 9.755 tokens
        # Sell all at $2: 9.755 * 2 = $19.51
        # Exit cost: 19.51 * 0.0245 = $0.478
        # Net: 19.51 - 0.478 = $19.032
        # P&L: 19.032 - 10 = $9.032
        
        assert abs(result['total_pnl'] - 9.032) < 0.01
        assert result['unrealized_pnl'] == 0  # All sold
    
    def test_hit_stop_loss_only(self):
        """Test when stop-loss is hit but not target."""
        result = calculate_strategy_a(
            alert_price=1.0,
            ath=1.8,  # Below 2x
            min_price=0.4,  # Below stop (0.5)
            current_price=0.45,
            status='active'
        )
        
        assert not result['hit_target']
        assert result['hit_stop']
        
        # Entry: $10 - 2.45% = $9.755 in tokens at $1 = 9.755 tokens
        # Stopped out at $0.5: 9.755 * 0.5 = $4.8775
        # Exit cost: 4.8775 * 0.0245 = $0.1195
        # Net: 4.8775 - 0.1195 = $4.758
        # P&L: 4.758 - 10 = -$5.242
        
        assert abs(result['total_pnl'] - (-5.242)) < 0.01
        assert result['unrealized_pnl'] == 0


class TestStrategyB:
    """Test Strategy B: Sell ALL at +50% with stop-loss."""
    
    def test_hit_target(self):
        """Test when +50% target is hit."""
        result = calculate_strategy_b(
            alert_price=1.0,
            ath=1.6,  # Above 1.5x
            min_price=0.7,  # Above stop
            current_price=1.5,
            status='active'
        )
        
        assert result['hit_target']
        assert not result['hit_stop']
        
        # Entry: $9.755 tokens
        # Sell all at $1.5: 9.755 * 1.5 = $14.6325
        # Exit cost: 14.6325 * 0.0245 = $0.358
        # Net: 14.6325 - 0.358 = $14.275
        # P&L: 14.275 - 10 = $4.275
        
        assert abs(result['total_pnl'] - 4.275) < 0.01


class TestStrategyC:
    """Test Strategy C: Sell HALF at 2x, hold rest with stop-loss."""
    
    def test_hit_2x_target_only(self):
        """Test when 2x target is hit but not stop."""
        result = calculate_strategy_c(
            alert_price=1.0,
            ath=2.5,  # Above 2x
            min_price=0.6,  # Above stop
            current_price=2.2,
            status='active'
        )
        
        assert result['hit_target']
        assert not result['hit_stop']
        
        # Entry: $9.755 tokens at $1 = 9.755 tokens
        # Sell half (4.8775 tokens) at $2 = $9.755
        # Exit cost: 9.755 * 0.0245 = $0.239
        # Realized: 9.755 - 0.239 - 5 = $4.516
        # Remaining 4.8775 tokens at $2.2 = $10.7305
        # Unrealized: 10.7305 - 5 = $5.7305
        # Total: 4.516 + 5.7305 = $10.246
        
        assert abs(result['realized_pnl'] - 4.516) < 0.01
        assert abs(result['unrealized_pnl'] - 5.7305) < 0.01
        assert abs(result['total_pnl'] - 10.246) < 0.01


class TestStrategyD:
    """Test Strategy D: Hold 24h baseline."""
    
    def test_after_24h_sell_with_profit(self):
        """Test selling after 24h with profit."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        alert_time = (now - timedelta(hours=25)).isoformat().replace('+00:00', 'Z')
        
        result = calculate_strategy_d(
            alert_price=1.0,
            current_price=2.0,
            status='active',
            alert_time=alert_time
        )
        
        assert result['exited']
        # Entry: $9.755 tokens at $1
        # Sell at $2: 9.755 * 2 = $19.51
        # Exit cost: 19.51 * 0.0245 = $0.478
        # Net: 19.51 - 0.478 = $19.032
        # P&L: 19.032 - 10 = $9.032
        assert abs(result['pnl'] - 9.032) < 0.01


class TestDataParsing:
    """Test CSV parsing and data handling."""
    
    def test_parse_alerts_csv(self):
        """Test parsing alerts CSV with correct columns."""
        fixtures_path = Path(__file__).parent / 'fixtures' / 'alerts.csv'
        
        with open(fixtures_path, 'r') as f:
            reader = csv.DictReader(f)
            alerts = list(reader)
        
        assert len(alerts) == 3
        
        # Check first alert
        assert alerts[0]['ca'] == 'TEST1111111111111111111111111111111111111'
        assert alerts[0]['symbol'] == 'MOON'
        assert alerts[0]['alert_type'] == 'new'
        assert float(alerts[0]['price_usd']) == 0.00005
        
        # Check all required columns exist
        required_cols = [
            'alert_time_utc', 'ca', 'symbol', 'name', 'alert_type',
            'mc_usd', 'price_usd', 'liq_usd', 'age_min', 'bonding_pct', 'reason'
        ]
        for col in required_cols:
            assert col in alerts[0]


class TestFormatting:
    """Test time formatting."""
    
    def test_format_zurich_time_summer(self):
        """Test UTC to Zurich conversion in summer (CEST, UTC+2)."""
        # July is definitely summer/DST
        utc_time = "2026-07-15T12:00:00Z"
        zurich = format_zurich_time(utc_time)
        
        # CEST is UTC+2, so 12:00 UTC = 14:00 CEST
        assert "14:00 CEST" in zurich
        assert "2026-07-15" in zurich
    
    def test_format_zurich_time_winter(self):
        """Test UTC to Zurich conversion in winter (CET, UTC+1)."""
        # January is definitely winter/no DST
        utc_time = "2026-01-15T12:00:00Z"
        zurich = format_zurich_time(utc_time)
        
        # CET is UTC+1, so 12:00 UTC = 13:00 CET
        assert "13:00 CET" in zurich
        assert "2026-01-15" in zurich


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
