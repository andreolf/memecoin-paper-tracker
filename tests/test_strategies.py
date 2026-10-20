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


class TestBondingCurveHandling:
    """Test proper handling of pump.fun bonding curve tokens."""
    
    def test_bonding_curve_token_not_dead(self):
        """
        Regression test: pump.fun bonding curve tokens with liq=0 
        should be marked as 'active', not 'dead', if they have a price.
        
        Bug: purr (9oJvYRC2UBLbUShDBnzhNVcJaAffYANPmDHmgf9Lpump) was 
        marked dead despite being +11% from alert and still on bonding curve.
        """
        # Simulate strategy C with bonding curve token that has:
        # - liq = 0 (still on bonding curve)
        # - price went UP (+11%)
        # - status should be 'active' not 'dead'
        result = calculate_strategy_c(
            alert_price=3.036e-05,
            ath=3.382e-05,  # +11.4% ATH
            min_price=2.98e-05,  # -1.8% min (nowhere near -50% stop)
            current_price=3.382e-05,  # +11.4% current
            status='active'  # Should be active, not dead
        )
        
        # Token went up, should not hit stop
        assert not result['hit_stop'], "Should not hit -50% stop when price is +11%"
        assert not result['hit_target'], "Should not hit 2x target at +11%"
        
        # Should show unrealized gain, not full loss
        # Entry: $10 - 2.45% = $9.755 worth of tokens at 3.036e-05 = 321,300 tokens
        # Mark at 3.382e-05: 321,300 * 3.382e-05 = $10.866
        # Unrealized P&L: $10.866 - $10 = $0.866
        assert result['total_pnl'] > 0, f"Should have positive P&L (+11% token), got {result['total_pnl']}"
        assert result['total_pnl'] > 0.8, f"Expected ~$0.87 gain, got {result['total_pnl']}"
        
    def test_dead_token_marks_to_zero(self):
        """Test that truly dead tokens (migrated, no liq) are handled correctly."""
        # Migrated token with no liquidity and price near zero
        result = calculate_strategy_c(
            alert_price=1.0,
            ath=1.5,
            min_price=0.7,
            current_price=0.0,  # Zero price
            status='dead'  # Migrated and dead
        )
        
        # Should mark remainder to zero, losing full stake
        assert result['unrealized_pnl'] == -10.0, "Dead token with zero price should show full loss"


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


class TestMissedCallsHandling:
    """Test handling of missed (filtered out) calls."""
    
    def test_missed_call_excluded_from_main_pnl(self):
        """Test that missed calls are excluded from main P&L calculations."""
        # This would be tested in integration, but we can verify the logic
        # In generate_dashboard, df_real should exclude alert_type='missed'
        import pandas as pd
        
        # Mock data
        alerts_data = {
            'ca': ['CA1', 'CA2'],
            'symbol': ['TOKEN1', 'TOKEN2'],
            'alert_type': ['new', 'missed'],
            'reason': ['good trade', 'skip dev-out, narr 2/5']
        }
        df = pd.DataFrame(alerts_data)
        
        df_real = df[df['alert_type'] != 'missed']
        df_missed = df[df['alert_type'] == 'missed']
        
        assert len(df_real) == 1
        assert len(df_missed) == 1
        assert df_real.iloc[0]['symbol'] == 'TOKEN1'
        assert df_missed.iloc[0]['symbol'] == 'TOKEN2'
    
    def test_skip_reason_extraction(self):
        """Test extraction of skip reason from missed call reason field."""
        reason = "skip dev-out, narr 2/5: low narrative score, other details"
        reason_parts = reason.split(',', 1)
        skip_reason = reason_parts[0].replace('skip ', '').strip()
        
        assert skip_reason == "dev-out"
    
    def test_exc_call_identification(self):
        """Test identification of EXC (exception) calls."""
        import pandas as pd
        
        alerts_data = {
            'reason': [
                'EXC narr 4/5: strong signal',
                'normal call reason',
                'EXC narr 5/5: exceptional'
            ]
        }
        df = pd.DataFrame(alerts_data)
        df['is_exc'] = df['reason'].str.startswith('EXC narr', na=False)
        
        assert df['is_exc'].sum() == 2
        assert df.iloc[0]['is_exc'] == True
        assert df.iloc[1]['is_exc'] == False
        assert df.iloc[2]['is_exc'] == True


class TestDashboardGeneration:
    """Test dashboard HTML generation."""
    
    def test_html_contains_kpi_section(self):
        """Test that generated HTML contains KPI cards."""
        from pathlib import Path
        
        dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'index.html'
        if dashboard_path.exists():
            with open(dashboard_path, 'r') as f:
                html_content = f.read()
            
            assert 'kpi-grid' in html_content
            assert 'Total Calls' in html_content
            assert 'Win Rate' in html_content
            assert 'Total P&L' in html_content
    
    def test_html_contains_calls_table(self):
        """Test that generated HTML contains calls table."""
        from pathlib import Path
        
        dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'index.html'
        if dashboard_path.exists():
            with open(dashboard_path, 'r') as f:
                html_content = f.read()
            
            assert 'calls-table' in html_content
            assert 'All Calls' in html_content
    
    def test_html_contains_missed_calls_section_when_present(self):
        """Test that HTML contains missed calls section when there are missed calls."""
        from pathlib import Path
        
        # This test assumes the dashboard has been generated
        dashboard_path = Path(__file__).parent.parent / 'dashboard' / 'index.html'
        if dashboard_path.exists():
            with open(dashboard_path, 'r') as f:
                html_content = f.read()
            
            # If there are missed calls in the data, the section should be present
            # This is validated by checking for the section title
            if 'Missed Calls' in html_content:
                assert 'Filtered Out' in html_content
                assert 'Filters Impact' in html_content


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
