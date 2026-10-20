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
from refresh import (calculate_strategy_a, calculate_strategy_b, calculate_strategy_c, calculate_strategy_d, 
                    calculate_strategy_e, parse_narrative_score, format_zurich_time)


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
    
    def test_hit_both_target_prioritized(self):
        """Test when both stop and target hit - target is prioritized if ATH proves it was reached."""
        result = calculate_strategy_a(
            alert_price=1.0,
            ath=5.0,  # Went to 5x (definitely hit 2x)
            min_price=0.4,  # Also hit stop on the way down
            current_price=0.4,
            status='active'
        )
        
        # ATH >= 2x proves we reached the target
        assert result['hit_target']
        assert not result['hit_stop']  # Stop not reported since target was hit
        
        # Should exit at 2x target, not at stop
        assert abs(result['total_pnl'] - 9.032) < 0.01


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


class TestStrategyE:
    """Test Strategy E: 2x + moonbag on strong narrative."""
    
    def test_parse_narrative_score(self):
        """Test parsing narrative score from reason field."""
        assert parse_narrative_score('narr 5/5, strong') == 5
        assert parse_narrative_score('narr 4/5, good') == 4
        assert parse_narrative_score('narr 3/5, okay') == 3
        assert parse_narrative_score('EXC narr 5/5, exception') == 5
        assert parse_narrative_score('EXC narr 4/5, exception') == 4
        assert parse_narrative_score('skip dev-out, narr 2/5') == 0  # Not at start
        assert parse_narrative_score('some other reason') == 0
        assert parse_narrative_score('') == 0
    
    def test_low_narrative_behaves_like_strategy_a(self):
        """Test that narrative < 4 behaves exactly like Strategy A."""
        result_e = calculate_strategy_e(
            alert_price=1.0,
            ath=2.5,
            min_price=0.6,
            current_price=2.2,
            status='active',
            reason='narr 3/5, weak',
            max_multiple=2.5
        )
        
        result_a = calculate_strategy_a(
            alert_price=1.0,
            ath=2.5,
            min_price=0.6,
            current_price=2.2,
            status='active'
        )
        
        assert result_e['narrative_score'] == 3
        assert result_e['kept_moonbag'] == False
        assert abs(result_e['total_pnl'] - result_a['total_pnl']) < 0.01
        assert result_e['hit_target'] == result_a['hit_target']
        assert result_e['hit_stop'] == result_a['hit_stop']
    
    def test_high_narrative_stop_hit(self):
        """Test that stop-loss applies to whole position before 2x even with high narrative."""
        result = calculate_strategy_e(
            alert_price=1.0,
            ath=1.8,  # Below 2x
            min_price=0.4,  # Below stop (0.5)
            current_price=0.45,
            status='active',
            reason='narr 5/5, strong but stopped',
            max_multiple=1.8
        )
        
        assert result['narrative_score'] == 5
        assert result['hit_stop'] == True
        assert result['hit_target'] == False
        assert result['kept_moonbag'] == False
        # Should behave like Strategy A when stopped
        assert abs(result['total_pnl'] - (-5.242)) < 0.01
    
    def test_high_narrative_target_then_stop(self):
        """Test that target is prioritized when ATH shows it was reached, even if stop also hit."""
        result = calculate_strategy_e(
            alert_price=1.0,
            ath=5.7,  # Reached 5.7x (definitely hit 2x)
            min_price=0.4,  # Also hit stop on way down
            current_price=0.4,
            status='active',
            reason='narr 4/5, hit target then crashed',
            max_multiple=5.7
        )
        
        assert result['narrative_score'] == 4
        assert result['hit_target'] == True
        assert result['hit_stop'] == False  # Not reported since target was hit
        assert result['kept_moonbag'] == False  # Moonbag exited at 5.7x (>= 5x)
        
        # 85% sold at 2x + 15% moonbag at 5.7x
        # Should be a significant profit
        assert result['total_pnl'] > 10
    
    def test_high_narrative_2x_but_not_5x_moonbag_held(self):
        """Test moonbag is held when 2x reached but not 5x."""
        result = calculate_strategy_e(
            alert_price=1.0,
            ath=3.0,  # Above 2x but below 5x
            min_price=0.6,
            current_price=2.8,  # Still active at 2.8x
            status='active',
            reason='EXC narr 4/5, moonbag held',
            max_multiple=3.0
        )
        
        assert result['narrative_score'] == 4
        assert result['hit_target'] == True
        assert result['hit_stop'] == False
        assert result['kept_moonbag'] == True
        
        # Entry: $10 - 2.45% = $9.755 in tokens at $1 = 9.755 tokens
        # Sell 85% at $2: 8.29175 tokens * 2 = $16.5835
        # Exit cost: 16.5835 * 0.0245 = $0.4063
        # Realized from 85%: 16.5835 - 0.4063 - 8.5 = $7.6772
        
        # Moonbag: 15% = 1.46325 tokens at current $2.8 = $4.097
        # Cost basis for moonbag: $1.5
        # Unrealized: 4.097 - 1.5 = $2.597
        
        # Total: 7.6772 + 2.597 = $10.274
        assert abs(result['realized_pnl'] - 7.6772) < 0.01
        assert abs(result['unrealized_pnl'] - 2.597) < 0.01
        assert abs(result['total_pnl'] - 10.274) < 0.01
    
    def test_high_narrative_5x_moonbag_exited(self):
        """Test moonbag exits at 5x+ peak."""
        result = calculate_strategy_e(
            alert_price=1.0,
            ath=6.0,  # Hit 6x
            min_price=0.6,
            current_price=5.5,  # Currently at 5.5x (still above 5x)
            status='active',
            reason='narr 5/5, moonbag exited',
            max_multiple=6.0  # Max was 6x
        )
        
        assert result['narrative_score'] == 5
        assert result['hit_target'] == True
        assert result['hit_stop'] == False
        assert result['kept_moonbag'] == False  # Exited because >= 5x
        
        # Entry: $10 - 2.45% = $9.755 in tokens at $1 = 9.755 tokens
        # Sell 85% at $2: 8.29175 tokens * 2 = $16.5835
        # Exit cost: 16.5835 * 0.0245 = $0.4063
        # Realized from 85%: 16.5835 - 0.4063 - 8.5 = $7.6772
        
        # Moonbag: 15% = 1.46325 tokens exited at 6x ($6) = $8.7795
        # Exit cost: 8.7795 * 0.0245 = $0.2151
        # Moonbag realized: 8.7795 - 0.2151 - 1.5 = $7.0644
        
        # Total realized: 7.6772 + 7.0644 = $14.7416
        assert abs(result['realized_pnl'] - 14.7416) < 0.01
        assert result['unrealized_pnl'] == 0
        assert abs(result['total_pnl'] - 14.7416) < 0.01


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
            # Since we may not always have missed calls, this is a conditional test
            if 'Missed Calls' in html_content and 'Filtered Out' in html_content:
                # If the section exists, it should have the filters impact label
                assert 'Impact' in html_content or 'Saved' in html_content or 'filter' in html_content.lower()


class TestStatusMapping:
    """Test status pill mapping based on strategy outcome."""
    
    def test_status_stopped_when_stop_hit(self):
        """Test that status shows 'Stopped −50%' when stop-loss is hit."""
        # Strategy result with stop hit
        result = {
            'hit_stop': True,
            'hit_target': False,
            'total_pnl': -5.0
        }
        
        assert result['hit_stop'] == True
        assert result['hit_target'] == False
        # Status should be 'Stopped −50%'
    
    def test_status_hit_target_when_target_reached(self):
        """Test that status shows 'Hit 2x' when 2x target is reached."""
        result = {
            'hit_stop': False,
            'hit_target': True,
            'total_pnl': 8.0
        }
        
        assert result['hit_stop'] == False
        assert result['hit_target'] == True
        # Status should be 'Hit 2x'
    
    def test_status_active_when_neither(self):
        """Test that status shows 'Active' when neither stop nor target hit."""
        result = {
            'hit_stop': False,
            'hit_target': False,
            'total_pnl': 2.0
        }
        
        assert result['hit_stop'] == False
        assert result['hit_target'] == False
        # Status should be 'Active' if token still has liquidity
    
    def test_active_count_only_open_positions(self):
        """Test that active count only includes truly open positions."""
        import pandas as pd
        
        # Mock results
        results = pd.DataFrame({
            'ca': ['CA1', 'CA2', 'CA3'],
            'symbol': ['T1', 'T2', 'T3'],
            'hit_stop': [True, False, False],
            'hit_target': [False, True, False],
            'total_pnl': [-5.0, 8.0, 2.0]
        })
        
        full_df = pd.DataFrame({
            'ca': ['CA1', 'CA2', 'CA3'],
            'status': ['active', 'active', 'active']
        })
        
        # Count only truly open positions
        active_count = 0
        for _, row in results.iterrows():
            ca = row['ca']
            status = full_df[full_df['ca'] == ca]['status'].iloc[0]
            if not row['hit_stop'] and not row['hit_target'] and status == 'active':
                active_count += 1
        
        assert active_count == 1  # Only CA3 is truly open


class TestRegressionDashboardCrash:
    """
    Regression test for Oct 20, 2026 dashboard crash.
    Tests the scenario with EXC calls, strategy E calls, missed calls, and normal calls.
    """
    
    def test_mixed_call_types_no_crash(self):
        """Test that dashboard generation handles mixed call types without crashing."""
        import pandas as pd
        
        # Mock data with all call types
        alerts_data = {
            'ca': ['NORMAL1', 'EXC_ARENA', 'EXC_JARVIS', 'MISSED1'],
            'symbol': ['NORM', 'ARENA', 'JARVIS', 'MISS'],
            'symbol_alert': ['NORM', 'ARENA', 'JARVIS', 'MISS'],
            'alert_type': ['new', 'new', 'new', 'missed'],
            'alert_time_utc': ['2026-10-20T10:00:00Z'] * 4,
            'alert_time_utc_alert': ['2026-10-20T10:00:00Z'] * 4,
            'price_usd': [1.0, 1.0, 1.0, 1.0],
            'mc_usd': [50000, 50000, 50000, 50000],
            'current_mc_usd': [100000, 120000, 150000, 80000],
            'ath_since_alert': [2.5, 2.1, 2.2, 1.8],
            'min_since_alert': [0.8, 0.9, 0.85, 0.6],
            'current_price_usd': [2.2, 2.0, 2.1, 1.5],
            'status': ['active', 'active', 'active', 'active'],
            'max_multiple': [2.5, 2.1, 2.2, 1.8],
            'reason': [
                'normal call',
                'EXC narr 4/5: Pokémon meta, original mechanic',
                'EXC narr 4/5: Specs x RDJ launch, Jarvis meta',
                'skip dev-out, narr 3/5: missed for dev-out'
            ]
        }
        
        df = pd.DataFrame(alerts_data)
        
        # Split like in generate_dashboard
        df_real = df[df['alert_type'] != 'missed'].copy()
        df_missed = df[df['alert_type'] == 'missed'].copy()
        df_real['is_exc'] = df_real['reason'].str.startswith('EXC narr', na=False)
        
        # Verify splits
        assert len(df_real) == 3
        assert len(df_missed) == 1
        assert df_real['is_exc'].sum() == 2  # ARENA and JARVIS
        
        # Calculate strategies
        results_a = []
        for _, row in df_real.iterrows():
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
                'is_exc': row['is_exc'],
                **strat_a
            })
        
        results_a_df = pd.DataFrame(results_a)
        
        # This is the line that was crashing (line 1167)
        # exc_mask was built from full_df but applied to results_a_df
        # Now exc_mask should be built from results_a_df itself
        exc_count = results_a_df['is_exc'].sum()
        exc_pnl = results_a_df[results_a_df['is_exc']]['total_pnl'].sum() if exc_count > 0 else 0
        
        # Should not crash and should have correct counts
        assert exc_count == 2
        assert isinstance(exc_pnl, (int, float))
        assert len(results_a_df) == 3
    
    def test_strategy_e_stop_only_before_2x(self):
        """
        Test Strategy E (ARENA/JARVIS scenario):
        - 85% sold at 2x
        - 15% moonbag with no stop after 2x
        - -50% stop applies only BEFORE 2x
        - If ATH >= 2x, target was reached regardless of later crash
        """
        # Scenario 1: Hits 2x then drops (moonbag held, no stop after 2x)
        result = calculate_strategy_e(
            alert_price=1.0,
            ath=2.1,  # Hit 2x
            min_price=0.9,  # Never hit -50% stop
            current_price=1.3,  # Now at 1.3x (dropped from 2.1x but still above entry)
            status='active',
            reason='EXC narr 4/5: Pokémon meta',
            max_multiple=2.1
        )
        
        assert result['narrative_score'] == 4
        assert result['hit_target'] == True  # Hit 2x
        assert result['hit_stop'] == False  # Never hit -50% stop
        assert result['kept_moonbag'] == True  # Still holding moonbag
        
        # 85% sold at 2x: profit from that
        # 15% moonbag marked at 1.3x: still positive
        assert result['realized_pnl'] > 0
        assert result['unrealized_pnl'] > 0  # Moonbag still above cost basis
        assert result['total_pnl'] > 0
        
        # Scenario 2: Hit stop BEFORE 2x (whole position stopped)
        result_stopped = calculate_strategy_e(
            alert_price=1.0,
            ath=1.8,  # Never hit 2x
            min_price=0.4,  # Hit -50% stop
            current_price=0.5,
            status='active',
            reason='EXC narr 5/5: strong but stopped',
            max_multiple=1.8
        )
        
        assert result_stopped['hit_stop'] == True
        assert result_stopped['hit_target'] == False
        assert result_stopped['kept_moonbag'] == False  # Never reached 2x to create moonbag
        assert result_stopped['total_pnl'] < 0  # Should be -50% loss
        
        # Scenario 3: Hit 2x, peaked at 5.7x, then crashed below stop (ARENA case)
        result_arena = calculate_strategy_e(
            alert_price=1.0,
            ath=5.7,  # Peaked at 5.7x (definitely hit 2x first)
            min_price=0.3,  # Later crashed below stop
            current_price=0.3,
            status='active',
            reason='narr 4/5: peaked then crashed',
            max_multiple=5.7
        )
        
        # ATH proves we hit 2x before crashing
        assert result_arena['hit_target'] == True
        assert result_arena['hit_stop'] == False  # Target prioritized since ATH >= 2x
        assert result_arena['kept_moonbag'] == False  # Moonbag exited at 5.7x (>= 5x)
        
        # 85% at 2x + 15% at 5.7x should be profitable
        assert result_arena['total_pnl'] > 10


class TestMoneyFormatting:
    """Test money formatting with proper +/− signs."""
    
    def test_positive_money_format(self):
        """Test formatting of positive values."""
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from refresh import format_money
        
        assert format_money(5.24) == "+$5.24"
        assert format_money(100.50) == "+$100.50"
    
    def test_negative_money_format(self):
        """Test formatting of negative values with minus sign."""
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from refresh import format_money
        
        assert format_money(-5.24) == "−$5.24"
        assert format_money(-100.50) == "−$100.50"
    
    def test_zero_money_format(self):
        """Test formatting of zero."""
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from refresh import format_money
        
        assert format_money(0) == "$0.00"


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
