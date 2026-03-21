from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskEvent:
    triggered: bool
    reason: Optional[str] = None
    metadata: Optional[Dict] = None


class LiveRiskManager:
    """Runtime risk controls: kill switch, circuit breakers, daily loss and trailing stop."""

    def __init__(self, config: Dict):
        self.config = config
        self.kill_switch = False
        self.kill_switch_reason: Optional[str] = None
        self.last_seen_data_ts: Optional[datetime] = None
        self.reject_count = 0
        self.day_start_equity: Optional[float] = None
        self.peak_equity_today: Optional[float] = None
        self.last_reset_date: Optional[date] = None

    def mark_data_heartbeat(self, timestamp: datetime) -> None:
        self.last_seen_data_ts = timestamp

    def record_order_reject(self) -> None:
        self.reject_count += 1

    def snapshot_state(self) -> Dict:
        """Serialize risk manager runtime state for persistence."""
        return {
            'reject_count': int(self.reject_count),
            'kill_switch': bool(self.kill_switch),
            'kill_switch_reason': self.kill_switch_reason,
            'last_reset_date': str(self.last_reset_date) if self.last_reset_date else None,
            'day_start_equity': float(self.day_start_equity) if self.day_start_equity is not None else None,
            'peak_equity_today': float(self.peak_equity_today) if self.peak_equity_today is not None else None,
            'last_seen_data_ts': self.last_seen_data_ts.isoformat() if self.last_seen_data_ts else None,
        }

    def restore_state(self, state: Dict) -> None:
        """Restore risk manager runtime state from a serialized snapshot."""
        if not state:
            return

        self.reject_count = int(state.get('reject_count', 0))
        self.kill_switch = bool(state.get('kill_switch', False))
        self.kill_switch_reason = state.get('kill_switch_reason')

        reset_raw = state.get('last_reset_date')
        self.last_reset_date = date.fromisoformat(reset_raw) if reset_raw else None

        self.day_start_equity = (
            float(state['day_start_equity'])
            if state.get('day_start_equity') is not None
            else None
        )
        self.peak_equity_today = (
            float(state['peak_equity_today'])
            if state.get('peak_equity_today') is not None
            else None
        )

        last_seen_raw = state.get('last_seen_data_ts')
        self.last_seen_data_ts = datetime.fromisoformat(last_seen_raw) if last_seen_raw else None

    def _trigger_kill_switch(self, reason: str, metadata: Optional[Dict] = None) -> RiskEvent:
        self.kill_switch = True
        self.kill_switch_reason = reason
        return RiskEvent(True, reason, metadata)

    def reset_daily(self, equity: float,
                    reset_date: Optional[date] = None,
                    clear_kill_switch: Optional[bool] = None) -> None:
        if clear_kill_switch is None:
            clear_kill_switch = bool(self.config.get('clear_kill_switch_on_reset', True))

        self.day_start_equity = equity
        self.peak_equity_today = equity
        self.reject_count = 0
        self.last_reset_date = reset_date or datetime.utcnow().date()
        if clear_kill_switch and self.kill_switch:
            logger.warning(
                "Clearing existing kill_switch during daily reset (reason=%s)",
                self.kill_switch_reason or 'unknown',
            )
        if clear_kill_switch:
            self.kill_switch = False
            self.kill_switch_reason = None

    def evaluate(self, current_equity: float, now: datetime) -> RiskEvent:
        if self.kill_switch:
            return RiskEvent(
                True,
                self.kill_switch_reason or 'kill_switch_active',
                {'kill_switch': True},
            )

        if self.last_reset_date is not None and now.date() != self.last_reset_date:
            logger.info("New trading day detected in evaluate; resetting daily risk baseline")
            self.reset_daily(current_equity, reset_date=now.date())

        if self.day_start_equity is None:
            logger.warning("day_start_equity missing during evaluate; performing fallback reset")
            self.reset_daily(current_equity, reset_date=now.date())

        self.peak_equity_today = max(self.peak_equity_today or current_equity, current_equity)

        stale_seconds = self.config.get('stale_data_seconds', 300)
        if self.last_seen_data_ts and (now - self.last_seen_data_ts).total_seconds() > stale_seconds:
            return self._trigger_kill_switch(
                'stale_data_breaker',
                {'stale_seconds': stale_seconds},
            )

        max_rejects = self.config.get('max_order_rejects', 3)
        if self.reject_count >= max_rejects:
            return self._trigger_kill_switch(
                'reject_count_breaker',
                {'reject_count': self.reject_count},
            )

        max_daily_loss = self.config.get('max_daily_loss_pct', 0.03)
        if self.day_start_equity and current_equity < self.day_start_equity * (1 - max_daily_loss):
            return self._trigger_kill_switch(
                'max_daily_loss_breaker',
                {'loss_pct': 1 - (current_equity / self.day_start_equity)},
            )

        trailing_stop = self.config.get('trailing_stop_loss_pct', 0.05)
        if self.peak_equity_today and current_equity < self.peak_equity_today * (1 - trailing_stop):
            return self._trigger_kill_switch(
                'trailing_stop_breaker',
                {'drawdown_from_peak': 1 - (current_equity / self.peak_equity_today)},
            )

        return RiskEvent(False)
