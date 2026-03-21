import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class StateBackupManager:
    def __init__(self, backup_dir: str = 'backups'):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup(self, state: Dict, config_path: str = 'config/config.yaml', logs_dir: str = 'logs') -> str:
        stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        state_file = self.backup_dir / f'state_{stamp}.json'
        with state_file.open('w') as f:
            json.dump(state, f, default=str, indent=2)

        archive = self.backup_dir / f'backup_{stamp}.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(state_file, arcname=state_file.name)
            cfg = Path(config_path)
            if cfg.exists():
                tar.add(cfg, arcname=str(cfg))
            logs = Path(logs_dir)
            if logs.exists():
                tar.add(logs, arcname=str(logs))

        return str(archive)

    def backup_runtime_state(self, state: Dict) -> str:
        """Persist lightweight runtime state snapshot as JSON."""
        stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        state_file = self.backup_dir / f'runtime_state_{stamp}.json'
        with state_file.open('w') as f:
            json.dump(state, f, default=str, indent=2)
        return str(state_file)

    def load_latest_runtime_state(self) -> Optional[Dict]:
        """Load most recent lightweight runtime state snapshot."""
        snapshots = sorted(self.backup_dir.glob('runtime_state_*.json'))
        if not snapshots:
            return None
        latest = snapshots[-1]
        with latest.open('r') as f:
            return json.load(f)

    def restore(self, archive_path: str, output_dir: str = '.') -> None:
        target_dir = Path(output_dir).resolve()
        with tarfile.open(archive_path, 'r:gz') as tar:
            members = tar.getmembers()
            for member in members:
                member_path = (target_dir / member.name).resolve()
                if not str(member_path).startswith(str(target_dir)):
                    raise ValueError(f"Unsafe archive member path: {member.name}")
            tar.extractall(path=target_dir)


class PositionDriftMonitor:
    def __init__(self, max_drift_pct: float = 0.02):
        self.max_drift_pct = max_drift_pct

    def check(self, target_positions: Dict[str, int], live_positions: Dict[str, Dict]) -> Dict[str, float]:
        drift = {}
        tickers = set(target_positions.keys()) | set(live_positions.keys())
        for ticker in tickers:
            target = target_positions.get(ticker, 0)
            live = int(live_positions.get(ticker, {}).get('shares', 0))
            if target == 0:
                # Closing positions is expected during rebalances; skip pre-trade drift alarms.
                continue
            denom = max(abs(target), abs(live), 1)
            drift_pct = abs(target - live) / denom
            if drift_pct > self.max_drift_pct:
                drift[ticker] = drift_pct
        return drift
