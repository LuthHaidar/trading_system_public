import unittest
import tarfile
import tempfile
from pathlib import Path

from utils.operations import PositionDriftMonitor, StateBackupManager


class OperationsRegressionTests(unittest.TestCase):
    def test_drift_monitor_skips_planned_closures(self):
        monitor = PositionDriftMonitor(max_drift_pct=0.02)
        drift = monitor.check(
            target_positions={'AAA': 0},
            live_positions={'AAA': {'shares': 150}},
        )
        self.assertEqual(drift, {})

    def test_drift_monitor_flags_material_non_close_drift(self):
        monitor = PositionDriftMonitor(max_drift_pct=0.02)
        drift = monitor.check(
            target_positions={'AAA': 100},
            live_positions={'AAA': {'shares': 90}},
        )
        self.assertIn('AAA', drift)
        self.assertAlmostEqual(drift['AAA'], 0.10, places=8)

    def test_backup_restore_rejects_unsafe_archive_paths(self):
        manager = StateBackupManager(backup_dir='backups')
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / 'unsafe.tar.gz'
            payload_path = Path(tmp) / 'payload.txt'
            payload_path.write_text('x', encoding='utf-8')

            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(payload_path, arcname='../evil.txt')

            with self.assertRaises(ValueError):
                manager.restore(str(archive_path), output_dir=tmp)


if __name__ == '__main__':
    unittest.main()
