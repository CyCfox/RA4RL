import json
import signal
import subprocess
import time
from pathlib import Path


def test_container_runner_reports_interruption_instead_of_completed(tmp_path):
    runner = Path("scripts/run_logged.sh").resolve()
    process = subprocess.Popen(["bash", str(runner), str(tmp_path), "sleep", "60"])
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "status.json").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / "status.json").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        status = json.loads((tmp_path / "status.json").read_text())
        assert status["phase"] == "interrupted" and status["exit_code"] == 143
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
