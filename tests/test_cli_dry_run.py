import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "mistral_fp16.yaml"


def test_dry_run_exits_zero_and_prints_run_id(tmp_path):
    target_csv = tmp_path / "should_not_exist.csv"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_quantization",
            "--config",
            str(CONFIG_PATH),
            "--csv-path",
            str(target_csv),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"
    assert "__fp16__16bit__" in res.stdout, f"stdout: {res.stdout!r}"
    assert not target_csv.exists()
