"""
report.py

Save one JSON experiment report per train.py execution.

Usage from train.py:
    from report import save_training_report
    report_path = save_training_report(hidden, report_data)
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def save_training_report(hidden, report_data, reports_dir="reports"):
    """Save a uniquely numbered JSON report for one training run."""
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^report_(\d+)_.*\.json$")
    existing_trials = []

    for path in reports_path.glob("report_*.json"):
        match = pattern.match(path.name)
        if match:
            existing_trials.append(int(match.group(1)))

    trial_no = max(existing_trials, default=0) + 1
    architecture = "_".join(str(width) for width in hidden)
    output_path = reports_path / f"report_{trial_no}_{architecture}.json"

    report = {
        "trial": trial_no,
        "architecture": list(hidden),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **report_data,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print(f"saved experiment report: {output_path}")
    return output_path
