"""
logger.py
Simple prediction logger. Records every API prediction to a CSV file
so we can monitor model behavior, debug issues, and prove usage.
"""

import csv
import os
from datetime import datetime, timezone

LOG_FILE = "predictions_log.csv"

# Column headers — created only if file doesn't exist
HEADERS = [
    "timestamp",
    "call_duration_sec",
    "time_of_day",
    "ivr_option",
    "agent_experience_yrs",
    "previous_calls",
    "wait_time_sec",
    "score",
    "route",
]


def log_prediction(inputs: dict, score: int, route: str):
    """Append one prediction record to the CSV log."""
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if first time
        if not file_exists:
            writer.writerow(HEADERS)

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            inputs.get("call_duration_sec"),
            inputs.get("time_of_day"),
            inputs.get("ivr_option"),
            inputs.get("agent_experience_yrs"),
            inputs.get("previous_calls"),
            inputs.get("wait_time_sec"),
            score,
            route,
        ])