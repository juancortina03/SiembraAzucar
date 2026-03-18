"""
Daily Update Script for Render Cron Job
========================================
Runs scrapers to refresh data, then retrains the ML models.
Designed to run headlessly on Render's cron scheduler.
"""

import subprocess
import sys
import time

STEPS = [
    ("SNIIM sugar prices", [sys.executable, "sniim_sugar_scraper.py"]),
    ("CONADESUCA balance index", [sys.executable, "conadesuca_balance_scraper.py"]),
    ("CONADESUCA politica comercial index", [sys.executable, "conadesuca_politica_comercial_scraper.py"]),
    ("ML model retrain", [sys.executable, "sugar_price_model.py"]),
]


def run():
    print("=" * 60)
    print("  Sugar Focars -- Daily Update")
    print("=" * 60)
    failed = []
    for name, cmd in STEPS:
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, timeout=600)
            print(f"  OK ({time.time() - t0:.1f}s)")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  FAILED: {e}")
            failed.append(name)

    if failed:
        print(f"\nWARNING: {len(failed)} step(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll steps completed successfully.")


if __name__ == "__main__":
    run()
