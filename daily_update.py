"""
Daily Update Script for Render Cron Job
========================================
Runs scrapers to refresh data, retrains ML models, then pushes
updated files back to GitHub so the web service auto-redeploys
with fresh data.
"""

import os
import subprocess
import sys
import time

STEPS = [
    ("SNIIM sugar prices", [sys.executable, "sniim_sugar_scraper.py"]),
    ("CONADESUCA balance index", [sys.executable, "conadesuca_balance_scraper.py"]),
    ("CONADESUCA politica comercial index", [sys.executable, "conadesuca_politica_comercial_scraper.py"]),
    ("ML model retrain", [sys.executable, "sugar_price_model.py"]),
]

# Files that get updated by scrapers / model and need to be pushed
DATA_FILES = [
    "sniim_sugar_prices.csv",
    "sniim_sugar_prices.xlsx",
    "conadesuca_balance_index.csv",
    "conadesuca_balance_index.xlsx",
    "politica_comercial_index.csv",
    "politica_comercial_index.xlsx",
    "model_results/",
]


def git_push():
    """Commit and push updated data files back to the repo."""
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "juancortina03/SiembraAzucar")

    if not token:
        print("\n  GITHUB_TOKEN not set -- skipping git push.")
        print("  Data was refreshed locally but won't persist across deploys.")
        return False

    # Configure git for the bot commit
    subprocess.run(["git", "config", "user.email", "bot@siembraazucar.com"], check=True)
    subprocess.run(["git", "config", "user.name", "SiembraAzucar Bot"], check=True)

    # Set remote URL with token for auth
    remote_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)

    # Stage data files
    for f in DATA_FILES:
        subprocess.run(["git", "add", f], check=False)

    # Check if there are changes
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
    if result.returncode == 0:
        print("\n  No data changes to push.")
        return True

    # Commit and push
    today = time.strftime("%Y-%m-%d")
    subprocess.run(
        ["git", "commit", "-m", f"Daily data update {today} [automated]"],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print(f"\n  Pushed data update to GitHub ({today}).")
    return True


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
    else:
        print("\nAll steps completed successfully.")

    # Push updated data to GitHub (triggers web service redeploy)
    print("\n--- Git push ---")
    try:
        git_push()
    except Exception as e:
        print(f"  Git push failed: {e}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run()
