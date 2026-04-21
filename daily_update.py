"""
Daily Update Script for Render Cron Job
========================================
Runs scrapers to refresh data, retrains ML models, then pushes
updated files back to GitHub so the web service auto-redeploys
with fresh data.

Environment variables required (set in Render dashboard):
  GITHUB_TOKEN  - GitHub PAT with 'repo' scope (required for git push)
  GITHUB_REPO   - "owner/repo" format (optional, defaults to juancortina03/SiembraAzucar)
  FRED_API_KEY  - for external market data scraper
  BANXICO_TOKEN - for USD/MXN exchange rate
"""

import os
import subprocess
import sys
import time

# Each step: (name, command, timeout_seconds)
# SNIIM scrapes 20+ years of daily prices so it needs a large timeout.
# CONADESUCA + extract are faster. ML retrain can also take a while.
STEPS = [
    ("SNIIM sugar prices",                       [sys.executable, "sniim_sugar_scraper.py"],                        1500),
    ("CONADESUCA balance index",                 [sys.executable, "conadesuca_balance_scraper.py"],                  900),
    ("CONADESUCA politica comercial index",      [sys.executable, "conadesuca_politica_comercial_scraper.py"],       900),
    ("Extract Excel reports from PDFs",          [sys.executable, "extract_all_reports.py", "skip-download"],        900),
    ("ML model retrain",                         [sys.executable, "sugar_price_model.py"],                           600),
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
    "excel_reports/",            # Monte Carlo & reference data
    "conadesuca_balance_pdfs/",  # PDFs fetched by balance scraper
    "politica_comercial_pdfs/",  # PDFs used by extract_all_reports
]

TARGET_BRANCH = "main"


def _run(cmd, check=False, capture=False):
    """Helper: run a subprocess, always print the command + status."""
    print(f"  $ {' '.join(cmd)}")
    if capture:
        r = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if r.stdout.strip():
            print(f"    stdout: {r.stdout.strip()}")
        if r.stderr.strip():
            print(f"    stderr: {r.stderr.strip()}")
        return r
    return subprocess.run(cmd, check=check)


def diagnose_environment():
    """Print a snapshot of the git/env state so cron log can be debugged."""
    print("\n--- Environment Diagnostics ---")
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "juancortina03/SiembraAzucar")
    print(f"  GITHUB_TOKEN set: {'YES' if token else 'NO'} ({'***' + token[-4:] if token else 'missing'})")
    print(f"  GITHUB_REPO:     {repo}")
    print(f"  FRED_API_KEY:    {'YES' if os.environ.get('FRED_API_KEY') else 'NO'}")
    print(f"  BANXICO_TOKEN:   {'YES' if os.environ.get('BANXICO_TOKEN') else 'NO'}")
    print(f"  CWD:             {os.getcwd()}")
    print(f"  Is git repo:     {os.path.isdir('.git')}")
    if os.path.isdir(".git"):
        _run(["git", "status", "--short", "-b"], capture=True)
        _run(["git", "branch", "-a"], capture=True)
        _run(["git", "log", "-1", "--oneline"], capture=True)


def git_push():
    """Commit and push updated data files back to the repo.

    Handles Render's detached HEAD state by explicitly forcing a local
    branch named `main` to point at the current HEAD before pushing.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "juancortina03/SiembraAzucar")

    if not token:
        print("\n  ERROR: GITHUB_TOKEN is not set in this environment.")
        print("  The cron cannot push data back to GitHub without a PAT.")
        print("  Fix: Render dashboard -> Cron -> Environment -> add GITHUB_TOKEN")
        return False

    if not os.path.isdir(".git"):
        print("\n  ERROR: Current directory is not a git repo.")
        print(f"  CWD: {os.getcwd()}")
        return False

    # Identity for the bot commit (may already be set; allow failure)
    _run(["git", "config", "user.email", "bot@siembraazucar.com"])
    _run(["git", "config", "user.name", "SiembraAzucar Bot"])

    # Set remote URL with token for auth
    remote_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    _run(["git", "remote", "set-url", "origin", remote_url])

    # CRITICAL: Render deploys in detached HEAD. Force-create/reset local `main`
    # branch pointing at current HEAD so `git push origin main` works.
    print(f"\n  Forcing local branch '{TARGET_BRANCH}' to current HEAD...")
    _run(["git", "checkout", "-B", TARGET_BRANCH])

    # Pull latest in case someone else pushed (rebase to avoid merge commits)
    # Use --no-rebase to accept a merge if rebase fails; keep going either way.
    print(f"\n  Fetching latest from origin/{TARGET_BRANCH}...")
    r = _run(["git", "fetch", "origin", TARGET_BRANCH], capture=True)
    if r.returncode != 0:
        print(f"  WARNING: fetch failed (code {r.returncode}). Continuing anyway.")
    else:
        # Try to reset to origin first, preserving our working tree changes
        # Actually we want to rebase our (nonexistent at this point) commits on top
        # The safest path is: stash our data files, pull, then re-add.
        # But since we haven't committed yet, unstaged data changes are untouched by pull.
        r2 = _run(["git", "pull", "--rebase", "origin", TARGET_BRANCH], capture=True)
        if r2.returncode != 0:
            print(f"  WARNING: pull --rebase failed. Will attempt push anyway.")

    # Stage data files (ignore missing files -- not every scraper produces every file)
    print("\n  Staging data files...")
    for f in DATA_FILES:
        _run(["git", "add", f])

    # Check if there are changes
    result = _run(["git", "diff", "--cached", "--quiet"], capture=True)
    if result.returncode == 0:
        print("\n  No data changes to push (all files already up to date).")
        return True

    # Show a summary of what's being committed
    _run(["git", "diff", "--cached", "--stat"], capture=True)

    # Commit and push
    today = time.strftime("%Y-%m-%d")
    commit_msg = f"Daily data update {today} [automated]"
    r = _run(["git", "commit", "-m", commit_msg], capture=True)
    if r.returncode != 0:
        print(f"  ERROR: commit failed (code {r.returncode})")
        return False

    print(f"\n  Pushing to origin/{TARGET_BRANCH}...")
    r = _run(["git", "push", "origin", TARGET_BRANCH], capture=True)
    if r.returncode != 0:
        print(f"  ERROR: push failed (code {r.returncode})")
        print("  Common causes:")
        print("    - GITHUB_TOKEN lacks 'repo' scope")
        print("    - GITHUB_TOKEN has expired")
        print("    - Branch protection rules blocking bot pushes")
        return False

    print(f"\n  SUCCESS: Pushed data update to GitHub ({today}).")
    print("  Render web service will redeploy automatically in ~1-2 minutes.")
    return True


def run():
    print("=" * 60)
    print("  Sugar Focars -- Daily Update")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 60)

    diagnose_environment()

    failed = []
    for name, cmd, timeout_s in STEPS:
        print(f"\n--- {name} ---  (timeout: {timeout_s}s)")
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, timeout=timeout_s)
            print(f"  OK ({time.time() - t0:.1f}s)")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
            failed.append(name)

    if failed:
        print(f"\nWARNING: {len(failed)} step(s) failed: {', '.join(failed)}")
    else:
        print("\nAll steps completed successfully.")

    # Push updated data to GitHub (triggers web service redeploy)
    # Always attempt push even if some steps failed -- partial data is better than none.
    print("\n--- Git push ---")
    try:
        pushed = git_push()
        if not pushed:
            print("  Push did not complete. Check errors above.")
    except Exception as e:
        print(f"  Git push crashed: {type(e).__name__}: {e}")

    if failed:
        print(f"\nExiting with code 1 due to {len(failed)} failed step(s).")
        sys.exit(1)

    print("\nDaily update complete.")


if __name__ == "__main__":
    run()
