"""Run all historical case study scripts and merge their results into one overview file."""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "case_study/baseline_compare_historical_25thNov.py",
    "case_study/baseline_compare_historical_Nov.py",
    "case_study/baseline_compare_historical_Mar23_Mar25.py",
]

RESULTS = [
    ("Barcelona — 25th November 2024", "results/results_Barcelona_25th_Nov.txt"),
    ("Barcelona — November 2024",      "results/results_Barcelona_Nov.txt"),
    ("Barcelona — March 2023 to March 2025", "results/results_Barcelona_Mar23_Mar25.txt"),
]

OVERVIEW = "results/results_overview.txt"


def run_scripts():
    python = sys.executable
    for script in SCRIPTS:
        print(f"\n>>> Running {script} ...")
        result = subprocess.run([python, script], check=True)


def merge_results():
    sections = []
    width = 72

    for label, path in RESULTS:
        header = f"{'=' * width}\n{label}\n{'=' * width}"
        body = Path(path).read_text(encoding="utf-8")
        # Strip the repeated label line already inside each file to avoid duplication
        lines = body.splitlines()
        # Drop first two lines if they are the label + underline already written by results_utils
        if lines and lines[0].startswith("Results:"):
            lines = lines[2:]
        sections.append(header + "\n" + "\n".join(lines))

    overview = ("\n\n" + "-" * width + "\n\n").join(sections)
    Path(OVERVIEW).write_text(overview, encoding="utf-8")
    print(f"\nOverview written: {OVERVIEW}")


if __name__ == "__main__":
    run_scripts()
    merge_results()
