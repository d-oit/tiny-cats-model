#!/usr/bin/env python
# .agents/skills/smart_lint.py

import sys
import subprocess
import re

ERROR_KEYWORDS = [
    r"error",
    r"fail",
    r"exception",
    r"traceback",
    r"line\s+\d+",
]

def run_and_filter(command: str) -> None:
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    output = (result.stdout or "") + (result.stderr or "")

    if result.returncode == 0:
        line_count = len(output.splitlines())
        print(f"✅ Success. No linting/build errors found. Tokens saved. ({line_count} lines suppressed)")
        return

    print(f"❌ Command failed (exit code {result.returncode}). Filtering output for errors...")

    pattern = re.compile("|".join(ERROR_KEYWORDS), re.IGNORECASE)
    filtered_lines = []

    for line in output.splitlines():
        if pattern.search(line):
            filtered_lines.append(line.rstrip())

    MAX_LINES = 40
    if not filtered_lines:
        print("No specific error lines matched filter; showing first 40 lines of raw output:")
        raw_lines = output.splitlines()
        snippet = raw_lines[:MAX_LINES]
        print("\n".join(snippet))
        if len(raw_lines) > MAX_LINES:
            print(f"\n...[Truncated {len(raw_lines) - MAX_LINES} additional lines]...")
        return

    if len(filtered_lines) > MAX_LINES:
        print("\n".join(filtered_lines[:MAX_LINES]))
        print(f"\n...[Truncated {len(filtered_lines) - MAX_LINES} more error lines]...")
    else:
        print("\n".join(filtered_lines))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python .agents/skills/smart_lint.py '<command>'")
        sys.exit(1)
    cmd = " ".join(sys.argv[1:])
    run_and_filter(cmd)
