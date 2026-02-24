#!/usr/bin/env python3
"""scripts/update-goap.py

Update GOAP.md with new action items from commits.

Usage:
    python scripts/update-goap.py [--dry-run]
"""

import argparse
import re
from pathlib import Path


def get_recent_commits(n: int = 3) -> list[dict]:
    """Get recent commits."""
    import subprocess

    result = subprocess.run(
        ["git", "log", f"-{n}", "--format=%H|%s|%ai"],
        capture_output=True,
        text=True,
        check=True,
    )

    commits = []
    for line in result.stdout.strip().split("\n"):
        if "|" in line:
            parts = line.split("|")
            commits.append({"hash": parts[0], "message": parts[1], "date": parts[2]})

    return commits


def extract_action_items(commits: list[dict]) -> list[str]:
    """Extract potential action items from commit messages."""
    items = []
    for commit in commits:
        msg = commit["message"]
        # Look for TODO, FIXME, HACK, XXX patterns
        patterns = [r"TODO[:\s]*(.+)", r"FIXME[:\s]*(.+)", r"HACK[:\s]*(.+)", r"XXX[:\s]*(.+)"]
        for pattern in patterns:
            match = re.search(pattern, msg, re.IGNORECASE)
            if match:
                items.append(f"- [ ] {match.group(1).strip()} (from {commit['hash'][:7]})")
    return items


def add_action_items(items: list[str], dry_run: bool = False) -> None:
    """Add action items to GOAP.md."""
    goap_path = Path(__file__).parent.parent / "plans" / "GOAP.md"

    if not goap_path.exists():
        print(f"Error: {goap_path} not found")
        return

    content = goap_path.read_text()

    # Find "## Current Action Items" section
    marker = "## Current Action Items\n"
    if marker not in content:
        print("No 'Current Action Items' section found. Creating one.")
        # Add before first ## section
        first_section = content.find("\n## ")
        if first_section > 0:
            insert_pos = first_section
            new_section = "\n## Current Action Items\n" + "\n".join(items) + "\n"
            content = content[:insert_pos] + new_section + content[insert_pos:]
        else:
            content += "\n## Current Action Items\n" + "\n".join(items) + "\n"
    else:
        # Find the next section after Current Action Items
        start = content.find(marker) + len(marker)
        next_section = content.find("\n## ", start)
        if next_section > 0:
            insert_pos = next_section
        else:
            insert_pos = len(content)

        new_items = "\n".join(items) + "\n"
        content = content[:insert_pos] + new_items + content[insert_pos:]

    if dry_run:
        print("Would update GOAP.md with:")
        print(new_items)
    else:
        goap_path.write_text(content)
        print(f"Updated {goap_path}")


def main():
    parser = argparse.ArgumentParser(description="Update GOAP.md with action items")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--commits", type=int, default=3, help="Number of commits to analyze")
    args = parser.parse_args()

    print("Analyzing commits for action items...")
    commits = get_recent_commits(args.commits)

    items = extract_action_items(commits)

    if not items:
        print("No action items found in recent commits.")
        return 0

    print(f"Found {len(items)} action item(s):")
    for item in items:
        print(f"  {item}")

    add_action_items(items, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
