#!/usr/bin/env python3
# mypy: ignore-errors
"""scripts/update-learnings.py

Update learnings.md with new patterns from commits.

Usage:
    python scripts/update-learnings.py [--dry-run]

This script:
1. Reads recent commit messages
2. Detects pattern changes (CI fixes, new features, bug fixes)
3. Suggests updates to agents-docs/learnings.md
4. Optionally applies updates automatically
"""

import argparse
import subprocess
from pathlib import Path


def get_recent_commits(n: int = 5) -> list[dict]:
    """Get recent commits with their changed files."""
    result = subprocess.run(
        ["git", "log", f"-{n}", "--name-only", "--format=%H|%s|%ai"],
        capture_output=True,
        text=True,
        check=True,
    )

    commits = []
    current = None
    files = []

    for line in result.stdout.strip().split("\n"):
        if "|" in line:
            if current:
                current["files"] = files
                commits.append(current)
            parts = line.split("|")
            current = {"hash": parts[0], "message": parts[1], "date": parts[2]}
            files = []
        elif line.strip():
            files.append(line.strip())

    if current:
        current["files"] = files
        commits.append(current)

    return commits


def categorize_commit(message: str, files: list[str]) -> dict:
    """Categorize a commit by type and affected areas."""
    categories = []

    # Conventional commit types
    if message.startswith("feat:"):
        categories.append("feature")
    elif message.startswith("fix:"):
        categories.append("bugfix")
    elif message.startswith("docs:"):
        categories.append("documentation")
    elif message.startswith("refactor:"):
        categories.append("refactoring")
    elif message.startswith("test:"):
        categories.append("testing")
    elif message.startswith("ci:"):
        categories.append("ci-cd")
    elif message.startswith("style:"):
        categories.append("style")

    # Area detection from files
    areas = []
    for f in files:
        if "src/" in f:
            areas.append("source")
        if "tests/" in f:
            areas.append("tests")
        if ".github/workflows/" in f:
            areas.append("ci-cd")
        if "agents-docs/" in f or "plans/" in f:
            areas.append("documentation")
        if "modal" in f.lower():
            areas.append("modal")
        if "security" in f.lower():
            areas.append("security")

    return {
        "type": categories[0] if categories else "other",
        "areas": list(set(areas)),
        "message": message,
        "files": files,
    }


def generate_learning_entry(commit: dict, categorization: dict) -> str:
    """Generate a learnings.md entry from a commit."""
    msg = commit["message"]
    commit_type = categorization["type"]
    areas = ", ".join(categorization["areas"])

    entry = f"""
### {msg}

**Date**: {commit["date"]}
**Type**: {commit_type}
**Areas**: {areas}

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]

"""
    return entry


def check_learnings_needs_update(commits: list[dict]) -> bool:
    """Check if learnings.md should be updated based on commits."""
    for commit in commits:
        cat = categorize_commit(commit["message"], commit["files"])
        # Significant changes that should be documented
        if cat["type"] in ["bugfix", "ci-cd", "feature"]:
            if "ci-cd" in cat["areas"] or "modal" in cat["areas"]:
                return True
            if (
                "fix" in commit["message"].lower()
                and "test" not in commit["message"].lower()
            ):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Update learnings.md from commits")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without applying"
    )
    parser.add_argument(
        "--commits", type=int, default=5, help="Number of recent commits to analyze"
    )
    args = parser.parse_args()

    print("Analyzing recent commits...")
    commits = get_recent_commits(args.commits)

    needs_update = False
    entries = []

    for commit in commits:
        cat = categorize_commit(commit["message"], commit["files"])
        if cat["type"] in ["bugfix", "ci-cd", "feature", "refactoring"]:
            needs_update = True
            entry = generate_learning_entry(commit, cat)
            entries.append(entry)
            print(f"  Found: {commit['message'][:60]}...")

    if not needs_update:
        print("No documentation updates needed.")
        return 0

    print(f"\n{len(entries)} commit(s) should be documented in learnings.md")

    if args.dry_run:
        print("\n--- Suggested entries ---")
        for entry in entries:
            print(entry)
        return 0

    # Append to learnings.md
    learnings_path = Path(__file__).parent.parent / "agents-docs" / "learnings.md"

    if not learnings_path.exists():
        print(f"Error: {learnings_path} not found")
        return 1

    content = learnings_path.read_text()

    # Find the "## Key Learnings" section and insert after it
    marker = "## Key Learnings\n"
    if marker in content:
        insert_pos = content.find(marker) + len(marker)
        # Skip existing entries
        new_content = content[:insert_pos] + "\n".join(entries) + content[insert_pos:]
    else:
        # Append before "## Reusable Patterns"
        marker = "## Reusable Patterns"
        if marker in content:
            insert_pos = content.find(marker)
            new_content = (
                content[:insert_pos] + "\n".join(entries) + "\n" + content[insert_pos:]
            )
        else:
            new_content = content + "\n" + "\n".join(entries)

    learnings_path.write_text(new_content)
    print(f"Updated {learnings_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
