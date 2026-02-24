#!/usr/bin/env python3
"""scripts/commit-msg-hook.py

Commit-msg hook to capture learnings from commit messages.

Usage (in .git/hooks/commit-msg):
    python scripts/commit-msg-hook.py <commit-msg-file>

Detects patterns in commit messages and suggests learnings.md updates.
"""

import sys
from pathlib import Path


def analyze_commit_message(msg: str) -> list[str]:
    """Analyze commit message for learning opportunities."""
    suggestions = []

    msg_lower = msg.lower()

    # CI/CD fixes
    if "ci" in msg_lower or "github actions" in msg_lower or "workflow" in msg_lower:
        if "fix" in msg_lower or "update" in msg_lower:
            suggestions.append(
                "📝 CI/CD change detected. Consider documenting the fix pattern in:\n"
                "   → agents-docs/learnings.md (Key Learnings section)\n"
                "   → Or create ADR if architectural decision"
            )

    # Modal/training changes
    if "modal" in msg_lower or "gpu" in msg_lower or "training" in msg_lower:
        suggestions.append(
            "📝 Modal/training change detected. Consider updating:\n"
            "   → agents-docs/training.md\n"
            "   → agents-docs/learnings.md with the pattern"
        )

    # Bug fixes with lessons
    if "fix" in msg_lower and (
        "timeout" in msg_lower or "error" in msg_lower or "fail" in msg_lower
    ):
        suggestions.append(
            "📝 Bug fix with error handling. Document the lesson:\n"
            "   → What was the root cause?\n"
            "   → What's the reusable pattern?\n"
            "   → Add to agents-docs/learnings.md"
        )

    # New features
    if msg.startswith("feat:"):
        suggestions.append(
            "📝 New feature added. Consider:\n"
            "   → Updating AGENTS.md if commands changed\n"
            "   → Adding to agents-docs/learnings.md"
        )

    return suggestions


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/commit-msg-hook.py <commit-msg-file>")
        return 1

    commit_msg_file = Path(sys.argv[1])

    if not commit_msg_file.exists():
        print(f"Error: {commit_msg_file} not found")
        return 1

    msg = commit_msg_file.read_text()

    # Skip merge commits
    if msg.startswith("Merge "):
        return 0

    suggestions = analyze_commit_message(msg)

    if suggestions:
        print("\n📋 Commit Analysis")
        print("=" * 50)
        for s in suggestions:
            print(s)
            print()
        print("=" * 50)
        print("\n💡 These are suggestions. Commit will proceed.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
