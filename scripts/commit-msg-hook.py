#!/usr/bin/env python3
"""scripts/commit-msg-hook.py

Commit-msg hook to capture learnings from commit messages.

Usage (in .git/hooks/commit-msg):
    python scripts/commit-msg-hook.py <commit-msg-file>

Detects patterns in commit messages and suggests learnings.md updates.
"""

import subprocess
import sys
from pathlib import Path


def get_current_commit_sha() -> str | None:
    """Get the current commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:7]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_changed_files() -> list[str]:
    """Get list of files changed in this commit (staged)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def check_quality_issues() -> list[str]:
    """Run quality checks on changed files. Returns suggestions if issues found."""
    suggestions: list[str] = []
    changed_files = get_changed_files()

    if not changed_files:
        return suggestions

    python_files = [f for f in changed_files if f.endswith(".py")]
    if not python_files:
        return suggestions

    files_arg = " ".join(python_files)

    ruff_result = subprocess.run(
        f"python -m ruff check {files_arg}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if ruff_result.returncode != 0:
        suggestions.append(
            "⚠️  Ruff lint errors detected in changed files:\n"
            f"   → Run: ruff check {files_arg} --fix\n"
            "   → Then re-stage and commit"
        )

    black_result = subprocess.run(
        f"python -m black --check {files_arg}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if black_result.returncode != 0:
        suggestions.append(
            "⚠️  Black format issues in changed files:\n"
            f"   → Run: black {files_arg}\n"
            "   → Then re-stage and commit"
        )

    return suggestions


def analyze_commit_message(msg: str, commit_sha: str | None = None) -> list[str]:
    """Analyze commit message for learning opportunities."""
    suggestions = []

    msg_lower = msg.lower()
    sha_info = f" (commit: {commit_sha})" if commit_sha else ""
    verify_cmd = f"git show {commit_sha}" if commit_sha else "git show HEAD"

    # CI/CD fixes
    if "ci" in msg_lower or "github actions" in msg_lower or "workflow" in msg_lower:
        if "fix" in msg_lower or "update" in msg_lower:
            suggestions.append(
                f"📝 CI/CD fix detected{sha_info}. Verify pattern, then document:\n"
                f"   → Review: {verify_cmd} --stat\n"
                "   → If verified: add to agents-docs/learnings.md (Key Learnings)\n"
                "   → Or create ADR if architectural decision"
            )

    # Modal/training changes
    if "modal" in msg_lower or "gpu" in msg_lower or "training" in msg_lower:
        suggestions.append(
            f"📝 Modal/training change detected{sha_info}. Verify, then update:\n"
            f"   → Review: {verify_cmd} --stat\n"
            "   → If verified: add pattern to agents-docs/training.md\n"
            "   → And to agents-docs/learnings.md"
        )

    # Bug fixes with lessons
    if "fix" in msg_lower and (
        "timeout" in msg_lower or "error" in msg_lower or "fail" in msg_lower
    ):
        suggestions.append(
            f"📝 Bug fix with error handling{sha_info}. Verify, then document:\n"
            f"   → Review: {verify_cmd}\n"
            "   → What was the root cause?\n"
            "   → What's the reusable pattern?\n"
            "   → If verified: add to agents-docs/learnings.md"
        )

    # New features
    if msg.startswith("feat:"):
        suggestions.append(
            f"📝 New feature{sha_info}. Verify, then consider:\n"
            f"   → Review: {verify_cmd} --stat\n"
            "   → If significant: update AGENTS.md if commands changed\n"
            "   → If verified pattern: add to agents-docs/learnings.md"
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

    suggestions = analyze_commit_message(msg, get_current_commit_sha())
    quality_issues = check_quality_issues()

    all_output = quality_issues + suggestions

    if all_output:
        print("\n📋 Commit Analysis")
        print("=" * 50)
        for s in all_output:
            print(s)
            print()
        print("=" * 50)
        print("\n💡 These are suggestions. Commit will proceed.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
