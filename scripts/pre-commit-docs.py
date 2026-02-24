#!/usr/bin/env python3
"""scripts/pre-commit-docs.py

Pre-commit hook to check if documentation needs updating.

Usage (in .git/hooks/pre-commit):
    python scripts/pre-commit-docs.py

Checks:
1. CI workflow changes → suggest ADR
2. Modal config changes → suggest training.md update
3. Security-related changes → suggest security.md update
4. New skills → suggest skills.md update
"""

import subprocess


def get_staged_files() -> list[str]:
    """Get list of staged files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def check_docs_needed(files: list[str]) -> list[str]:
    """Check if documentation updates are needed."""
    warnings = []

    for f in files:
        # CI workflow changes
        if ".github/workflows/" in f:
            warnings.append(
                f"⚠️  CI workflow changed ({f})\n" f"   → Consider updating agents-docs/ci-cd.md or creating an ADR"
            )

        # Modal config changes
        if "modal" in f.lower() and f.endswith((".py", ".yml", ".yaml")):
            warnings.append(f"⚠️  Modal config changed ({f})\n" f"   → Consider updating agents-docs/training.md")

        # Security-related changes
        if "security" in f.lower() or "secret" in f.lower() or "token" in f.lower():
            if not f.endswith("test"):
                warnings.append(f"⚠️  Security-related change ({f})\n" f"   → Consider updating agents-docs/security.md")

        # Agent skills changes
        if ".agents/skills/" in f:
            warnings.append(f"⚠️  Agent skill changed ({f})\n" f"   → Consider updating agents-docs/skills.md")

        # GOAP/ADR changes
        if "plans/GOAP.md" in f or "plans/ADR-" in f:
            warnings.append(
                f"ℹ️  Planning document changed ({f})\n"
                f"   → Consider updating agents-docs/learnings.md with key insights"
            )

    return warnings


def main():
    files = get_staged_files()

    if not files:
        print("No staged files.")
        return 0

    warnings = check_docs_needed(files)

    if warnings:
        print("\n📋 Documentation Check")
        print("=" * 50)
        for w in warnings:
            print(w)
            print()
        print("=" * 50)
        print("\n💡 Tip: Run 'python scripts/update-learnings.py' to auto-update learnings")
        print("   Or 'python scripts/adr-scaffold.py \"Title\"' to create an ADR\n")

        # Don't block commit, just warn
        return 0
    else:
        print("✓ No documentation updates required.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
