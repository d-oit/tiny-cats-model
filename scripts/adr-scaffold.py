#!/usr/bin/env python3
"""scripts/adr-scaffold.py

Create a new Architecture Decision Record (ADR).

Usage:
    python scripts/adr-scaffold.py "Title of the decision"
    python scripts/adr-scaffold.py "Add new feature" --status proposed
"""

import argparse
from pathlib import Path


def get_next_adr_number(plans_dir: Path) -> int:
    """Get the next ADR number."""
    existing = list(plans_dir.glob("ADR-*.md"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("-")[1])
            numbers.append(num)
        except (ValueError, IndexError):
            pass
    return max(numbers) + 1 if numbers else 1


def create_adr(title: str, status: str = "Proposed") -> Path:
    """Create a new ADR file."""
    plans_dir = Path(__file__).parent.parent / "plans"
    num = get_next_adr_number(plans_dir)

    filename = f"ADR-{num:03d}-{title.lower().replace(' ', '-')}.md"
    filepath = plans_dir / filename

    content = f"""# ADR-{num:03d}: {title}

## Status
{status}

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?

- **Positive**:
- **Negative**:

## Alternatives Considered
1. [Alternative 1] - Why not chosen
2. [Alternative 2] - Why not chosen

## Related
- [Link to related ADRs or GOAP.md]

## References
- [Links to external resources]
"""

    filepath.write_text(content)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Create a new ADR")
    parser.add_argument("title", help="Title of the ADR")
    parser.add_argument(
        "--status",
        choices=["Proposed", "Accepted", "Rejected", "Deprecated"],
        default="Proposed",
        help="Status of the ADR",
    )
    args = parser.parse_args()

    filepath = create_adr(args.title, args.status)
    print(f"Created: {filepath}")
    print("Edit the file to fill in the details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
