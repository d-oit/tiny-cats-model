"""Build and atomically persist a deterministic classifier split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splits import build_split_manifest, split_manifest_summary, write_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="ImageFolder dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--stratify-by", choices=("label", "breed"), default="label")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_split_manifest(
        args.root,
        seed=args.seed,
        val_split=args.val_split,
        test_split=args.test_split,
        version=args.version,
        stratify_by=args.stratify_by,
    )
    if args.output is None:
        path = write_split_manifest(args.root, manifest)
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        path = output
    print(json.dumps({**split_manifest_summary(manifest), "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
