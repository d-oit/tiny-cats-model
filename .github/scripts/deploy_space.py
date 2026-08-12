#!/usr/bin/env python3
"""Deploy a static demo page for the cats classifier to a HuggingFace Space.

The page is generated from evaluation_report.json (committed at repo root) and
pushed as a free Static Space via the `hf upload` CLI (huggingface_hub>=1.0).
Static Spaces need no hardware and cost nothing, matching the project rule of
never using paid HuggingFace services.

Usage (run in a checkout that has evaluation_report.json):
    HF_TOKEN=... ACTION=deploy SPACE_ID=d4oit/tiny-cats-model-demo \
        python .github/scripts/deploy_space.py
    ACTION=list   # print the Spaces the token can see (discovery, no writes)
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPACE = "d4oit/tiny-cats-model-demo"
REPORT_PATH = REPO_ROOT / "evaluation_report.json"

EMOJI = "🐈"
PAGE_TITLE = "Tiny Cats Model — Classifier Demo"
SHORT_DESC = "Binary cat-vs-dog classifier metrics demo"


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    return token


def _pct(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return str(value)


def build_static_page(out_dir: Path) -> None:
    """Generate README.md + index.html for a Static Space from the eval report."""
    report = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            report = json.load(f)

    accuracy = _pct(report.get("accuracy", "N/A"))
    macro_f1 = _pct(report.get("macro_f1", "N/A"))
    weighted_f1 = _pct(report.get("weighted_f1", "N/A"))
    confusion = report.get("confusion_matrix") or []
    class_names = report.get("class_names") or []
    total = report.get("total", "N/A")
    num_failures = report.get("num_failures", "N/A")
    timestamp = report.get("timestamp", "unknown")

    (out_dir / "README.md").write_text(
        f"""---
title: {PAGE_TITLE}
emoji: {EMOJI}
colorFrom: blue
colorTo: pink
sdk: static
short_description: {SHORT_DESC}
pinned: false
---

# {PAGE_TITLE}

Static demo page for the binary cat-vs-dog classifier.
Source: [d-oit/tiny-cats-model](https://huggingface.co/d-oit/tiny-cats-model).
"""
    )

    header = (
        "<tr>"
        + "".join(f"<th>{html.escape(c)}</th>" for c in ["", *class_names])
        + "</tr>"
    )
    body = "".join(
        "<tr><th>{}</th>{}</tr>".format(
            html.escape(class_names[i]),
            "".join(f"<td>{cell}</td>" for cell in row),
        )
        for i, row in enumerate(confusion[: len(class_names)])
    )
    rows = header + body

    per_class_rows = ""
    if class_names:
        prec = report.get("precision") or {}
        rec = report.get("recall") or {}
        f1 = report.get("f1") or {}
        per_class_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(c)}</td>"
            f"<td>{_pct(prec.get(c, 'N/A'))}</td>"
            f"<td>{_pct(rec.get(c, 'N/A'))}</td>"
            f"<td>{_pct(f1.get(c, 'N/A'))}</td>"
            "</tr>"
            for c in class_names
        )

    (out_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(PAGE_TITLE)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 3rem auto;
        padding: 0 1rem; color: #1f2937; line-height: 1.55; }}
  h1 {{ font-size: 1.6rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
  th, td {{ border: 1px solid #d1d5db; padding: .5rem .75rem; text-align: left; }}
  th {{ background: #eff6ff; }}
  a {{ color: #1d4ed8; }}
  .muted {{ color: #6b7280; font-size: .85rem; }}
</style>
</head>
<body>
<h1>{html.escape(PAGE_TITLE)}</h1>
<p>Binary <strong>cat vs dog</strong> classifier (ResNet-18, {report.get("total")} images per
split eval, Oxford IIIT Pet), trained with class-balanced loss on the full
dataset. Evaluation report in <code>evaluation_report.json</code> at the
<a href="https://github.com/d-oit/tiny-cats-model">repo</a> or the
<a href="https://huggingface.co/d-oit/tiny-cats-model">model hub repo</a>.</p>

<h2>Headline metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Accuracy</td><td>{accuracy} ({report.get("correct")}/{total})</td></tr>
<tr><td>Macro F1</td><td>{macro_f1}</td></tr>
<tr><td>Weighted F1</td><td>{weighted_f1}</td></tr>
<tr><td>Misclassifications</td><td>{num_failures}</td></tr>
</table>

<h2>Per-class precision / recall / F1</h2>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{per_class_rows}
</table>

<h2>Confusion matrix (rows=true, cols=predicted)</h2>
<table>
{rows}
</table>

<p class="muted">Report timestamp: {html.escape(str(timestamp))}</p>
</body>
</html>
"""
    )


def list_spaces() -> None:
    """Print the Spaces the token can see (for discovery; no writes)."""
    from huggingface_hub import HfApi

    api = HfApi(token=_require_token())
    me = api.whoami()
    print(f"whoami: {me.get('name')} (type={me.get('type')})")
    authors = ["d-oit", me.get("name", "")]
    seen: set[str] = set()
    for author in authors:
        if not author or author in seen:
            continue
        seen.add(author)
        print(f"== spaces under {author} ==")
        found = False
        for space in api.list_spaces(author=author):
            found = True
            rt = space.runtime or {}
            print(
                f"  {space.id}  private={space.private}  sdk={space.sdk}  "
                f"stage={rt.get('stage')}"
            )
        if not found:
            print("  (none)")


def main() -> None:
    if os.environ.get("ACTION", "deploy").lower() == "list":
        list_spaces()
        return

    _require_token()
    space_id = os.environ.get("SPACE_ID", DEFAULT_SPACE)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "site"
        out_dir.mkdir()
        build_static_page(out_dir)

        print(f"Uploading static page to {space_id} ...")
        env = {**os.environ, "HF_TOKEN": os.environ["HF_TOKEN"]}
        result = subprocess.run(
            [
                "hf",
                "upload",
                space_id,
                str(out_dir),
                "--repo-type",
                "space",
                "--commit-message",
                "Deploy static classifier demo page",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        print(result.stdout[-2000:])
        if result.returncode != 0:
            print(result.stderr[-2000:], file=sys.stderr)
            sys.exit(result.returncode)

    print(f"Deployed to https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
