#!/usr/bin/env python3
"""Deploy the Gradio demo app (src/app_gradio.py) to a HuggingFace Space.

This is used by .github/workflows/spaces-deploy.yml. The space is created as
a private Space by default because the model repo `d4oit/tiny-cats-model` it
downloads from is private.

Usage:
    HF_TOKEN=... SPACE_ID=d-oit/tiny-cats-model-demo \
        PRIVATE=true python .github/scripts/deploy_space.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_SRC = REPO_ROOT / "src" / "app_gradio.py"

REQUIREMENTS = """\
gradio>=5.0.0
numpy>=1.24.0
pillow>=10.0.0
huggingface_hub>=0.30.0
onnxruntime>=1.20.0
torch>=2.0.0
"""

SPACE_README = """\
---
title: Tiny Cats Model Demo
emoji: 🐈
colorFrom: blue
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# 🐈 Tiny Cats Model Demo

Classify a cat image (binary cat vs other) or generate cats with TinyDiT.
Models are loaded from the `d4oit/tiny-cats-model` Hub repository.
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    space_id = os.environ.get("SPACE_ID", "d-oit/tiny-cats-model-demo")
    private = os.environ.get("PRIVATE", "true").lower() == "true"

    if not APP_SRC.exists():
        print(f"ERROR: app source not found: {APP_SRC}")
        sys.exit(1)

    print(f"Ensuring Space {space_id} (private={private})...")
    create_repo(
        space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=private,
        token=token,
    )

    with tempfile.TemporaryDirectory() as tmp:
        app_dir = Path(tmp)
        shutil.copy(APP_SRC, app_dir / "app.py")
        (app_dir / "requirements.txt").write_text(REQUIREMENTS)
        (app_dir / "README.md").write_text(SPACE_README)

        upload_folder(
            folder_path=str(app_dir),
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Deploy Gradio demo (binary classifier fix)",
        )

    print(f"Deployed to https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
