# tiny-cats-model

[![CI](https://github.com/d-oit/tiny-cats-model/actions/workflows/train.yml/badge.svg)](https://github.com/d-oit/tiny-cats-model/actions/workflows/train.yml)

A cats classifier built on tiny vision model infrastructure, following 2026 best practices for AI-agent-friendly repos.

## Features

- ResNet-18 fine-tuned for cat classification (cat / not-cat or breed labels)
- Dataset download script (Oxford IIIT Pet compatible)
- Modal-based GPU training support
- Full CI via GitHub Actions

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & prepare dataset
bash data/download.sh

# 3. Train
python src/train.py data/cats

# 4. Evaluate
python src/eval.py
```

## Project Structure

```
tiny-cats-model/
├── data/
│   ├── cats/              # local / script-generated dataset
│   └── download.sh        # dataset download/prepare script
├── src/
│   ├── train.py           # training entrypoint
│   ├── eval.py            # evaluation script
│   ├── dataset.py         # DataLoader factory
│   └── model.py           # model definition
├── tests/
│   └── test_dataset.py    # unit tests
├── .agents/
│   └── skills/
│       ├── testing-workflow/
│       │   ├── SKILL.md
│       │   └── verify.sh
│       ├── gh-actions/
│       │   └── SKILL.md
│       └── cli-usage/
│           └── SKILL.md
├── .github/
│   └── workflows/
│       └── train.yml
├── AGENTS.md
├── README.md
├── modal.yml
└── requirements.txt
```

## Modal Training (GPU)

```bash
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret
modal run src/train.py
```

See `modal.yml` for full config. Never commit secrets — use env vars or GitHub Secrets.

## Dataset

Default: Oxford IIIT Pet Dataset (cats subset). The `data/download.sh` script downloads and prepares the dataset. Replace the URL with your own source if needed.

## License

MIT
