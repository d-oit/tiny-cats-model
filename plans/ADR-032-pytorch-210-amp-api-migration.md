# ADR-032: PyTorch 2.10 AMP API Migration

## Status
Accepted

## Context
PyTorch 2.10 deprecated the `torch.cuda.amp` module functions:
- `torch.cuda.amp.GradScaler()` → Use `torch.amp.GradScaler('cuda')`
- `torch.cuda.amp.autocast()` → Use `torch.amp.autocast('cuda')`

Also discovered `volume_utils.py` was not included in Modal container (same root cause as ADR-031).

## Decision
1. Migrate all `torch.cuda.amp.GradScaler()` calls to `torch.amp.GradScaler('cuda')`
2. Migrate all `torch.cuda.amp.autocast()` calls to `torch.amp.autocast('cuda')`
3. Add `src/volume_utils.py` to Modal container via `add_local_file`

## Changes Made
- `src/train.py`: Lines 200, 672 + added volume_utils.py to container
- `src/train_dit.py`: Lines 625, 683 + added volume_utils.py to container

## Consequences
- **Positive**: Eliminates FutureWarnings, code compatible with PyTorch 2.10+
- **Positive**: Volume cleanup now works in Modal container
- **Negative**: LSP (pyright) shows false positive errors (runtime works fine)

## Related
- ADR-030: Modal Container Python Path Fix
- ADR-031: Modal Container Download Scripts Fix
- GOAP.md Phase 13
