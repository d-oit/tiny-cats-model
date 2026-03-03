# ADR-050: Fix train_dit.py CLI Argument Mismatch

## Status
Accepted

## Context
After implementing ADR-048 to fix Modal CLI syntax in GitHub Actions, the Train DiT workflow started failing with "Process completed with exit code 1". Investigation revealed that ADR-048 only updated the workflow file (train.yml) to use `--data-dir` syntax, but `train_dit.py` was not updated to accept this as an optional argument.

### The Mismatch
| Component | Argument Type | Syntax |
|-----------|--------------|--------|
| `train.py` | Optional (`--data-dir`) | ✅ Correct - matches workflow |
| `train_dit.py` | Positional (`data_dir`) | ❌ Broken - workflow passes `--data-dir` but script expects positional |
| `train.yml` workflow | Optional (`--data-dir`) | ✅ Correct after ADR-048 |

### Error Manifestation
When the workflow runs:
```bash
modal run src/train_dit.py --data-dir data/cats --steps 400000
```

The script receives:
```python
# argparse sees --data-dir as unknown or misplaced
# data_dir positional argument is missing
```

Result: Training fails immediately with argument parsing error.

## Decision
Update `src/train_dit.py` to use `--data-dir` as a required optional argument instead of a positional argument. This maintains consistency with:
1. ADR-048 workflow changes
2. `train.py` argument structure
3. Modal 1.0+ CLI conventions

### Change Required
```python
# BEFORE (line 159):
parser.add_argument("data_dir", type=str, help="Path to dataset root")

# AFTER:
parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root")
```

### Documentation Updates
1. Update script docstring to show correct usage
2. Update all examples in AGENTS.md
3. Update GOAP.md Phase 18 status
4. Update learnings.md with this pattern

## Consequences

### Positive
- **Consistency**: Both training scripts use the same argument pattern
- **CI/CD Working**: GitHub Actions workflow will function correctly
- **User Experience**: Single syntax pattern for all Modal CLI commands
- **Maintainability**: One pattern to document and support

### Negative
- **Breaking Change**: Any existing scripts using positional `data/cats` will need updating
- **Documentation Updates Required**: Multiple files need updates

### Migration Path
For users with existing scripts:
```bash
# OLD (broken):
python src/train_dit.py data/cats --steps 100

# NEW (correct):
python src/train_dit.py --data-dir data/cats --steps 100
```

## Verification Checklist
- [x] Update `src/train_dit.py` argument parsing
- [x] Update script docstring examples
- [x] Run quality gate locally
- [x] Test `python src/train_dit.py --help`
- [x] Update GOAP.md Phase 18
- [x] Update learnings.md
- [ ] Verify CI passes after push
- [ ] Re-run 400k training via GitHub Actions

## Related
- ADR-048: Modal CLI Syntax Fix (incomplete implementation)
- ADR-046: GitHub Actions Missing Dependencies Fix
- ADR-047: Local Testing vs GitHub Actions Strategy
- train.yml: GitHub Actions workflow
- GOAP.md Phase 18: High-Accuracy Training

## Date
2026-03-03
