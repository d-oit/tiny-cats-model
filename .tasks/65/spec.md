# Specification for Issue #65

## Issue Summary
Two persistent bugs in `src/train_dit.py`:
1. **Loss stuck at 0.0000**: Loss values are logged as `0.0000` because the format string uses `:.4f` which truncates small values.
2. **Scheduler deprecation warning**: PyTorch emits a deprecation warning around `scheduler.step(epoch)`.

## Requirements
1. **Fix Loss Logging**:
   - Change the loss format string from `{avg_loss:.4f}` to `{avg_loss:.6e}`.
   - Also, ensure `loss.item()` is properly used.

2. **Fix Scheduler Warning**:
   - Replace the `LinearLR`, `CosineAnnealingLR`, and `SequentialLR` implementation with `torch.optim.lr_scheduler.LambdaLR`.
   - Implement the `lr_lambda` function directly as suggested in the issue.
   - Ensure the `LambdaLR` scheduler is stepped without arguments.

## Plan
1. Edit `src/train_dit.py`.
2. Locate the scheduler creation and replace it with `LambdaLR` and the custom `lr_lambda` function. Include the required `import math` if needed.
3. Update the logging string format for loss.
4. Run formatting/linting using `.agents/skills/smart_lint.py "src/train_dit.py"`.
5. Run tests using `.agents/skills/token_safe_exec.sh "pytest tests/ -v"`.
