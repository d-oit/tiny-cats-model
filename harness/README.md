# Harness

Test harness and automation scripts for the tiny-cats-model project.

## Structure

```
harness/
├── README.md              # This file
├── memory-persist.sh      # Memory persistence and cleanup
├── checkpoint-enhance.sh  # Checkpoint post-processing (../scripts/)
└── ...
```

## Scripts

### memory-persist.sh

Manages MiMoCode session memory lifecycle:

```bash
# Preview what would be archived
bash harness/memory-persist.sh --dry-run

# Archive checkpoints older than 14 days (default)
bash harness/memory-persist.sh

# Archive checkpoints older than 7 days
bash harness/memory-persist.sh --days 7
```

## Integration

These scripts can be run:
- Manually after significant work sessions
- Via cron for automated maintenance
- As part of pre-commit or post-commit hooks
