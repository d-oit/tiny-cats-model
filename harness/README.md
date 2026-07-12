# Harness

Test harness and automation scripts for the tiny-cats-model project.

## Structure

```
harness/
├── README.md              # This file
├── memory-persist.sh      # Memory persistence and cleanup
├── memory-search.sh       # Search and discover patterns in memory
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

### memory-search.sh

Search and discover patterns across the memory system:

```bash
# Search for a term across all memory
bash harness/memory-search.sh "checkpoint"

# Search only in session history
bash harness/memory-search.sh "Modal" --scope sessions

# Search with context lines
bash harness/memory-search.sh "early stopping" --context --limit 5
```

Options:
- `--scope <scope>`: Search scope (all|projects|sessions|current)
- `--type <type>`: Filter by content type (checkpoint|notes|memory|all)
- `--limit <n>`: Max results per category (default: 10)
- `--context`: Show surrounding context lines

## Integration

These scripts can be run:
- Manually after significant work sessions
- Via cron for automated maintenance
- As part of pre-commit or post-commit hooks
