# Harness

Test harness and automation scripts for the tiny-cats-model project.

## Structure

```
harness/
├── README.md              # This file
├── memory-persist.sh      # Memory persistence and cleanup (Area 1)
├── memory-search.sh       # Ranked search + "what do we know" digest (Area 3)
├── memory-share.sh        # Cross-project sharing + learnings library (Area 4)
└── ...
```

All scripts operate on the MiMoCode memory store at
`~/.local/share/mimocode/memory/`. See Issue #119 for the surrounding
multi-area memory-system work.

## Scripts

### memory-persist.sh

Manages MiMoCode session memory lifecycle (Area 1):

```bash
# Preview what would be archived
bash harness/memory-persist.sh --dry-run

# Archive checkpoints older than 14 days (default)
bash harness/memory-persist.sh

# Archive checkpoints older than 7 days
bash harness/memory-persist.sh --days 7
```

### memory-search.sh

Ranked, technical-term-aware search across the memory system (Area 3).
Instead of raw `grep`, results are **scored** — lines with more matching
query tokens, whole-word hits, and technical tokens (`ADR-\d+`, file paths,
`fn()` calls) rank higher; `MEMORY.md` outranks `notes.md` outranks
`checkpoint.md`.

```bash
# Search for a term across all memory
bash harness/memory-search.sh "checkpoint"

# Search only in session history
bash harness/memory-search.sh "Modal" --scope sessions

# Search with context lines
bash harness/memory-search.sh "early stopping" --context --limit 5

# Digest mode: "what do we know about X?"
bash harness/memory-search.sh "Modal" --ask
```

Options:
- `--scope <scope>`: Search scope (all|projects|sessions|current|global)
- `--type <type>`: Filter by content type (checkpoint|notes|memory|all)
- `--limit <n>`: Max results (default: 10)
- `--context`: Show surrounding context lines
- `--ask`: Digest mode — prints a ranked summary plus related durable
  knowledge (section headings + ADR anchors) instead of raw line hits
- `--project <id>` / `--session <id>`: Override auto-detected project /
  session ids

The current project and session are auto-detected (repo name match for the
project, most-recently-modified for the session).

### memory-share.sh

Cross-project memory sharing (Area 4):

```bash
# List all projects + how many MEMORY files / ADRs each holds
bash harness/memory-share.sh --list

# Show reusable patterns + ADR refs for one project
bash harness/memory-share.sh --extract <project-id>

# Rebuild the cross-project learnings library
bash harness/memory-share.sh --learnings

# Suggest other projects' patterns relevant to the current one (by ADR overlap)
bash harness/memory-share.sh --suggest

# Append reusable patterns from one project's MEMORY to another's
bash harness/memory-share.sh --copy <src-project> <dst-project>
```

- `--learnings` writes a persistent, auto-generated
  `~/.local/share/mimocode/user/learnings.md` aggregating every project's
  reusable patterns under a `## Project:` heading per project.
- The current project id is auto-detected by repo name; override with
  `--project <id>`.

## Integration

These scripts can be run:
- Manually after significant work sessions
- Via cron for automated maintenance
- As part of pre-commit or post-commit hooks
