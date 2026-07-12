#!/usr/bin/env bash
# scripts/memory-persist.sh
# Memory persistence and cleanup for MiMoCode sessions.
#
# Usage:
#   bash scripts/memory-persist.sh [--dry-run] [--days N]
#
# What it does:
#   1. Archives old session checkpoints (>N days, default 14)
#   2. Extracts durable learnings from notes.md files
#   3. Reports memory statistics
#
# Options:
#   --dry-run   Show what would be done without making changes
#   --days N    Days before archiving (default: 14)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}▶ $1${NC}"; }
log_success() { echo -e "${GREEN}✓ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
log_error() { echo -e "${RED}✗ $1${NC}"; }
log_stat() { echo -e "${CYAN}  $1${NC}"; }

# Defaults
DRY_RUN=false
ARCHIVE_DAYS=14
MEMORY_BASE="${HOME}/.local/share/mimocode/memory"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --days) ARCHIVE_DAYS="$2"; shift 2 ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Memory Persistence${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Memory Statistics
# ─────────────────────────────────────────────────────────────────────────────
log_info "Memory Statistics"
echo ""

SESSIONS_DIR="$MEMORY_BASE/sessions"
PROJECTS_DIR="$MEMORY_BASE/projects"
GLOBAL_DIR="$MEMORY_BASE/global"

# Count sessions
TOTAL_SESSIONS=$(find "$SESSIONS_DIR" -maxdepth 1 -type d -name "ses_*" 2>/dev/null | wc -l)
TOTAL_CHECKPOINTS=$(find "$SESSIONS_DIR" -name "checkpoint.md" -type f 2>/dev/null | wc -l)
TOTAL_NOTES=$(find "$SESSIONS_DIR" -name "notes.md" -type f -size +2c 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$SESSIONS_DIR" 2>/dev/null | cut -f1)

log_stat "Sessions: $TOTAL_SESSIONS"
log_stat "Checkpoints: $TOTAL_CHECKPOINTS"
log_stat "Notes files (non-empty): $TOTAL_NOTES"
log_stat "Total size: $TOTAL_SIZE"
echo ""

# Count by age
RECENT_7D=$(find "$SESSIONS_DIR" -name "checkpoint.md" -type f -mtime -7 2>/dev/null | wc -l)
OLD_7_14D=$(find "$SESSIONS_DIR" -name "checkpoint.md" -type f -mtime +7 -mtime -14 2>/dev/null | wc -l)
OLD_14D=$(find "$SESSIONS_DIR" -name "checkpoint.md" -type f -mtime +14 2>/dev/null | wc -l)

log_stat "Last 7 days: $RECENT_7D"
log_stat "7-14 days: $OLD_7_14D"
log_stat ">14 days: $OLD_14D (candidates for archival)"
echo ""

# Project memory
if [[ -d "$PROJECTS_DIR" ]]; then
    PROJECT_COUNT=$(find "$PROJECTS_DIR" -maxdepth 1 -type d | wc -l)
    MEMORY_FILES=$(find "$PROJECTS_DIR" -name "MEMORY*.md" -type f 2>/dev/null | wc -l)
    log_stat "Projects: $((PROJECT_COUNT - 1))"
    log_stat "MEMORY files: $MEMORY_FILES"
fi

if [[ -f "$GLOBAL_DIR/MEMORY.md" ]]; then
    GLOBAL_SIZE=$(wc -c < "$GLOBAL_DIR/MEMORY.md")
    log_stat "Global MEMORY.md: ${GLOBAL_SIZE} bytes"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Archive Old Checkpoints
# ─────────────────────────────────────────────────────────────────────────────
log_info "Archiving checkpoints older than $ARCHIVE_DAYS days"

ARCHIVE_DIR="$SESSIONS_DIR/.archived"
OLD_CHECKPOINTS=$(find "$SESSIONS_DIR" -name "checkpoint.md" -type f -mtime +$ARCHIVE_DAYS 2>/dev/null)

if [[ -z "$OLD_CHECKPOINTS" ]]; then
    log_success "No old checkpoints to archive"
else
    COUNT=$(echo "$OLD_CHECKPOINTS" | wc -l)
    log_info "Found $COUNT checkpoint(s) to archive"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "$OLD_CHECKPOINTS" | while read -r f; do
            log_stat "Would archive: $(dirname "$f" | xargs basename)"
        done
    else
        mkdir -p "$ARCHIVE_DIR"
        echo "$OLD_CHECKPOINTS" | while read -r f; do
            SESSION_ID=$(dirname "$f" | xargs basename)
            # Move entire session directory to archive
            SESSION_DIR=$(dirname "$f")
            if [[ -d "$SESSION_DIR" ]]; then
                mv "$SESSION_DIR" "$ARCHIVE_DIR/" 2>/dev/null && \
                    log_success "Archived: $SESSION_ID" || \
                    log_warning "Failed to archive: $SESSION_ID"
            fi
        done
    fi
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract Durable Learnings
# ─────────────────────────────────────────────────────────────────────────────
log_info "Scanning for durable learnings in notes.md files"

LEARNINGS_FOUND=0
find "$SESSIONS_DIR" -name "notes.md" -type f -size +100c 2>/dev/null | while read -r notes_file; do
    SESSION_ID=$(echo "$notes_file" | grep -oP 'ses_[a-f0-9]+' | head -1)
    
    # Check if notes contain learnings markers
    if grep -qiE "(learning|discovered|pattern|reusable|always|never|fix:|bug:)" "$notes_file" 2>/dev/null; then
        LEARNINGS_FOUND=$((LEARNINGS_FOUND + 1))
        if [[ "$DRY_RUN" == true ]]; then
            log_stat "Would extract from: $SESSION_ID"
        else
            log_info "Durable learnings in: $SESSION_ID"
        fi
    fi
done

if [[ $LEARNINGS_FOUND -eq 0 ]]; then
    log_success "No new durable learnings found"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}  Dry Run Complete${NC}"
else
    echo -e "${GREEN}  Persistence Complete${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  - Review archived sessions: ls $ARCHIVE_DIR"
echo "  - Promote learnings: Edit $PROJECTS_DIR/*/MEMORY.md"
echo "  - Run with --dry-run first to preview changes"
