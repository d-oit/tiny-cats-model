#!/usr/bin/env bash
# harness/memory-search.sh
# Search and discover patterns across the MiMoCode memory system.
#
# Usage:
#   bash harness/memory-search.sh <query> [options]
#
# Options:
#   --scope <scope>     Search scope: all|projects|sessions|current (default: all)
#   --type <type>       Filter by content type: checkpoint|notes|memory|all (default: all)
#   --limit <n>         Max results per category (default: 10)
#   --context           Show surrounding context lines (default: off)

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
log_found() { echo -e "${CYAN}  $1${NC}"; }

# Defaults
QUERY=""
SCOPE="all"
TYPE="all"
LIMIT=10
CONTEXT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scope) SCOPE="$2"; shift 2 ;;
        --type) TYPE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --context) CONTEXT=true; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$QUERY" ]]; then
                QUERY="$1"
            else
                QUERY="$QUERY $1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$QUERY" ]]; then
    echo "Usage: bash harness/memory-search.sh <query> [options]"
    echo ""
    echo "Options:"
    echo "  --scope <scope>    Search scope: all|projects|sessions|current"
    echo "  --type <type>      Filter: checkpoint|notes|memory|all"
    echo "  --limit <n>        Max results per category (default: 10)"
    echo "  --context          Show surrounding context"
    exit 1
fi

MEMORY_BASE="${HOME}/.local/share/mimocode/memory"
SESSIONS_DIR="$MEMORY_BASE/sessions"
PROJECTS_DIR="$MEMORY_BASE/projects"
GLOBAL_DIR="$MEMORY_BASE/global"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Memory Search: $QUERY${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL_HITS=0
CONTEXT_FLAG=""
if [[ "$CONTEXT" == true ]]; then
    CONTEXT_FLAG="-C 2"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. Search Project Memory
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "projects" ]]; then
    if [[ "$TYPE" == "all" || "$TYPE" == "memory" ]]; then
        log_info "Searching project MEMORY.md files"
        
        if [[ -d "$PROJECTS_DIR" ]]; then
            # Use grep directly for speed
            grep -r $CONTEXT_FLAG -n -i --include="MEMORY*.md" "$QUERY" "$PROJECTS_DIR" 2>/dev/null | head -$((LIMIT * 5)) | while IFS=: read -r file line content; do
                PROJECT_ID=$(echo "$file" | grep -oP '[a-f0-9-]{36}' | head -1)
                echo -e "    ${CYAN}$PROJECT_ID${NC}:$line: $content"
                TOTAL_HITS=$((TOTAL_HITS + 1))
            done
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Search Current Session
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "current" ]]; then
    log_info "Searching current session"
    
    CURRENT_SESSION="$SESSIONS_DIR/ses_0a8cdc2e7ffeWD5k4XynOo5okb"
    if [[ -d "$CURRENT_SESSION" ]]; then
        for f in checkpoint.md notes.md; do
            if [[ -f "$CURRENT_SESSION/$f" ]]; then
                RESULTS=$(grep $CONTEXT_FLAG -n -i "$QUERY" "$CURRENT_SESSION/$f" 2>/dev/null | head -$LIMIT)
                if [[ -n "$RESULTS" ]]; then
                    MATCH_COUNT=$(echo "$RESULTS" | wc -l)
                    TOTAL_HITS=$((TOTAL_HITS + MATCH_COUNT))
                    echo ""
                    log_found "$f ($MATCH_COUNT hits)"
                    echo "$RESULTS" | while read -r line; do
                        echo -e "    $line"
                    done
                fi
            fi
        done
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Search Session History
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "sessions" ]]; then
    log_info "Searching session history"
    
    # Use grep directly for speed
    if [[ "$TYPE" == "all" || "$TYPE" == "checkpoint" ]]; then
        grep -r $CONTEXT_FLAG -n -i --include="checkpoint.md" "$QUERY" "$SESSIONS_DIR" 2>/dev/null | grep -v "ses_0a8cdc2e7ffeWD5k4XynOo5okb" | head -$((LIMIT * 3)) | while IFS=: read -r file line content; do
            SESSION_ID=$(echo "$file" | grep -oP 'ses_[a-f0-9]+' | head -1)
            echo -e "    ${CYAN}${SESSION_ID}${NC}:$line: $content"
        done
    fi
    
    if [[ "$TYPE" == "all" || "$TYPE" == "notes" ]]; then
        grep -r $CONTEXT_FLAG -n -i --include="notes.md" "$QUERY" "$SESSIONS_DIR" 2>/dev/null | grep -v "ses_0a8cdc2e7ffeWD5k4XynOo5okb" | head -$((LIMIT * 3)) | while IFS=: read -r file line content; do
            SESSION_ID=$(echo "$file" | grep -oP 'ses_[a-f0-9]+' | head -1)
            echo -e "    ${CYAN}${SESSION_ID}${NC}:$line: $content"
        done
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Search Global Memory
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "projects" ]]; then
    if [[ -f "$GLOBAL_DIR/MEMORY.md" ]]; then
        log_info "Searching global memory"
        
        RESULTS=$(grep $CONTEXT_FLAG -n -i "$QUERY" "$GLOBAL_DIR/MEMORY.md" 2>/dev/null | head -$LIMIT)
        if [[ -n "$RESULTS" ]]; then
            MATCH_COUNT=$(echo "$RESULTS" | wc -l)
            TOTAL_HITS=$((TOTAL_HITS + MATCH_COUNT))
            
            echo ""
            log_found "global/MEMORY.md ($MATCH_COUNT hits)"
            echo "$RESULTS" | while read -r line; do
                echo -e "    $line"
            done
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Search Complete: $TOTAL_HITS total matches${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
