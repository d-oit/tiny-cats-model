#!/usr/bin/env bash
# harness/memory-share.sh
# Cross-project memory sharing for MiMoCode.
#
# Usage:
#   bash harness/memory-share.sh [options]
#
# Options:
#   --list              List all projects and their MEMORY files
#   --extract <project> Extract reusable patterns from a project
#   --suggest           Suggest which patterns might apply to current project
#   --copy <src> <dst>  Copy a pattern from one project to another

set -euo pipefail

# Colors (NC = no-color reset, used by all log_* functions)
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}▶ $1${NC}"; }
log_success() { echo -e "${GREEN}✓ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
log_found() { echo -e "${CYAN}  $1${NC}"; }

MEMORY_BASE="${HOME}/.local/share/mimocode/memory"
PROJECTS_DIR="$MEMORY_BASE/projects"

# Current project (tiny-cats-model)
CURRENT_PROJECT="ad632b7b-0b2a-4ed3-9ff0-cffc2ba8058b"

# Parse arguments
ACTION=""
ARG1=""
ARG2=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --list) ACTION="list"; shift ;;
        --extract) ACTION="extract"; ARG1="$2"; shift 2 ;;
        --suggest) ACTION="suggest"; shift ;;
        --copy) ACTION="copy"; ARG1="$2"; ARG2="$3"; shift 3 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo "Usage: bash harness/memory-share.sh [options]"
    echo ""
    echo "Options:"
    echo "  --list              List all projects and their MEMORY files"
    echo "  --extract <project> Extract reusable patterns from a project"
    echo "  --suggest           Suggest which patterns might apply to current project"
    echo "  --copy <src> <dst>  Copy a pattern from one project to another"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Cross-Project Memory Sharing${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. List Projects
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "list" ]]; then
    log_info "Listing all projects"
    echo ""
    
    find "$PROJECTS_DIR" -maxdepth 1 -type d | while read -r project_dir; do
        PROJECT_ID=$(basename "$project_dir")
        
        # Skip global
        if [[ "$PROJECT_ID" == "global" ]]; then
            continue
        fi
        
        # Find MEMORY files
        MEMORY_COUNT=$(find "$project_dir" -name "MEMORY*.md" -type f 2>/dev/null | wc -l)
        
        if [[ $MEMORY_COUNT -gt 0 ]]; then
            log_found "$PROJECT_ID ($MEMORY_COUNT MEMORY files)"
            
            # List MEMORY files
            find "$project_dir" -name "MEMORY*.md" -type f 2>/dev/null | while read -r mem_file; do
                MEM_NAME=$(basename "$mem_file")
                MEM_SIZE=$(wc -c < "$mem_file")
                echo -e "    $MEM_NAME ($MEM_SIZE bytes)"
            done
            echo ""
        fi
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract Reusable Patterns
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "extract" ]]; then
    PROJECT_ID="$ARG1"
    PROJECT_DIR="$PROJECTS_DIR/$PROJECT_ID"
    
    if [[ ! -d "$PROJECT_DIR" ]]; then
        log_warning "Project not found: $PROJECT_ID"
        exit 1
    fi
    
    log_info "Extracting reusable patterns from $PROJECT_ID"
    echo ""
    
    # Find main MEMORY.md
    MEMORY_FILE="$PROJECT_DIR/MEMORY.md"
    
    if [[ ! -f "$MEMORY_FILE" ]]; then
        log_warning "No MEMORY.md found in $PROJECT_ID"
        exit 1
    fi
    
    # Extract patterns (lines with keywords)
    PATTERNS=$(grep -iE "(pattern|reusable|always|never|fix:|bug:|discovered|learning)" "$MEMORY_FILE" 2>/dev/null | head -20)
    
    if [[ -n "$PATTERNS" ]]; then
        log_found "Reusable patterns found:"
        echo "$PATTERNS" | while read -r line; do
            echo -e "    $line"
        done
    else
        log_warning "No reusable patterns found"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Suggest Patterns for Current Project
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "suggest" ]]; then
    log_info "Suggesting patterns for current project"
    echo ""
    
    CURRENT_MEMORY="$PROJECTS_DIR/$CURRENT_PROJECT/MEMORY.md"
    
    if [[ ! -f "$CURRENT_MEMORY" ]]; then
        log_warning "No MEMORY.md found for current project"
        exit 1
    fi
    
    # Get current project's topics
    CURRENT_TOPICS=$(grep -iE "(Modal|training|DiT|checkpoint|GPU)" "$CURRENT_MEMORY" 2>/dev/null | head -10)
    
    if [[ -n "$CURRENT_TOPICS" ]]; then
        log_found "Current project focuses on:"
        echo "$CURRENT_TOPICS" | while read -r line; do
            echo -e "    $line"
        done
        echo ""
    fi
    
    # Search other projects for related patterns
    log_found "Related patterns from other projects:"
    echo ""
    
    find "$PROJECTS_DIR" -name "MEMORY*.md" -type f 2>/dev/null | grep -v "$CURRENT_PROJECT" | while read -r mem_file; do
        PROJECT_ID=$(echo "$mem_file" | grep -oP '[a-f0-9-]{36}' | head -1)
        
        # Search for Modal, training, CI/CD patterns
        RESULTS=$(grep -iE "(Modal|training|CI/CD|GPU|checkpoint|deployment)" "$mem_file" 2>/dev/null | head -3)
        
        if [[ -n "$RESULTS" ]]; then
            log_found "$PROJECT_ID:"
            echo "$RESULTS" | while read -r line; do
                echo -e "    $line"
            done
            echo ""
        fi
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Copy Pattern
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "copy" ]]; then
    SRC_PROJECT="$ARG1"
    DST_PROJECT="$ARG2"
    
    SRC_DIR="$PROJECTS_DIR/$SRC_PROJECT"
    DST_DIR="$PROJECTS_DIR/$DST_PROJECT"
    
    if [[ ! -d "$SRC_DIR" ]]; then
        log_warning "Source project not found: $SRC_PROJECT"
        exit 1
    fi
    
    if [[ ! -d "$DST_DIR" ]]; then
        log_warning "Destination project not found: $DST_PROJECT"
        exit 1
    fi
    
    log_info "Copying patterns from $SRC_PROJECT to $DST_PROJECT"
    echo ""
    
    # Find reusable patterns in source
    SRC_MEMORY="$SRC_DIR/MEMORY.md"
    
    if [[ ! -f "$SRC_MEMORY" ]]; then
        log_warning "No MEMORY.md found in source project"
        exit 1
    fi
    
    # Extract patterns
    PATTERNS=$(grep -iE "(pattern|reusable|always|never|fix:|bug:|discovered|learning)" "$SRC_MEMORY" 2>/dev/null | head -10)
    
    if [[ -n "$PATTERNS" ]]; then
        log_found "Patterns to copy:"
        echo "$PATTERNS" | while read -r line; do
            echo -e "    $line"
        done
        echo ""
        
        # Append to destination MEMORY.md
        DST_MEMORY="$DST_DIR/MEMORY.md"

        if [[ -f "$DST_MEMORY" ]]; then
            {
                echo ""
                echo "## Cross-project patterns (from $SRC_PROJECT)"
                echo "_Auto-copied on $(date +%Y-%m-%d)_"
                echo ""
                echo "$PATTERNS"
            } >> "$DST_MEMORY"

            log_success "Patterns copied to $DST_PROJECT/MEMORY.md"
        fi
    else
        log_warning "No reusable patterns found in source project"
    fi
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
