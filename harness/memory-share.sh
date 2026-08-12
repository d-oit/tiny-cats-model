#!/usr/bin/env bash
# harness/memory-share.sh
# Cross-project memory sharing for MiMoCode (Issue #119 Area 4).
#
# Each MiMoCode project dir keeps MEMORY*.md files. This tool extracts
# reusable patterns and ADR references from them, builds a persistent
# cross-project "learnings library" in the global store, and can suggest /
# copy patterns between projects.
#
# Usage:
#   bash harness/memory-share.sh [options]
#
# Options:
#   --list              List all projects and their MEMORY files
#   --extract <project> Show reusable patterns + ADR refs from a project
#   --learnings         Rebuild the cross-project learnings library
#                       (~/.local/share/mimocode/user/learnings.md)
#   --suggest           Show likely-relevant patterns for the current project
#   --copy <src> <dst>  Append patterns from one project MEMORY to another
#   --project <id>      Project id for the current-project scope
#                       (default: auto-detect the repo's project)

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
LEARNINGS_FILE="${HOME}/.local/share/mimocode/user/learnings.md"

# Current project (tiny-cats-model) — auto-detect from repo name if possible,
# else default to the small, curated id used across the harness.
CURRENT_PROJECT="ad632b7b-0b2a-4ed3-9ff0-cffc2ba8058b"
if [[ -d "$PROJECTS_DIR" ]]; then
    REPO_NAME="$(basename "$(git rev-parse --show-toplevel 2>/dev/null || echo .)")"
    FOUND_ID="$(grep -rl -i --include="MEMORY.md" "$REPO_NAME" "$PROJECTS_DIR" 2>/dev/null \
        | head -1 | grep -oP '(?<=projects/)[a-f0-9-]{36}' || true)"
    if [[ -n "$FOUND_ID" ]]; then
        CURRENT_PROJECT="$FOUND_ID"
    fi
fi

# Parse arguments
ACTION=""
ARG1=""
ARG2=""
PROJECT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --list) ACTION="list"; shift ;;
        --extract) ACTION="extract"; ARG1="$2"; shift 2 ;;
        --learnings) ACTION="learnings"; shift ;;
        --suggest) ACTION="suggest"; shift ;;
        --copy) ACTION="copy"; ARG1="$2"; ARG2="$3"; shift 3 ;;
        --project) PROJECT_OVERRIDE="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "$PROJECT_OVERRIDE" ]]; then
    CURRENT_PROJECT="$PROJECT_OVERRIDE"
fi

if [[ -z "$ACTION" ]]; then
    echo "Usage: bash harness/memory-share.sh [options]"
    echo ""
    echo "Options:"
    echo "  --list              List all projects and their MEMORY files"
    echo "  --extract <project> Extract reusable patterns from a project"
    echo "  --learnings         Rebuild the cross-project learnings library"
    echo "  --suggest           Suggest patterns for the current project"
    echo "  --copy <src> <dst>  Copy a pattern from one project to another"
    echo "  --project <id>      Override current project id"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Cross-Project Memory Sharing${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Extract reusable patterns from a project's MEMORY files: lines flagged with
# durable/pattern keywords plus any ADR references found in them.
extract_patterns() {
    local project="$1"
    local pdir="$PROJECTS_DIR/$project"

    if [[ ! -d "$pdir" ]]; then
        log_warning "Project not found: $project"
        return 1
    fi

    local files=()
    while IFS= read -r f; do
        files+=("$f")
    done < <(find "$pdir" -maxdepth 1 -name 'MEMORY*.md' -type f 2>/dev/null | sort)

    if [[ ${#files[@]} -eq 0 ]]; then
        log_warning "No MEMORY files in $project"
        return 1
    fi

    # Pattern-flagged content lines (exclude markdown headers + the
    # italic instruction lines the memory writer embeds), dedup, strip
    # leading "- "* noise.
    grep -hiE "(pattern|reusable|always|never|fix:|bug:|discovered|learning)" \
        "${files[@]}" 2>/dev/null \
        | grep -vE '^\s*#{1,6}\s|^_[A-Z]|_Edit only|_Durable|_What is' \
        | sed 's/^[[:space:]]*[-*] //' | sed 's/^[[:space:]]*//' | sort -u
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. List Projects
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "list" ]]; then
    log_info "Listing all projects"
    echo ""

    if [[ ! -d "$PROJECTS_DIR" ]]; then
        log_warning "No projects dir at $PROJECTS_DIR"
        exit 0
    fi

    while IFS= read -r project_dir; do
        PROJECT_ID="$(basename "$project_dir")"
        [[ "$PROJECT_ID" == "global" ]] && continue

        MEMORY_COUNT="$(find "$project_dir" -maxdepth 1 -name 'MEMORY*.md' -type f 2>/dev/null | wc -l)"
        ADR_COUNT="$(grep -rho 'ADR-[0-9]\+' "$project_dir" --include='MEMORY*.md' 2>/dev/null | sort -u | wc -l || true)"

        if [[ "$MEMORY_COUNT" -gt 0 ]]; then
            MARK=""
            [[ "$PROJECT_ID" == "$CURRENT_PROJECT" ]] && MARK=" (current)"
            log_found "$PROJECT_ID ($MEMORY_COUNT MEMORY files, $ADR_COUNT ADRs)$MARK"
            while IFS= read -r mem_file; do
                MEM_NAME="$(basename "$mem_file")"
                MEM_SIZE="$(wc -c < "$mem_file")"
                echo -e "    $MEM_NAME ($MEM_SIZE bytes)"
            done < <(find "$project_dir" -maxdepth 1 -name 'MEMORY*.md' -type f 2>/dev/null | sort)
            echo ""
        fi
    done < <(find "$PROJECTS_DIR" -maxdepth 1 -type d 2>/dev/null | sort)
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract Reusable Patterns
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "extract" ]]; then
    PROJECT_ID="$ARG1"
    log_info "Extracting reusable patterns + ADRs from $PROJECT_ID"
    echo ""

    if ! PATTERNS="$(extract_patterns "$PROJECT_ID")"; then
        exit 1
    fi

    echo -e "${CYAN}Reusable patterns:${NC}"
    sed -n '1,20p' <<< "$PATTERNS"
    echo ""

    ADRS="$(grep -rho 'ADR-[0-9]\+' "$PROJECTS_DIR/$PROJECT_ID" --include='MEMORY*.md' 2>/dev/null | sort -u | tr '\n' ' ')"
    echo -e "${CYAN}ADR references:${NC} ${ADRS:-none}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Rebuild the cross-project learnings library
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "learnings" ]]; then
    log_info "Rebuilding cross-project learnings library → $LEARNINGS_FILE"
    echo ""

    mkdir -p "$(dirname "$LEARNINGS_FILE")"

    # Header
    {
        echo "# Cross-project learnings library (auto-generated)"
        echo "_Built $(date -u +%Y-%m-%dT%H:%M:%SZ) by harness/memory-share.sh — do not edit by hand._"
        echo ""
    } > "$LEARNINGS_FILE"

    FOUND_ANY=false
    while IFS= read -r project_dir; do
        PROJECT_ID="$(basename "$project_dir")"
        [[ "$PROJECT_ID" == "global" ]] && continue

        if ! PATTERNS="$(extract_patterns "$PROJECT_ID")"; then
            continue
        fi

        FOUND_ANY=true
        {
            echo "## Project: $PROJECT_ID"
            echo ""
            sed -n '1,15p' <<< "$PATTERNS"
            echo ""
        } >> "$LEARNINGS_FILE"
    done < <(find "$PROJECTS_DIR" -maxdepth 1 -type d 2>/dev/null | sort)

    if [[ "$FOUND_ANY" == false ]]; then
        log_warning "No reusable patterns found in any project"
        rm -f "$LEARNINGS_FILE"
        exit 0
    fi

    log_success "Learnings library written: $LEARNINGS_FILE"
    echo -e "    $(wc -l < "$LEARNINGS_FILE") lines across $(grep -c '^## Project:' "$LEARNINGS_FILE") projects"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Suggest Patterns for Current Project
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "suggest" ]]; then
    log_info "Suggesting patterns for current project ($CURRENT_PROJECT)"
    echo ""

    CURRENT_MEMORY="$PROJECTS_DIR/$CURRENT_PROJECT/MEMORY.md"

    if [[ ! -f "$CURRENT_MEMORY" ]]; then
        log_warning "No MEMORY.md found for current project $CURRENT_PROJECT"
        exit 1
    fi

    # Current project's own durable technical vocabulary: ADRs + key tokens.
    CURRENT_ADRS="$(grep -ho 'ADR-[0-9]\+' "$CURRENT_MEMORY" 2>/dev/null | sort -u)"
    log_found "Current project ADRs: $(echo "$CURRENT_ADRS" | tr '\n' ' ' | sed 's/ $//')"
    echo ""

    log_found "Related patterns from other projects:"
    echo ""

    FOUND=false
    while IFS= read -r mem_file; do
        PROJECT_ID="$(echo "$mem_file" | grep -oP '(?<=projects/)[a-f0-9-]{36}' | head -1 || true)"
        [[ "$PROJECT_ID" == "$CURRENT_PROJECT" ]] && continue
        [[ -z "$PROJECT_ID" ]] && continue

        # Score each other project's MEMORY by overlap with current ADRs.
        OVERLAP="$(grep -ho 'ADR-[0-9]\+' "$mem_file" 2>/dev/null \
            | sort -u | grep -cE "$(echo "$CURRENT_ADRS" | paste -sd'|' -)" || true)"
        if [[ "${OVERLAP:-0}" -gt 0 ]]; then
            FOUND=true
            log_found "$PROJECT_ID (shares $(grep -ho 'ADR-[0-9]\+' "$mem_file" 2>/dev/null | sort -u | grep -cE "$(echo "$CURRENT_ADRS" | paste -sd'|' -)") ADR topics)"
            grep -iE "$(echo "$CURRENT_ADRS" | paste -sd'|' -)" "$mem_file" 2>/dev/null | head -3 \
                | sed 's/^/    /'
            echo ""
        fi
    done < <(find "$PROJECTS_DIR" -name 'MEMORY*.md' -type f 2>/dev/null | sort)

    if [[ "$FOUND" == false ]]; then
        log_warning "No cross-project ADR overlap found yet"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Copy Pattern
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

    log_info "Copying reusable patterns from $SRC_PROJECT to $DST_PROJECT"
    echo ""

    if ! PATTERNS="$(extract_patterns "$SRC_PROJECT")"; then
        exit 1
    fi

    DST_MEMORY="$DST_DIR/MEMORY.md"
    if [[ ! -f "$DST_MEMORY" ]]; then
        log_warning "No MEMORY.md in destination; creating one"
        mkdir -p "$DST_DIR"
        { echo "# Project memory"; echo ""; } > "$DST_MEMORY"
    fi

    {
        echo ""
        echo "## Cross-project patterns (from $SRC_PROJECT)"
        echo "_Auto-copied on $(date +%Y-%m-%d)_"
        echo ""
        sed -n '1,10p' <<< "$PATTERNS"
    } >> "$DST_MEMORY"

    log_success "Patterns appended to $DST_PROJECT/MEMORY.md"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
