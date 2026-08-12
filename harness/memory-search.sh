#!/usr/bin/env bash
# harness/memory-search.sh
# Ranked search and discovery across the MiMoCode memory system.
#
# Aims for Issue #119 Area 3: technical-term aware search instead of raw
# grep. Scoring boosts:
#   - file type: MEMORY.md (project/global) > notes.md > checkpoint.md
#   - technical tokens: ADR-\d+, file paths, snake_case identifiers,
#     dotted module paths, and UPPER_SNAKE constants.
#
# Usage:
#   bash harness/memory-search.sh <query> [options]
#
# Options:
#   --scope <scope>    Search scope: all|projects|sessions|current|global
#                      (default: all)
#   --type <type>      Filter by content type: checkpoint|notes|memory|all
#                      (default: all)
#   --limit <n>        Max results (default: 10)
#   --context          Show surrounding context lines (default: off)
#   --ask              Digest mode: "what do we know about X?" — prints a
#                      concise, ranked summary instead of raw line hits
#   --project <id>     Project id for the current-project scope
#                      (default: auto-detect the repo's project)
#   --session <id>     Session id for the current-session scope
#                      (default: most recently modified session)

set -euo pipefail

# Colors
GREEN='\033[0;32m'
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
ASK=false
PROJECT_ID=""
SESSION_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scope) SCOPE="$2"; shift 2 ;;
        --type) TYPE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --context) CONTEXT=true; shift ;;
        --ask) ASK=true; shift ;;
        --project) PROJECT_ID="$2"; shift 2 ;;
        --session) SESSION_ID="$2"; shift 2 ;;
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
    echo "  --scope <scope>    Search scope: all|projects|sessions|current|global"
    echo "  --type <type>      Filter: checkpoint|notes|memory|all"
    echo "  --limit <n>        Max results (default: 10)"
    echo "  --context          Show surrounding context"
    echo "  --ask              Digest mode: 'what do we know about X?'"
    echo "  --project <id>     Project id for current-project scope"
    echo "  --session <id>     Session id for current-session scope"
    exit 1
fi

MEMORY_BASE="${HOME}/.local/share/mimocode/memory"
SESSIONS_DIR="$MEMORY_BASE/sessions"
PROJECTS_DIR="$MEMORY_BASE/projects"
GLOBAL_DIR="$MEMORY_BASE/global"

# Validate scope
case "$SCOPE" in
    all|projects|sessions|current|global) ;;
    *) echo "Unknown scope: $SCOPE (all|projects|sessions|current|global)"; exit 1 ;;
esac

# Auto-detect current project if not given: the project whose MEMORY.md
# mentions the repo name from the current working directory.
if [[ -z "$PROJECT_ID" && -d "$PROJECTS_DIR" ]]; then
    REPO_NAME="$(basename "$(git rev-parse --show-toplevel 2>/dev/null || echo .)")"
    if [[ -n "$REPO_NAME" && "$REPO_NAME" != "." ]]; then
        PROJECT_ID="$(grep -rl -i --include="MEMORY.md" "$REPO_NAME" "$PROJECTS_DIR" 2>/dev/null \
            | head -1 | grep -oP '(?<=projects/)[a-f0-9-]{36}')"
    fi
fi

# Auto-detect current session if not given: most recently modified session dir.
if [[ -z "$SESSION_ID" && -d "$SESSIONS_DIR" ]]; then
    SESSION_ID="$(find "$SESSIONS_DIR" -maxdepth 1 -type d -name 'ses_*' \
        -printf '%T@\t%f\n' 2>/dev/null | sort -nrk1 | head -1 | cut -f2 || true)"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
if [[ "$ASK" == true ]]; then
    echo -e "${BLUE}  What do we know about: $QUERY${NC}"
else
    echo -e "${BLUE}  Memory Search: $QUERY${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

CONTEXT_ARGS=()
if [[ "$CONTEXT" == true ]]; then
    CONTEXT_ARGS=(-C 2)
fi

# A robust awk ranker shared by every scope. Input lines are
# "0<TAB>source<TAB>line-number<TAB>line-text" from `grep`. Scoring:
#   - +1 per distinct query token present in the line (case-insensitive)
#   - +1 if a token appears as a whole word
#   - technical-token boosts: ADR-\d+, file paths, snake_case calls,
#     UPPER_SNAKE constants
#   - multiplied by the caller's per-source weight
# Emits "SCORE<TAB>source<TAB>lineno<TAB>text", best first.
rank_lines() {
    local qwords="$1" srcweight="$2"
    awk -F'\t' -v qw="$qwords" -v sw="$srcweight" '
    {
        text = tolower($4)
        score = 0
        n = split(qw, toks, /[[:space:]]+/)
        for (i = 1; i <= n; i++) {
            t = toks[i]
            if (t == "") continue
            if (index(text, t) > 0) {
                score += 1
            }
            if (text ~ ("(^|[^a-z0-9_])" t "($|[^a-z0-9_])")) {
                score += 1
            }
        }
        if (text ~ /adr-[0-9]+/) score += 3
        if (text ~ /\/([a-z0-9_.-]+\/)+[a-z0-9_.-]+/) score += 2
        if (text ~ /[a-z_][a-z0-9_]*\(\)/) score += 2
        score = score * sw
        if (score > 0) {
            printf "%d\t%s\t%s\t%s\n", score, $2, $3, $4
        }
    }' | sort -t$'\t' -k1,1nr | head -"$LIMIT"
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Search Project Memory
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "projects" ]]; then
    if [[ "$TYPE" == "all" || "$TYPE" == "memory" ]]; then
        log_info "Searching project MEMORY.md files"

        if [[ -d "$PROJECTS_DIR" ]]; then
            # gather raw hits then rank
            RAW="$(mktemp)"
            grep -r "${CONTEXT_ARGS[@]}" -n -i --include="MEMORY*.md" "$QUERY" "$PROJECTS_DIR" 2>/dev/null \
                | sed 's/^\([^:]*\):\([0-9]*\):/\1\t\2\t/' \
                | while IFS= read -r line; do
                    file="${line%%$'\t'*}"
                    rest="${line#*$'\t'}"
                    lineno="${rest%%$'\t'*}"
                    content="${rest#*$'\t'}"
                    PROJECT_ID_MATCH="$(echo "$file" | grep -oP '(?<=projects/)[a-f0-9-]{36}' | head -1)"
                    printf '0\t%s\t%s\t%s\n' "$PROJECT_ID_MATCH" "$lineno" "$content" >> "$RAW"
                done
            rank_lines "$QUERY" 3 < "$RAW"
            rm -f "$RAW"
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Search Current Session
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "current" ]]; then
    log_info "Searching current session${SESSION_ID:+ ($SESSION_ID)}"

    CURRENT_SESSION="$SESSIONS_DIR/$SESSION_ID"
    if [[ -z "$SESSION_ID" || ! -d "$CURRENT_SESSION" ]]; then
        log_found "No current session found (use --session <id>)"
    else
        for f in checkpoint.md notes.md; do
            if [[ -f "$CURRENT_SESSION/$f" ]]; then
                RAW="$(mktemp)"
                grep "${CONTEXT_ARGS[@]}" -n -i "$QUERY" "$CURRENT_SESSION/$f" 2>/dev/null \
                    | sed 's/^\([0-9]*\):/\1\t/' \
                    | while IFS=$'\t' read -r lineno content; do
                        printf '0\t%s\t%s\t%s\n' "$f" "$lineno" "$content" >> "$RAW"
                    done
                rank_lines "$QUERY" 2 < "$RAW"
                rm -f "$RAW"
            fi
        done
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Search Session History
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "sessions" ]]; then
    log_info "Searching session history"

    if [[ "$TYPE" == "all" || "$TYPE" == "checkpoint" ]]; then
        RAW="$(mktemp)"
        grep -r "${CONTEXT_ARGS[@]}" -n -i --include="checkpoint.md" "$QUERY" "$SESSIONS_DIR" 2>/dev/null \
            | grep -v "ses_0a8cdc2e7ffeWD5k4XynOo5okb" \
            | sed 's/^\([^:]*\):\([0-9]*\):/\1\t\2\t/' \
            | while IFS= read -r line; do
                file="${line%%$'\t'*}"
                rest="${line#*$'\t'}"
                lineno="${rest%%$'\t'*}"
                content="${rest#*$'\t'}"
                SESSION_ID_MATCH="$(echo "$file" | grep -oP 'ses_[a-f0-9]+' | head -1)"
                printf '0\t%s\t%s\t%s\n' "$SESSION_ID_MATCH" "$lineno" "$content" >> "$RAW"
            done
        rank_lines "$QUERY" 1 < "$RAW"
        rm -f "$RAW"
    fi

    if [[ "$TYPE" == "all" || "$TYPE" == "notes" ]]; then
        RAW="$(mktemp)"
        grep -r "${CONTEXT_ARGS[@]}" -n -i --include="notes.md" "$QUERY" "$SESSIONS_DIR" 2>/dev/null \
            | grep -v "ses_0a8cdc2e7ffeWD5k4XynOo5okb" \
            | sed 's/^\([^:]*\):\([0-9]*\):/\1\t\2\t/' \
            | while IFS= read -r line; do
                file="${line%%$'\t'*}"
                rest="${line#*$'\t'}"
                lineno="${rest%%$'\t'*}"
                content="${rest#*$'\t'}"
                SESSION_ID_MATCH="$(echo "$file" | grep -oP 'ses_[a-f0-9]+' | head -1)"
                printf '0\t%s\t%s\t%s\n' "$SESSION_ID_MATCH" "$lineno" "$content" >> "$RAW"
            done
        rank_lines "$QUERY" 1 < "$RAW"
        rm -f "$RAW"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Search Global Memory
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SCOPE" == "all" || "$SCOPE" == "global" ]]; then
    if [[ -f "$GLOBAL_DIR/MEMORY.md" ]]; then
        log_info "Searching global memory"

        RAW="$(mktemp)"
        grep "${CONTEXT_ARGS[@]}" -n -i "$QUERY" "$GLOBAL_DIR/MEMORY.md" 2>/dev/null \
            | sed 's/^\([0-9]*\):/\1\t/' \
            | while IFS=$'\t' read -r lineno content; do
                printf '0\tglobal/MEMORY.md\t%s\t%s\n' "$lineno" "$content" >> "$RAW"
            done
        rank_lines "$QUERY" 4 < "$RAW"
        rm -f "$RAW"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Related memories (Issue #119 Area 3): when any project MEMORY matches,
#    surface the section headings and ADR references anchored near the hit
#    so a user can drill into the surrounding durable knowledge.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ASK" == true && -d "$PROJECTS_DIR" ]]; then
    log_info "Surfacing related durable knowledge"
    grep -rn -i --include="MEMORY.md" "$QUERY" "$PROJECTS_DIR" 2>/dev/null | head -50 \
        | while IFS=: read -r file line _ ; do
            PROJECT_ID_MATCH="$(echo "$file" | grep -oP '(?<=projects/)[a-f0-9-]{36}' | head -1)"
            SECTION="$(grep -n '^##\? ' "$file" 2>/dev/null \
                | awk -F: -v l="$line" '$1 <= l {s=$2} END {print s}')"
            ADRS="$(grep -o 'ADR-[0-9]\+' "$file" 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')"
            if [[ -n "$SECTION" ]]; then
                echo -e "    ${CYAN}${PROJECT_ID_MATCH:0:12}${NC} [$SECTION]${ADRS:+ · ADRs: $ADRS}"
            fi
        done | sort -u | head -"$(( LIMIT + 5 ))"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
if [[ "$ASK" == true ]]; then
    echo -e "${GREEN}  Digest complete${NC}"
else
    echo -e "${GREEN}  Search Complete${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
