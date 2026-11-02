#!/usr/bin/env bash
# scripts/check-commitlint.sh
#
# Adapter required by the commit-msg hook that `do-harness hook install`
# generates. `do-harness init` seeds an invariant referencing this path but
# never scaffolds it, so every commit was blocked (d-o-hub/do-harness#420).
#
# Delegate to the repository's existing validator instead of duplicating commit
# rules: scripts/commit-msg-hook.py already owns them.
#
# Interface expected by the generated hook:
#   bash scripts/check-commitlint.sh --message <commit-msg-file>

set -euo pipefail

message_file=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --message)
            message_file="${2:-}"
            shift 2
            ;;
        *)
            message_file="$1"
            shift
            ;;
    esac
done

if [[ -z "$message_file" ]]; then
    echo "usage: check-commitlint.sh --message <commit-msg-file>" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$root/scripts/commit-msg-hook.py" "$message_file"
