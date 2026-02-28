#!/bin/bash
# .agents/skills/token_safe_exec.sh
# Usage: ./token_safe_exec.sh "cargo test"

COMMAND=$1
if [ -z "$COMMAND" ]; then
  echo "Usage: token_safe_exec.sh \"<command>\""
  exit 1
fi

TEMP_OUT=$(mktemp)

# Run the command and capture both stdout and stderr
eval "$COMMAND" > "$TEMP_OUT" 2>&1
EXIT_CODE=$?

LINE_COUNT=$(wc -l < "$TEMP_OUT")

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: '$COMMAND'"
    echo "Output suppressed to save tokens. ($LINE_COUNT lines generated)."
else
    echo "❌ FAILED: '$COMMAND' (Exit Code: $EXIT_CODE)"
    if [ "$LINE_COUNT" -gt 50 ]; then
        echo "--- OUTPUT TRUNCATED (Showing first 20 and last 30 lines) ---"
        head -n 20 "$TEMP_OUT"
        echo
        echo "... [TRUNCATED $LINE_COUNT LINES TO SAVE TOKENS] ..."
        echo
        tail -n 30 "$TEMP_OUT"
    else
        cat "$TEMP_OUT"
    fi
fi

rm "$TEMP_OUT"
exit $EXIT_CODE
