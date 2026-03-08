export PATH="/opt/homebrew/bin:$PATH"

#!/bin/bash
set -euo pipefail

# ---- config ----
OSASCRIPT="/usr/bin/osascript"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
MAIN_PY="$SCRIPT_DIR/src/main.py"
DEBUG_LOG="$HOME/mlx-srt-debug.log"
AUTO_LOG="$SCRIPT_DIR/mlx-srt-automator-debug.log"

# ---- logging ----
exec 2>>"$AUTO_LOG"
echo "===== $(date) =====" >>"$DEBUG_LOG"
echo "ARGS COUNT: $#" >>"$DEBUG_LOG"
echo "ARGS: $@" >>"$DEBUG_LOG"

# ---- guard ----
[ "$#" -eq 0 ] && exit 0

cd "$SCRIPT_DIR" || exit 1

# ---- process files ----
for f in "$@"; do
  BASENAME="$(basename "$f")"

  "$OSASCRIPT" -e "display notification \"Starting: $BASENAME\" with title \"MLX-SRT\""

  TEMP_LOG="$(mktemp)"
  if "$VENV_PY" -u "$MAIN_PY" "$f" >"$TEMP_LOG" 2>&1; then
    "$OSASCRIPT" -e "display notification \"Completed: $BASENAME\" with title \"MLX-SRT\""
    rm -f "$TEMP_LOG"
  else
    ERR_LOG="${f%.*}_error.log"
    mv "$TEMP_LOG" "$ERR_LOG"
    "$OSASCRIPT" -e "display notification \"Failed: $BASENAME\" with title \"MLX-SRT\" sound name \"Basso\""
  fi
done
