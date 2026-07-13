#!/bin/zsh
# Launched nightly by launchd (~/Library/LaunchAgents/com.harmonia.nightly-agent.plist).
# launchd gives us a bare environment (no .zshrc sourced), so PATH is set explicitly here.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH:/usr/bin:/bin:/usr/sbin:/sbin"

REPO_DIR="/Users/vincente/Documents/Projets Perso/Code/harmonia"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/nightly_agent.log"
PROMPT_FILE="$REPO_DIR/scripts/nightly_agent_prompt.txt"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR" || { echo "$(date -u +%FT%TZ) FATAL: could not cd to $REPO_DIR" >> "$LOG_FILE"; exit 1; }

echo "===== $(date -u +%FT%TZ) nightly agent run starting =====" >> "$LOG_FILE"

claude -p "$(cat "$PROMPT_FILE")" \
  --dangerously-skip-permissions \
  --max-budget-usd 15 \
  --output-format text \
  --no-session-persistence \
  >> "$LOG_FILE" 2>&1

echo "===== $(date -u +%FT%TZ) nightly agent run finished (exit $?) =====" >> "$LOG_FILE"
