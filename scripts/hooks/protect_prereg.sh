#!/usr/bin/env bash
# Blocks edits to PREREGISTRATION.md once gates have been run.
INPUT=$(cat)
if echo "$INPUT" | grep -q "PREREGISTRATION.md"; then
  if [ -s results/GATE_LOG.md ]; then
    echo "BLOCKED: PREREGISTRATION.md is frozen because gates have already run." >&2
    echo "Amending pre-registered claims after seeing results invalidates them." >&2
    exit 2
  fi
fi
exit 0
