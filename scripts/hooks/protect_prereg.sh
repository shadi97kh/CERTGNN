#!/usr/bin/env bash
# Blocks edits to PREREGISTRATION.md once gates have been run.
#
# Matches the TARGET PATH of the tool call, not the payload. The previous
# version grepped the whole stdin blob for the string "PREREGISTRATION.md",
# which fired on any file whose text merely mentioned the name -- it blocked
# writing an unrelated source file in a different repository because a docstring
# referred to the pre-registration. A guard that cries wolf on mentions trains
# people to bypass it, which is worse than no guard.
INPUT=$(cat)

# Nothing to protect until a gate has produced results.
[ -s results/GATE_LOG.md ] || exit 0

TARGET=$(printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
ti = d.get("tool_input") or {}
# Write / Edit / NotebookEdit address a file directly.
for k in ("file_path", "notebook_path", "path"):
    if ti.get(k):
        print(ti[k]); break
' 2>/dev/null)

blocked() {
  echo "BLOCKED: PREREGISTRATION.md is frozen because gates have already run." >&2
  echo "It records claims and thresholds before results exist; amending it now" >&2
  echo "invalidates them. To append a closure record, do it deliberately and say" >&2
  echo "so in the commit message." >&2
  exit 2
}

# Direct file edits: block only when the target IS the pre-registration.
if [ -n "$TARGET" ] && [ "$(basename "$TARGET")" = "PREREGISTRATION.md" ]; then
  blocked
fi

# Shell commands: block only writes aimed at it, not mentions of it.
CMD=$(printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    print((json.load(sys.stdin).get("tool_input") or {}).get("command", ""))
except Exception:
    pass
' 2>/dev/null)
if [ -n "$CMD" ]; then
  if printf '%s' "$CMD" | grep -Eq '(>>?[[:space:]]*|tee[[:space:]]+(-a[[:space:]]+)?|sed[[:space:]]+-i[^|]*|mv[[:space:]]+[^|]*|cp[[:space:]]+[^|]*|rm[[:space:]]+[^|]*)([^[:space:];|&]*/)?PREREGISTRATION\.md'; then
    blocked
  fi
fi
exit 0
