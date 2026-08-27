---
name: theorem-audit
description: Check that a theorem's assumptions actually hold in the implementation. Use when the user asks to audit a theorem, verify assumptions, or check whether code matches the math.
---

# Theorem audit

For the named theorem in `paper/theorems.md`:

1. List every assumption in the statement.
2. For each, find the code that must satisfy it and quote the lines.
3. Flag any assumption with no corresponding enforcement in code.
4. Specifically check: soft masks not hard (Assumption 5), spectral
   normalization active if a Lipschitz constant is used, latent-space output if
   the link function appears, exchangeability of the calibration split.
5. Write findings to `paper/audits/<theorem>.md`. Report unverifiable
   assumptions as unverifiable rather than assuming they hold.
