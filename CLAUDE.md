# certgnn

Certified faithful explanations for message-passing GNNs. Target: ICLR 2027.

## Non-negotiables

- **Never edit `PREREGISTRATION.md` after a gate has been run.** It records claims
  and thresholds before results exist. Amending it post hoc invalidates the paper.
- **Every experiment writes to `results/runs/<timestamp>_<gitsha>_<config>/`.**
  No result exists unless it has a git SHA and a config hash.
- **Minimum 5 seeds** for any reported number. Report mean and 95% CI, never a
  single run.
- **Soft masks only.** Hard binary masking invalidates the Jacobian arguments in
  Theorems 3 through 5. See `certgnn/explain/masks.py`.
- **All model outputs are in latent (logit) space.** Probability-space outputs
  break Theorem 2. The head returns logits; convert only at display time.
- **Sanity checks are not optional.** Any explanation result must ship with the
  model-randomization and label-randomization controls from `certgnn/eval/sanity.py`.

## Substrate-agnostic design

The core (`topology`, `certify`, `explain`, `eval`) must not import anything
substrate-specific. Substrates implement the protocol in
`certgnn/substrates/base.py`. Adding a substrate must require zero changes to core.

Substrates: `splice`, `connectome`, `synthetic`. The go/no-go between splice and
connectome is open; do not hardcode either.

## Commands

- `make test` runs unit tests
- `make gate1` / `gate2` / `gate3` runs the decision gates
- `python -m experiments.<name> --config configs/<...>.yaml`

## Style

- Type hints on all public functions. `numpy` docstrings.
- No notebooks in version control. Analysis goes in `experiments/`.
- Config via yaml + hydra-style overrides, never hardcoded constants.
