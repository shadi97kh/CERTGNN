"""Which experiment module implements which ABLATIONS.md rows.

``mode="row_experiment"``: the module implements the listed rows internally
(all cells of those rows in one run), so the launcher generates one config
per seed and does not expand the rows' axes. ``mode="cell"`` (none yet): the
launcher expands the row's axes into cells and the module must honour the
resulting dotted overrides. Rows absent from this registry cannot be
launched; the launcher lists them loudly instead of dropping them.
"""

from __future__ import annotations

RUNNERS: dict[str, dict[str, object]] = {
    "experiments.tier0_controls": {
        "rows": ["0.6", "0.7", "0.8"],
        "mode": "row_experiment",
    },
    "experiments.gate2_link": {
        "rows": ["1.6", "1.7", "1.8", "1.9"],
        "mode": "row_experiment",
    },
}


def runner_for(row_id: str) -> tuple[str, str] | None:
    for module, spec in RUNNERS.items():
        if row_id in spec["rows"]:  # type: ignore[operator]
            return module, str(spec["mode"])
    return None
