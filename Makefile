.PHONY: setup test lint gate1 gate2 gate3 gate4 gate5 sweep aggregate clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

test:
	pytest -q --cov=certgnn

lint:
	ruff check . && mypy certgnn

gate1:
	python -m experiments.gate1_resistance --config configs/base.yaml
gate2:
	python -m experiments.gate2_link --config configs/base.yaml
gate3:
	python -m experiments.gate3_vacuity --config configs/base.yaml
gate4:
	python -m experiments.gate4_pareto --config configs/base.yaml
gate5:
	python -m experiments.gate5_retrospective --config configs/base.yaml

sweep:
	python scripts/launch_sweep.py --matrix ABLATIONS.md --tier $(TIER)

aggregate:
	python scripts/aggregate.py --runs results/runs --out paper/tables
