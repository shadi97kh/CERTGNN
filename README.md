# certgnn

Research code for faithful graph explanations and the reliability of attributions
from genotype–phenotype models. The repository contains two connected lines of
work:

- **GNN explanation and certification:** effective resistance, differentiable
  node masks, latent-space faithfulness, conformal calibration, synthetic graph
  controls, and a real MFASS splice-graph substrate.
- **Splicing attribution reliability:** independently trained sequence models,
  accuracy-matched attribution comparisons, monotone reparameterization and
  closure experiments, assay-noise separation estimates, and a reproducible
  paper-generation pipeline.

The current manuscript is **“Reading Positions Off a Genotype–Phenotype Model:
A Negative Result on Attribution Reliability in a Multiplexed Splicing Assay”**
([LaTeX source](paper/neurips_2026.tex)). It studies the BRCA2 5′ splice-site
assay. The original GNN claims and decision gates remain in the frozen
[preregistration](PREREGISTRATION.md); they are not all implemented or validated.

## Contents

- [Current findings and scope](#current-findings-and-scope)
- [Installation and quick start](#installation-and-quick-start)
- [Repository map](#repository-map)
- [Core library](#core-library)
- [Configuration](#configuration)
- [Data](#data)
- [Experiment catalog](#experiment-catalog)
- [Running experiments](#running-experiments)
- [Sweeps and aggregation](#sweeps-and-aggregation)
- [Run records and reproducibility](#run-records-and-reproducibility)
- [Paper, figures, and audits](#paper-figures-and-audits)
- [Development and known limitations](#development-and-known-limitations)

## Current findings and scope

These are summaries of recorded experiments, with their limitations. A passing
synthetic check, an uninformative real-data run, and a proposed experiment carry
different evidential weight.

| Study | Recorded result | Interpretation and source |
|---|---|---|
| BRCA2 attribution occurrence | In the paper's 10-seed run, accuracy-tied models disagree on the top three positions for **43%** of sequences at the best-fitting cell and up to **94%** across the grid. | Predictive accuracy alone does not select a reproducible attribution in this setting. See the [paper](paper/neurips_2026.tex) and its [occurrence run](results/runs/20260830T002223Z_6f98de3_6e25cf7d/). |
| Attribution concentration | The best-fitting cell places a median **98%** of attribution magnitude in three of nine positions. | Full-position rank correlation can obscure disagreement about the positions that carry the mass. Top-k overlap, exact-set agreement, concentration, and per-instance distributions are reported together. |
| Accuracy versus function agreement | Accuracy-tied models can still make pointwise predictions that are separable under the assay-count noise model. | Failure to reject an accuracy difference is not equivalence of functions. The independent-model result and the constructed-twin result must remain separate; see the [claim correction](paper/prior_art/claim_correction.md). |
| Closure and optimization controls | A warm-started identity-target refit starts at its own answer; cold-start controls expose search failures. | High in-sample closure does not establish functional containment, and failed refitting does not establish non-containment. See [closure search](paper/tables/closure_search.md). |
| Synthetic G2 | Probability-space conditional-coverage gap **0.1622 [0.1527, 0.1721]**, latent-space gap **0.0203 [0.0165, 0.0240]**, 10 seeds: **PASS**. | This uses the generator's oracle, not a trained model. The [gate log](results/GATE_LOG.md) records thresholds, provenance, and disclosures. |
| Trained-model G2 follow-up | The informative, baseline-observable arm reverses the oracle's gap ordering; the gap statistic also fails its intended sanity interpretation. | The oracle result did not transfer to the trained model. This follow-up does not re-decide the preregistered gate. |
| MFASS topology controls and G2 | Held-out predictive R² is at or below zero: **UNINFORMATIVE**. | The reference model did not generalize, so these runs cannot establish that topology is unnecessary or confirm G2 on real data. See the [splice analysis](paper/prior_art/splice_negative.md). |
| Epistasis spectrum | The exploratory mechanistic verdict is **retracted** and the direction is retired. | Measurement code and artifacts remain for traceability; see the [prior-art and interpretation audit](paper/prior_art/spectral_discriminator.md). |

The BRCA2 result concerns attribution **reproducibility**, not biological truth.
The library, model classes, finite accuracy-test resolution, concentration, and
the confounding between depth and fit quality limit its scope. The top-3 headline
above belongs to the full 10-seed paper run; later per-instance dump runs serve
separate diagnostics and must not replace that population implicitly.

## Installation and quick start

Use **Python 3.11 or newer**, Git, and a checkout of this repository. Run commands
from the repository root unless a command explicitly changes directory. Package
metadata is in [pyproject.toml](pyproject.toml), currently version `0.1.0`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`make setup` creates the same environment and installs the development extra;
activate it with `source .venv/bin/activate` afterward. Later Make targets use
the active interpreter and executables.

Declared dependencies include PyTorch, PyTorch Geometric, NetworkX, NumPy, SciPy,
pandas, scikit-learn, PyYAML, Hydra/OmegaConf, Matplotlib, Seaborn, TorchCP,
Captum, tqdm, and Weights & Biases. Development dependencies include pytest,
pytest-cov, Hypothesis, Ruff, and mypy. Historical Excel datasets additionally
need `openpyxl`; optional random access to the Otari reference genome needs
`pyfaidx`. These two packages are not declared in the project dependencies:

```bash
python -m pip install openpyxl pyfaidx
```

Start with tests and a small synthetic run; neither requires downloading data:

```bash
python -m pytest -q -m "not slow"

python -m experiments.gate2_link --config configs/base.yaml --allow-dirty \
  seeds=1 bootstrap_resamples=50 \
  gate2.link_sweep.n_instances=60 gate2.link_sweep.n_pairs=500 \
  gate2.curve_fit.n_instances=60 gate2.rank_reversal.n_per_baseline=20 \
  gate2.coverage.n_cal=40 gate2.coverage.n_test=100 \
  output.runs=/tmp/certgnn-smoke/runs \
  output.figures=/tmp/certgnn-smoke/figures \
  output.tables=/tmp/certgnn-smoke/tables
```

This is a one-seed **pipeline smoke check**, not a reportable gate result. G2
returns exit status `0` for PASS and `1` for FAIL; either can be a completed
experiment. Inspect its saved verdict and logs before treating a nonzero status
as a crash. `--allow-dirty` permits development runs and records the modified-tree
status. Reportable runs should use a clean, committed source tree.

The BRCA2 capacity, closure, separation, and occurrence experiments share
`closure_capacity.device`. The checked-in value is `cuda:1`, reflecting the
original machine. Set it explicitly for your hardware:

```bash
python -m experiments.occurrence --help
# Select the first CUDA device when available, otherwise CPU:
python -m experiments.occurrence --config configs/base.yaml \
  closure_capacity.device=auto
```

The second command requires the BRCA2 data and launches the full default
experiment. `closure_capacity.device=cpu` and `closure_capacity.device=cuda:0`
are also supported. Full neural sweeps are substantially more expensive than the
synthetic smoke check.

## Repository map

```text
certgnn/
  topology/       Effective-resistance solvers
  certify/        Link transforms and conformal quantiles
  explain/        Soft-mask initialization and optimization
  eval/           Paired faithfulness metrics and sanity controls
  models/         GCN/GIN/sum-SAGE, edge-free MLP, BQN, training
  substrates/     Shared protocol; synthetic and MFASS implementations
  viz/            Reserved package; plotting currently lives in experiments
configs/          Base configuration and selectable YAML groups
experiments/      Experiment entry points and paper generators
scripts/          Sweep launcher, aggregation, data fetch, historical figures
tests/            Unit, property, Monte Carlo, and experiment smoke tests
data/raw/         Local datasets and tracked provenance document
results/runs/     Configured runs, per-seed measurements, and provenance
results/sweeps/   Sweep manifests
results/diagnostics/  Development diagnostics, separate from reported results
paper/            Manuscript, appendix, style, figures, tables, and audits
docs/             Numeric-claim provenance
PREREGISTRATION.md  Frozen original claims and gate thresholds
ABLATIONS.md      Experiment matrix, statuses, controls, and reporting rules
```

## Core library

The core is substrate-agnostic: `topology`, `certify`, `explain`, and `eval`
must not import a concrete substrate. Domain-specific loading belongs under
`certgnn/substrates/`.

| Module | Implemented functionality |
|---|---|
| [topology/resistance.py](certgnn/topology/resistance.py) | Series-parallel graph reduction, reducibility checks, general Laplacian pseudoinverse, and resistance to a target. |
| [certify/link.py](certgnn/certify/link.py) | Logit and arcsine-square-root transforms, sigmoid Jacobian, latent/probability fidelity gaps, and baseline strata. |
| [certify/conformal.py](certgnn/certify/conformal.py) | Finite-sample split quantiles, Mondrian per-stratum quantiles, test-point-weighted conformal quantiles, empirical coverage, and conditional-coverage gaps. |
| [explain/masks.py](certgnn/explain/masks.py) | Resistance-ball and cold initializations; sigmoid-parameterized masks; augmented-Lagrangian optimization for latent ε-sufficiency; convergence and feasibility reporting. |
| [eval/faithfulness.py](certgnn/eval/faithfulness.py) | Comprehensiveness, sufficiency, AOPC, normalized AOPC, deletion/insertion AUC, and combined reports. Metrics return both probability- and latent-space scores through `PairedScore`. |
| [eval/sanity.py](certgnn/eval/sanity.py) | Model and label randomization, topology shuffling, degree-preserving rewiring, random explanations, and degenerate-mask checks. |
| [models/gnn.py](certgnn/models/gnn.py) | GCN, GIN, sum-aggregation SAGE, and an edge-free node MLP; target or pooled readout; soft masks applied at the input and after each layer. |
| [models/bqn.py](certgnn/models/bqn.py) | Brain Quadratic Network and Hadamard encoder controls for fixed-size connectivity matrices. The cluster readout is a documented simplification of the reference BQN. |
| [models/train.py](certgnn/models/train.py) | Seeded training, validation-based model selection, binary BCE-with-logits/AUROC/accuracy or MSE/R² regression. |

Graph-model heads return a scalar **latent/logit**, with no sigmoid in the head.
The mask interface is a differentiable callable `latent_fn(mask) -> scalar
Tensor`; the caller binds the graph, features, target, and model. The target
mask is fixed at one. Hard masks are available only through an explicitly
acknowledged negative-control path and do not satisfy the soft-mask assumptions.

A minimal example of the explainer interface:

```python
import networkx as nx
import torch

from certgnn.explain.masks import explain

graph = nx.path_graph(4)
weights = torch.tensor([0.0, 0.8, 0.1, 0.0])

def latent_fn(mask: torch.Tensor) -> torch.Tensor:
    return (weights * mask).sum()

result = explain(latent_fn, graph, target=0, eps=0.1)
print(result.mask, result.latent_gap, result.sufficient)
```

`MaskResult.sufficient` checks the achieved gap against ε for this callable.
It is not an implementation of the proposed certified-radius theorem.
Conformal calibration likewise requires appropriately held-out scores;
exchangeability or the required covariate-shift weighting is a statistical
assumption, not something the quantile function can establish from a tensor.

### Substrates

The [Substrate protocol](certgnn/substrates/base.py) requires `name`,
`load(split)`, `target_node`, `candidate_nodes`, `ground_truth_window`,
`to_networkx`, and `baseline_rate`. `load` returns PyTorch Geometric `Data`
objects with node features `x` and a `target_idx`. Existing loaders expose
`train`, `val`, `cal`, and `test` splits.

| Substrate | Status and behavior |
|---|---|
| [Synthetic](certgnn/substrates/synthetic/) | Implemented. Path, bubble, nested-bubble, multi-parallel, and pendant motifs; identity/logit/probit/cloglog links; an electrical-conductance oracle; known sufficient sets; Gaussian baseline shift with exact likelihood ratios. Ground truth is constructed. |
| [Splice](certgnn/substrates/splice/substrate.py) | Implemented on MFASS. Sequence-window nodes, genomic-backbone edges, and an exon-skipping junction. Variants are split by exon identifier to avoid sharing an exon across splits. The requested 300 nt radius is clipped to the 170 nt assayed construct. |
| [Connectome](certgnn/substrates/connectome/) | Package stub and configuration only. No `ConnectomeSubstrate` loader is implemented; a BQN implementation alone does not make this substrate runnable. |

Add a substrate by implementing the protocol without changing the core. Provide
its configuration and wire it into the relevant experiment loader separately;
there is no automatic plugin discovery for a new substrate.

## Configuration

[configs/base.yaml](configs/base.yaml) is loaded by
[`experiments._common.load_config`](experiments/_common.py). It uses OmegaConf
with Hydra-style overrides; it is not a Hydra multirun application.

| Group | Available selections | Default |
|---|---|---|
| `substrate` | `synthetic`, `splice`, `connectome` | `synthetic` |
| `model` | `gcn`, `gin`, `sage_sum`, `mlp` | `sage_sum` |
| `explainer` | `soft_mask`, `soft_mask_cold` | `soft_mask` |
| `rewire` | `none`, `add_only`, `swap` | `none` |
| `conformal` | `split`, `mondrian`, `weighted` | `split` |

Group files merge under their group name. `model=gin` chooses a YAML file;
`model.hidden=128` changes a field. Experiment-specific settings use dotted
paths such as `gate2.coverage.n_test=2000`. Quote list overrides so the shell
passes them intact:

```bash
python -m experiments.tier0_controls --config configs/base.yaml \
  model=gin model.hidden=128 'tier0.substrates=[synthetic]' seeds=5
```

Defaults include 10 seeds, 2,000 bootstrap resamples, conformal `alpha=0.1`,
five strata, and one PyTorch CPU thread. `seed=7` selects only seed 7;
otherwise `seeds=N` means seeds `0` through `N-1`. Different hypothesis tests
have their own α values: the occurrence accuracy filter and separation test use
`0.05` and must agree with each other.

Configuration availability does not imply that every runner implements every
group. The graph experiments and the BRCA2 sequence-model experiments consume
different sections. In particular, the BRCA2 architecture grid is controlled by
`closure_capacity.hidden` and `closure_capacity.depth`, not `model.hidden` or
`model.depth`. Rewiring YAML files specify planned settings; they are not a
complete add-only rewiring experiment pipeline.

## Data

Raw data is excluded from Git. A fresh clone contains
[data/raw/PROVENANCE.md](data/raw/PROVENANCE.md), not the large local data
collection. That document records sources, phenotype definitions, checksums,
library discrepancies, and acquisition history.

| Dataset/resource | Expected local path | Role and scope |
|---|---|---|
| BRCA2 5′ splice-site MPSA | `data/raw/mpsa_data.csv.gz` | Main attribution study: 30,483 measured nine-nucleotide RNA sequences; `y` is log10 percent-spliced-in. The designed library has 32,768 variants, a different count. |
| BRCA2 replicate | `data/raw/mpsa_replicate_data.csv.gz` | Separate released replicate with 30,697 rows; not silently pooled with the primary assay. |
| MFASS | `data/raw/mfass_snv_data_clean.txt` | 32,669 variant rows over 2,339 exons before loader filtering; splice indices, sequence, exon/intron lengths, and read support. |
| FAS/CD95 exon 6 | `data/raw/fas_supp/` | Julien et al. single/double-mutant supplementary tables; enrichment scores, not PSI. This is not the later 3,072-genotype combinatorial library. |
| GB1 | `data/raw/gb1_olson2014_mmc2.xlsx` | Count-based single/double-mutant data used by the retired epistasis-spectrum experiment. |
| eqFP611 | `data/raw/poelwijk_supp/` | Complete 8,192-genotype, 13-bit landscape used in the same historical measurement. |
| Otari | `data/raw/otari_repo/`, `data/raw/otari_resources.tar.gz`, `data/raw/resources/` | Pinned upstream code, hg38, annotations, isoform data, and pretrained feature models. The resource archive alone is about 5 GB; the BRCA2 experiments do not require it. |
| ProteinGym inventory | `data/raw/pg_raw/`, `data/raw/pg_proc/`, `data/raw/pg_ref.csv` | An exploratory [assay-reliability inventory](paper/tables/proteingym_reliability_inventory.md), not a completed noise-corrected benchmark. |

To fetch only the two small BRCA2 files from the source recorded in provenance:

```bash
mkdir -p data/raw
curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/jbkinney/mavenn/master/mavenn/examples/datasets/mpsa_data.csv.gz \
  --output data/raw/mpsa_data.csv.gz
curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/jbkinney/mavenn/master/mavenn/examples/datasets/mpsa_replicate_data.csv.gz \
  --output data/raw/mpsa_replicate_data.csv.gz

sha256sum data/raw/mpsa_data.csv.gz data/raw/mpsa_replicate_data.csv.gz
```

Expected SHA-256 values from the recorded acquisition:

```text
df1251425d7bae15cbae5922c9b05e7765344fc4bb13970fb44b51370c09cf74  data/raw/mpsa_data.csv.gz
c18a824798617fff07806132aadf9e8c1e6547c770dce8b8d515bb3189e80415  data/raw/mpsa_replicate_data.csv.gz
```

For the broader historical acquisition, `bash scripts/fetch_data.sh` clones
Otari at its recorded commit, downloads its resource archive, extracts FAS
supplementary files, and downloads BRCA2. Its coverage is narrower than the
full inventory: it does **not** fetch MFASS, GB1, eqFP611, or ProteinGym, or
extract the Otari resource archive. Follow the recorded sources for those files
and verify their checksums. The MFASS loader's suggestion to run this script is
therefore insufficient on a fresh checkout.

## Experiment catalog

Research runners below use `python -m experiments.<name>` from the repository
root. Unless noted otherwise, they accept `--config`, positional `key=value`
overrides, and `--allow-dirty`. Use `--help` for runner-specific switches.

| Module | Purpose | Inputs/status |
|---|---|---|
| [gate2_link](experiments/gate2_link.py) | Link sweep, effect-versus-baseline fit, rank reversal, and conditional coverage; ablations 1.6–1.9. | Synthetic oracle; recorded G2. |
| [gate2_model](experiments/gate2_model.py) | Repeats the coverage comparison with a trained GNN, hidden/observable baseline arms, and randomization controls. | Synthetic trained-model follow-up. |
| [gate2_splice](experiments/gate2_splice.py) | Soft-masked trained GNN on real measured splice baselines. | MFASS; recorded outcome uninformative. |
| [tier0_controls](experiments/tier0_controls.py) | Degree-preserving shuffle, edge-free MLP with target/mean readout, and connectome BQN control; rows 0.6–0.8. | Synthetic/MFASS available; connectome unavailable. Supports `--retable` and sweep `--aggregate`. |
| [rank_reversal_standardized](experiments/rank_reversal_standardized.py) | Adds per-instance standardization to raw probability- and latent-space rank comparisons. | Synthetic; prior-art-motivated third arm. |
| [benchmark_reeval](experiments/benchmark_reeval.py) | Compares explainer rankings in probability and latent space; `--precheck` examines operating-point distributions. | Candidate C.5–C.8 work. Some benchmark/explainer adapters are explicitly unavailable; this is not a completed benchmark suite. |
| [identifiability_probe](experiments/identifiability_probe.py) | Linear, pairwise, and neural latent maps; monotone warps, closure, restart agreement, and numerical-floor/radius diagnostics. | Synthetic genotype–phenotype probe. |
| [ism_invariance](experiments/ism_invariance.py) | Tests finite in-silico mutagenesis against within-locus gradient-invariance reasoning over effect sizes. | Synthetic probe. |
| [identifiability_mpsa](experiments/identifiability_mpsa.py) | Applies latent-map and attribution diagnostics to a fitted real-assay latent, with null controls. | BRCA2 MPSA. |
| [closure_capacity](experiments/closure_capacity.py) | Width/depth sweep of in-sample monotone-warp approximation, reference-fit quality, and learning-rate pilots. | BRCA2; supports `--retable`. |
| [closure_shuffled](experiments/closure_shuffled.py) | Compares warp re-representation with non-monotone/shuffled targets. | BRCA2; supports `--retable`. |
| [closure_heldout](experiments/closure_heldout.py) | Splits fitting from evaluation to distinguish in-sample fit from held-out agreement. | BRCA2; its warm-started ceiling limitation is studied by `closure_search`; supports `--retable`. |
| [closure_search](experiments/closure_search.py) | Cold-start null and larger-class controls for whether the optimizer can find an in-class target. | BRCA2; supports `--retable`. |
| [separation](experiments/separation.py) | Estimates assay-noise-scaled separation between a reference model and a constructed twin; compares attribution sets. | BRCA2; supports `--retable`. |
| [occurrence](experiments/occurrence.py) | Trains independent models, filters accuracy-tied pairs, and measures per-sequence attribution disagreement and function separation. | Main BRCA2 experiment; supports `--retable` and per-instance dumps. |
| [dead_positions](experiments/dead_positions.py) | Audits attribution to constrained library positions and never-trained one-hot input weights. | Reads an existing position-indexed dump; accepts `--run` and `--out`, not the common experiment CLI. |
| [epistasis_spectrum](experiments/epistasis_spectrum.py) | Historical cross-landscape epistasis measurements; `--only fas`, `gb1`, or `eqfp`. | Retired; interpret with the retraction, not as new mechanistic evidence. |

The [ablation matrix](ABLATIONS.md) additionally specifies sanity controls,
theorem validation, architecture, explanation, conformal calibration, shift,
and efficiency studies. A specified row is a plan, not evidence that its runner
or result exists. Draft and candidate work retains the status recorded there.

## Running experiments

### Graph controls and the implemented gate

With a clean source tree:

```bash
# Full synthetic oracle G2, using the configured 10 seeds.
python -m experiments.gate2_link --config configs/base.yaml
# Equivalent Make target:
make gate2

# Trained-model follow-up and synthetic topology controls.
python -m experiments.gate2_model --config configs/base.yaml
python -m experiments.tier0_controls --config configs/base.yaml \
  'tier0.substrates=[synthetic]'

# After acquiring the MFASS file.
python -m experiments.tier0_controls --config configs/base.yaml \
  'tier0.substrates=[splice]'
python -m experiments.gate2_splice --config configs/base.yaml
```

The Makefile also advertises `gate1`, `gate3`, `gate4`, and `gate5`, but their
referenced modules (`gate1_resistance`, `gate3_vacuity`, `gate4_pareto`, and
`gate5_retrospective`) do not exist. Those targets are placeholders, not runnable
gate implementations. The original README's `make gate1` quick start is obsolete.

### BRCA2 attribution study

The default occurrence protocol subsamples 4,000 sequences per outer seed,
uses 3,000 for fitting and 1,000 held out, trains 20 initializations per cell,
and evaluates widths 16/32/64/128 at depths 1/2/3. It compares up to 190 pairs
per cell and computes ISM profiles on 400 held-out sequences. These sequence
models are distinct from the message-passing GNN backbones.

```bash
python -m experiments.occurrence --config configs/base.yaml \
  closure_capacity.device=auto

python -m experiments.separation --config configs/base.yaml \
  closure_capacity.device=auto

python -m experiments.closure_search --config configs/base.yaml \
  closure_capacity.device=auto
```

To reproduce a particular historical run, use its saved `config.yaml`, recorded
source commit, and environment lock. Current defaults and code may include
later diagnostics or interpretation fixes; rerunning current code is not an
exact recreation of an older code state.

Per-instance dumps support concentration figures and position-specific audits:

```bash
python -m experiments.occurrence --config configs/base.yaml \
  --dump-per-instance --dump-cells 128x1,128x3 --dump-seeds 0,1,2,3,4 \
  seeds=5 closure_capacity.device=auto

python -m experiments.dead_positions
```

Dump mode retrains the requested cells, writes `per_instance.npz`, and marks its
results `dump_only`. It is not a cheap export of already-trained models. The CLI
default dump seed list is only `0,1,2`; the example explicitly uses five seeds.
`dead_positions` selects the latest local dump containing `__ism` arrays, or
accepts an explicit `--run`. Dump artifacts retain both sorted concentration profiles
and position-indexed attribution magnitudes.

Supported runners can rebuild their aggregates and tables from existing
`seed_*.json` records:

```bash
python -m experiments.occurrence \
  --retable results/runs/20260830T002223Z_6f98de3_6e25cf7d
```

`--retable` skips model fitting but **writes regenerated aggregates/tables**.
Review the resulting diff, particularly when interpretation logic has changed.
Default output locations can overwrite the corresponding summary under
`paper/tables/`; use output overrides for exploratory training runs.

## Sweeps and aggregation

[`scripts/launch_sweep.py`](scripts/launch_sweep.py) parses the YAML blocks in
`ABLATIONS.md`, produces resolved configurations, and records a manifest before
launching. The [runner registry](experiments/registry.py) currently maps only:

| Ablation rows | Runner |
|---|---|
| `0.6`, `0.7`, `0.8` | `experiments.tier0_controls` |
| `1.6`, `1.7`, `1.8`, `1.9` | `experiments.gate2_link` |

Both registrations are `row_experiment`: the runner implements its row set
internally. Selecting rows for the launcher does not necessarily restrict the
runner to only those rows. Other research modules are invoked directly.

```bash
# Generate configs and a manifest without training.
python scripts/launch_sweep.py --matrix ABLATIONS.md --tier 0 \
  --rows 0.6,0.7 --seeds 10 --dry-run

# Launch the registered synthetic-link row set.
python scripts/launch_sweep.py --matrix ABLATIONS.md --tier 1 \
  --rows 1.6,1.7,1.8,1.9 --seeds 10

# Set this to the actual manifest path printed by the launcher.
CERTGNN_MANIFEST=results/sweeps/REPLACE_WITH_SWEEP_ID.json
python scripts/launch_sweep.py --resume "$CERTGNN_MANIFEST"
python scripts/aggregate.py --runs results/runs --out paper/tables \
  --manifest "$CERTGNN_MANIFEST" --expected-seeds 10
```

A dry run writes files. Missing runners cause refusal unless `--partial` is
supplied; omitted rows are reported. Candidates are excluded, and draft rows
require `--include-draft` as well as the project approval described in the
matrix. Resume retries entries that did not exit with status zero, which can
include a completed gate that returned FAIL.

The aggregator checks planned-cell completeness, combines `per_seed_values.json`,
reports seed means with 95% bootstrap intervals, applies Holm–Bonferroni to
available gate statistics, and writes Markdown/LaTeX tables plus tuning-budget
records. Its default minimum is five seeds. `make aggregate` uses its default
paths; manifest-specific aggregation is preferable for a defined sweep. Some
later experiments use richer `results.json`/`seed_*.json` schemas and their own
aggregation or paper generators.

## Run records and reproducibility

Experiments using the shared runner create:

```text
results/runs/<UTC-timestamp>_<git-sha>_<config-hash>/
  config.yaml             Resolved configuration
  meta.json               Experiment, SHA, dirty status, hash, seeds, arguments
  environment.lock        Python/platform description and pip freeze
  results.json            Experiment-specific aggregate results
  per_seed_values.json    Flat metrics for shared aggregation, when emitted
  seed_<n>.json           Detailed per-seed records, when emitted
  tuning_budget.json      Search effort, when emitted
  gate_stats.json         Gate statistics, when applicable
  verdict.md / table.md   Human-readable interpretation, when emitted
  per_instance.npz        Occurrence dump arrays, when requested
```

The metadata, configuration, and environment files are created by
[`experiments/_common.py`](experiments/_common.py); additional artifacts depend
on the runner. Dirty runs include `-dirty` in the directory name. Timestamp
collisions receive a suffix rather than sharing a directory. Some early
historical runs predate the complete provenance schema; the gate log identifies
the regenerated G2 record.

Repository research conventions from [CLAUDE.md](CLAUDE.md),
[PREREGISTRATION.md](PREREGISTRATION.md), and [ABLATIONS.md](ABLATIONS.md):

- Preserve the preregistration after the first gate; record follow-ups and
  corrections separately.
- Use at least five seeds for reported measurements, ten for headline results,
  with the estimator and 95% interval stated. Pooled per-instance summaries and
  averages of per-seed summaries need explicit labels.
- Select models on validation data, keep calibration separate, and follow the
  preregistered test-use rules. Each later experiment's saved protocol should
  be read on its own terms.
- Report randomization and degeneracy controls alongside explanation results.
- Record tuning effort and departures from equal budgets. The historical
  three-backbone-versus-one-MLP discrepancy is disclosed in
  [tuning-budget notes](results/tuning_budget_notes.md).
- Treat a search over zero candidates as **VOID**, not as a negative result;
  the shared `void_if_unsearched` helper supports this distinction.

Raw data, caches, new run directories, checkpoints, and generated sweep configs
are ignored by default. Selected historical run records are nevertheless
tracked explicitly and support the manuscript. Development logs under
[`results/diagnostics/`](results/diagnostics/README.md) are not reportable
multi-seed results.

## Paper, figures, and audits

The current manuscript uses [`paper/neurips_2026.tex`](paper/neurips_2026.tex),
[`paper/appendix.tex`](paper/appendix.tex), and the **included**
[`paper/neurips_2026.sty`](paper/neurips_2026.sty). Its current template is an
ICBINB/NeurIPS workshop manuscript; older project notes still mention the
original ICLR target.

Regenerate the paper's derived assets from committed run records:

```bash
python experiments/make_tables.py
python experiments/make_figures.py
python experiments/make_workflow_figure.py
python experiments/make_appendix.py
```

| Generator | Outputs and behavior |
|---|---|
| `make_tables.py` | Replaces seven named `% >>> GENERATED` blocks directly inside `neurips_2026.tex`; writes `paper/TABLES_PROVENANCE.json`. Missing markers are errors. |
| `make_figures.py` | Main occurrence/quality, concentration, and function-separation figures, plus twin and warm/cold-start appendix figures; PDF/PNG outputs with JSON data/provenance sidecars. The concentration panel requires a per-instance dump. |
| `make_workflow_figure.py` | Workflow figure with actual model/pair counts and disagreement values; defaults to cell `128x1`. |
| `make_appendix.py` | Regenerates `paper/appendix.tex` from run records. |
| `scripts/make_paper_figures.py` | Historical synthetic-identifiability figures and summary table; separate from the current manuscript's generators. |

The current generators inspect local run directories with the required fields,
normally preferring records marked as having a clean source tree. They do not
verify that the artifacts themselves are tracked in Git. Full-result selection
excludes `dump_only` occurrence records; a separate resolver finds the
per-instance dump. New usable local runs can change the selected input, so
inspect the printed source paths and provenance sidecars after regeneration.
`make_figures.py` also accepts explicit `--occurrence-run` and
`--separation-run` paths.

Build with a separately installed TeX toolchain, for example Tectonic:

```bash
cd paper
tectonic -X compile neurips_2026.tex --outdir .
```

[`paper/README.md`](paper/README.md) is historical: its statements that the real
style file is absent and tables are emitted to `tex/*.tex` no longer describe
the current tree. `paper/pagecount_proxy/neurips_2026.sty` is a geometry proxy,
not the submission style. Follow the current generators and the included style
for the manuscript build.

Further research documentation:

| Resource | What it records |
|---|---|
| [Number provenance](docs/number_provenance.md) and [numeric audit script](paper/audits/number_audit.py) | Claim-to-artifact tracing and recomputation of headline values. The audit script targets a specific occurrence run; read the dated correction entries when numbers or seed counts changed. |
| [Theorem ledger](paper/theorems.md) and [individual audits](paper/audits/) | Original claims, assumptions, enforcement gaps, and falsification checks. Formal statements marked `PENDING PROPOSAL` remain incomplete. |
| [Gate log](results/GATE_LOG.md) | G2 decisions, model follow-ups, disclosures, and the real-splice negative. |
| [Tuning budgets](results/tuning_budget.md) | Recorded baseline search effort, with qualifications in the linked tuning notes. |
| [SQUID/MAVE-NN comparison](paper/prior_art/squid_collision.md) | Prior-art overlap for link correction and standardized attribution comparisons. |
| [Occurrence gap](paper/prior_art/occurrence_gap.md) and [claim correction](paper/prior_art/claim_correction.md) | The scope of the independent-fit claim and the distinction from constructed twins. |
| [Noise-ceiling review](paper/prior_art/noise_ceiling.md) and [ProteinGym inventory](paper/tables/proteingym_reliability_inventory.md) | An exploratory reliability-normalization direction and limits of available assay uncertainties. |
| [Spectral discriminator review](paper/prior_art/spectral_discriminator.md) | Prior art, retraction, and known interpretation errors in the retired epistasis direction. |
| [Signed rectangle-cover note](paper/prior_art/sign_rectangle_cover.md) | A separate exploratory prior-art note on sign-constrained network width. |

## Development and known limitations

```bash
make test                       # pytest with certgnn coverage, including slow tests
python -m pytest -q -m "not slow"
python -m pytest -q tests/test_experiments_smoke.py
make lint                       # ruff check . && mypy certgnn
```

Tests cover resistance identities and solver comparisons, synthetic mechanisms,
conformal corrections and coverage, soft-mask optimization, paired faithfulness
metrics, model controls, sweep completeness, and experiment provenance. Slow
tests include Monte Carlo and end-to-end runs. These checks establish software
behavior; they do not turn incomplete theorem assumptions into guarantees.

Public functions use type hints and NumPy-style docstrings. Keep analysis in
Python experiment modules rather than versioned notebooks, and expose research
settings through configuration. Register a new ablation runner explicitly if
it should be available to the sweep launcher.

The main unfinished or restricted areas are:

- **Certification scope:** split/Mondrian/weighted calibration and soft-mask
  optimization exist; a complete certified-radius/Lipschitz pipeline and the
  G1/G3/G4/G5 experiment entry points do not. The theorem audits document
  remaining assumption-enforcement gaps.
- **Resistance scope:** use the validated connected, unit-weight graph setting
  when comparing solver paths. The reduction reads edge resistances, whereas
  the pseudoinverse path builds an unweighted adjacency; disconnected-graph
  fallback behavior also needs the qualifications in the audits.
- **Substrate scope:** connectome remains unimplemented. MFASS is implemented
  despite stale comments in `configs/substrate/splice.yaml` and the early
  `results/tier0_verdict.md`; its later real-data runs were uninformative.
- **Attribution support:** BRCA2 fixes one position and restricts another.
  Some ISM substitutions activate one-hot inputs never seen during training;
  `dead_positions.py` and the saved position diagnostics investigate this
  additional source of instability.
- **Evidence scope:** synthetic oracle results, trained-model results,
  independent-model occurrence, and constructed-twin separation answer
  different questions. Historical notes can retain superseded interpretations;
  consult their corrections and the exact source run.
- **Reproduction scope:** lower-bounded dependencies in `pyproject.toml` are
  not a universal environment lock. Use the saved per-run environment and
  configuration for historical reproduction; CPU/GPU changes and later source
  changes can alter training outcomes.
