"""Models. Every head returns a latent (logit); probabilities are display-only."""

from certgnn.models.bqn import (
    BrainQuadraticNetwork,
    HadamardEncoderLayer,
    QuadraticPerceptron,
)
from certgnn.models.gnn import NodeFeatureMLP, TargetReadoutGCN
from certgnn.models.train import (
    FitResult,
    fit,
    infer_task,
    metrics,
    predict,
    primary_metric,
)

__all__ = [
    "BrainQuadraticNetwork",
    "HadamardEncoderLayer",
    "QuadraticPerceptron",
    "NodeFeatureMLP",
    "TargetReadoutGCN",
    "FitResult",
    "fit",
    "infer_task",
    "metrics",
    "predict",
    "primary_metric",
]
