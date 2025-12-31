"""Torch metrics for regression."""

from pydantic import TypeAdapter
from typing import Optional

import torchmetrics

from remote_sensing_processor.common.types import ListOfMetrics


regression_metrics = [
    "concordance_correlation_coefficient",
    "cosine_similarity",
    "critical_success_index",
    "explained_variance",
    "kendall_rank_correlation_coefficient_a",
    "kendall_rank_correlation_coefficient_b",
    "kendall_rank_correlation_coefficient_c",
    "kl_divergence",
    "log_cosh_error",
    "mae",
    "mape",
    "mse",
    "msle",
    "manhattan_distance",
    "euclidean_distance",
    "minkowski_distance_3",
    "minkowski_distance_10",
    "minkowski_distance_100",
    "nrmse",
    "pearson_correlation_coefficient",
    "r2",
    "rse",
    "rmse",
    "rrse",
    "spearman_correlation_coefficient",
    "smape",
    "tweedie_deviance_score",
    "weighted_mape",
]


class RegresssionMetrics:
    """Regression metrics base class."""

    def setup_metrics(
        self,
        metrics: Optional[list[dict]],
        sklearn: Optional[bool] = False,
    ) -> None:
        """Setup metrics."""
        # Setting up default metrics if not set
        if metrics is None:
            metrics = [
                {"name": "r2", "log": "verbose"},
                {"name": "rmse", "log": "verbose"},
            ]
        metrics = TypeAdapter(ListOfMetrics).validate_python(metrics)

        metrics_dict_epoch = {}
        metrics_dict_step = {}
        metrics_dict_verbose = {}

        for metric in metrics:
            name = metric.name
            log = metric.log
            if name in regression_metrics:
                # Setting up supported metrics
                if name == "concordance_correlation_coefficient":
                    m = torchmetrics.ConcordanceCorrCoef()
                elif name == "cosine_similarity":
                    m = torchmetrics.CosineSimilarity()
                elif name == "critical_success_index":
                    m = torchmetrics.CriticalSuccessIndex(0.5)
                elif name == "explained_variance":
                    m = torchmetrics.ExplainedVariance()
                elif name == "kendall_rank_correlation_coefficient_a":
                    m = torchmetrics.KendallRankCorrCoef(variant="a")
                elif name == "kendall_rank_correlation_coefficient_b":
                    m = torchmetrics.KendallRankCorrCoef(variant="b")
                elif name == "kendall_rank_correlation_coefficient_c":
                    m = torchmetrics.KendallRankCorrCoef(variant="c")
                elif name == "kl_divergence":
                    m = torchmetrics.KLDivergence()
                elif name == "log_cosh_error":
                    m = torchmetrics.LogCoshError()
                elif name == "mae":
                    m = torchmetrics.MeanAbsoluteError()
                elif name == "mape":
                    m = torchmetrics.MeanAbsolutePercentageError()
                elif name == "mse":
                    m = torchmetrics.MeanSquaredError(squared=True)
                elif name == "msle":
                    m = torchmetrics.MeanSquaredLogError()
                elif name == "manhattan_distance":
                    m = torchmetrics.MinkowskiDistance(1)
                elif name == "euclidean_distance":
                    m = torchmetrics.MinkowskiDistance(2)
                elif name == "minkowski_distance_3":
                    m = torchmetrics.MinkowskiDistance(3)
                elif name == "minkowski_distance_10":
                    m = torchmetrics.MinkowskiDistance(10)
                elif name == "minkowski_distance_100":
                    m = torchmetrics.MinkowskiDistance(100)
                elif name == "nrmse":
                    m = torchmetrics.NormalizedRootMeanSquaredError()
                elif name == "pearson_correlation_coefficient":
                    m = torchmetrics.PearsonCorrCoef()
                elif name == "r2":
                    m = torchmetrics.R2Score()
                elif name == "rse":
                    m = torchmetrics.RelativeSquaredError(squared=True)
                elif name == "rmse":
                    m = torchmetrics.MeanSquaredError(squared=False)
                elif name == "rrse":
                    m = torchmetrics.RelativeSquaredError(squared=False)
                elif name == "spearman_correlation_coefficient":
                    m = torchmetrics.SpearmanCorrCoef()
                elif name == "smape":
                    m = torchmetrics.SymmetricMeanAbsolutePercentageError()
                elif name == "tweedie_deviance_score":
                    m = torchmetrics.TweedieDevianceScore()
                elif name == "weighted_mape":
                    m = torchmetrics.WeightedMeanAbsolutePercentageError()
                else:
                    raise ValueError(name + " is not a valid semantic segmentation metric")
            else:
                # Custom metrics
                if metric.metric is None:
                    raise ValueError(f"Custom metric {name} is not set.")
                m = metric.metric

            if log == "epoch":
                metrics_dict_epoch[name] = m
            elif log == "step":
                metrics_dict_step[name] = m
            elif log == "verbose":
                metrics_dict_verbose[name] = m

        if not sklearn:
            if metrics_dict_epoch:
                self.metrics_epoch_train = torchmetrics.MetricCollection(metrics_dict_epoch, prefix="train_")
                self.metrics_epoch_val = self.metrics_epoch_train.clone(prefix="val_")
                self.metrics_epoch_test = self.metrics_epoch_train.clone(prefix="test_")
            else:
                self.metrics_epoch_train = None
                self.metrics_epoch_val = None
                self.metrics_epoch_test = None

            if metrics_dict_step:
                self.metrics_step_train = torchmetrics.MetricCollection(metrics_dict_step, prefix="train_")
                self.metrics_step_val = self.metrics_step_train.clone(prefix="val_")
                self.metrics_step_test = self.metrics_step_train.clone(prefix="test_")
            else:
                self.metrics_step_train = None
                self.metrics_step_val = None
                self.metrics_step_test = None

            if metrics_dict_verbose:
                self.metrics_verbose_train = torchmetrics.MetricCollection(metrics_dict_verbose, prefix="train_")
                self.metrics_verbose_val = self.metrics_verbose_train.clone(prefix="val_")
                self.metrics_verbose_test = self.metrics_verbose_train.clone(prefix="test_")
            else:
                self.metrics_verbose_train = None
                self.metrics_verbose_val = None
                self.metrics_verbose_test = None
        else:
            all_metrics = {}
            all_metrics.update(metrics_dict_epoch)
            all_metrics.update(metrics_dict_step)
            all_metrics.update(metrics_dict_verbose)
            self.metrics = torchmetrics.MetricCollection(all_metrics)
