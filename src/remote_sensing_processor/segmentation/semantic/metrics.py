"""Torch metrics for semantic segmentation."""

from pydantic import TypeAdapter
from typing import Optional

import torchmetrics
import torchmetrics.segmentation

from remote_sensing_processor.common.types import ListOfMetrics


semantic_metrics = [
    "accuracy_macro",
    "accuracy_micro",
    "cohen_kappa",
    "exact_math",
    "f1_macro",
    "f1_micro",
    "hamming_distance_macro",
    "hamming_distance_micro",
    "jaccard_index_macro",
    "jaccard_index_micro",
    "matthews_correlation_coefficient",
    "negative_predictive_value_macro",
    "negative_predictive_value_micro",
    "precision_macro",
    "precision_micro",
    "recall_macro",
    "recall_micro",
    "dice_score_macro",
    "dice_score_micro",
    "generalized_dice_score",
    "mean_iou",
]


class SemanticSegmentationMetrics:
    """Semantic segmentation metrics base class."""

    num_classes: Optional[int]
    y_nodata: Optional[int]

    def setup_metrics(
        self,
        metrics: Optional[list[dict]],
        sklearn: Optional[bool] = False,
    ) -> None:
        """Setup metrics."""
        # Setting up default metrics if not set
        if metrics is None:
            metrics = [
                {"name": "accuracy_micro", "log": "verbose"},
                {"name": "accuracy_macro", "log": "step"},
                {"name": "precision_macro", "log": "epoch"},
                {"name": "precision_micro", "log": "epoch"},
                {"name": "recall_macro", "log": "epoch"},
                {"name": "recall_micro", "log": "epoch"},
                {"name": "f1_macro", "log": "epoch"},
                {"name": "f1_micro", "log": "epoch"},
                {"name": "mean_iou", "log": "verbose"},
            ]
        metrics = TypeAdapter(ListOfMetrics).validate_python(metrics)

        metrics_dict_epoch = {}
        metrics_dict_step = {}
        metrics_dict_verbose = {}

        for metric in metrics:
            name = metric.name
            log = metric.log
            if name in semantic_metrics:
                # Setting up supported metrics
                if name == "accuracy_macro":
                    m = torchmetrics.Accuracy(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="macro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "accuracy_micro":
                    m = torchmetrics.Accuracy(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="micro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "cohen_kappa":
                    m = torchmetrics.CohenKappa(
                        task="multiclass",
                        num_classes=self.num_classes,
                        ignore_index=self.y_nodata,
                    )
                elif name == "exact_math":
                    m = torchmetrics.ExactMatch(
                        task="multiclass",
                        num_classes=self.num_classes,
                        ignore_index=self.y_nodata,
                    )
                elif name == "f1_macro":
                    m = torchmetrics.F1Score(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="macro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "f1_micro":
                    m = torchmetrics.F1Score(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="micro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "hamming_distance_macro":
                    m = torchmetrics.HammingDistance(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="macro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "hamming_distance_micro":
                    m = torchmetrics.HammingDistance(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="micro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "jaccard_index_macro":
                    m = torchmetrics.JaccardIndex(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="macro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "jaccard_index_micro":
                    m = torchmetrics.JaccardIndex(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="micro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "matthews_correlation_coefficient":
                    m = torchmetrics.MatthewsCorrCoef(
                        task="multiclass",
                        num_classes=self.num_classes,
                        ignore_index=self.y_nodata,
                    )
                elif name == "negative_predictive_value_macro":
                    m = torchmetrics.NegativePredictiveValue(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="macro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "negative_predictive_value_micro":
                    m = torchmetrics.NegativePredictiveValue(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="micro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "precision_macro":
                    m = torchmetrics.Precision(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="macro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "precision_micro":
                    m = torchmetrics.Precision(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="micro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "recall_macro":
                    m = torchmetrics.Recall(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="macro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "recall_micro":
                    m = torchmetrics.Recall(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="micro",
                        ignore_index=self.y_nodata,
                    )
                elif name == "dice_score_macro":
                    m = torchmetrics.segmentation.DiceScore(
                        num_classes=self.num_classes,
                        average="macro",
                        input_format="index" if sklearn else "mixed",
                        include_background=False,
                    )
                elif name == "dice_score_micro":
                    m = torchmetrics.segmentation.DiceScore(
                        num_classes=self.num_classes,
                        average="micro",
                        input_format="index" if sklearn else "mixed",
                        include_background=False,
                    )
                elif name == "generalized_dice_score":
                    m = torchmetrics.segmentation.GeneralizedDiceScore(
                        num_classes=self.num_classes,
                        input_format="index" if sklearn else "mixed",
                        include_background=False,
                    )
                elif name == "mean_iou":
                    m = torchmetrics.segmentation.MeanIoU(
                        num_classes=self.num_classes,
                        input_format="index" if sklearn else "mixed",
                        include_background=False,
                    )
                else:
                    raise ValueError(name + " is not a valid semantic segmentation metric")
            else:
                # Custom metrics
                if metric.metric is None:
                    raise ValueError(f"Custom metric {name} is not set.")
                m = metric.metric

                # Check if custom metric has right nodata and num_classes values
                if hasattr(m, "ignore_index") and m.ignore_index != self.y_nodata:
                    raise ValueError(
                        f"Looks like you set wrong ignore index to metric {name}. "
                        f"The value is {m.ignore_index}, should be {self.y_nodata}",
                    )

                if hasattr(m, "num_classes") and m.num_classes != self.num_classes:
                    raise ValueError(
                        f"Looks like you set wrong num_classes to metric {name}. "
                        f"The value is {m.num_classes}, should be {self.num_classes}",
                    )

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
