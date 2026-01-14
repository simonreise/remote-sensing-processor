Semantic segmentation
=====================
.. automodule:: remote_sensing_processor.semantic
   :members: generate_tiles, train, test, generate_map, band_importance, confusion_matrix
   :show-inheritance:
   
List of available NN models
---------------------------

.. list-table:: Supported ML Models
   :widths: 25 50 25
   :header-rows: 1

   * - Model Name
     - Backbone
     - Reference
   * - BEiT
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/beit>`_
   * - ConditionalDETR
     - See Transformers backbones
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/conditional_detr>`_
   * - Data2Vec
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/data2vec>`_
   * - DETR
     - See Transformers backbones
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/data2vec>`_
   * - DPT
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/dpt>`_
   * - EoMT
     - See Transformers backbones
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/eomt>`_
   * - Mask2Former
     - See Transformers backbones
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/mask2former>`_
   * - MaskFormer
     - See Transformers backbones
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/maskformer>`_
   * - MobileNetV2
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/mobilenet_v2>`_
   * - MobileViT
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/mobilevit>`_
   * - MobileViTV2
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/mobilevitv2>`_
   * - OneFormer
     - See Transformers backbones
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/oneformer>`_
   * - SegFormer
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/segformer>`_
   * - UperNet
     - See Transformers backbones
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/upernet>`_
   * - DeepLabV3
     - "MobileNet_V3_Large", "ResNet50", "ResNet101"
     - `Torchvision <https://docs.pytorch.org/vision/stable/models/deeplabv3.html>`_
   * - FCN
     - "ResNet50", "ResNet101"
     - `Torchvision <https://docs.pytorch.org/vision/stable/models/fcn.html>`_
   * - LRASPP
     - Not available
     - `Torchvision <https://docs.pytorch.org/vision/stable/models/lraspp.html>`_
   * - UNet
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#unet>`_
   * - UNet++
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#unetplusplus>`_
   * - FPN
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#fpn>`_
   * - PSPNet
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#pspnet>`_
   * - DeepLabV3_smp
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#deeplabv3>`_
   * - DeepLabV3+
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus>`_
   * - Linknet
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#linknet>`_
   * - MAnet
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#manet>`_
   * - PAN
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#pan>`_
   * - UperNet_smp
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#upernet>`_
   * - SegFormer_smp
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#segformer>`_
   * - DPT_smp
     - `SMP <https://smp.readthedocs.io/en/latest/encoders.html>`_ and `TIMM <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/models.html#dpt>`_
   * - FarSeg
     - Not available
     - `TorchGeo <https://torchgeo.readthedocs.io/en/stable/api/models/farseg.html>`_

Transformers backbones are:

- BEiT
- BiT
- ConvNeXT
- ConvNeXTV2
- DiNAT
- DINOV2
- DINOV2WithRegisters
- DINOV3ViT
- DINOV3ConvNeXT
- FocalNet
- HGNet-V2
- Hiera
- MaskFormer-Swin
- NAT
- PVTV2
- ResNet
- RT-DETR-ResNet
- Swin
- SwinV2
- ViTDet
- Any TIMM backbone (experimental support)

You can fine-tune pre-trained model by defining ``weights``. For models from Transformers you can get available weights from `Huggingface Hub <https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads>`_, for Torchvision models you just set ``weights = True``.

``rsp.segmentation.train`` also saves CSV and Tensorboard logs in directory where checkpoint file is saved.

DiNAT and NAT backbones require ``natten`` library, that is not available on Windows and Mac and not available via Conda. RSP supports DiNAT backbone, but you need to install ``natten`` in your python env manually.


List of available Scikit-learn models
-------------------------------------

.. list-table:: Supported Scikit-Learn Models
   :widths: 20 40 20 20
   :header-rows: 1

   * - Model Name
     - Kernel/solver
     - Warm start
     - Reference
   * - Logistic Regression
     - "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
     - Only for lbfgs, newton-cg, sag, saga solvers
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
   * - Ridge
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html>`_
   * - SGD
     - "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_
   * - Nearest Neighbors
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
   * - Radius Neighbors
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html>`_
   * - SVM
     - "rbf", "linear", "poly", "sigmoid"
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
   * - Gaussian Process
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html>`_
   * - Naive Bayes
     - "gaussian", "bernoulli", "categorical", "complement", "multinomial"
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/naive_bayes.html>`_
   * - QDA
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html>`_
   * - LDA
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html>`_
   * - Decision Tree
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_
   * - Extra Tree
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html>`_
   * - Random Forest
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
   * - Extra Trees
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>`_
   * - AdaBoost
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html>`_
   * - Gradient Boosting
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`_
   * - Multilayer Perceptron
     - "adam", "sgd", "lbfgs"
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`_
   * - XGBoost
     - Not available
     - Not supported
     - `XGBoost <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier>`_
   * - XGB Random Forest
     - Not available
     - Not supported
     - `XGBoost <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRFClassifier>`_

Model kernel or solver is defined with ``backbone`` argument.

Models that support warm start can be fine-tuned using pre-trained models with ``checkpoint`` argument.

Some models can have issues while saving, especially when trained on big datasets. Some models (like SVM) can train for a very long time or (like Gaussian process) can have memory issues with big datasets. So we recommend using Scikit-learn models only for small datasets.

For Random Forest and Extra Trees models ``max_depth`` is by default set to 6, because it is unlimited by default and the training could be very slow. To train with unlimited tree depth ``set max_depth = None``.

List of available losses
------------------------

.. list-table:: Supported loss functions
   :widths: 50 50
   :header-rows: 1

   * - Loss
     - Reference
   * - cross_entropy
     - `Torch <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_
   * - nll
     - `Torch <https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html>`_
   * - jaccard
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/losses.html#jaccardloss>`_
   * - dice
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/losses.html#diceloss>`_
   * - tversky
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/losses.html#tverskyloss>`_
   * - focal
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/losses.html#focalloss>`_
   * - lovasz
     - `Segmentation Models Pytorch <https://smp.readthedocs.io/en/latest/losses.html#lovaszloss>`_

You can also use your custom loss. It can be useful if you want to initialize a loss with custom parameters.
You also can pass any custom function as a loss. The only limit - it must inherit ``torch.nn.modules.loss._Loss``.

.. code-block:: python
   # Here we just use one of the default losses
   loss = "tversky"

   # Here we initialize a loss that is already supported, but set the custom parameters
   # Don't forget to add the "mode" and "ignore_index" parameters
   loss = segmentation_models_pytorch.losses.FocalLoss(
       mode="multiclass",
       ignore_index=0,
       alpha=0.25,
       normalized=True,
       reduced_threshold=0.5,
   )

   # And here we use the loss that is not supported by default
   loss = monai.losses.GeneralizedDiceFocalLoss()

   # And here we create our own custom loss function
   from torch.nn.modules.loss import _Loss

   class SoftBCEWithLogitsLoss(_Loss):
       """
       Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
       - Support of ignore_index value
       - Support of label smoothing
       Copied from https://github.com/BloodAxe/pytorch-toolbelt
       """

       __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

       def __init__(
           self, weight=None, ignore_index: Optional[int] = -100, reduction="mean", smooth_factor=None, pos_weight=None
       ):
           super().__init__()
           self.ignore_index = ignore_index
           self.reduction = reduction
           self.smooth_factor = smooth_factor
           self.register_buffer("weight", weight)
           self.register_buffer("pos_weight", pos_weight)

       def forward(self, input: Tensor, target: Tensor) -> Tensor:
           if self.smooth_factor is not None:
               soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
           else:
               soft_targets = target.type_as(input)

           loss = F.binary_cross_entropy_with_logits(
               input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction="none"
           )

           if self.ignore_index is not None:
               not_ignored_mask: Tensor = target != self.ignore_index
               loss *= not_ignored_mask.type_as(loss)

           if self.reduction == "mean":
               loss = loss.mean()

           if self.reduction == "sum":
               loss = loss.sum()

           return loss

       loss = SoftBCEWithLogitsLoss(ignore_index=0, smooth_factor=0.5)

List of available metrics
-------------------------

.. list-table:: Supported metrics
   :widths: 40 30 30
   :header-rows: 1

   * - Metric
     - Additional parameters
     - Reference
   * - accuracy_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html>`_
   * - accuracy_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html>`_
   * - cohen_kappa
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/cohen_kappa.html>`_
   * - exact_math
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/exact_match.html>`_
   * - f1_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html>`_
   * - f1_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html>`_
   * - hamming_distance_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/hamming_distance.html>`_
   * - hamming_distance_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/hamming_distance.html>`_
   * - jaccard_index_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html>`_
   * - jaccard_index_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html>`_
   * - matthews_correlation_coefficient
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/matthews_corr_coef.html>`_
   * - negative_predictive_value_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/negative_predictive_value.html>`_
   * - negative_predictive_value_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/negative_predictive_value.html>`_
   * - precision_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/precision.html>`_
   * - precision_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/precision.html>`_
   * - recall_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/recall.html>`_
   * - recall_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/classification/recall.html>`_
   * - dice_score_macro
     - average="macro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html>`_
   * - dice_score_micro
     - average="micro"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html>`_
   * - generalized_dice_score
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/segmentation/generalized_dice.html>`_
   * - mean_iou
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html>`_

For most of the semantic segmentation metrics micro and macro-averaged versions are available by default.

You also can use any custom metric for evaluation. The only limit - it must inherit ``torchmetrics.metric.Metric``.

.. code-block:: python
   # A custom metric (top-2 accuracy) - a supported metric, but with custom parameter - ``top-k=2``
   # Don't forget to set task, num_classes and ignore_index
   a_micro_2 = torchmetrics.Accuracy(
       task="multiclass",
       num_classes=10,
       average="micro",
       ignore_index=0,
       top_k=2,
   )

   # A custom metric (f-beta score) - a metric, that is not supported by default
   # Don't forget to set task, num_classes and ignore_index
   f_beta = torchmetrics.FBetaScore(
       task="multiclass",
       beta=2.0,
       num_classes=10,
       average="micro",
       ignore_index=0,
   )

   metrics=[
       {"name": "accuracy_micro", "log": "verbose"}, # Supported metric, logged on each step and printed in the progress bar
       {"name": "accuracy_macro", "log": "step"}, # Supported metric, logged each step and saved to logs
       {"name": "mean_iou", "log": "epoch"}, # Supported metric, logged each epoch and saved to logs
       {"name": "accuracy_micro_2", "metric": a_micro_2, "log": "verbose"}, # A custom metric
       {"name": "f_beta_2", "metric": f_beta, "log": "step"}, # Another custom metric
   ]

Supported augmentations
-----------------------

.. list-table:: Supported augmentations
   :widths: 40 30 30
   :header-rows: 1

   * - Augmentation
     - Additional parameters
     - Reference
   * - ScaleJitter
     - None
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.ScaleJitter.html>`_
   * - RandomResizedCrop
     - antialias=True
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomResizedCrop.html>`_
   * - RandomHorizontalFlip
     - p=0.5
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomHorizontalFlip.html>`_
   * - RandomVerticalFlip
     - p=0.5
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomVerticalFlip.html>`_
   * - RandomZoomOut
     - None
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomZoomOut.html>`_
   * - RandomRotation
     - degrees=90
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomRotation.html>`_
   * - RandomAffine
     - degrees=90, translate=(0.5, 0.5), shear=0.5
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html>`_
   * - RandomPerspective
     - None
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomPerspective.html>`_
   * - ElasticTransform
     - None
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.ElasticTransform.html>`_
   * - GaussianBlur
     - kernel_size=(5, 9)
     - `Torchvision <https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.GaussianBlur.html>`_

If you just pass ``augment=True``, RSP will use a default sequence of augmentations: ("RandomResizedCrop", "RandomHorizontalFlip").
You can pass your own sequence of augmentations, they will be applied to data in the given order.
You can use both supported augmentation names or custom augmentations.
You can use any custom augmentations, but they must inherit ``torchvision.transforms.v2.Transform``.

.. code-block:: python
   augment = (
       "RandomResizedCrop", # Supported augmentation name
       v2.ScaleJitter(target_size=(128, 128), scale_range=(0.5, 1.5)) # Custom augmentation - a supported one with custom parameter
       v2.RandomRotation(45, fill={tv_tensors.Image: 0, tv_tensors.Mask: 0}) # Custom augmentation - a supported one with custom parameter
       v2.RandomSolarize(), # Custom augmentation - not supported by default, works only with 3-channel images
   )
