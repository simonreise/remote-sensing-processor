Regression
=====================
.. automodule:: remote_sensing_processor.regression
   :members: generate_tiles, train, test, generate_map, band_importance
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
   * - Data2Vec
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/data2vec>`_
   * - DPT
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/dpt>`_
   * - MobileNetV2
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/mobilenet_v2>`_
   * - MobileViT
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/mobilevit>`_
   * - MobileViTV2
     - Not available
     - `Huggingface Transformers <https://huggingface.co/docs/transformers/main/en/model_doc/mobilevitv2>`_
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
   * - Linear Regression
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_
   * - Ridge
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_
   * - Bayesian Ridge
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html>`_
   * - Lasso
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_
   * - Multitask Lasso
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html>`_
   * - Lars
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html>`_
   * - LassoLars
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html>`_
   * - LassoLarsIC
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html>`_
   * - ElasticNet
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_
   * - Multitask ElasticNet
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html>`_
   * - Orthogonal Matching Pursuit
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html>`_
   * - ARD
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html>`_
   * - Huber
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html>`_
   * - RANSAC
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html>`_
   * - Theil-Sen
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html>`_
   * - Gamma
     - "lbfgs", "newton-cholesky"
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html>`_
   * - Poisson
     - "lbfgs", "newton-cholesky"
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html>`_
   * - Tweedie
     - "lbfgs", "newton-cholesky"
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html>`_
   * - SGD
     - "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html>`_
   * - Nearest Neighbors
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`_
   * - Radius Neighbors
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html>`_
   * - SVM
     - "rbf", "linear", "poly", "sigmoid"
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_
   * - Gaussian Process
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_
   * - Decision Tree
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html>`_
   * - Extra Tree
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html>`_
   * - Random Forest
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_
   * - Extra Trees
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html>`_
   * - AdaBoost
     - Not available
     - Not supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html>`_
   * - Gradient Boosting
     - Not available
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html>`_
   * - Multilayer Perceptron
     - "adam", "sgd", "lbfgs"
     - Supported
     - `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`_
   * - XGBoost
     - Not available
     - Not supported
     - `XGBoost <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor>`_
   * - XGB Random Forest
     - Not available
     - Not supported
     - `XGBoost <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRFRegressor>`_

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
   * - mse
     - `Torch <https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_
   * - mae
     - `Torch <https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html>`_

You can also use your custom loss. It can be useful if you want to initialize a loss with custom parameters.
You also can pass any custom function as a loss. The only limit - it must inherit ``torch.nn.modules.loss._Loss``.

.. code-block:: python
   # Here we just use one of the default losses
   loss = "mae"

   # Here we initialize a loss that is already supported, but set the custom parameters
   loss = torch.nn.MSELoss(reduction="sum")

   # And here we use the loss that is not supported by default
   loss = torch.nn.modules.loss.SmoothL1Loss()

   # And here we create our own custom loss function
   class XTanhLoss(_Loss):
       def __init__(self):
           super().__init__()

       def forward(self, y_t, y_prime_t):
           ey_t = y_t - y_prime_t
           return torch.mean(ey_t * torch.tanh(ey_t))

   loss = XTanhLoss()

List of available metrics
-------------------------

.. list-table:: Supported metrics
   :widths: 40 30 30
   :header-rows: 1
   * - Metric
     - Additional parameters
     - Reference
   * - concordance_correlation_coefficient
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/concordance_corr_coef.html>`_
   * - cosine_similarity
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/cosine_similarity.html>`_
   * - critical_success_index
     - threshold=0.5
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/critical_success_index.html>`_
   * - explained_variance
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/explained_variance.html>`_
   * - kendall_rank_correlation_coefficient_a
     - variant="a"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/kendall_rank_corr_coef.html>`_
   * - kendall_rank_correlation_coefficient_b
     - variant="b"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/kendall_rank_corr_coef.html>`_
   * - kendall_rank_correlation_coefficient_c
     - variant="c"
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/kendall_rank_corr_coef.html>`_
   * - kl_divergence
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/kl_divergence.html>`_
   * - log_cosh_error
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/log_cosh_error.html>`_
   * - mae
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_error.html>`_
   * - mape
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_percentage_error.html>`_
   * - mse
     - squared=True
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html>`_
   * - msle
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_log_error.html>`_
   * - manhattan_distance
     - p=1
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/minkowski_distance.html>`_
   * - euclidean_distance
     - p=2
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/minkowski_distance.html>`_
   * - minkowski_distance_3
     - p=1
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/minkowski_distance.html>`_
   * - minkowski_distance_10
     - p=10
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/minkowski_distance.html>`_
   * - minkowski_distance_100
     - p=100
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/minkowski_distance.html>`_
   * - nrmse
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/normalized_root_mean_squared_error.html>`_
   * - pearson_correlation_coefficient
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/pearson_corr_coef.html>`_
   * - r2
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html>`_
   * - rse
     - squared=True
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/rse.html>`_
   * - rmse
     - squared=False
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html>`_
   * - rrse
     - squared=False
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/rse.html>`_
   * - spearman_correlation_coefficient
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/spearman_corr_coef.html>`_
   * - smape
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/symmetric_mean_absolute_percentage_error.html>`_
   * - tweedie_deviance_score
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/tweedie_deviance_score.html>`_
   * - weighted_mape
     - None
     - `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/regression/weighted_mean_absolute_percentage_error.html>`_

You also can use any custom metric for evaluation. The only limit - it must inherit ``torchmetrics.metric.Metric``.

.. code-block:: python
   # A custom metric (top-2 accuracy) - a supported metric, but with custom parameter - ``top-k=2``
   # Don't forget to set task, num_classes and ignore_index
   distance4 = torchmetrics.MinkowskiDistance(4)

   metrics=[
       {"name": "r2", "log": "verbose"}, # Supported metric, logged on each step and printed in the progress bar
       {"name": "rmse", "log": "step"}, # Supported metric, logged each step and saved to logs
       {"name": "kendall_rank_correlation_coefficient_a", "log": "epoch"}, # Supported metric, logged each epoch and saved to logs
       {"name": "minkowski_distance_4", "metric": distance4, "log": "verbose"}, # A custom metric
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
