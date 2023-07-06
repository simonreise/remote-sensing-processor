Semantic segmentation
=====================
.. automodule:: remote_sensing_processor.segmentation
   :members: generate_tiles, train, test, generate_map
   :show-inheritance:
   
## List of available NN models

| Model           | Backbone           | Classification | Regression | No metrics-related freeze issues  | No convergence issues  | Fine-tuning available  | Reference                                                                                                                           |
|-----------------|--------------------|----------------|------------|-----------------------------------|------------------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| BEiT            |                    | +              | -[^4]      | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/beit)                                         |
| ConditionalDETR |                    | +              | +          | +-[^1]                            | +                      | Not tested             | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/conditional_detr)                             |
| Data2Vec        |                    | +              | -[^4]      | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec)                                     |
| DETR            |                    | +              | +          | +-[^1]                            | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/detr)                                         |
| DPT             |                    | +              | -[^4]      | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/dpt)                                          |
| Mask2Former     |                    | +              | -[^4]      | +-[^1]                            | +-[^2]                 | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mask2former)                                  |
| MaskFormer      |                    | +              | -[^4]      | +                                 | +-[^2]                 | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/maskformer)                                   |
| MobileNetV2     |                    | +              | +          | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mobilenet_v2)                                 |
| MobileViT       |                    | +              | +          | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mobilevit)                                    |
| MobileViTV2     |                    | +              | +          | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mobilevitv2)                                  |
| OneFormer       | Swin               | +              | -[^4]      | +-[^1]                            | +-[^2]                 | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer)                                    |
|                 | ConvNeXT           | +              | -[^4]      | +-[^1]                            | +-[^2]                 | Not tested             |                                                                                                                                     |
|                 | ConvNeXTV2         | +              | -[^4]      | +-[^1]                            | +-[^2]                 | Not tested             |                                                                                                                                     |
|                 | DiNAT              | -[^3]          |            |                                   |                        |                        |                                                                                                                                     |
| SegFormer       |                    | +              | -[^4]      | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/segformer)                                    |
| UperNet         | Swin               | +              | -[^4]      | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/upernet)                                      |
|                 | ResNet             | +              | +          | +                                 | +                      | Not tested             |                                                                                                                                     |
|                 | ConvNeXT           | +              | +          | +                                 | +                      | +                      |                                                                                                                                     |
|                 | ConvNeXTV2         | +              | +          | +                                 | +                      | Not tested             |                                                                                                                                     |
| DeepLabV3       | MobileNet_V3_Large | +              | +          | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html) |
|                 | ResNet50           | +              | +          | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html)           |
|                 | ResNet101          | +              | +          | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html)          |
| FCN             | ResNet50           | +              | +          | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet50.html)                 |
|                 | ResNet101          | +              | +          | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet101.html)                |
| LRASPP          |                    | +              | +          | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)    |

You can fine-tune pre-trained model by defining `weights`. For models from Transformers you can get available weights from [Huggingface Hub](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads), for Torchvision models you just set `weights = True`.

`rsp.segmentation.train` also saves CSV and Tensorboard logs in directory where checkpoint file is saved.

## List of available Scikit-learn models

| Model                 | Kernel/solver[^5] | Classification | Regression | Warm start | Reference                                                                                                                          |
|-----------------------|-------------------|----------------|------------|------------|------------------------------------------------------------------------------------------------------------------------------------|
| Nearest Neighbors     |                   | +              | +          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)                      |
| Logistic Regression   | lbfgs             | +              | -          | +          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)                     |
|                       | liblinear         | +              | -          | -          |                                                                                                                                    |
|                       | newton-cg         | +              | -          | +          |                                                                                                                                    |
|                       | newton-cholesky   | +              | -          | +          |                                                                                                                                    |
|                       | sag               | +              | -          | +          |                                                                                                                                    |
|                       | saga              | +              | -          | +          |                                                                                                                                    |
| Ridge                 |                   | -              | +          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)                                  |
| Lasso                 |                   | -              | +          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)                                  |
| ElasticNet            |                   | -              | +          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)                             |
| SVM                   | rbf               | +              | +          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)                                             |
|                       | linear            | +              | +          | -          |                                                                                                                                    |
|                       | poly              | +              | +          | -          |                                                                                                                                    |
|                       | sigmoid           | +              | +          | -          |                                                                                                                                    |
| Gaussian Process      |                   | +              | +          | +          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)          |
| Naive Bayes           |                   | +              | -          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)                              |
| QDA                   |                   | +              | -          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html) |
| Decision Tree         |                   | +              | +          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)                         |
| Random Forest         |                   | +              | +          | +          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)                     |
| AdaBoost              |                   | +              | +          | -          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)                         |
| Gradient Boosting     |                   | +              | +          | +          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)             |
| Multilayer Perceptron |                   | +              | +          | +          | [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)                        |

Models that support warm start can be fine-tuned using pre-trained models with `checkpoint` arg.

Some models can have issues while saving, especially when trained on big datasets. Some models (like SVM) can train for a very long time or (like Gaussian process) can have memory issues with big datasets. So we recommend using Scikit-learn models only for small datasets.

[^1]: These models can freeze forever on one of first steps. It happens due to some confusion matrix-based metrics related error. Try to restart training with `less_metrics = True`.

[^2]: These models showed very poor performance on tests. They only converged slowly with `lr=1e-5`, but much more slow than other models. I do not recommend using them.

[^3]: DiNAT requires `natten` library, that is not available on Windows and Mac and not available via Conda. RSP supports DiNAT backbone, but you need to install `natten` in your python env manually.

[^4]: Loss becomes `nan` while training several transformers if the task is regression.

[^5]: Is defined with `backbone`.