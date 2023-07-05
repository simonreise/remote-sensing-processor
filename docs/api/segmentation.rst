Semantic segmentation
=====================
.. automodule:: remote_sensing_processor.segmentation
   :members: generate_tiles, train, test, generate_map
   :show-inheritance:
   
List of available models

| Model           | Backbone           | Works | No metrics-related freeze issues  | No convergence issues  | Fine-tuning available  | Reference                                                                                                                           |
|-----------------|--------------------|-------|-----------------------------------|------------------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| BEiT            |                    | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/beit)                                         |
| ConditionalDETR |                    | +     | +-[^1]                            | +                      | Not tested             | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/conditional_detr)                             |
| Data2Vec        |                    | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec)                                     |
| DETR            |                    | +     | +-[^1]                            | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/detr)                                         |
| DPT             |                    | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/dpt)                                          |
| Mask2Former     |                    | +     | +-[^1]                            | +-[^2]                 | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mask2former)                                  |
| MaskFormer      |                    | +     | +                                 | +-[^2]                 | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/maskformer)                                   |
| MobileNetV2     |                    | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mobilenet_v2)                                 |
| MobileViT       |                    | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mobilevit)                                    |
| MobileViTV2     |                    | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/mobilevitv2)                                  |
| OneFormer       | Swin               | +     | +-[^1]                            | +-[^2]                 | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer)                                    |
|                 | ConvNeXT           | +     | +-[^1]                            | +-[^2]                 | Not tested             |                                                                                                                                     |
|                 | ConvNeXTV2         | +     | +-[^1]                            | +-[^2]                 | Not tested             |                                                                                                                                     |
|                 | DiNAT              | -[^3] |                                   |                        |                        |                                                                                                                                     |
| SegFormer       |                    | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/segformer)                                    |
| UperNet         | Swin               | +     | +                                 | +                      | +                      | [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/upernet)                                      |
|                 | ResNet             | +     | +                                 | +                      | Not tested             |                                                                                                                                     |
|                 | ConvNeXT           | +     | +                                 | +                      | +                      |                                                                                                                                     |
|                 | ConvNeXTV2         | +     | +                                 | +                      | Not tested             |                                                                                                                                     |
| DeepLabV3       | MobileNet_V3_Large | +     | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html) |
|                 | ResNet50           | +     | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html)           |
|                 | ResNet101          | +     | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html)          |
| FCN             | ResNet50           | +     | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet50.html)                 |
|                 | ResNet101          | +     | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet101.html)                |
| LRASPP          |                    | +     | +                                 | +                      | +                      | [Torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)    |

You can fine-tune pre-trained model by defining `weights`. For models from Transformers you can get available weights from [Huggingface Hub](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads), for Torchvision models you just set `weights = True`.

`rsp.segmentation.train` also saves CSV and Tensorboard logs in directory where checkpoint file is saved.

[^1]: These models can freeze forever on one of first steps. It happens due to some confusion matrix-based metrics related error. Try to restart training with `less_metrics = True`.

[^2]: These models showed very poor performance on tests. They only converged slowly with `lr=1e-5`, but much more slow than other models. I do not recommend using them.

[^3]: DiNAT requires `natten` library, that is not available on Windows and Mac and not available via Conda. RSP supports DiNAT backbone, but you need to install `natten` in your python env manually.