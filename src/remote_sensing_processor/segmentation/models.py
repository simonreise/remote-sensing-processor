import torch
import torchvision
import transformers


def load_model(model, bb, weights, input_shape, input_dims, num_classes):
    if model == 'BEiT':
        if weights != None:
            model = transformers.BeitForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.BeitConfig(image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
            model = transformers.BeitForSemanticSegmentation(config)
    elif model == 'ConditionalDETR':
        if weights != None:
            model = transformers.ConditionalDetrForSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.ConditionalDetrConfig(num_channels = input_dims, num_labels = num_classes)
            model = transformers.ConditionalDetrForSegmentation(config)        
    elif model == 'Data2Vec':
        if weights != None:
            model = transformers.Data2VecVisionForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.Data2VecVisionConfig(image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
            model =  transformers.Data2VecVisionForSemanticSegmentation(config)
    elif model == 'DETR':
        if weights != None:
            model = transformers.DetrForSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.DetrConfig(num_channels = input_dims, num_labels = num_classes)
            model = transformers.DetrForSegmentation(config)
    elif model == 'DPT':
        if weights != None:
            model = transformers.DPTForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.DPTConfig(image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
            model = transformers.DPTForSemanticSegmentation(config)
    elif model == 'Mask2Former':
        if weights != None:
            config = transformers.AutoConfig.from_pretrained(weights, num_labels = num_classes)
            if hasattr(config.backbone_config, "image_size"):
                config.backbone_config.image_size = input_shape
            if hasattr(config.backbone_config, "num_channels"):
                config.backbone_config.num_channels = input_dims
            model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, config = config)
        else:
            backbone = transformers.SwinConfig(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
            config = transformers.Mask2FormerConfig(backbone_config = backbone, num_labels = num_classes)
            model = transformers.Mask2FormerForUniversalSegmentation(config)
    elif model == 'MaskFormer':
        if weights != None:
            config = transformers.AutoConfig.from_pretrained(weights, num_labels = num_classes)
            if hasattr(config.backbone_config, "image_size"):
                config.backbone_config.image_size = input_shape
            if hasattr(config.backbone_config, "num_channels"):
                config.backbone_config.num_channels = input_dims
            model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, config = config)
        else:
            backbone = transformers.SwinConfig(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
            config = transformers.MaskFormerConfig(backbone_config = backbone, num_labels = num_classes)
            model = transformers.MaskFormerForInstanceSegmentation(config)
    elif model == 'MobileNetV2':
        if weights != None:
            model = transformers.MobileNetV2ForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.MobileNetV2Config(image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
            model = transformers.MobileNetV2ForSemanticSegmentation(config)
    elif model == 'MobileViT':
        if weights != None:
            model = transformers.MobileViTForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.MobileViTConfig(image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
            model = transformers.MobileViTForSemanticSegmentation(config)
    elif model == 'MobileViTV2':
        if weights != None:
            model = transformers.MobileViTV2ForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.MobileViTV2Config(image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
            model = transformers.MobileViTV2ForSemanticSegmentation(config)
    elif model == 'OneFormer':
        if weights != None:
            config = transformers.AutoConfig.from_pretrained(weights, num_labels = num_classes)
            if hasattr(config.backbone_config, "image_size"):
                config.backbone_config.image_size = input_shape
            if hasattr(config.backbone_config, "num_channels"):
                config.backbone_config.num_channels = input_dims
            model = transformers.OneFormerForUniversalSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, config = config)
        else:
            if bb == 'Swin' or bb == None:
                backbone = transformers.SwinConfig(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.OneFormerConfig(backbone_config = backbone, num_labels = num_classes)
            elif bb == 'ConvNeXT':
                backbone = transformers.ConvNextConfig(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.OneFormerConfig(backbone_config = backbone, num_labels = num_classes)
            elif bb == 'ConvNeXTV2':
                backbone = transformers.ConvNextV2Config(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.OneFormerConfig(backbone_config = backbone, num_labels = num_classes)
            #currently not supported because there's no natten package in conda and no windows support
            elif bb == 'DiNAT':
                backbone = transformers.DinatConfig(num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.OneFormerConfig(backbone_config = backbone, num_labels = num_classes)
            model = transformers.OneFormerForUniversalSegmentation(config)
    elif model == 'SegFormer':
        if weights != None:
            model = transformers.SegformerForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, image_size = input_shape, num_channels = input_dims, num_labels = num_classes)
        else:
            config = transformers.SegformerConfig(num_channels = input_dims, num_labels = num_classes)
            model = transformers.SegformerForSemanticSegmentation(config)
    elif model == 'UperNet':
        if weights != None:
            config = transformers.AutoConfig.from_pretrained(weights, num_labels = num_classes)
            if hasattr(config.backbone_config, "image_size"):
                config.backbone_config.image_size = input_shape
            if hasattr(config.backbone_config, "num_channels"):
                config.backbone_config.num_channels = input_dims
            model = transformers.UperNetForSemanticSegmentation.from_pretrained(weights, ignore_mismatched_sizes=True, config = config)
        else:
            if bb == 'ResNet':
                backbone = transformers.ResNetConfig(num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.UperNetConfig(backbone_config = backbone, num_labels = num_classes, auxiliary_in_channels = 1024)
            elif bb == 'Swin' or bb == None:
                backbone = transformers.SwinConfig(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.UperNetConfig(backbone_config = backbone, num_labels = num_classes)
            elif bb == 'ConvNeXT':
                backbone = transformers.ConvNextConfig(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.UperNetConfig(backbone_config = backbone, num_labels = num_classes)
            elif bb == 'ConvNeXTV2':
                backbone = transformers.ConvNextV2Config(image_size = input_shape, num_channels = input_dims, out_features=["stage1", "stage2", "stage3", "stage4"])
                config = transformers.UperNetConfig(backbone_config = backbone, num_labels = num_classes)
            model = transformers.UperNetForSemanticSegmentation(config)
    elif model == 'DeepLabV3':
        if bb == 'MobileNet_V3_Large' or bb == None:
            if weights != None:
                weights = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights = weights)
                model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1, stride=(1, 1))
                model.aux_classifier[4] = torch.nn.Conv2d(10, num_classes, kernel_size=1, stride=(1, 1))
            else:
                model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(num_classes = num_classes)
            model.backbone['0'][0] = torch.nn.Conv2d(input_dims, 16, kernel_size=3, stride=(2, 2), padding=(1, 1), bias=False)
        elif bb == 'ResNet50':
            if weights != None:
                weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                model = torchvision.models.segmentation.deeplabv3_resnet50(weights = weights)
                model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1, stride=(1, 1))
            else:    
                model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes = num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(input_dims, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
        elif bb == 'ResNet101':
            if weights != None:
                weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
                model = torchvision.models.segmentation.deeplabv3_resnet101(weights = weights)
                model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1, stride=(1, 1))
            else:    
                model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes = num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(input_dims, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
    elif model == 'FCN':
        if bb == 'ResNet50' or bb == None:
            if weights != None:
                weights = torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                model = torchvision.models.segmentation.fcn_resnet50(weights = weights)
                model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1, stride=(1, 1))
            else:    
                model = torchvision.models.segmentation.fcn_resnet50(num_classes = num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(input_dims, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
        elif bb == 'ResNet101':
            if weights != None:
                weights = torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
                model = torchvision.models.segmentation.fcn_resnet101(weights = weights)
                model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=1, stride=(1, 1))
            else:    
                model = torchvision.models.segmentation.fcn_resnet101(num_classes = num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(input_dims, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
    elif model == 'LRASPP':
        if weights != None:
            weights = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights = weights)
            model.classifier.low_classifier = torch.nn.Conv2d(40, num_classes, kernel_size=1, stride=(1, 1))
            model.classifier.high_classifier = torch.nn.Conv2d(128, num_classes, kernel_size=1, stride=(1, 1))
        else:    
            model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(num_classes = num_classes)
        model.backbone['0'][0] = torch.nn.Conv2d(input_dims, 16, kernel_size=3, stride=(2, 2), padding=(1, 1), bias=False)
    return model