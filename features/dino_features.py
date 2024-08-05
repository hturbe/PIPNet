import ssl

import torch

dict_model = {
    "dinov2_vits14_reg": ["facebookresearch/dinov2", "dinov2_vits14_reg"],
    "dinov2_vitb14_reg": ["facebookresearch/dinov2", "dinov2_vitb14_reg"],
    "dinov2_vitl14_reg": ["facebookresearch/dinov2", "dinov2_vitl14_reg"],
    "dinov2_vitg14_reg": ["facebookresearch/dinov2", "dinov2_vitg14_reg"],
}


def dino_features(pretrained=False, **kwargs):
    arch = "dinov2_vitl14_reg"
    if arch in dict_model:
        ssl._create_default_https_context = ssl._create_unverified_context
        image_encoder = torch.hub.load(*dict_model[arch])
    else:
        raise NotImplementedError

    return image_encoder


def dino_features_b(pretrained=False, **kwargs):
    arch = "dinov2_vitb14_reg"
    if arch in dict_model:
        ssl._create_default_https_context = ssl._create_unverified_context
        image_encoder = torch.hub.load(*dict_model[arch])
    else:
        raise NotImplementedError

    return image_encoder
