#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.hub.registry import RegistryHub

# Zenodo link : https://zenodo.org/record/8020843
# Hash type : md5
CNEXT_REGISTRY = RegistryHub(
    infos={
        "cnext_nobl": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1",
            "hash_value": "e069ecd1c7b880268331119521c549f2",
            "hash_type": "md5",
            "fname": "convnext_tiny_471mAP.pth",
            "state_dict_key": "model",
        },
        "cnext_bl_70": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
            "hash_value": "0688ae503f5893be0b6b71cb92f8b428",
            "hash_type": "md5",
            "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
            "state_dict_key": "model",
        },
        "cnext_bl_75": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/records/10987498/files/convnext_tiny_465mAP_BL_AC_75kit.pth?download=1",
            "hash_value": "f6f57c87b7eb664a23ae8cad26eccaa0",
            "hash_type": "md5",
            "fname": "convnext_tiny_465mAP_BL_AC_75kit.pth",
        },
    },
)

# Zenodo link : https://zenodo.org/record/3987831
# Hash type : md5
PANN_REGISTRY = RegistryHub(
    infos={
        "Cnn10": {
            "architecture": "Cnn10",
            "url": "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth?download=1",
            "hash_value": "bfb1f1f9968938fa8ef4012b8471f5f6",
            "hash_type": "md5",
            "fname": "Cnn10_mAP_0.380.pth",
            "state_dict_key": "model",
        },
        "Cnn14_DecisionLevelAtt": {
            "architecture": "Cnn14_DecisionLevelAtt",
            "url": "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth?download=1",
            "hash_value": "c8281ca2b9967244b91d557aa941e8ca",
            "hash_type": "md5",
            "fname": "Cnn14_DecisionLevelAtt_mAP_0.425.pth",
            "state_dict_key": "model",
        },
        "Cnn14": {
            "architecture": "Cnn14",
            "url": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
            "hash_value": "541141fa2ee191a88f24a3219fff024e",
            "hash_type": "md5",
            "fname": "Cnn14_mAP_0.431.pth",
            "state_dict_key": "model",
        },
        "Cnn6": {
            "architecture": "Cnn6",
            "url": "https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth?download=1",
            "hash_value": "e25e26b84585b14c7754c91e48efc9be",
            "hash_type": "md5",
            "fname": "Cnn6_mAP_0.343.pth",
            "state_dict_key": "model",
        },
        "ResNet22": {
            "architecture": "ResNet22",
            "url": "https://zenodo.org/record/3987831/files/ResNet22_mAP%3D0.430.pth?download=1",
            "hash_value": "cf36d413096793c4e15dc752a3abd599",
            "hash_type": "md5",
            "fname": "ResNet22_mAP_0.430.pth",
            "state_dict_key": "model",
        },
        "ResNet38": {
            "architecture": "ResNet38",
            "url": "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1",
            "hash_value": "bf12f36aaabac4e0855e22d3c3239c1b",
            "hash_type": "md5",
            "fname": "ResNet38_mAP_0.434.pth",
            "state_dict_key": "model",
        },
        "ResNet54": {
            "architecture": "ResNet54",
            "url": "https://zenodo.org/record/3987831/files/ResNet54_mAP%3D0.429.pth?download=1",
            "hash_value": "4f1f1406d37a29e2379916885e18c5f3",
            "hash_type": "md5",
            "fname": "ResNet54_mAP_0.429.pth",
            "state_dict_key": "model",
        },
        "Wavegram_Cnn14": {
            "architecture": "Wavegram_Cnn14",
            "url": "https://zenodo.org/record/3987831/files/Wavegram_Cnn14_mAP%3D0.389.pth?download=1",
            "hash_value": "1e3506ab640371e0b5a417b15fd66d21",
            "hash_type": "md5",
            "fname": "Wavegram_Cnn14_mAP_0.389.pth",
            "state_dict_key": "model",
        },
        "Wavegram_Logmel_Cnn14": {
            "architecture": "Wavegram_Logmel_Cnn14",
            "url": "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1",
            "hash_value": "17fa9ab65af3c0eb5ffbc5f65552c4e1",
            "hash_type": "md5",
            "fname": "Wavegram_Logmel_Cnn14_mAP_0.439.pth",
            "state_dict_key": "model",
        },
    },
)
