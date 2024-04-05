#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.utils.ckpt import ModelCheckpointRegister

# Zenodo link : https://zenodo.org/record/8020843
# Hash type : md5
CNEXT_REGISTER = ModelCheckpointRegister(
    infos={
        "cnext_bl": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
            "hash": "0688ae503f5893be0b6b71cb92f8b428",
            "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
        },
        "cnext_nobl": {
            "architecture": "ConvNeXt",
            "url": "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1",
            "hash": "e069ecd1c7b880268331119521c549f2",
            "fname": "convnext_tiny_471mAP.pth",
        },
    },
    state_dict_key="model",
)

# Zenodo link : https://zenodo.org/record/3987831
# Hash type : md5
PANN_REGISTER = ModelCheckpointRegister(
    infos={
        "Cnn10": {
            "architecture": "Cnn10",
            "url": "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth?download=1",
            "hash": "bfb1f1f9968938fa8ef4012b8471f5f6",
            "fname": "Cnn10_mAP_0.380.pth",
        },
        "Cnn14_DecisionLevelAtt": {
            "architecture": "Cnn14_DecisionLevelAtt",
            "url": "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth?download=1",
            "hash": "c8281ca2b9967244b91d557aa941e8ca",
            "fname": "Cnn14_DecisionLevelAtt_mAP_0.425.pth",
        },
        "Cnn14": {
            "architecture": "Cnn14",
            "url": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
            "hash": "541141fa2ee191a88f24a3219fff024e",
            "fname": "Cnn14_mAP_0.431.pth",
        },
        "Cnn6": {
            "architecture": "Cnn6",
            "url": "https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth?download=1",
            "hash": "e25e26b84585b14c7754c91e48efc9be",
            "fname": "Cnn6_mAP_0.343.pth",
        },
        "ResNet22": {
            "architecture": "ResNet22",
            "url": "https://zenodo.org/record/3987831/files/ResNet22_mAP%3D0.430.pth?download=1",
            "hash": "cf36d413096793c4e15dc752a3abd599",
            "fname": "ResNet22_mAP_0.430.pth",
        },
        "ResNet38": {
            "architecture": "ResNet38",
            "url": "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1",
            "hash": "bf12f36aaabac4e0855e22d3c3239c1b",
            "fname": "ResNet38_mAP_0.434.pth",
        },
        "ResNet54": {
            "architecture": "ResNet54",
            "url": "https://zenodo.org/record/3987831/files/ResNet54_mAP%3D0.429.pth?download=1",
            "hash": "4f1f1406d37a29e2379916885e18c5f3",
            "fname": "ResNet54_mAP_0.429.pth",
        },
        "Wavegram_Cnn14": {
            "architecture": "Wavegram_Cnn14",
            "url": "https://zenodo.org/record/3987831/files/Wavegram_Cnn14_mAP%3D0.389.pth?download=1",
            "hash": "1e3506ab640371e0b5a417b15fd66d21",
            "fname": "Wavegram_Cnn14_mAP_0.389.pth",
        },
        "Wavegram_Logmel_Cnn14": {
            "architecture": "Wavegram_Logmel_Cnn14",
            "url": "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1",
            "hash": "17fa9ab65af3c0eb5ffbc5f65552c4e1",
            "fname": "Wavegram_Logmel_Cnn14_mAP_0.439.pth",
        },
    },
    state_dict_key="model",
)
