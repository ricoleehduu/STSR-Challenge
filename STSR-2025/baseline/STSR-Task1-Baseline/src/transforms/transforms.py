# src/transforms/transforms.py
from typing import Dict, Any
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    EnsureTyped,
    Activationsd,
    AsDiscreted, # 使用字典版本
)
KEYS = ["image", "label"]

def get_train_transforms(config: Dict[str, Any]) -> Compose:
    cfg_t = config['transforms']
    num_samples = 1

    return Compose([
        LoadImaged(keys=KEYS, image_only=True),
        EnsureChannelFirstd(keys=KEYS),
                
        Orientationd(keys=KEYS, axcodes=cfg_t['orientation']),
        Spacingd(
            keys=KEYS,
            pixdim=cfg_t['spacing'],
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg_t['intensity_norm']['a_min'],
            a_max=cfg_t['intensity_norm']['a_max'],
            b_min=cfg_t['intensity_norm']['b_min'],
            b_max=cfg_t['intensity_norm']['b_max'],
            clip=cfg_t['intensity_norm']['clip']
        ),
        CropForegroundd(keys=KEYS, source_key="label", allow_smaller=True) if cfg_t.get('crop_foreground', False) else lambda x: x,

        RandCropByPosNegLabeld(
            keys=KEYS,
            label_key="label",
            spatial_size=cfg_t['spatial_crop_size'],
            pos=cfg_t.get('rand_crop_pos_ratio', 1.0),
            neg=1.0 - cfg_t.get('rand_crop_pos_ratio', 1.0),
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=KEYS, prob=cfg_t.get('rand_flip_prob', 0.0), spatial_axis=0),
        RandRotate90d(keys=KEYS, prob=cfg_t.get('rand_rotate90_prob', 0.0), max_k=3, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=cfg_t.get('rand_scale_intensity_prob', 0.0)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=cfg_t.get('rand_shift_intensity_prob', 0.0)),
        
        EnsureTyped(keys=KEYS, track_meta=False),
    ])

def get_val_transforms(config: Dict[str, Any]) -> Compose:
    cfg_t = config['transforms']
    return Compose([
        LoadImaged(keys=KEYS, image_only=True),
        EnsureChannelFirstd(keys=KEYS),

        Orientationd(keys=KEYS, axcodes=cfg_t['orientation']),
        Spacingd(
            keys=KEYS,
            pixdim=cfg_t['spacing'],
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg_t['intensity_norm']['a_min'],
            a_max=cfg_t['intensity_norm']['a_max'],
            b_min=cfg_t['intensity_norm']['b_min'],
            b_max=cfg_t['intensity_norm']['b_max'],
            clip=cfg_t['intensity_norm']['clip']
        ),
        
        EnsureTyped(keys=KEYS, track_meta=False),
    ])

# 修改 src/transforms/transforms.py 中 get_post_transforms 函数
def get_post_transforms(config):
    transforms = []
    
    # 添加 softmax 激活（假设是多分类任务）
    if config['transforms'].get('apply_softmax', True):
        transforms.append(
            Activationsd(keys=["pred", "label"], softmax=True)
        )
    
    # 添加 argmax 操作
    transforms.append(
        AsDiscreted(
            keys=["pred", "label"],
            argmax=(True, True),  # 对 pred 和 label 都应用 argmax
            to_onehot=config['model']['out_channels']
        )
    )
    
    return Compose(transforms)