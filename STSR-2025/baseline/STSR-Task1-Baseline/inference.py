import os
import argparse
import yaml
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import glob
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, VNet, SegResNet
from monai.transforms import (
    Compose,
    LoadImaged,
    SaveImage,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Activationsd,
    AsDiscrete,
    EnsureTyped
)
from monai.data.utils import decollate_batch # 明确导入 decollate_batch


def main(config_path, model_path, input_dir, output_dir):
    """
    主推理函数
    """
    # --- 1. 加载配置 ---
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded for inference:")
    print(yaml.dump(config, indent=2))

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 准备数据文件列表 ---
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    if not input_files:
        print(f"Error: No .nii.gz files found in {input_dir}")
        return
    
    files = [{"image": f} for f in input_files]

    # --- 3. 定义推理用的 Transforms ---
    val_transforms_config = config['transforms']
    infer_transforms = Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=val_transforms_config['orientation']),
        Spacingd(
            keys=["image"],
            pixdim=val_transforms_config['spacing'],
            mode="bilinear"
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=val_transforms_config['intensity_norm']['a_min'],
            a_max=val_transforms_config['intensity_norm']['a_max'],
            b_min=val_transforms_config['intensity_norm']['b_min'],
            b_max=val_transforms_config['intensity_norm']['b_max'],
            clip=True
        ),
        EnsureTyped(keys=["image"], track_meta=False)
    ])
    
    # --- 使用 DataLoader，并将 num_workers 设为 0 以获得最佳兼容性 ---
    # 在 Windows 上，num_workers > 0 可能导致 multiprocessing 错误。
    # 对于推理，性能影响很小，但稳定性大大提高。
    infer_ds = Dataset(data=files, transform=infer_transforms)
    infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- 4. 模型初始化 (与训练时完全一致) ---
    model_config = config['model']
    model_name = model_config.get('name', 'UNet').lower()
    
    # 动态创建模型
    if model_name == 'unet':
        model = UNet(
            spatial_dims=model_config['spatial_dims'], in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'], channels=model_config['channels'],
            strides=model_config['strides'], num_res_units=model_config['num_res_units'],
            norm=model_config['norm']
        ).to(device)
    elif model_name == 'vnet':
        model = VNet(
            spatial_dims=model_config['spatial_dims'], in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'], dropout_prob=model_config.get('dropout_prob', 0.5)
        ).to(device)
    elif model_name == 'segresnet':
        model = SegResNet(
            spatial_dims=model_config['spatial_dims'], in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'], init_filters=model_config.get('init_filters', 16),
            blocks_down=model_config.get('blocks_down', [1, 2, 2, 4]), blocks_up=model_config.get('blocks_up', [1, 1, 1]),
            dropout_prob=model_config.get('dropout_prob', 0.2)
        ).to(device)
    else:
        raise ValueError(f"Unsupported model name in config: {model_config['name']}")

    # --- 5. 加载模型权重 (增加对 PyTorch 2.6+ 的兼容性) ---
    print(f"Loading model weights from: {model_path}")
    try:
        # 默认尝试以安全模式加载，这在 PyTorch 2.6+ 是默认行为
        checkpoint = torch.load(model_path, map_location=device)
    except Exception:
        # 如果失败（例如，因为文件包含非张量对象），则回退到不安全的加载模式
        # 这对于加载我们自己训练的可信检查点是安全的。
        print("Default torch.load failed. Retrying with 'weights_only=False'.")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 兼容直接保存 state_dict 和保存在 'state_dict' key 下的情况
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # --- 6. 定义后处理 Transform (已简化) ---
    # 只需进行 argmax，无需任何标签映射
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscrete(keys="pred", argmax=True), # 直接输出类别索引 0-12
    ])

    # --- 7. 创建输出目录 & 开始推理 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    saver = SaveImage(
        output_dir=output_dir,
        output_postfix="seg",
        resample=False, # 使用推理后的图像仿射矩阵
        output_dtype=np.uint8, # 保存为8位无符号整数，足以容纳0-12的标签
        separate_folder=False
    )

    with torch.no_grad():
        for batch_data in tqdm(infer_loader, desc="Inferencing"):
            val_inputs = batch_data["image"].to(device)
            
            roi_size = tuple(config['transforms']['spatial_crop_size'])
            sw_batch_size = config['training'].get('validation_sw_batch_size', 4)
            
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.5,
                mode="gaussian"
            )
            
            # 分离 batch
            decollated_data = decollate_batch(batch_data)
            for i, item in enumerate(decollated_data):
                item["pred"] = val_outputs[i] # 将对应的预测结果放回字典
                processed_item = post_transforms(item)
                saver(processed_item["pred"])

    print(f"\nInference complete. All predictions saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MONAI CBCT Segmentation Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the training configuration YAML file.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth.tar).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input NIfTI images for inference.")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions",
                        help="Directory to save the segmentation predictions.")
    
    args = parser.parse_args()
    
    main(args.config, args.model_path, args.input_dir, args.output_dir)