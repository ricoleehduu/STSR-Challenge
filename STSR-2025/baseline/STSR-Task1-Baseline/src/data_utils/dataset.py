# src/data_utils/dataset.py
import os
import glob
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
import logging
from monai.data import Dataset, DataLoader


def get_all_data_files(
    image_dir: str, label_dir: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:

    images = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    
    labeled_files = []
    unlabeled_files = []

    for img_path in images:
        img_basename = os.path.basename(img_path)
        name_part = img_basename.replace(".nii.gz", "")
        expected_mask_name = f"{name_part}_Masks.nii.gz"
        expected_mask_path = os.path.join(label_dir, expected_mask_name)

        if os.path.exists(expected_mask_path):
            labeled_files.append({"image": img_path, "label": expected_mask_path})
        else:
            unlabeled_files.append({"image": img_path})

    logging.info(f"Found {len(labeled_files)} labeled image-label pairs based on the '*_Masks.nii.gz' rule.")
    logging.info(f"Found {len(unlabeled_files)} unlabeled images (no corresponding mask found).")
    
    if not labeled_files and images:
        logging.warning("No matching image/label pairs found! Please check that your label files are named correctly (e.g., 'image_name_Masks.nii.gz') and are in the correct directory.")
    elif not images:
        raise ValueError(f"No images found in {image_dir}. Please check the image directory path.")
        
    return labeled_files, unlabeled_files

def prepare_dataloaders(
    config: Dict[str, Any], train_transforms, val_transforms, unlabeled_transforms=None
) -> Dict[str, DataLoader]:
    labeled_files, unlabeled_files = get_all_data_files(
        config['data']['image_dir'], config['data']['label_dir']
    )

    if not labeled_files:
        raise ValueError("Cannot proceed with training as no labeled data was found.")

    train_files, val_files = train_test_split(
        labeled_files,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed']
    )

    logging.info(f"Labeled training samples: {len(train_files)}")
    logging.info(f"Validation samples: {len(val_files)}")
    logging.info(f"Unlabeled training samples available: {len(unlabeled_files)}")

    num_workers = config['data'].get('num_workers', 4)

    logging.warning("Dataset caching is force-disabled in dataset.py for debugging purposes.")
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    logging.info("Created standard training Dataset (no caching).")
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    logging.info("Created standard validation Dataset (no caching).")

    dataloaders = {}

    dataloaders['train'] = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dataloaders['val'] = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if unlabeled_files and unlabeled_transforms:
        unlabeled_ds = Dataset(data=unlabeled_files, transform=unlabeled_transforms)
        logging.info("Created standard unlabeled Dataset (no caching).")
        
        unlabeled_batch_size = config['training'].get('unlabeled_batch_size', config['training']['batch_size'])
        dataloaders['unlabeled'] = DataLoader(
            unlabeled_ds,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        logging.info("Created a DataLoader for unlabeled data.")

    return dataloaders