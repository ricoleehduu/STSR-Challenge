# evaluate_local.py
#
# A simplified script for local evaluation of segmentation performance.
#
# Pre-requisites:
#   - Python 3.x
#   - Required packages: pip install numpy simpleitk scipy
#   - The 'SurfaceDice.py' script must be in the same directory as this script.
#
# Usage:
#   python evaluate_local.py -p /path/to/your/predictions -g /path/to/ground_truth -o /path/to/output/scores.json
#
# Filename Matching Logic:
#   This script identifies cases by splitting filenames by '_' and using the
#   second-to-last element as the unique case ID.
#   e.g., 'STS_case_01_prediction.nii.gz' -> ID: '01'
#         'STS_case_01_gt.nii.gz'         -> ID: '01'
#

import json
import os
import numpy as np
import SimpleITK as sitk
import time
import argparse
from typing import Sequence, List, Dict
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

# --- METRIC CONFIGURATION ---
NSD_TOLERANCE_MM = 2.0  # Tolerance in millimeters for Normalized Surface Dice

def get_itk_reader(filename: str) -> sitk.Image:
    """Reads a NIfTI file using SimpleITK."""
    return sitk.ReadImage(filename)

def resample_to_reference_grid(
    image_to_resample_itk: sitk.Image,
    reference_image_itk: sitk.Image,
    interpolator: int = sitk.sitkNearestNeighbor,
    default_pixel_value: float = 0.0
) -> sitk.Image:
    """Resamples an image to match the grid of a reference image."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image_itk)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    return resampler.Execute(image_to_resample_itk)

def calculate_nsd_robust(gt_binary_mask_np, pred_binary_mask_np, spacing_zyx, tolerance):
    """Calculates Normalized Surface Dice, handling empty mask cases."""
    if not gt_binary_mask_np.any() and not pred_binary_mask_np.any():
        return 1.0
    if not gt_binary_mask_np.any() or not pred_binary_mask_np.any():
        return 0.0

    surface_distances = compute_surface_distances(gt_binary_mask_np, pred_binary_mask_np, spacing_mm=spacing_zyx)
    nsd_val = compute_surface_dice_at_tolerance(surface_distances, tolerance)
    return nsd_val if not np.isnan(nsd_val) else 0.0

def calculate_iou(gt_binary_mask_np, pred_binary_mask_np):
    """Calculates Intersection over Union (IoU)."""
    if not gt_binary_mask_np.any() and not pred_binary_mask_np.any():
        return 1.0
    intersection = np.sum(gt_binary_mask_np & pred_binary_mask_np)
    union = np.sum(gt_binary_mask_np | pred_binary_mask_np)
    return intersection / union if union > 0 else 0.0

def robust_mean(data_list: List[float]) -> float:
    """Calculates mean of a list, ignoring NaNs and handling empty lists."""
    if not data_list: return 0.0
    valid_data = [x for x in data_list if not np.isnan(x)]
    return np.mean(valid_data).item() if valid_data else 0.0

def create_file_map(directory: str) -> Dict[str, str]:
    """Creates a map from case ID to file path based on the naming convention."""
    file_map = {}
    for f in os.listdir(directory):
        if f.endswith('.nii.gz'):
            parts = f.split('_')
            if len(parts) >= 2:
                case_id = parts[-2]
                file_map[case_id] = os.path.join(directory, f)
    return file_map

def evaluate(prediction_dir: str, groundtruth_dir: str, output_file: str):
    """Main evaluation function."""
    try:
        gt_map = create_file_map(groundtruth_dir)
        pred_map = create_file_map(prediction_dir)
    except FileNotFoundError as e:
        print(f"Error: Directory not found - {e}. Please check your paths.")
        return

    if not gt_map:
        print(f"Error: No ground truth .nii.gz files found in {groundtruth_dir}")
        return

    all_scores = {
        'dice_image': [], 'iou_image': [], 'nsd_image': [],
        'dice_instance': [], 'iou_instance': [], 'nsd_instance': [],
        'ia': []
    }

    for case_id, gt_path in gt_map.items():
        print(f"Processing Case ID: {case_id}")
        pred_path = pred_map.get(case_id)

        if not pred_path:
            print(f"  - Prediction file for case {case_id} not found. Assigning 0 scores.")
            all_scores['dice_image'].append(0.0)
            all_scores['iou_image'].append(0.0)
            all_scores['nsd_image'].append(0.0)
            all_scores['ia'].append(0.0)
            continue

        try:
            gt_itk = get_itk_reader(gt_path)
            pred_itk = get_itk_reader(pred_path)

            if gt_itk.GetSize() != pred_itk.GetSize() or gt_itk.GetSpacing() != pred_itk.GetSpacing():
                 print(f"  - Resampling prediction to match ground truth grid.")
                 pred_itk = resample_to_reference_grid(pred_itk, gt_itk)

            gt_np = sitk.GetArrayFromImage(gt_itk)
            pred_np = sitk.GetArrayFromImage(pred_itk)
            spacing_zyx = list(reversed(gt_itk.GetSpacing()))

        except Exception as e:
            print(f"  - Error processing file: {e}. Assigning 0 scores.")
            all_scores['dice_image'].append(0.0)
            all_scores['iou_image'].append(0.0)
            all_scores['nsd_image'].append(0.0)
            all_scores['ia'].append(0.0)
            continue

        # --- Image-level metrics ---
        gt_binary = (gt_np > 0)
        pred_binary = (pred_np > 0)
        all_scores['dice_image'].append(compute_dice_coefficient(gt_binary, pred_binary))
        all_scores['iou_image'].append(calculate_iou(gt_binary, pred_binary))
        all_scores['nsd_image'].append(calculate_nsd_robust(gt_binary, pred_binary, spacing_zyx, NSD_TOLERANCE_MM))

        # --- Instance-level metrics ---
        gt_labels = np.unique(gt_np[gt_np > 0])
        pred_labels = np.unique(pred_np[pred_np > 0])
        num_iou_gte_05 = 0

        if len(gt_labels) > 0:
            for label_id in gt_labels:
                gt_inst_mask = (gt_np == label_id)
                pred_inst_mask = (pred_np == label_id)

                dice_inst = compute_dice_coefficient(gt_inst_mask, pred_inst_mask)
                iou_inst = calculate_iou(gt_inst_mask, pred_inst_mask)
                nsd_inst = calculate_nsd_robust(gt_inst_mask, pred_inst_mask, spacing_zyx, NSD_TOLERANCE_MM)

                all_scores['dice_instance'].append(dice_inst)
                all_scores['iou_instance'].append(iou_inst)
                all_scores['nsd_instance'].append(nsd_inst)

                if iou_inst >= 0.5:
                    num_iou_gte_05 += 1

        # --- IA (Instance-level Agreement) Calculation ---
        union_labels_count = len(set(gt_labels).union(set(pred_labels)))
        if union_labels_count == 0:
            ia_score = 1.0
        else:
            ia_score = float(num_iou_gte_05) / float(union_labels_count)
        all_scores['ia'].append(ia_score)
        
        print(f"  - Image Dice: {all_scores['dice_image'][-1]:.4f}, Image NSD: {all_scores['nsd_image'][-1]:.4f}, IA: {ia_score:.4f}")

    # --- Final Score Aggregation ---
    final_scores = {
        'Dice_Image': robust_mean(all_scores['dice_image']),
        'IoU_Image': robust_mean(all_scores['iou_image']),
        'NSD_Image': robust_mean(all_scores['nsd_image']),
        'Dice_Instance': robust_mean(all_scores['dice_instance']),
        'IoU_Instance': robust_mean(all_scores['iou_instance']),
        'NSD_Instance': robust_mean(all_scores['nsd_instance']),
        'IA': robust_mean(all_scores['ia']),
    }

    numeric_scores = [v for k, v in final_scores.items()]
    final_scores['All_Average'] = robust_mean(numeric_scores)

    print("\n" + "="*30)
    print("      FINAL RESULTS")
    print("="*30)
    for key, value in final_scores.items():
        print(f"{key:<15}: {value:.4f}")
    print("="*30)

    try:
        with open(output_file, 'w') as f:
            json.dump(final_scores, f, indent=4)
        print(f"\nScores successfully saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving scores to file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Local evaluation script for segmentation challenge.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-p', '--prediction_dir', type=str, required=True,
        help="Path to the directory containing your predicted segmentation masks."
    )
    parser.add_argument(
        '-g', '--groundtruth_dir', type=str, required=True,
        help="Path to the directory containing the ground truth masks."
    )
    parser.add_argument(
        '-o', '--output_file', type=str, default='scores.json',
        help="Path to save the output JSON file with the results. (default: scores.json)"
    )
    
    args = parser.parse_args()

    start_time = time.time()
    evaluate(args.prediction_dir, args.groundtruth_dir, args.output_file)
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")