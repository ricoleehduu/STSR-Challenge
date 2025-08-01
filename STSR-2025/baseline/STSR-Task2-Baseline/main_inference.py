import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import trimesh
import nibabel as nib
from pathlib import Path
import yaml
import shutil
from tqdm import tqdm

from models.main_model import RegistrationModel

def load_model_for_inference(config, jaw_type):
    """Load the model for the specified jaw type."""
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = RegistrationModel(feat_dim=config['feature_dim']).to(device)

    exp_name = config[f'{jaw_type}_jaw_experiment_name']
    checkpoint_path = Path(config['output_dir']) / exp_name / "checkpoints" / "latest_model.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model weights not found for {jaw_type} jaw: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device

def predict_single_case(model, device, config, stl_path, cbct_path):
    """Perform inference on a single STL and CBCT pair, return the final transformation matrix."""
    # --- Load and preprocess input data ---
    mesh = trimesh.load(str(stl_path), force='mesh')  # Ensure loading as mesh
    p_src, _ = trimesh.sample.sample_surface(mesh, config['num_points_stl'])
    
    cbct_img = nib.load(str(cbct_path))
    cbct_data = cbct_img.get_fdata()
    coords = np.argwhere(cbct_data > 800)
    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
    p_tgt = (cbct_img.affine @ coords_h.T).T[:, :3]

    if p_tgt.shape[0] > config['num_points_cbct']:
        indices = np.random.choice(p_tgt.shape[0], config['num_points_cbct'], replace=False)
        p_tgt = p_tgt[indices]
    elif p_tgt.shape[0] < config['num_points_cbct']:
        indices = np.random.choice(p_tgt.shape[0], config['num_points_cbct'], replace=True)
        p_tgt = p_tgt[indices]
        
    # --- Normalization ---
    center = p_tgt.mean(axis=0)
    p_src_norm = p_src - center
    p_tgt_norm = p_tgt - center
    scale = np.max(np.linalg.norm(p_tgt_norm, axis=1))
    # Avoid division by zero
    if scale < 1e-6:
        scale = 1.0
    p_src_norm /= scale
    p_tgt_norm /= scale

    p_src_tensor = torch.from_numpy(p_src_norm).float().unsqueeze(0).to(device)
    p_tgt_tensor = torch.from_numpy(p_tgt_norm).float().unsqueeze(0).to(device)

    # --- Model inference ---
    with torch.no_grad():
        transform_pred_norm = model(p_src_tensor, p_tgt_tensor)

    # --- Denormalization ---
    T_pred_norm_np = transform_pred_norm.squeeze(0).cpu().numpy()
    R_pred = T_pred_norm_np[:3, :3]
    t_pred_norm = T_pred_norm_np[:3, 3]
    t_final = scale * t_pred_norm - R_pred @ center + center
    
    T_final_corrected = np.eye(4)
    T_final_corrected[:3,:3] = R_pred
    T_final_corrected[:3, 3] = t_final
    
    return T_final_corrected

def main():

    # --- 1. Load configuration ---
    config_path = 'configs/inference_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 2. Prepare paths and temporary folder ---
    inference_root = Path(config['inference_data_root'])
    submission_zip_path = Path(config['submission_zip_path'])
    
    # Create a temporary folder to store results, then zip and delete it
    temp_submission_dir = Path("./temp_submission")
    if temp_submission_dir.exists():
        shutil.rmtree(temp_submission_dir)
    temp_submission_dir.mkdir()

    print("--- Batch inference and packaging started ---")
    
    # --- 3. Load both models (upper and lower jaw) ---
    print("Loading upper jaw model...")
    upper_model, device = load_model_for_inference(config, 'upper')
    print("Loading lower jaw model...")
    lower_model, _ = load_model_for_inference(config, 'lower')  # device is the same

    # --- 4. Iterate over all cases and perform inference ---
    case_folders = sorted([p for p in inference_root.iterdir() if p.is_dir()])
    
    for case_folder in tqdm(case_folders, desc="Processing cases"):
        case_id = case_folder.name
        print(f"\nProcessing case: {case_id}")
        
        # Create a subfolder for this case's results
        result_case_dir = temp_submission_dir / case_id
        result_case_dir.mkdir()

        cbct_path = case_folder / "CBCT.nii.gz"

        # a. Predict upper jaw
        upper_stl_path = case_folder / "upper.stl"
        if upper_stl_path.exists() and cbct_path.exists():
            print("  - Predicting upper jaw...")
            upper_transform = predict_single_case(upper_model, device, config, upper_stl_path, cbct_path)
            np.save(result_case_dir / "upper_gt.npy", upper_transform)
            print("    -> Upper jaw matrix saved.")
        else:
            print(f"  - Warning: Missing upper jaw or CBCT file for {case_id}, skipping.")

        # b. Predict lower jaw
        lower_stl_path = case_folder / "lower.stl"
        if lower_stl_path.exists() and cbct_path.exists():
            print("  - Predicting lower jaw...")
            lower_transform = predict_single_case(lower_model, device, config, lower_stl_path, cbct_path)
            np.save(result_case_dir / "lower_gt.npy", lower_transform)
            print("    -> Lower jaw matrix saved.")
        else:
            print(f"  - Warning: Missing lower jaw or CBCT file for {case_id}, skipping.")

    # --- 5. Package into ZIP file ---
    print("\nAll cases processed. Creating ZIP archive...")
    
    # shutil.make_archive uses base_name without extension
    # root_dir is the folder to compress
    # format is 'zip'
    shutil.make_archive(
        base_name=str(submission_zip_path.with_suffix('')),
        format='zip',
        root_dir=temp_submission_dir
    )
    
    print(f"Packaging completed! Submission file saved at: {submission_zip_path}")
    
    # --- 6. Clean up temporary folder ---
    shutil.rmtree(temp_submission_dir)
    print("Temporary folder cleaned up.")
    print("--- Task completed ---")

if __name__ == '__main__':
    main()