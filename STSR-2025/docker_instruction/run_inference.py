# run_inference.py
import os
import torch
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

# --> ACTION REQUIRED: Import your model definition <--
# Example: from model.UNet import UNet
# Make sure your model definition file is copied in the Dockerfile.
from model.UNet import UNet # Replace with your actual model import

# Define constants for input/output directories
INPUT_DIR = '/inputs'
OUTPUT_DIR = '/outputs'
# --> ACTION REQUIRED: Specify your model weight's path <--
MODEL_PATH = 'your_model_weights.pth' # IMPORTANT: Must match the filename in the Dockerfile

def preprocess(image_np: np.ndarray) -> np.ndarray:
    """
    Pre-process the input image numpy array.
    --> ACTION REQUIRED: Implement your pre-processing logic here <--
    This is just an example. Replace it with your own logic (e.g., normalization, resizing).
    """
    # Example: Normalization
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)
    
    # Example: Resizing (if your model requires a fixed input size)
    # from scipy.ndimage import zoom
    # original_shape = image_np.shape
    # target_shape = (128, 128, 128)
    # zoom_factors = [t / o for t, o in zip(target_shape, original_shape)]
    # image_np = zoom(image_np, zoom_factors, order=1)

    return image_np

def postprocess(prediction_np: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Post-process the model's prediction.
    --> ACTION REQUIRED: Implement your post-processing logic here <--
    This is just an example. Replace it with your own logic (e.g., resizing back, thresholding).
    """
    # Example: Resizing back to original shape
    # from scipy.ndimage import zoom
    # zoom_factors = [o / p for o, p in zip(original_shape, prediction_np.shape)]
    # prediction_np = zoom(prediction_np, zoom_factors, order=1)
    
    # Example: Thresholding to create a binary mask
    prediction_np[prediction_np >= 0.5] = 1
    prediction_np[prediction_np < 0.5] = 0
    
    return prediction_np.astype(np.uint8)

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    # --> ACTION REQUIRED: Initialize your model and load weights <--
    # The number of classes and channels should match your trained model.
    model = UNet(in_channels=1, n_class=32).to(device) # Example initialization
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Find Input Files ---
    # The evaluation environment mounts one case at a time in the /inputs folder.
    # We use glob to find the .nii.gz file robustly.
    input_files = glob(os.path.join(INPUT_DIR, '*.nii.gz'))
    if not input_files:
        print("Error: No .nii.gz file found in /inputs. Exiting.")
        return

    input_path = input_files[0]
    input_filename = os.path.basename(input_path)
    print(f"Processing file: {input_filename}")

    # --- Inference Loop ---
    with torch.no_grad():
        # Load image
        input_sitk = sitk.ReadImage(input_path)
        input_np = sitk.GetArrayFromImage(input_sitk)
        original_shape = input_np.shape

        # Pre-process
        processed_np = preprocess(input_np)
        
        # Convert to tensor and add batch/channel dimensions
        input_tensor = torch.from_numpy(processed_np).unsqueeze(0).unsqueeze(0).to(device).float()

        # Run inference
        prediction_tensor = model(input_tensor)

        # Convert back to numpy and remove batch/channel dimensions
        # NOTE: Adjust indices if your model output is different, e.g., for multi-class segmentation.
        prediction_np = prediction_tensor.squeeze(0).squeeze(0).cpu().numpy()

        # Post-process
        final_mask_np = postprocess(prediction_np, original_shape)

        # Save the final mask
        output_sitk = sitk.GetImageFromArray(final_mask_np)
        output_sitk.CopyInformation(input_sitk) # Preserve spacing, origin, etc.
        
        output_filename = input_filename.replace(".nii.gz", "_mask.nii.gz")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        sitk.WriteImage(output_sitk, output_path)
        print(f"Prediction saved to: {output_path}")

if __name__ == "__main__":
    main()