import open3d as o3d
import numpy as np
import nibabel as nib
from pathlib import Path

def visualize_registration(patient_id, image_dir, label_dir):
    """
    Load data for the specified patient, apply registration, and visualize the results.
    """
    print(f"--- Validating patient: {patient_id} ---")

    # --- 1. Define file paths ---
    cbct_path = image_dir / patient_id / "CBCT.nii.gz"
    lower_stl_path = image_dir / patient_id / "lower.stl"
    upper_stl_path = image_dir / patient_id / "upper.stl"
    lower_gt_path = label_dir / patient_id / "lower_gt.npy"
    upper_gt_path = label_dir / patient_id / "upper_gt.npy"

    # --- 2. Load data ---
    # Load STL mesh files
    lower_mesh_orig = o3d.io.read_triangle_mesh(str(lower_stl_path))
    upper_mesh_orig = o3d.io.read_triangle_mesh(str(upper_stl_path))

    # Load transformation matrices
    lower_transform = np.load(str(lower_gt_path))
    upper_transform = np.load(str(upper_gt_path))

    print("Lower jaw transformation matrix:\n", lower_transform)
    print("Upper jaw transformation matrix:\n", upper_transform)

    # --- 3. Apply transformations ---
    # Copy original mesh for transformation, keep original for comparison
    lower_mesh_transformed = o3d.geometry.TriangleMesh(lower_mesh_orig)
    lower_mesh_transformed.transform(lower_transform)

    upper_mesh_transformed = o3d.geometry.TriangleMesh(upper_mesh_orig)
    upper_mesh_transformed.transform(upper_transform)

    # --- 4. Process CBCT data for visualization ---
    # Load NIfTI file
    cbct_img = nib.load(str(cbct_path))
    cbct_data = cbct_img.get_fdata()

    # Extract point cloud from CBCT volume using thresholding
    # Threshold should be adjusted based on actual data (e.g., bone/teeth HU values)
    threshold = 800  # Example Hounsfield Unit threshold
    points = np.argwhere(cbct_data > threshold)

    # Create Open3D point cloud object
    cbct_pcd = o3d.geometry.PointCloud()
    cbct_pcd.points = o3d.utility.Vector3dVector(points)

    # Convert from NIfTI (i, j, k) indices to world coordinates using affine matrix
    cbct_pcd.transform(cbct_img.affine)

    # --- 5. Visualization ---
    # Color different models for better distinction
    cbct_pcd.paint_uniform_color([0.7, 0.7, 0.7])          # CBCT point cloud: gray
    lower_mesh_orig.paint_uniform_color([1, 0, 0])        # Original lower jaw: red
    upper_mesh_orig.paint_uniform_color([1, 0, 0])        # Original upper jaw: red
    lower_mesh_transformed.paint_uniform_color([0, 1, 0]) # Transformed lower jaw: green
    upper_mesh_transformed.paint_uniform_color([0, 0, 1]) # Transformed upper jaw: blue

    print("\nVisualization window guide:")
    print("Gray points: CBCT data")
    print("Red mesh: Original intraoral scan position")
    print("Green mesh: Registered lower jaw")
    print("Blue mesh: Registered upper jaw")
    print("\nIf the crowns of green and blue meshes align well with the gray CBCT teeth, the registration is correct.")

    o3d.visualization.draw_geometries(
        [cbct_pcd, lower_mesh_orig, upper_mesh_orig, lower_mesh_transformed, upper_mesh_transformed],
        window_name=f"Registration Validation - Patient {patient_id}"
    )


if __name__ == "__main__":
    # --- Set your directories and test patient ID ---
    # Replace these paths with your actual dataset locations
    IMAGE_DIR = Path("path/to/your/images")
    LABEL_DIR = Path("path/to/your/labels")

    # Replace with a valid patient ID present in your dataset
    PATIENT_ID_TO_TEST = "027"

    visualize_registration(PATIENT_ID_TO_TEST, IMAGE_DIR, LABEL_DIR)