import open3d as o3d
import numpy as np
import nibabel as nib
from pathlib import Path
from data.dataset import compute_aabb, center_align_points
import trimesh

def visualize_registration(patient_id, image_dir, label_dir):
    """
    Load data for the specified patient, apply registration, center-align using bounding boxes,
    and visualize the results.
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
    threshold = 800  # Example Hounsfield Unit threshold
    points = np.argwhere(cbct_data > threshold)

    # Create Open3D point cloud object
    cbct_pcd = o3d.geometry.PointCloud()
    cbct_pcd.points = o3d.utility.Vector3dVector(points)

    # Convert from NIfTI (i, j, k) indices to world coordinates using affine matrix
    cbct_pcd.transform(cbct_img.affine)

    # --- 5. Calculate bounding boxes and centers ---
    # Calculate CBCT point cloud center
    cbct_points = np.asarray(cbct_pcd.points)
    _, _, cbct_center = compute_aabb(cbct_points)

    # Calculate original STL mesh centers
    lower_points = np.asarray(lower_mesh_orig.vertices)
    _, _, lower_center = compute_aabb(lower_points)

    upper_points = np.asarray(upper_mesh_orig.vertices)
    _, _, upper_center = compute_aabb(upper_points)

    # Calculate transformed STL mesh centers
    lower_transformed_points = np.asarray(lower_mesh_transformed.vertices)
    _, _, lower_transformed_center = compute_aabb(lower_transformed_points)

    upper_transformed_points = np.asarray(upper_mesh_transformed.vertices)
    _, _, upper_transformed_center = compute_aabb(upper_transformed_points)

    # Find the overall center (average of all centers)
    all_centers = np.vstack([
        cbct_center,
        lower_center,
        upper_center,
        lower_transformed_center,
        upper_transformed_center
    ])
    overall_center = np.mean(all_centers, axis=0)

    # --- 6. Center-align all geometries ---
    # Center-align CBCT point cloud
    cbct_points_aligned = center_align_points(cbct_points, overall_center)
    cbct_pcd.points = o3d.utility.Vector3dVector(cbct_points_aligned)

    # Center-align original STL meshes
    lower_vertices_aligned = center_align_points(lower_points, overall_center)
    lower_mesh_orig.vertices = o3d.utility.Vector3dVector(lower_vertices_aligned)

    upper_vertices_aligned = center_align_points(upper_points, overall_center)
    upper_mesh_orig.vertices = o3d.utility.Vector3dVector(upper_vertices_aligned)

    # Center-align transformed STL meshes
    lower_transformed_vertices_aligned = center_align_points(lower_transformed_points, overall_center)
    lower_mesh_transformed.vertices = o3d.utility.Vector3dVector(lower_transformed_vertices_aligned)

    upper_transformed_vertices_aligned = center_align_points(upper_transformed_points, overall_center)
    upper_mesh_transformed.vertices = o3d.utility.Vector3dVector(upper_transformed_vertices_aligned)

    # --- 7. Recalculate normals for better visualization ---
    lower_mesh_orig.compute_vertex_normals()
    upper_mesh_orig.compute_vertex_normals()
    lower_mesh_transformed.compute_vertex_normals()
    upper_mesh_transformed.compute_vertex_normals()

    # --- 8. Visualization ---
    # Color different models for better distinction
    cbct_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # CBCT point cloud: gray
    lower_mesh_orig.paint_uniform_color([1, 0.7, 0.7])  # Original lower jaw: light red
    upper_mesh_orig.paint_uniform_color([1, 0.7, 0.7])  # Original upper jaw: light red
    lower_mesh_transformed.paint_uniform_color([0, 1, 0])  # Transformed lower jaw: green
    upper_mesh_transformed.paint_uniform_color([0, 0.5, 1])  # Transformed upper jaw: blue

    # Create coordinate frame for reference (shows origin)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0)

    # Create bounding box for CBCT for reference
    cbct_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(cbct_pcd.points)
    cbct_bbox.color = (0.5, 0.5, 0.5)  # Gray

    print("\nVisualization window guide:")
    print("Gray points: CBCT data")
    print("Gray box: CBCT bounding box")
    print("Light red mesh: Original intraoral scan position")
    print("Green mesh: Registered lower jaw")
    print("Blue mesh: Registered upper jaw")
    print("Coordinate frame: World origin (0,0,0)")
    print("\nAll geometries are center-aligned to the same point (origin).")
    print("If the crowns of green and blue meshes align well with the gray CBCT teeth, the registration is correct.")

    # Visualize all together
    o3d.visualization.draw_geometries(
        [
            cbct_pcd,
            cbct_bbox,
            lower_mesh_orig,
            upper_mesh_orig,
            lower_mesh_transformed,
            upper_mesh_transformed,
            coordinate_frame
        ],
        window_name=f"Registration Validation - Patient {patient_id}"
    )


if __name__ == "__main__":
    # --- Set your directories and test patient ID ---
    # Replace these paths with your actual dataset locations
    # IMAGE_DIR = Path("./data/Task2/Train-Labeled/Images")
    # LABEL_DIR = Path("./data/Task2/Train-Labeled/Labels")
    # PATIENT_ID_TO_TEST = "027"

    IMAGE_DIR = Path("./data/Task2/Validation/Images")
    LABEL_DIR = Path("./data/Task2/Validation/Labels")
    # Replace with a valid patient ID present in your dataset
    PATIENT_ID_TO_TEST = "002"

    visualize_registration(PATIENT_ID_TO_TEST, IMAGE_DIR, LABEL_DIR)