import numpy as np
from scipy.spatial.transform import Rotation

class RandomRigidTransform:
    def __init__(self, mag_trans=0.5, mag_rot=45):
        self.mag_trans = mag_trans  # Translation magnitude in meters
        self.mag_rot = mag_rot      # Rotation magnitude in degrees

    def __call__(self, points):
        # Random rotation
        angle_x = np.random.uniform(-self.mag_rot, self.mag_rot)
        angle_y = np.random.uniform(-self.mag_rot, self.mag_rot)
        angle_z = np.random.uniform(-self.mag_rot, self.mag_rot)
        rotation = Rotation.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)
        R = rotation.as_matrix()

        # Random translation
        t = np.random.uniform(-self.mag_trans, self.mag_trans, size=3)

        # Build 4x4 transformation matrix
        transform = np.identity(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        # Apply transformation
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = (transform @ points_homogeneous.T).T[:, :3]

        return transformed_points, transform