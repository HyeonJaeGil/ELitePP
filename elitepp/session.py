"""Session I/O: a session is `Scans/NNNNNN.pcd` (sensor frame) + KITTI-format `poses.txt`."""
import os

import numpy as np
import open3d as o3d


def load_poses(path):
    """Read KITTI poses (12 values per line) into an (N, 4, 4) array."""
    rows = np.loadtxt(path, dtype=np.float64).reshape(-1, 12)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses


def save_poses(path, poses):
    np.savetxt(path, np.asarray(poses)[:, :3, :].reshape(-1, 12), fmt="%.9g")


def transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def load_scans(session_dir, poses):
    """Load every scan and move it into the session (world) frame with its pose."""
    scans_dir = os.path.join(session_dir, "Scans")
    scans = []
    for i, T in enumerate(poses):
        pcd = o3d.io.read_point_cloud(os.path.join(scans_dir, f"{i:06d}.pcd"))
        scans.append(transform(np.asarray(pcd.points), T))
    return scans


def save_pcd(path, points):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


def load_pcd(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)
