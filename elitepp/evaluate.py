"""Multi-session alignment metrics, following the ELite protocol (ICRA 2025, Sec. IV-A).

Correspondences are nearest neighbors from source to target; pairs closer than `inlier`
(0.5 m) are inliers.
  AC   = inlier ratio (source -> target)
  RMSE = root mean squared inlier distance (source -> target)
  CD   = mean inlier distance source -> target + mean inlier distance target -> source

Both maps are built from raw scans + poses, so any pose file (ELite, ELite++, ...) can be
compared on the same footing.

  elitepp-eval --src ../dataset/parkinglot/02 outputs/parkinglot/02/aligned_poses.txt \
               --tgt ../dataset/parkinglot/01 ../dataset/parkinglot/01/poses.txt
"""
import argparse

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from elitepp.session import load_poses, load_scans


def build_map(session_dir, pose_file, voxel):
    pts = np.concatenate(load_scans(session_dir, load_poses(pose_file)))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return np.asarray(pcd.voxel_down_sample(voxel).points)


def alignment_metrics(src, tgt, inlier=0.5):
    d_st = cKDTree(tgt).query(src, workers=-1)[0]
    d_ts = cKDTree(src).query(tgt, workers=-1)[0]
    in_st, in_ts = d_st[d_st < inlier], d_ts[d_ts < inlier]
    return {
        "AC": len(in_st) / len(src),
        "RMSE": float(np.sqrt(np.mean(in_st ** 2))),
        "CD": float(np.mean(in_st) + np.mean(in_ts)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", nargs=2, required=True, metavar=("SESSION_DIR", "POSES"))
    parser.add_argument("--tgt", nargs=2, required=True, metavar=("SESSION_DIR", "POSES"))
    parser.add_argument("--voxel", type=float, default=0.2, help="map downsampling before evaluation")
    parser.add_argument("--inlier", type=float, default=0.5)
    args = parser.parse_args()

    src = build_map(*args.src, args.voxel)
    tgt = build_map(*args.tgt, args.voxel)
    m = alignment_metrics(src, tgt, args.inlier)
    print(f"AC {m['AC']:.3f}  RMSE {m['RMSE']:.3f}  CD {m['CD']:.3f}  (src {len(src)} pts, tgt {len(tgt)} pts)")


if __name__ == "__main__":
    main()
