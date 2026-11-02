"""Voxel statistics of one aligned session (paper Sec. 3.3, eq. 4 inputs).

For every scan: hit counts of occupied voxels, exposure counts of voxels traversed by rays
(range-image test on a local grid), and ground voxels from normal-based anchors.
Numerically identical to `analyzer.py` (run_analysis), vectorized over voxels.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

from elitepp.dor.params import StatsParams
from elitepp.voxel_map import VoxelMap, encode, sum_by_code

_REDUCE_EVERY = 20_000_000  # buffered voxel records before compaction


def downsample(points, size):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(pcd.voxel_down_sample(size).points)


def downsample_scans(scans, size):
    """Per-scan voxel downsampling, concatenated in scan order (the points DOR classifies)."""
    pts = [downsample(s, size) for s in scans]
    return np.concatenate([p for p in pts if len(p)]).astype(np.float32)


def _exposure_grid(p):
    r_val = np.arange(-p.local_max_dist, p.local_max_dist + p.voxel_size, p.voxel_size, dtype=np.float32)
    z_val = np.arange(-p.local_z_range, p.local_z_range + p.voxel_size, p.voxel_size, dtype=np.float32)
    gx, gy, gz = np.meshgrid(r_val, r_val, z_val, indexing="ij")
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    r = np.linalg.norm(pts, axis=1)
    mask = (r <= p.local_max_dist) & (r > p.min_range)
    pts, r = pts[mask], r[mask]
    azi = np.floor((np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) + 180) / p.ang_res).astype(np.int32)
    ele = np.floor((np.degrees(np.arcsin(np.clip(pts[:, 2] / (r + 1e-6), -1.0, 1.0))) + 90) / p.ang_res).astype(np.int32)
    return pts, r, azi, ele


def _frame(pts_global, grid, p):
    """Hit and exposure records of one scan: (codes, hit increments, exposure increments)."""
    # NOTE: ray origin is the point median, not the sensor pose (as in the original code)
    scanner_pos = np.median(pts_global, axis=0).astype(np.float32)
    pts_local = (pts_global - scanner_pos).astype(np.float32)
    r_p = np.linalg.norm(pts_local, axis=1)

    hit_keys, counts = np.unique(np.floor(pts_global / p.voxel_size).astype(np.int32), axis=0, return_counts=True)
    hit_codes = encode(hit_keys)

    # range image: closest return per angular bin
    azi_p = np.floor((np.degrees(np.arctan2(pts_local[:, 1], pts_local[:, 0])) + 180) / p.ang_res).astype(np.int32)
    ele_p = np.floor((np.degrees(np.arcsin(np.clip(pts_local[:, 2] / (r_p + 1e-6), -1.0, 1.0))) + 90) / p.ang_res).astype(np.int32)
    max_azi, max_ele = int(360 / p.ang_res), int(180 / p.ang_res)
    ang_map = np.full((max_azi + 1, max_ele + 1), 1000.0, dtype=np.float32)
    valid = (azi_p >= 0) & (azi_p < max_azi) & (ele_p >= 0) & (ele_p < max_ele)
    order = np.argsort(r_p[valid])[::-1]
    ang_map[azi_p[valid][order], ele_p[valid][order]] = r_p[valid][order]

    # grid cells in front of the first return are exposed (traversed by a ray)
    v_pts, v_r, v_azi, v_ele = grid
    ref_r = ang_map[v_azi, v_ele]
    exp_mask = (ref_r < 999.0) & (v_r < (ref_r - p.voxel_size))
    exp_codes = np.unique(encode(np.floor((v_pts[exp_mask] + scanner_pos) / p.voxel_size).astype(np.int32)))
    exp_codes = exp_codes[~np.isin(exp_codes, hit_codes)]

    codes = np.concatenate([hit_codes, exp_codes])
    hit_inc = np.concatenate([counts, np.zeros(len(exp_codes), np.int64)])
    exp_inc = np.concatenate([counts, np.ones(len(exp_codes), np.int64)])
    return codes, hit_inc, exp_inc


def _ground_codes(pts, p):
    """Voxels of flat points near the local ground height (anchors on an xy grid)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=p.normal_radius, max_nn=30))
    is_flat = np.abs(np.asarray(pcd.normals)[:, 2]) > p.ground_threshold
    cells, inv = np.unique(np.floor(pts[:, :2] / p.anchor_step).astype(np.int32), axis=0, return_inverse=True)
    n_flat = np.bincount(inv, weights=is_flat.astype(float), minlength=len(cells))
    anchors, zs = [], []
    for i in np.flatnonzero(n_flat >= p.min_ground_points):
        mask = (inv == i) & is_flat
        anchors.append(np.mean(pts[mask, :2], axis=0))
        zs.append(np.median(pts[mask, 2]))
    if len(anchors) < 3:
        return np.zeros(0, np.int64)
    _, idx = cKDTree(np.array(anchors)).query(pts[:, :2], k=min(p.ground_knn, len(anchors)))
    ref_z = np.mean(np.array(zs)[idx], axis=1)
    dz = pts[:, 2] - ref_z
    ground = is_flat & (dz <= p.upper_tolerance) & (dz >= -p.lower_tolerance)
    return encode(np.floor(pts[ground] / p.voxel_size).astype(int))


def accumulate(scans, params=None):
    """Voxel statistics of a session whose scans are already in the map frame.

    Returns (VoxelMap, points): `points` are the downsampled scans that DOR classifies.
    Ground sampling uses the global NumPy RNG (seed it for reproducibility).
    """
    p = params or StatsParams()
    grid = _exposure_grid(p)
    total = (np.zeros(0, np.int64),) * 3
    buf, buf_len, ground, batches = [], 0, [], []

    for i in tqdm(range(0, len(scans), p.batch_size), desc="DOR stats", ncols=100):
        batch = []
        for scan in scans[i:i + p.batch_size]:
            pts = downsample(scan, p.downsample_size)
            if len(pts):
                rec = _frame(pts, grid, p)
                buf.append(rec)
                buf_len += len(rec[0])
                batch.append(pts)
        if batch:
            combined = np.concatenate(batch)
            batches.append(combined)
            idx = np.random.choice(len(combined), min(len(combined), p.ground_samples), replace=False)
            ground.append(_ground_codes(combined[idx], p))
        if buf_len > _REDUCE_EVERY:
            total, buf, buf_len = _reduce(total, buf), [], 0
    codes, hits, exps = _reduce(total, buf)

    is_ground = np.isin(codes, np.concatenate(ground)) if ground else np.zeros(len(codes), bool)
    points = np.concatenate(batches).astype(np.float32) if batches else np.zeros((0, 3), np.float32)
    return VoxelMap(codes, hits, exps, is_ground, p.voxel_size), points


def _reduce(total, records):
    parts = [total] + records
    return sum_by_code(*(np.concatenate([r[k] for r in parts]) for k in range(3)))
