"""The vectorized accumulation must equal the original dict-based analyzer.py, bit for bit."""
import numpy as np
from scipy.spatial import cKDTree

from elitepp.dor.params import StatsParams
from elitepp.dor.stats import _exposure_grid, accumulate, downsample, downsample_scans
from elitepp.voxel_map import decode


def reference_accumulate(scans, p):
    """Original algorithm (analyzer.py run_analysis), kept verbatim apart from names."""
    import open3d as o3d
    hit, exp, ground, batches = {}, {}, set(), []
    v_pts, v_r = _exposure_grid(p)[:2]

    def frame(pts_global):
        scanner_pos = np.median(pts_global, axis=0).astype(np.float32)
        pts_local = (pts_global - scanner_pos).astype(np.float32)
        r_p = np.linalg.norm(pts_local, axis=1)
        u_v, counts = np.unique(np.floor(pts_global / p.voxel_size).astype(np.int32), axis=0, return_counts=True)
        hit_tuples = set()
        for v, c in zip(u_v, counts):
            vt = tuple(v)
            hit[vt] = hit.get(vt, 0) + c
            exp[vt] = exp.get(vt, 0) + c
            hit_tuples.add(vt)
        azi_p = np.floor((np.degrees(np.arctan2(pts_local[:, 1], pts_local[:, 0])) + 180) / p.ang_res).astype(np.int32)
        ele_p = np.floor((np.degrees(np.arcsin(np.clip(pts_local[:, 2] / (r_p + 1e-6), -1.0, 1.0))) + 90) / p.ang_res).astype(np.int32)
        max_azi, max_ele = int(360 / p.ang_res), int(180 / p.ang_res)
        ang_map = np.full((max_azi + 1, max_ele + 1), 1000.0, dtype=np.float32)
        valid = (azi_p >= 0) & (azi_p < max_azi) & (ele_p >= 0) & (ele_p < max_ele)
        s = np.argsort(r_p[valid])[::-1]
        ang_map[azi_p[valid][s], ele_p[valid][s]] = r_p[valid][s]
        v_azi = np.floor((np.degrees(np.arctan2(v_pts[:, 1], v_pts[:, 0])) + 180) / p.ang_res).astype(np.int32)
        v_ele = np.floor((np.degrees(np.arcsin(np.clip(v_pts[:, 2] / (v_r + 1e-6), -1.0, 1.0))) + 90) / p.ang_res).astype(np.int32)
        ref_r = ang_map[v_azi, v_ele]
        m = (ref_r < 999.0) & (v_r < (ref_r - p.voxel_size))
        for v in np.unique(np.floor((v_pts[m] + scanner_pos) / p.voxel_size).astype(np.int32), axis=0):
            vt = tuple(v)
            if vt not in hit_tuples:
                exp[vt] = exp.get(vt, 0) + 1

    def ground_batch(pts):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=p.normal_radius, max_nn=30))
        is_flat = np.abs(np.asarray(pcd.normals)[:, 2]) > p.ground_threshold
        u, inv = np.unique(np.floor(pts[:, :2] / p.anchor_step).astype(np.int32), axis=0, return_inverse=True)
        v_flat = np.bincount(inv, weights=is_flat.astype(float), minlength=len(u))
        anchors, zs = [], []
        for i in range(len(u)):
            if v_flat[i] >= p.min_ground_points:
                mask = (inv == i) & is_flat
                anchors.append(np.mean(pts[mask, :2], axis=0))
                zs.append(np.median(pts[mask, 2]))
        if len(anchors) >= 3:
            _, idx = cKDTree(np.array(anchors)).query(pts[:, :2], k=min(5, len(anchors)))
            ref_z = np.mean(np.array(zs)[idx], axis=1)
            gm = is_flat & (pts[:, 2] - ref_z <= p.upper_tolerance) & (pts[:, 2] - ref_z >= -p.lower_tolerance)
            for v in np.floor(pts[gm] / p.voxel_size).astype(int):
                ground.add(tuple(v))

    for i in range(0, len(scans), p.batch_size):
        batch = []
        for scan in scans[i:i + p.batch_size]:
            pts = downsample(scan, p.downsample_size)
            frame(pts)
            batch.append(pts)
        combined = np.concatenate(batch)
        batches.append(combined)
        idx = np.random.choice(len(combined), min(len(combined), p.ground_samples), replace=False)
        ground_batch(combined[idx])
    return hit, exp, ground, np.concatenate(batches).astype(np.float32)


def test_accumulate_matches_reference(session):
    p = StatsParams(local_max_dist=20.0)
    np.random.seed(0)
    ref_hit, ref_exp, ref_ground, ref_pts = reference_accumulate(session, p)
    np.random.seed(0)
    vmap, pts = accumulate(session, p)

    keys = [tuple(k) for k in decode(vmap.codes).tolist()]
    assert set(keys) == set(ref_exp)
    assert [int(h) for h in vmap.hits] == [int(ref_hit.get(k, 0)) for k in keys]
    assert [int(e) for e in vmap.exps] == [int(ref_exp[k]) for k in keys]
    assert {k for k, g in zip(keys, vmap.ground) if g} == ref_ground
    assert np.array_equal(pts, ref_pts)


def test_downsample_scans_matches_accumulate_points(session):
    np.random.seed(0)
    _, pts = accumulate(session, StatsParams(local_max_dist=20.0))
    assert np.array_equal(downsample_scans(session, 0.3), pts)
