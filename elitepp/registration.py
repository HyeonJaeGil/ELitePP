"""Window-based multi-session registration (ELite++ Sec. 3.2).

Ported from `initRT_finder_new.py` (adaptive_scan_alignment) with the algorithm unchanged.
Instead of writing corrected scans, it returns one correction transform per scan,
so the aligned pose is `T_corr[i] @ pose[i]`.
"""
import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from kiss_matcher import KISSMatcher

from elitepp.params import Params
from elitepp.viz import rerun_logger as rrl

log = logging.getLogger("elitepp")


@dataclass
class RegistrationParams(Params):
    window_size: int = 200          # scans per query submap
    stride: int = 100               # scans between window starts (< window_size -> overlap)
    fitness_threshold: float = 0.93  # inlier ratio needed to trust a window
    map_voxel_size: float = 0.35    # downsampling of submaps and the target map [m]
    matcher_voxel_size: float = 1.0  # KISS-Matcher voxel size [m]
    first_eval_distance: float = 50.0  # inlier distance before the first trusted window [m]
    eval_distance: float = 2.0     # inlier distance once tracking [m]
    min_correspondences: int = 10


def solve_rt_from_correspondences(src_pts, tgt_pts):
    if len(src_pts) < 3:
        return np.eye(4)
    centroid_src = np.mean(src_pts, axis=0)
    centroid_tgt = np.mean(tgt_pts, axis=0)
    H = (src_pts - centroid_src).T @ (tgt_pts - centroid_tgt)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centroid_tgt - R @ centroid_src
    return T


def average_transforms(transforms, weights):
    if not transforms:
        return np.eye(4)
    weights = np.array(weights, dtype=np.float64)
    sum_w = np.sum(weights)
    weights = weights / sum_w if sum_w > 0 else np.ones(len(weights)) / len(weights)

    t_avg = np.zeros(3)
    R_sum = np.zeros((3, 3))
    for T, w in zip(transforms, weights):
        t_avg += w * T[:3, 3]
        R_sum += w * T[:3, :3]

    U, S, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        Vt[2, :] *= -1
        R_avg = U @ Vt

    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    return T_avg


def _to_pcd(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def align_session(query_scans, target_points, params=None):
    """Align a session (scans already in its own world frame) to the previous map.

    query_scans: list of (N_i, 3) arrays in the query session frame.
    target_points: (M, 3) array of the previous lifelong static map.
    Returns a list of 4x4 correction transforms, one per scan (aligned pose = T @ pose).
    """
    p = params or RegistrationParams()
    window_size, stride, map_voxel_size = p.window_size, p.stride, p.map_voxel_size
    num_q = len(query_scans)
    log.info("[registration] aligning %d scans in windows of %d (stride %d)", num_q, window_size, stride)
    if num_q == 0:
        return []

    scan_trusted_rts = [[] for _ in range(num_q)]
    tgt_down = _to_pcd(target_points).voxel_down_sample(map_voxel_size)
    tgt_pts = np.asarray(tgt_down.points, dtype=np.float64)
    matcher = KISSMatcher(voxel_size=p.matcher_voxel_size)

    # Step 1: sliding-window matching with fitness verification
    start_indices = list(range(0, num_q - window_size + 1, stride))
    if not start_indices or (start_indices[-1] + window_size < num_q):
        start_indices.append(max(0, num_q - window_size))
    start_indices = sorted(set(start_indices))

    T_initial_guess = np.eye(4)
    is_initial_offset_found = False

    for w_idx, start_idx in enumerate(start_indices):
        end_idx = min(start_idx + window_size, num_q)
        q_local_down = _to_pcd(np.concatenate(query_scans[start_idx:end_idx])).voxel_down_sample(map_voxel_size)
        src_pts = np.asarray(q_local_down.points, dtype=np.float64)
        if is_initial_offset_found:
            src_pts = src_pts @ T_initial_guess[:3, :3].T + T_initial_guess[:3, 3]

        try:
            corr_src, corr_tgt = matcher.match(src_pts, tgt_pts)
            if len(corr_src) < p.min_correspondences:
                log.info("[window %03d~%03d] skipped: %d correspondences", start_idx, end_idx, len(corr_src))
                continue

            T_rel = solve_rt_from_correspondences(np.array(corr_src), np.array(corr_tgt))
            T_absolute = T_rel @ T_initial_guess if is_initial_offset_found else T_rel
            eval_max_dist = p.eval_distance if is_initial_offset_found else p.first_eval_distance

            q_temp = _to_pcd(np.asarray(q_local_down.points)).transform(T_absolute)
            fitness = o3d.pipelines.registration.evaluate_registration(
                q_temp, tgt_down, max_correspondence_distance=eval_max_dist).fitness
            passed = fitness >= p.fitness_threshold
            rrl.window(w_idx, start_idx, end_idx, np.asarray(q_local_down.points), T_absolute, fitness, passed)
            log.info("[window %03d~%03d] fitness %.5f -> %s", start_idx, end_idx, fitness, "PASS" if passed else "FAIL")

            if passed:
                T_initial_guess = T_absolute
                is_initial_offset_found = True
                for i in range(start_idx, end_idx):
                    scan_trusted_rts[i].append(T_absolute)
        except Exception as e:  # the original skips windows the matcher cannot handle
            log.warning("[window %03d~%03d] skipped: matching error (%s)", start_idx, end_idx, e)

    # Step 2: fuse trusted window estimates per scan
    final_transforms = [None] * num_q
    trusted_positions, trusted_transforms = [], []
    centroids = [np.mean(s, axis=0) if len(s) > 0 else np.zeros(3) for s in query_scans]
    for i in range(num_q):
        if scan_trusted_rts[i]:
            T_final = average_transforms(scan_trusted_rts[i], [1.0] * len(scan_trusted_rts[i]))
            final_transforms[i] = T_final
            trusted_positions.append(centroids[i])
            trusted_transforms.append(T_final)
    log.info("[registration] trusted scans: %d/%d", len(trusted_transforms), num_q)

    if not trusted_transforms:
        log.warning("[registration] no trusted window; applying the last guess to all scans")
        trusted_positions.append(np.zeros(3))
        trusted_transforms.append(T_initial_guess)

    # Step 3: inverse-distance interpolation for low-confidence scans
    trusted_positions = np.array(trusted_positions)
    for i in range(num_q):
        if final_transforms[i] is None:
            distances = np.linalg.norm(trusted_positions - centroids[i], axis=1)
            min_idx = np.argmin(distances)
            if distances[min_idx] < 1e-4:
                final_transforms[i] = trusted_transforms[min_idx]
            else:
                final_transforms[i] = average_transforms(trusted_transforms, 1.0 / distances ** 2)

    return final_transforms

