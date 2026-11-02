"""Static / dynamic classification on cached voxel statistics (paper Sec. 3.3).

Stages (all on the GPU when available):
  1. initial labels from the stability ratio R (low R -> dynamic, high R + many hits -> solid)
  2. cluster refinement: connected regions near the ground with a low mean R -> dynamic
  3. floating fragment removal: small unsupported static fragments -> dynamic
  4. ground column propagation: low voxels under mostly-dynamic columns -> dynamic
  5. majority smoothing of weakly observed voxels
  6. overrides: low R always dynamic, solid always static
  7. point level: bottom residue (ankle) under dynamic voxels, probabilistic in-voxel recovery

Numerically identical to `analyzer.py` (calculate_metrics_and_save_gpu) with default params.
"""
from dataclasses import dataclass

import numpy as np
import torch

from elitepp.dor.params import ClassifyParams
from elitepp.voxel_map import encode

# Reason codes: which stage made a point dynamic (0 = static).
REASONS = {0: "static", 1: "low ratio", 2: "cluster", 3: "floating", 4: "column", 5: "smoothing",
           6: "ankle"}

_OFFSETS_6 = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
_SUPPORT_OFFSETS = [[0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]


@dataclass
class DORResult:
    is_static: np.ndarray  # (N,) bool per input point
    reason: np.ndarray     # (N,) uint8, see REASONS


class VoxelGrid:
    """Unique voxels of a point set, with neighbor lookup by linear hash (original layout)."""

    def __init__(self, points, voxel_size):
        v_idx = torch.floor(points / voxel_size).long()
        self.v_min = v_idx.min(0).values
        v_rel = v_idx - self.v_min
        self.dims = v_rel.max(0).values + 1
        self.m2, self.m1 = self.dims[1] * self.dims[2], self.dims[2]
        self.hashes, self.point_voxel = torch.unique(self.hash(v_rel), return_inverse=True)
        self.num = self.hashes.shape[0]
        self.rel = torch.zeros((self.num, 3), dtype=torch.long, device=points.device)
        self.rel.index_copy_(0, self.point_voxel, v_rel)

    def hash(self, rel):
        return rel[..., 0] * self.m2 + rel[..., 1] * self.m1 + rel[..., 2]

    def find(self, rel, table=None, check_bounds=False):
        """Index into `table` (default: all voxels, sorted hashes) and whether it matched.

        Without bounds checking, out-of-grid neighbors can alias onto other voxels; this is
        kept where the original code did it so results stay identical.
        """
        table = self.hashes if table is None else table
        h = self.hash(rel)
        idx = torch.clamp(torch.searchsorted(table, h), 0, table.shape[0] - 1)
        found = table[idx] == h
        if check_bounds:
            found &= (rel >= 0).all(-1) & (rel < self.dims).all(-1)
        return idx, found


def _label_propagation(grid, members, iters, max_size=None):
    """Max-label propagation over 6-connected `members` (voxel indices, ascending)."""
    rel, table = grid.rel[members], grid.hashes[members]
    labels = torch.arange(members.shape[0], device=members.device)
    offsets = torch.tensor(_OFFSETS_6, device=members.device)
    for _ in range(iters):
        old = labels.clone()
        if max_size is None:  # floating-fragment variant: Jacobi update, no bounds check
            for off in offsets:
                idx, match = grid.find(rel + off, table)
                labels = torch.where(match, torch.maximum(labels, old[idx]), labels)
        else:  # cluster variant: Gauss-Seidel update, labels stop growing at max_size
            oversized = torch.bincount(labels, minlength=members.shape[0]) >= max_size
            for off in offsets:
                idx, match = grid.find(rel + off, table, check_bounds=True)
                neighbor = torch.where(match, labels[idx], labels)
                neighbor = torch.where(oversized[neighbor], labels, neighbor)
                labels = torch.maximum(labels, neighbor)
        if torch.equal(old, labels):
            break
    return labels


def classify(points, vmap, params=None, device=None):
    """Classify points (N, 3) with voxel statistics `vmap` (VoxelMap). Returns DORResult."""
    p = params or ClassifyParams()
    if len(points) == 0:
        return DORResult(np.zeros(0, bool), np.zeros(0, np.uint8))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).to(device)
    vs = vmap.voxel_size
    grid = VoxelGrid(pts, vs)
    num_v, point_voxel = grid.num, grid.point_voxel

    # voxel statistics of the occupied voxels
    hits, exps, ground = vmap.query(encode((grid.rel + grid.v_min).cpu().numpy()))
    h_v = torch.tensor(hits, dtype=torch.float32, device=device)
    e_v = torch.tensor(exps, dtype=torch.float32, device=device)
    is_g = torch.tensor(ground, dtype=torch.bool, device=device)
    ratio = h_v / (torch.max(h_v, e_v) + 1e-6)

    # 1) initial labels
    low = (ratio < p.dynamic_ratio) & (~is_g) | (ratio < p.absolute_dynamic_ratio)
    solid = (ratio > p.solid_ratio) & (h_v > p.solid_min_hits)
    dyn = low.clone()
    reason = torch.where(low, 1, 0).to(torch.uint8)

    def mark(target, mask_or_idx, code):
        sel = torch.zeros(num_v, dtype=torch.bool, device=device)
        sel[mask_or_idx] = True
        reason[sel & ~target] = torch.where(reason[sel & ~target] == 0, code, reason[sel & ~target])
        target[sel] = True

    # height above the highest ground voxel of the same xy column
    g_idx = torch.where(is_g)[0]
    v_z = grid.rel[:, 2]
    if g_idx.shape[0] > 0:
        g_rel = grid.rel[g_idx]
        g_xy = g_rel[:, 0] * grid.m2 + g_rel[:, 1] * grid.m1
        uniq_xy, xy_inv = torch.unique(g_xy, return_inverse=True)
        g_top = torch.zeros_like(uniq_xy, dtype=torch.long)
        g_top.scatter_reduce_(0, xy_inv, g_rel[:, 2], reduce="amax", include_self=False)
        v_xy = grid.rel[:, 0] * grid.m2 + grid.rel[:, 1] * grid.m1
        col = torch.clamp(torch.searchsorted(uniq_xy, v_xy), 0, uniq_xy.shape[0] - 1)
        has_ground = uniq_xy[col] == v_xy
        ground_z = g_top[col]
        near_ground = has_ground & (v_z > ground_z) & (v_z <= ground_z + int(p.cluster_max_height / vs))
    else:
        near_ground = torch.zeros(num_v, dtype=torch.bool, device=device)
        ground_z = torch.zeros(num_v, dtype=torch.long, device=device)

    # 2) cluster refinement
    targets = torch.where(near_ground & ~is_g & ~solid & ~low)[0]
    if p.enable_cluster_removal and targets.shape[0] > 0:
        labels = _label_propagation(grid, targets, p.cluster_iters, max_size=p.cluster_max_size)
        _, inv = torch.unique(labels, return_inverse=True)
        reliable = h_v[targets] >= p.cluster_min_hits
        n_clusters = int(inv.max()) + 1
        hit_sum = torch.zeros(n_clusters, device=device).index_add_(0, inv[reliable], h_v[targets[reliable]])
        exp_sum = torch.zeros(n_clusters, device=device).index_add_(
            0, inv[reliable], torch.max(h_v[targets[reliable]], e_v[targets[reliable]]))
        dynamic_cluster = (exp_sum > 0) & (hit_sum / (exp_sum + 1e-6) <= p.cluster_dynamic_ratio)
        mark(dyn, targets[dynamic_cluster[inv]], 2)

    # 3) floating fragments: small static clusters without static support outside themselves
    height = v_z - ground_z
    iso = torch.where(~dyn & ~is_g & (height >= int(p.floating_min_height / vs))
                      & (height <= int(p.floating_max_height / vs)))[0]
    if p.enable_floating_removal and iso.shape[0] > 0:
        labels = _label_propagation(grid, iso, p.floating_iters)
        uniq, counts = torch.unique(labels, return_counts=True)
        small = torch.isin(labels, uniq[counts <= p.floating_max_size])
        label_of = torch.full((num_v,), -1, dtype=torch.long, device=device)
        label_of[iso] = labels
        cand, cand_labels = iso[small], labels[small]
        offs = torch.tensor(_SUPPORT_OFFSETS, device=device)
        n_idx, found = grid.find(grid.rel[cand][:, None, :] + offs[None])
        support = found & ~dyn[n_idx] & (label_of[n_idx] != cand_labels[:, None])
        supported = torch.unique(cand_labels[support.any(1)])
        mark(dyn, cand[~torch.isin(cand_labels, supported)], 3)

    # 4) ground column propagation
    corrected = dyn.clone()
    ground_keys = vmap.keys[vmap.ground]
    if p.enable_column_propagation and len(ground_keys) > 0:
        g_rel = torch.tensor(ground_keys, dtype=torch.long, device=device) - grid.v_min
        for dz in range(1, int(p.column_height / vs) + 1):
            idx, found = grid.find(g_rel + torch.tensor([0, 0, dz], device=device))
            idx = idx[found & ~dyn[idx]]
            if idx.shape[0] == 0:
                continue
            n_dyn = torch.zeros(idx.shape[0], device=device)
            n_all = torch.zeros(idx.shape[0], device=device)
            for up in range(1, p.column_lookup + 1):
                a_idx, exists = grid.find(grid.rel[idx] + torch.tensor([0, 0, up], device=device))
                n_all += exists.float()
                n_dyn += (exists & dyn[a_idx]).float()
            mark(corrected, idx[(n_all > 0) & (n_dyn / (n_all + 1e-6) >= p.column_dynamic_ratio)], 4)

    # 5) majority smoothing of weakly observed voxels
    weak = torch.where((h_v <= p.smoothing_max_hits) & ~is_g & ~low & ~solid)[0]
    if p.enable_smoothing and weak.shape[0] > 0:
        r = torch.tensor([-1, 0, 1], device=device)
        off26 = torch.cartesian_prod(r, r, r)
        off26 = off26[torch.any(off26 != 0, dim=1)]
        n_idx, found = grid.find(grid.rel[weak].unsqueeze(1) + off26)
        n_dyn = torch.zeros_like(found)
        n_dyn[found] = corrected[n_idx[found]]
        vote = torch.where(found.sum(1) > 0, n_dyn.sum(1) > (found.sum(1) / 2), corrected[weak])
        mark(corrected, weak[vote & ~corrected[weak]], 5)
        corrected[weak] = vote

    # 6) overrides
    corrected[low] = True
    corrected[solid] = False
    reason[~corrected] = 0
    point_dyn = corrected[point_voxel].clone()
    point_reason = reason[point_voxel].clone()

    # 7a) ankle: inside ground voxels under a dynamic voxel, drop points above the voxel mean
    if p.enable_ankle_removal and g_idx.shape[0] > 0:
        above, exists = grid.find(grid.rel[g_idx] + torch.tensor([0, 0, 1], device=device))
        above_dyn = exists & corrected[above]
        tgt = g_idx[above_dyn]
        if tgt.shape[0] > 0:
            in_tgt = torch.isin(point_voxel, tgt)
            if in_tgt.any():
                z, vox = pts[in_tgt, 2], point_voxel[in_tgt]
                z_sum = torch.zeros(num_v, device=device).scatter_add_(0, vox, z)
                z_cnt = torch.zeros(num_v, device=device).scatter_add_(0, vox, torch.ones_like(z))
                residue = z > (z_sum / (z_cnt + 1e-6))[vox] + p.ankle_tolerance
                point_dyn[in_tgt] = residue
                point_reason[in_tgt] = torch.where(residue, 6, 0).to(torch.uint8)
        locked = torch.isin(point_voxel, g_idx[~above_dyn])
        point_dyn[locked] = False

    # 7b) in-voxel recovery: restore points of isolated low-ratio voxels with probability
    #     min(1, R / 2 * (1 + planarity))  (paper eq. 6)
    cand = torch.where(low)[0]
    if p.enable_in_voxel_recovery and point_dyn.any() and cand.shape[0] > 0:
        r = torch.tensor([-1, 0, 1], device=device)
        off26 = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
        off26 = off26[~(off26 == 0).all(1)]
        low_table = torch.sort(grid.hashes[low])[0]
        n_low = torch.zeros(cand.shape[0], dtype=torch.int32, device=device)
        for off in off26:
            _, match = grid.find(grid.rel[cand] + off, low_table, check_bounds=True)
            n_low += match.int()
        isolated = cand[n_low <= p.recovery_max_dynamic_neighbors]
        if isolated.shape[0] > 0:
            planarity = torch.zeros(num_v, device=device)
            for v in isolated[(h_v[isolated] >= p.recovery_planarity_min_hits) & ~is_g[isolated]]:
                vp = pts[point_voxel == v]
                if len(vp) < p.recovery_planarity_min_points:
                    continue
                c = vp - vp.mean(dim=0)
                ev = torch.linalg.eigvalsh(c.t() @ c / (len(vp) - 1))
                planarity[v] = (ev[1] - ev[0]) / (ev[2] + 1e-6)
            sel = torch.where(torch.isin(point_voxel, isolated) & point_dyn)[0]
            if sel.shape[0] > 0:
                vox = point_voxel[sel]
                prob = torch.clamp(ratio[vox] / 2.0 * (1.0 + planarity[vox]), 0.0, 1.0)
                gen = torch.Generator(device=device).manual_seed(p.seed)
                restored = sel[torch.rand(sel.shape[0], device=device, generator=gen) < prob]
                point_dyn[restored] = False

    # always dynamic: almost never hit although traversed
    absolute = ((ratio < p.absolute_dynamic_ratio) & ~is_g)[point_voxel]
    point_dyn[absolute] = True
    point_reason[absolute & (point_reason == 0)] = 1
    point_reason[~point_dyn] = 0
    return DORResult(~point_dyn.cpu().numpy(), point_reason.cpu().numpy())
