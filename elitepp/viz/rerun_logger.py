"""Optional Rerun recording of a pipeline run. Every function is a no-op until `init()`.

File size is kept small: clouds are voxel-downsampled before logging (raw scans are never
logged) and maps are logged once as static data. About 110 MB per ParkingLot session.
"""
import numpy as np
import open3d as o3d

from elitepp.dor import REASONS
from elitepp.viz.style import BLUE, GRAY, ORANGE, REASON_COLORS, ratio_colors  # noqa: F401 (re-exported)

_rr = None
VOXEL = 0.3  # display resolution [m]


def init(rrd_path, recording="elitepp"):
    global _rr
    import rerun as rr
    rr.init(recording)
    rr.save(rrd_path)
    _rr = rr


def _down(points, voxel=VOXEL):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    return np.asarray(pcd.voxel_down_sample(voxel).points, dtype=np.float32)


def points(path, pts, color, voxel=VOXEL):
    if _rr:
        _rr.log(path, _rr.Points3D(_down(pts, voxel), colors=color, radii=voxel / 3), static=True)


def trajectory(path, poses, color):
    if _rr:
        _rr.log(path, _rr.LineStrips3D([np.asarray(poses)[:, :3, 3]], colors=color), static=True)


def window(idx, start, end, pts, T, fitness, passed):
    """One registration window: its submap under the candidate transform (green pass / red fail)."""
    if not _rr:
        return
    _rr.set_time("window", sequence=idx)
    pts = np.asarray(pts) @ T[:3, :3].T + T[:3, 3]
    _rr.log("registration/window", _rr.Points3D(pts, colors=(0, 200, 0) if passed else (230, 0, 0), radii=0.15))
    _rr.log("registration/log", _rr.TextLog(f"[{start}~{end}] fitness {fitness:.4f} {'PASS' if passed else 'FAIL'}"))


def dor_result(prefix, pts, result):
    """Points split by the DOR stage that removed them (prefix/static, prefix/removed/<reason>)."""
    if not _rr:
        return
    for code, name in REASONS.items():
        sel = result.reason == code
        if sel.any():
            path = f"{prefix}/static" if code == 0 else f"{prefix}/removed/{name.replace(' ', '_')}"
            points(path, pts[sel], tuple(REASON_COLORS[code]))


def voxel_ratio(path, vmap):
    """Voxel centers colored by stability ratio (red unstable .. blue stable); hit voxels only."""
    if not _rr:
        return
    hit = vmap.hits > 0
    centers = (vmap.keys[hit] + 0.5).astype(np.float32) * vmap.voxel_size
    _rr.log(path, _rr.Points3D(centers, colors=ratio_colors(vmap.ratio()[hit]), radii=vmap.voxel_size / 3),
            static=True)
