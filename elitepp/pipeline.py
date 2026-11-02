"""ELite++ end-to-end pipeline: registration -> DOR -> Voxel Ratio Map integration, per session.

    elitepp-run configs/parkinglot.yaml [--rrd]

Outputs per session in <output_dir>/<session>/ (see README "Outputs").
"""
import argparse
import logging
import os
import time

import numpy as np
import yaml

from elitepp import session as io
from elitepp.dor import ClassifyParams, StatsParams, accumulate, classify
from elitepp.registration import RegistrationParams, align_session
from elitepp.viz import rerun_logger as rrl
from elitepp.voxel_map import VoxelMap

log = logging.getLogger("elitepp")

FILES = {
    "poses": "aligned_poses.txt",
    "session_map": "session_map.pcd",
    "session_vrm": "session_vrm.npz",
    "lifelong_map": "lifelong_map.pcd",
    "lifelong_vrm": "lifelong_vrm.npz",
    "rrd": "debug.rrd",
}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    dor = cfg.get("dor") or {}
    cfg["registration"] = RegistrationParams.from_dict(cfg.get("registration"))
    cfg["stats"] = StatsParams.from_dict(dor.get("stats"))
    cfg["classify"] = ClassifyParams.from_dict(dor.get("classify"))
    cfg.setdefault("seed", 0)
    return cfg


def session_paths(cfg, session):
    out = os.path.join(cfg["output_dir"], session)
    return {k: os.path.join(out, v) for k, v in FILES.items()}


def previous_session(cfg, session):
    i = cfg["sessions"].index(session)
    return cfg["sessions"][i - 1] if i > 0 else None


def run_session(cfg, session, rrd=False):
    paths = session_paths(cfg, session)
    os.makedirs(os.path.dirname(paths["poses"]), exist_ok=True)
    np.random.seed(cfg["seed"])
    if rrd:
        rrl.init(paths["rrd"], f"elitepp/{session}")

    session_dir = os.path.join(cfg["dataset_dir"], session)
    poses = io.load_poses(os.path.join(session_dir, "poses.txt"))
    scans = io.load_scans(session_dir, poses)
    log.info("[%s] loaded %d scans", session, len(scans))

    # 1) registration to the previous lifelong map
    prev = previous_session(cfg, session)
    prev_paths = session_paths(cfg, prev) if prev else None
    if prev:
        prev_map = io.load_pcd(prev_paths["lifelong_map"])
        rrl.points("map/previous", prev_map, rrl.GRAY, voxel=0.5)
        rrl.trajectory("trajectory/odometry", poses, rrl.ORANGE)
        corrections = align_session(scans, prev_map, cfg["registration"])
        scans = [io.transform(s, T) for s, T in zip(scans, corrections)]
        poses = np.array([T @ P for T, P in zip(corrections, poses)])
    io.save_poses(paths["poses"], poses)
    rrl.trajectory("trajectory/aligned", poses, rrl.BLUE)

    # 2) session DOR
    session_vrm, points = accumulate(scans, cfg["stats"])
    result = classify(points, session_vrm, cfg["classify"])
    io.save_pcd(paths["session_map"], points[result.is_static])
    session_vrm.save(paths["session_vrm"])
    rrl.dor_result("session", points, result)
    rrl.voxel_ratio("session/voxel_ratio", session_vrm)
    log.info("[%s] session map: %d / %d static points", session, result.is_static.sum(), len(points))

    # 3) integration: previous lifelong points + this session, classified with merged statistics
    if prev:
        lifelong_vrm = VoxelMap.load(prev_paths["lifelong_vrm"]).merge(session_vrm)
        # NOTE: lifelong points grow each session; voxel-dedup them if memory becomes an issue
        points = np.concatenate([prev_map.astype(np.float32), points])
        result = classify(points, lifelong_vrm, cfg["classify"])
    else:
        lifelong_vrm = session_vrm
    io.save_pcd(paths["lifelong_map"], points[result.is_static])
    lifelong_vrm.save(paths["lifelong_vrm"])
    rrl.dor_result("lifelong", points, result)
    rrl.voxel_ratio("lifelong/voxel_ratio", lifelong_vrm)
    log.info("[%s] lifelong map: %d points, %d voxels", session, result.is_static.sum(), len(lifelong_vrm))


def save_resolved_config(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out = {k: v for k, v in cfg.items() if k not in ("registration", "stats", "classify")}
    out["registration"] = cfg["registration"].to_dict()
    out["dor"] = {"stats": cfg["stats"].to_dict(), "classify": cfg["classify"].to_dict()}
    with open(os.path.join(cfg["output_dir"], "config.yaml"), "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="YAML config, e.g. configs/parkinglot.yaml")
    parser.add_argument("--rrd", action="store_true", help="write <session>/debug.rrd for Rerun")
    parser.add_argument("--force", action="store_true", help="re-run sessions that already have outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    save_resolved_config(cfg)
    for session in cfg["sessions"]:
        if os.path.exists(session_paths(cfg, session)["lifelong_vrm"]) and not args.force:
            log.info("[%s] outputs exist, skipping (use --force to re-run)", session)
            continue
        t0 = time.time()
        run_session(cfg, session, args.rrd)
        log.info("[%s] done in %.1fs", session, time.time() - t0)


if __name__ == "__main__":
    main()
