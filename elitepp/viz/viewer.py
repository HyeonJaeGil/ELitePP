"""Interactive Viser viewer and DOR tuner for pipeline outputs.

    elitepp-view configs/parkinglot.yaml --session 02 [--mode lifelong|session]

Shows the points DOR classified (colored by the stage that removed them, or by voxel
stability), and all session trajectories. Every ClassifyParams field gets a GUI control;
changing one re-runs the classification on the cached voxel statistics (about 0.5 s for
15 M points on a desktop GPU) and recolors the cloud. Tuned parameters can be saved as YAML
and the tuned static map exported.
"""
import argparse
import dataclasses
import os
import threading
import time

import numpy as np
import viser
import yaml

from elitepp import session as io
from elitepp.dor import REASONS, ClassifyParams, classify, downsample_scans
from elitepp.pipeline import load_config, previous_session, session_paths
from elitepp.viz.style import REASON_COLORS, ratio_colors
from elitepp.voxel_map import VoxelMap, encode

COLOR_MODES = ("removal stage", "stability ratio", "static only")
SESSION_COLORS = [(0, 120, 230), (230, 120, 0), (0, 170, 80), (200, 0, 160), (120, 80, 40), (0, 160, 170)]
GROUPS = ["initial", "cluster", "floating", "column", "smoothing", "ankle", "recovery", "misc"]


def load_inputs(cfg, session, mode):
    """Points and voxel statistics exactly as the pipeline classified them."""
    paths = session_paths(cfg, session)
    poses = io.load_poses(paths["poses"])
    scans = io.load_scans(os.path.join(cfg["dataset_dir"], session), poses)
    points = downsample_scans(scans, cfg["stats"].downsample_size)
    if mode == "session":
        return points, VoxelMap.load(paths["session_vrm"])
    prev = previous_session(cfg, session)
    if prev:
        prev_map = io.load_pcd(session_paths(cfg, prev)["lifelong_map"]).astype(np.float32)
        points = np.concatenate([prev_map, points])
    return points, VoxelMap.load(paths["lifelong_vrm"])


def group_of(name):
    key = name.removeprefix("enable_")
    if key.startswith(("dynamic_ratio", "absolute_dynamic", "solid")):
        return "initial"
    if key.startswith("in_voxel"):
        return "recovery"
    for g in GROUPS:
        if key.startswith(g):
            return g
    return "misc"


def add_control(gui, field, value):
    """Checkbox for bools, slider for numbers (range guessed from the default)."""
    if isinstance(value, bool):
        return gui.add_checkbox(field.name, value)
    if isinstance(value, int):
        return gui.add_slider(field.name, 0, max(3 * value, 10), 1, value)
    hi = 1.0 if value <= 1.0 else round(3 * value, 1)
    return gui.add_slider(field.name, 0.0, hi, 0.01 if hi <= 1.0 else 0.1, value)


class Viewer:
    def __init__(self, cfg, session, mode, points, vmap, max_points, port):
        self.cfg, self.session, self.mode = cfg, session, mode
        self.points, self.vmap = points, vmap
        rng = np.random.default_rng(0)
        n = len(points)
        self.shown = np.sort(rng.choice(n, max_points, replace=False)) if n > max_points else np.arange(n)
        codes = encode(np.floor(points[self.shown] / vmap.voxel_size).astype(np.int64))
        hits, exps, _ = vmap.query(codes)
        self.shown_ratio = hits / (np.maximum(hits, exps) + 1e-6)
        self.result = None
        self.lock = threading.Lock()
        self.dirty = False

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        lo, hi = np.percentile(points[self.shown], [2, 98], axis=0)
        center, extent = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
        self.server.initial_camera.look_at = tuple(center)
        self.server.initial_camera.position = tuple(center + np.array([0.0, -0.5, 0.7]) * extent)
        self._build_gui()
        self._add_trajectories()
        self.update()

    # ---- GUI -------------------------------------------------------------------------------
    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown(f"**ELite++** · session `{self.session}` · {self.mode} · "
                         f"{len(self.points):,} points ({len(self.shown):,} shown)")
        with gui.add_folder("Display"):
            self.color_mode = gui.add_dropdown("color", COLOR_MODES, initial_value=COLOR_MODES[0])
            self.point_size = gui.add_slider("point size", 0.01, 0.5, 0.01, 0.05)
            self.show_traj = gui.add_checkbox("trajectories", True)
        self.color_mode.on_update(lambda _: self.redraw())
        self.point_size.on_update(lambda _: self.redraw())
        self.show_traj.on_update(lambda _: [setattr(h, "visible", self.show_traj.value) for h in self.traj])

        self.stats_md = gui.add_markdown("")
        with gui.add_folder("DOR parameters"):
            self.auto = gui.add_checkbox("auto update", True)
            apply_btn = gui.add_button("apply")
            reset_btn = gui.add_button("reset defaults")
            defaults = ClassifyParams()
            self.controls = {}
            folders = {g: gui.add_folder(g, expand_by_default=False) for g in GROUPS}
            for f in dataclasses.fields(ClassifyParams):
                with folders[group_of(f.name)]:
                    ctrl = add_control(gui, f, getattr(defaults, f.name))
                ctrl.on_update(lambda _: self.auto.value and self.request_update())
                self.controls[f.name] = ctrl
        apply_btn.on_click(lambda _: self.request_update())
        reset_btn.on_click(lambda _: self._reset())

        with gui.add_folder("Export"):
            gui.add_button("save params (YAML)").on_click(lambda _: self._save_params())
            gui.add_button("export static map (PCD)").on_click(lambda _: self._export_map())
            self.export_md = gui.add_markdown("")

    def _add_trajectories(self):
        self.traj = []
        for i, s in enumerate(self.cfg["sessions"]):
            path = session_paths(self.cfg, s)["poses"]
            if not os.path.exists(path):
                continue
            xyz = io.load_poses(path)[:, :3, 3].astype(np.float32)
            segs = np.stack([xyz[:-1], xyz[1:]], axis=1)
            self.traj.append(self.server.scene.add_line_segments(
                f"/trajectory/{s}", segs, SESSION_COLORS[i % len(SESSION_COLORS)], thickness=0.15))

    def params(self):
        return ClassifyParams(**{k: c.value for k, c in self.controls.items()})

    def _reset(self):
        auto = self.auto.value
        self.auto.value = False  # avoid one re-run per control
        for k, v in ClassifyParams().to_dict().items():
            self.controls[k].value = v
        self.auto.value = auto
        self.request_update()

    # ---- classification ---------------------------------------------------------------------
    def request_update(self):
        """Run in a background thread; coalesce changes made while a run is in progress."""
        self.dirty = True
        if self.lock.locked():
            return
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        with self.lock:
            while True:
                self.dirty = False
                t = time.time()
                self.result = classify(self.points, self.vmap, self.params())
                dt = time.time() - t
                self.redraw()
                counts = np.bincount(self.result.reason, minlength=len(REASONS))
                lines = [f"**classify** {dt:.2f} s · static {counts[0]:,} / {len(self.points):,}", ""]
                lines += [f"- <span style='color:rgb{tuple(int(c) for c in REASON_COLORS[k])}'>■</span> "
                          f"{REASONS[k]}: {counts[k]:,}" for k in REASONS if k]
                self.stats_md.content = "\n".join(lines)
                if not self.dirty:
                    break

    def redraw(self):
        if self.result is None:
            return
        pts = self.points[self.shown]
        static = self.result.is_static[self.shown]
        mode = self.color_mode.value
        if mode == "stability ratio":
            colors = ratio_colors(self.shown_ratio)
        else:
            colors = REASON_COLORS[self.result.reason[self.shown]]
        if mode == "static only":
            pts, colors = pts[static], colors[static]
        self.server.scene.add_point_cloud("/points", pts, colors, point_size=self.point_size.value,
                                          precision="float32")  # float16 is too coarse for map coordinates

    # ---- export -------------------------------------------------------------------------------
    def _out(self, name):
        return os.path.join(os.path.dirname(session_paths(self.cfg, self.session)["poses"]), name)

    def _save_params(self):
        changed = {k: v for k, v in self.params().to_dict().items() if ClassifyParams().to_dict()[k] != v}
        path = self._out(f"tuned_classify_{self.mode}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"dor": {"classify": changed}}, f, sort_keys=False)
        self.export_md.content = f"saved `{path}` ({len(changed)} changed)"

    def _export_map(self):
        path = self._out(f"tuned_{self.mode}_map.pcd")
        io.save_pcd(path, self.points[self.result.is_static])
        self.export_md.content = f"wrote `{path}`"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--session", help="session to inspect (default: last finished one)")
    parser.add_argument("--mode", choices=("lifelong", "session"), default="lifelong")
    parser.add_argument("--max-points", type=int, default=3_000_000, help="points sent to the browser")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    done = [s for s in cfg["sessions"] if os.path.exists(session_paths(cfg, s)["lifelong_vrm"])]
    if not done:
        raise SystemExit("no finished session found; run elitepp-run first")
    session = args.session or done[-1]
    print(f"loading session {session} ({args.mode}) ...", flush=True)
    points, vmap = load_inputs(cfg, session, args.mode)
    viewer = Viewer(cfg, session, args.mode, points, vmap, args.max_points, args.port)
    print(f"viewer at http://localhost:{viewer.server.get_port()}", flush=True)
    viewer.server.sleep_forever()


if __name__ == "__main__":
    main()
