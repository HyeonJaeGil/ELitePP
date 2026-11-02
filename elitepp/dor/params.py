"""DOR parameters. Defaults reproduce the original analyzer scripts exactly.

StatsParams change the accumulated voxel statistics (expensive: re-runs ray casting).
ClassifyParams only change the decision made on cached statistics (cheap: seconds on GPU),
so they are what the interactive tuner exposes.
"""
from dataclasses import dataclass

from elitepp.params import Params


@dataclass
class StatsParams(Params):
    voxel_size: float = 0.3            # voxel edge [m] of the Voxel Ratio Map
    downsample_size: float = 0.3       # per-scan voxel downsampling before analysis [m]
    batch_size: int = 5                # scans per ground-estimation batch
    # ray casting (exposure)
    local_max_dist: float = 35.0       # max range of the exposure grid [m]
    local_z_range: float = 3.5         # vertical half-extent of the exposure grid [m]
    min_range: float = 0.5             # grid cells closer than this are ignored [m]
    ang_res: float = 0.7               # angular bin of the range image [deg]
    # ground estimation
    ground_samples: int = 50000        # points sampled per batch for normal estimation
    normal_radius: float = 0.8         # normal estimation radius [m]
    ground_threshold: float = 0.5      # |n_z| above this counts as flat
    anchor_step: float = 5.0           # xy cell size of ground anchors [m]
    min_ground_points: int = 100       # flat points needed for an anchor
    ground_knn: int = 5                # anchors averaged for the local ground height
    upper_tolerance: float = 0.1       # ground band above the local height [m]
    lower_tolerance: float = 1.0       # ground band below the local height [m]


@dataclass
class ClassifyParams(Params):
    # initial voxel labels from the stability ratio R = hits / max(hits, exps)
    dynamic_ratio: float = 0.4         # R below this (non-ground) -> dynamic candidate
    absolute_dynamic_ratio: float = 0.05  # R below this -> dynamic, even on ground
    solid_ratio: float = 0.9           # R above this ...
    solid_min_hits: int = 10           # ... with more hits -> locked static
    # cluster-level refinement (paper: DBSCAN; code: 6-connected label propagation)
    enable_cluster_removal: bool = True
    cluster_max_height: float = 4.5    # only voxels up to this height above ground [m]
    cluster_max_size: int = 100        # labels stop growing at this many voxels
    cluster_iters: int = 15
    cluster_min_hits: int = 2          # voxels with fewer hits are ignored in the cluster score
    cluster_dynamic_ratio: float = 0.65  # mean cluster ratio at or below this -> dynamic
    # floating fragment removal
    enable_floating_removal: bool = True
    floating_min_height: float = 0.5   # height band above ground searched for fragments [m]
    floating_max_height: float = 4.0
    floating_iters: int = 10
    floating_max_size: int = 20        # fragments up to this many voxels can be removed
    # ground column propagation: low static voxels under mostly-dynamic columns
    enable_column_propagation: bool = True
    column_height: float = 1.5         # voxels this high above ground are re-checked [m]
    column_lookup: int = 14            # voxels looked at above each of them
    column_dynamic_ratio: float = 0.8
    # majority smoothing of weakly observed voxels (26-neighborhood vote)
    enable_smoothing: bool = True
    smoothing_max_hits: int = 30
    # point-level refinement
    enable_ankle_removal: bool = True  # bottom residue under dynamic voxels
    ankle_tolerance: float = 0.03      # points this far above the voxel mean height are removed [m]
    enable_in_voxel_recovery: bool = True  # probabilistic restore in isolated dynamic voxels
    recovery_max_dynamic_neighbors: int = 1
    recovery_planarity_min_hits: int = 30
    recovery_planarity_min_points: int = 10
    seed: int = 0                      # RNG seed of the recovery step
