# Parameters

Every parameter lives in a dataclass; the YAML config overrides fields by name.
Defaults reproduce the authors' original scripts exactly.

## What it costs to change a parameter

Timings for one ParkingLot session (~700 scans, 16 M analyzed points) on an RTX 3080.

| Tier | Parameters | What re-runs | Time | Interactive? |
|---|---|---|---|---|
| **Live** | `dor.classify.*` | classification on cached voxel statistics | ~0.5 s (session), ~1 s (lifelong, 26 M points) | yes: `elitepp-view` sliders |
| **Per session** | `dor.stats.*` | ray casting + ground estimation, then classification | ~110 s | no (re-run `elitepp-run --force`) |
| **Per session + downstream** | `registration.*` | KISS-Matcher per window, then DOR | ~30 s + DOR | no |

Changing a stats or registration parameter changes every later session, because each session
is aligned to and merged with the previous lifelong map.

## `registration` — `elitepp.registration.RegistrationParams`

| Name | Default | Meaning |
|---|---|---|
| `window_size` | 200 | consecutive scans aggregated into one query submap |
| `stride` | 100 | scans between window starts; smaller than `window_size` gives overlapping windows whose estimates are averaged |
| `fitness_threshold` | 0.93 | inlier ratio a window needs to be trusted |
| `map_voxel_size` | 0.35 | downsampling of submaps and the target map [m] |
| `matcher_voxel_size` | 1.0 | KISS-Matcher voxel size [m] |
| `first_eval_distance` | 50.0 | inlier distance before any window is trusted [m] (see implementation_notes.md) |
| `eval_distance` | 2.0 | inlier distance once tracking [m] |
| `min_correspondences` | 10 | windows with fewer matches are skipped |

## `dor.stats` — `elitepp.dor.StatsParams`

| Name | Default | Meaning |
|---|---|---|
| `voxel_size` | 0.3 | voxel edge of the Voxel Ratio Map [m]; must match across sessions |
| `downsample_size` | 0.3 | per-scan downsampling before analysis [m] |
| `batch_size` | 5 | scans per ground-estimation batch |
| `local_max_dist` | 35.0 | range of the exposure grid around the ray origin [m] |
| `local_z_range` | 3.5 | vertical half-extent of the exposure grid [m] |
| `min_range` | 0.5 | grid cells closer than this are ignored [m] |
| `ang_res` | 0.7 | angular bin of the per-scan range image [deg]; coarser bins register more exposure on sparse scans |
| `ground_samples` | 50000 | points sampled per batch for normal estimation |
| `normal_radius` | 0.8 | normal estimation radius [m] |
| `ground_threshold` | 0.5 | `abs(n_z)` above this counts as flat |
| `anchor_step` | 5.0 | xy cell of a ground anchor [m] |
| `min_ground_points` | 100 | flat points needed for an anchor |
| `ground_knn` | 5 | anchors averaged for the local ground height |
| `upper_tolerance`, `lower_tolerance` | 0.1, 1.0 | ground band above / below the local height [m] |

## `dor.classify` — `elitepp.dor.ClassifyParams`

Stages run in this order; each `enable_*` switch turns one off. The viewer colors points by
the stage that removed them.

| Stage | Name | Default | Meaning |
|---|---|---|---|
| initial | `dynamic_ratio` | 0.4 | R below this (non-ground) → dynamic |
| | `absolute_dynamic_ratio` | 0.05 | R below this → dynamic even on the ground (also re-applied at the very end) |
| | `solid_ratio`, `solid_min_hits` | 0.9, 10 | R above and hits above → always static |
| cluster | `enable_cluster_removal` | true | regions near the ground judged by their mean R |
| | `cluster_max_height` | 4.5 | only voxels up to this height above ground [m] |
| | `cluster_max_size` | 100 | labels stop growing at this many voxels (bounds region size) |
| | `cluster_iters` | 15 | label propagation iterations |
| | `cluster_min_hits` | 2 | voxels with fewer hits do not count in the mean |
| | `cluster_dynamic_ratio` | 0.65 | mean R at or below this → whole region dynamic |
| floating | `enable_floating_removal` | true | small static fragments with no static neighbor outside themselves → dynamic |
| | `floating_min_height`, `floating_max_height` | 0.5, 4.0 | height band searched [m] |
| | `floating_iters` | 10 | label propagation iterations |
| | `floating_max_size` | 20 | larger fragments are kept |
| column | `enable_column_propagation` | true | low static voxels under mostly dynamic columns → dynamic |
| | `column_height` | 1.5 | voxels up to this height above ground are re-checked [m] |
| | `column_lookup` | 14 | voxels examined above each |
| | `column_dynamic_ratio` | 0.8 | share of dynamic voxels above needed |
| smoothing | `enable_smoothing` | true | weakly observed voxels take the 26-neighborhood majority |
| | `smoothing_max_hits` | 30 | voxels with at most this many hits are smoothed |
| ankle | `enable_ankle_removal` | true | in a ground voxel under a dynamic voxel, points above the voxel's mean height are removed; ground voxels under static / empty space are locked static |
| | `ankle_tolerance` | 0.03 | height margin above the mean [m] |
| recovery | `enable_in_voxel_recovery` | true | points of isolated low-R voxels are restored with probability min(1, R/2 · (1 + planarity)) |
| | `recovery_max_dynamic_neighbors` | 1 | "isolated": at most this many low-R voxels among the 26 neighbors |
| | `recovery_planarity_min_hits`, `recovery_planarity_min_points` | 30, 10 | planarity is computed only for voxels with enough hits / points |
| | `seed` | 0 | RNG seed of the recovery draw |

## Tuning workflow

1. `elitepp-run configs/parkinglot.yaml` once (computes and saves the voxel statistics).
2. `elitepp-view configs/parkinglot.yaml --mode session`, adjust sliders, watch the counts per stage.
3. "save params (YAML)" writes `tuned_classify_session.yaml` next to the outputs; copy its
   `dor.classify` block into the config.
4. `elitepp-run configs/parkinglot.yaml --force` to propagate through all sessions.
