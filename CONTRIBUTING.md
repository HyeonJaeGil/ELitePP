# Contributing

How to change ELite++ without silently changing its results. README.md explains what the
pipeline does; this file is about working on the code.

## Setup and commands

```bash
conda create -n elitepp python=3.12 -y && conda activate elitepp
pip install -e ".[viz,dev]"

pytest                                         # unit tests on synthetic data, < 10 s
elitepp-run configs/parkinglot.yaml            # full pipeline, ~5 min for two ParkingLot sessions
elitepp-run configs/parkinglot.yaml --rrd      # + Rerun recording per session
elitepp-eval --src <session_dir> <poses> --tgt <session_dir> <poses>
elitepp-view configs/parkinglot.yaml           # browser viewer with live DOR tuning
```

The pipeline logs one timestamped line per registration window and per stage, so long runs
are easy to follow in a log file.

## Where things live

| To change | Look at |
|---|---|
| session loop, output files | `elitepp/pipeline.py` (`FILES`, `run_session`) |
| registration | `elitepp/registration.py` (`align_session`, `RegistrationParams`) |
| what counts as hit / exposure / ground | `elitepp/dor/stats.py` |
| static / dynamic decision | `elitepp/dor/classify.py`, one block per stage as listed in its docstring |
| parameters | `elitepp/dor/params.py`, `RegistrationParams`; document them in `docs/parameters.md` |
| merging sessions | `elitepp/voxel_map.py` (`VoxelMap.merge`) |
| viewer GUI | `elitepp/viz/viewer.py`; controls are generated from `ClassifyParams` fields |
| Rerun recordings | `elitepp/viz/rerun_logger.py` (no-ops unless `--rrd` is given) |

## Rules that keep results trustworthy

- **Default parameters reproduce the original research scripts.** `tests/test_stats.py`
  contains the original dict-based accumulation as a reference and must stay bit-identical.
  To change behavior on purpose, add a parameter whose default keeps the old behavior, or
  update the reference test and `docs/implementation_notes.md` in the same commit.
- **Classification must stay fast.** The viewer re-runs `classify` on every slider change;
  avoid Python loops over voxels and time it on ~15 M points before and after a change.
- **Voxel keys** are `floor(p / voxel_size)` packed by `voxel_map.encode` (21 bits per axis).
  Maps can only be merged when their `voxel_size` is equal.
- **Poses** are KITTI 3x4 text files; `aligned_pose = correction @ odometry_pose`.
- **Reproducibility:** each session seeds NumPy with `seed`, the recovery step uses
  `ClassifyParams.seed`. KISS-Matcher is not deterministic, so aligned poses vary slightly.
- Config keys must match a parameter dataclass field; unknown keys raise an error.

## Checking a change

1. `pytest`
2. DOR changes: compare the logged `[<session>] session map: <static> / <total>` line with a
   previous run. ParkingLot 01 with default parameters gives `14172530 / 16415743`.
3. Registration changes: compare `elitepp-eval` against session 01 before and after
   (KISS-Matcher varies slightly between runs, so compare a few runs).
4. Look at it: `elitepp-view` for DOR, `rerun <session>/debug.rrd` for registration windows.

## Style

- Comments explain *why*. Known simplifications are marked `NOTE:` with their limit and how
  to lift it (`grep -rn "NOTE:" elitepp`); implementation choices are described in `docs/implementation_notes.md`.
- Small functions, flat modules, no new dependency for what NumPy, Open3D or PyTorch already do.
- One commit per change; mention measured effects (timings, point counts) in the message.
