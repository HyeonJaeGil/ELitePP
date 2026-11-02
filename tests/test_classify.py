import numpy as np

from conftest import car_mask
from elitepp.dor import ClassifyParams, StatsParams, accumulate, classify


# The synthetic scans are sparse in angle, so a coarser range image is needed to register
# exposure behind the car (real LiDAR scans fill the default 0.7 deg bins).
STATS = StatsParams(local_max_dist=20.0, ang_res=2.0)


def run(session, params=None):
    np.random.seed(0)
    vmap, pts = accumulate(session, STATS)
    return pts, classify(pts, vmap, params)


def test_removes_car_keeps_wall(session):
    pts, r = run(session)
    car = car_mask(pts)
    wall = np.abs(pts[:, 1]) > 9.5
    assert car.any() and wall.any()
    assert r.is_static[car].mean() < 0.2, "car should be mostly removed"
    assert r.is_static[wall].mean() > 0.9, "wall should be kept"


def test_reason_is_zero_exactly_for_static(session):
    _, r = run(session)
    assert np.array_equal(r.reason == 0, r.is_static)


def test_deterministic_with_seed(session):
    _, a = run(session)
    _, b = run(session)
    assert np.array_equal(a.is_static, b.is_static)


def test_all_stages_off_runs(session):
    off = {k: False for k in ClassifyParams().to_dict() if k.startswith("enable_")}
    _, r = run(session, ClassifyParams(**off))
    assert r.is_static.dtype == bool
