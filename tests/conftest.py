import numpy as np
import pytest


def make_session(n_frames=12, car_frames=4, seed=0):
    """Synthetic session in the map frame: ground + two walls seen in every frame, a car that leaves.

    The sensor drives along x (y = 0); the car stands between the path and the y = 10 wall only
    in the first `car_frames` frames, so later rays to the wall traverse its voxels. The scene is
    symmetric because DOR uses the point median as the ray origin.
    """
    rng = np.random.default_rng(seed)
    scans = []
    for i in range(n_frames):
        x0 = i * 1.0
        ground = np.c_[rng.uniform(x0 - 15, x0 + 15, 8000), rng.uniform(-12, 12, 8000), rng.normal(-1.8, 0.01, 8000)]
        walls = np.c_[rng.uniform(x0 - 15, x0 + 15, 8000), rng.choice([-10.0, 10.0], 8000) + rng.normal(0, 0.01, 8000),
                      rng.uniform(-1.8, 2.0, 8000)]
        parts = [ground, walls]
        if i < car_frames:
            parts.append(np.c_[rng.uniform(4, 8, 1500), rng.uniform(4, 6, 1500), rng.uniform(-1.8, -0.3, 1500)])
        scans.append(np.concatenate(parts))
    return scans


@pytest.fixture
def session():
    return make_session()


def car_mask(points):
    return (points[:, 0] > 4) & (points[:, 0] < 8) & (points[:, 1] > 4) & (points[:, 1] < 6) & (points[:, 2] > -1.4)
