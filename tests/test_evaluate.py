import numpy as np

from elitepp.evaluate import alignment_metrics


def plane():
    g = np.stack(np.meshgrid(np.arange(0, 10, 0.05), np.arange(0, 10, 0.05)), -1).reshape(-1, 2)
    return np.c_[g, np.zeros(len(g))]


def test_identical_clouds():
    assert alignment_metrics(plane(), plane()) == {"AC": 1.0, "RMSE": 0.0, "CD": 0.0}


def test_shifted_plane():
    m = alignment_metrics(plane() + [0, 0, 0.1], plane())
    assert abs(m["RMSE"] - 0.1) < 1e-6 and abs(m["CD"] - 0.2) < 1e-6
