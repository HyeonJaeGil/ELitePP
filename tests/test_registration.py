import numpy as np
import open3d as o3d

from elitepp.registration import RegistrationParams, align_session, average_transforms


def test_recovers_shift_of_structured_scene():
    rng = np.random.default_rng(0)
    parts = [np.c_[rng.uniform(-30, 30, (20000, 2)), np.zeros(20000)]]
    for c in rng.uniform(-25, 25, (25, 2)):
        mesh = o3d.geometry.TriangleMesh.create_box(*rng.uniform(1, 4, 3)).translate([c[0], c[1], 0])
        parts.append(np.asarray(mesh.sample_points_uniformly(3000).points))
    scene = np.concatenate(parts)
    shift = np.array([1.5, -2.0, 0.0])
    scans = np.array_split(rng.permutation(scene) - shift, 10)
    Ts = align_session(scans, scene, RegistrationParams(window_size=5, stride=5))
    assert len(Ts) == 10
    assert np.allclose(Ts[0][:3, 3], shift, atol=0.3)


def test_average_transforms_of_identical_is_identity_op():
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    assert np.allclose(average_transforms([T, T], [1, 3]), T)
