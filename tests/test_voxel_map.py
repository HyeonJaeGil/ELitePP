import numpy as np

from elitepp.voxel_map import VoxelMap, decode, encode


def vm(keys, hits, exps, ground):
    codes = encode(keys)
    order = np.argsort(codes)
    return VoxelMap(codes[order], np.array(hits)[order], np.array(exps)[order], np.array(ground)[order], 0.3)


def test_encode_roundtrip():
    keys = np.array([[0, 0, 0], [-5, 7, -1], [1000, -1000, 3], [-(2 ** 20), 2 ** 20 - 1, 0]])
    assert np.array_equal(decode(encode(keys)), keys)


def test_merge_sums_counts_and_takes_current_ground():
    a = vm([[0, 0, 0], [1, 0, 0]], [3, 1], [4, 5], [True, False])
    b = vm([[0, 0, 0], [2, 0, 0]], [2, 7], [2, 7], [False, True])
    m = a.merge(b)
    got = {tuple(k): (h, e, g) for k, h, e, g in zip(m.keys.tolist(), m.hits, m.exps, m.ground)}
    assert got == {(0, 0, 0): (5, 6, False), (1, 0, 0): (1, 5, False), (2, 0, 0): (7, 7, True)}


def test_query_missing_voxels_read_as_zero():
    a = vm([[0, 0, 0]], [3], [4], [True])
    hits, exps, ground = a.query(encode([[0, 0, 0], [9, 9, 9]]))
    assert hits.tolist() == [3, 0] and exps.tolist() == [4, 0] and ground.tolist() == [True, False]


def test_save_load_roundtrip(tmp_path):
    a = vm([[0, 0, 0], [-3, 2, 1]], [3, 1], [4, 5], [True, False])
    a.save(tmp_path / "v.npz")
    b = VoxelMap.load(tmp_path / "v.npz")
    for f in ("codes", "hits", "exps", "ground"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert b.voxel_size == 0.3
