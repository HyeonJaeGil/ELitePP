"""Voxel Ratio Map (paper Sec. 3.4): per-voxel hit / exposure counts and ground flags.

Voxels are integer keys (floor(p / voxel_size)). Internally each key is packed into one
int64 code so that lookups and merges are sorted-array operations instead of dict loops.
"""
from dataclasses import dataclass

import numpy as np

_BITS = 21
_OFF = 1 << (_BITS - 1)  # keys must lie in [-2^20, 2^20): +-300 km at 0.3 m voxels
_MASK = (1 << _BITS) - 1


def encode(keys):
    k = np.asarray(keys, dtype=np.int64).reshape(-1, 3) + _OFF
    return (k[:, 0] << (2 * _BITS)) | (k[:, 1] << _BITS) | k[:, 2]


def decode(codes):
    c = np.asarray(codes, dtype=np.int64)
    return np.stack([(c >> (2 * _BITS)) & _MASK, (c >> _BITS) & _MASK, c & _MASK], axis=1) - _OFF


def sum_by_code(codes, *values):
    """Group equal codes; return (unique sorted codes, summed value arrays...)."""
    uniq, inv = np.unique(codes, return_inverse=True)
    return (uniq, *[np.bincount(inv, weights=v, minlength=len(uniq)).astype(np.int64) for v in values])


@dataclass
class VoxelMap:
    codes: np.ndarray    # (N,) int64, sorted, unique
    hits: np.ndarray     # (N,) int64
    exps: np.ndarray     # (N,) int64, includes the hits (exposure = hit or traversed)
    ground: np.ndarray   # (N,) bool
    voxel_size: float

    def __len__(self):
        return len(self.codes)

    @property
    def keys(self):
        return decode(self.codes)

    def lookup(self, codes):
        """Index of each code in this map and whether it exists."""
        idx = np.clip(np.searchsorted(self.codes, codes), 0, max(len(self.codes) - 1, 0))
        found = (self.codes[idx] == codes) if len(self.codes) else np.zeros(len(codes), bool)
        return idx, found

    def query(self, codes):
        """(hits, exps, ground) for arbitrary codes; missing voxels read as 0 / False."""
        idx, found = self.lookup(codes)
        if not len(self.codes):
            return np.zeros(len(codes), np.int64), np.zeros(len(codes), np.int64), found
        return np.where(found, self.hits[idx], 0), np.where(found, self.exps[idx], 0), found & self.ground[idx]

    def ratio(self):
        """Stability score R(v) = hits / max(hits, exps)."""
        return self.hits / (np.maximum(self.hits, self.exps) + 1e-6)

    def merge(self, current):
        """Integrate a newer session map: counts are summed, ground flags come from `current`.

        NOTE: plain sum as in the original scripts; density-adaptive decay (eq. 8-9) is TODO.
        """
        assert self.voxel_size == current.voxel_size, "voxel sizes differ"
        codes = np.concatenate([self.codes, current.codes])
        uniq, hits, exps = sum_by_code(codes, np.concatenate([self.hits, current.hits]),
                                       np.concatenate([self.exps, current.exps]))
        return VoxelMap(uniq, hits, exps, current.query(uniq)[2], self.voxel_size)

    def save(self, path):
        np.savez_compressed(path, keys=self.keys.astype(np.int32), hits=self.hits.astype(np.int32),
                            exps=self.exps.astype(np.int32), grounds=self.ground, voxel_size=self.voxel_size)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        codes = encode(d["keys"])
        order = np.argsort(codes)
        return cls(codes[order], d["hits"][order].astype(np.int64), d["exps"][order].astype(np.int64),
                   d["grounds"][order].astype(bool), float(d["voxel_size"]))
