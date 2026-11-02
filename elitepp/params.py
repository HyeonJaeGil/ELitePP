"""Base class for parameter dataclasses loaded from YAML."""
from dataclasses import asdict, fields


class Params:
    @classmethod
    def from_dict(cls, d=None):
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self):
        return asdict(self)
