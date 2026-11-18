"""Versioned pipeline configuration; no learned parameters are fitted here."""
from dataclasses import dataclass, asdict
import json

SCHEMA_VERSION = '1.0'

@dataclass(frozen=True)
class Config:
    radius_m: float = 50.0
    top_k: int = 8
    fallback: bool = True
    fallback_max_m: float = 300.0
    gps_gap_s: float = 5.0
    gps_jump_mps: float = 65.0
    stationary_mps: float = 0.5
    weak_label_distance_m: float = 12.0
    weak_label_margin_m: float = 8.0
    weak_label_heading_deg: float = 55.0
    weak_label_max_route_regression: int = 0
    # Never infer ground-truth from GPS/route row positions.

    def validate(self):
        if self.radius_m <= 0 or self.top_k < 1 or self.fallback_max_m < 0:
            raise ValueError('Invalid spatial search configuration')
        if self.weak_label_distance_m <= 0 or self.weak_label_margin_m < 0:
            raise ValueError('Invalid weak-label configuration')
        return self

    def as_json(self):
        return { 'schema_version': SCHEMA_VERSION, **asdict(self) }

    @classmethod
    def from_path(cls, path):
        if not path:
            return cls().validate()
        with open(path, encoding='utf8') as f:
            d = json.load(f)
        d.pop('schema_version', None)
        return cls(**d).validate()
