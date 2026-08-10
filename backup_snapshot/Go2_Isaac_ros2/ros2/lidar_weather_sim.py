"""Realtime adverse-weather LiDAR degradation and local elevation grids."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nav_msgs.msg import OccupancyGrid


WEATHER_PRESETS = {
    "clear": {"alpha": 0.0, "friction": 0.80, "noise_std": 0.00, "drop_scale": 0.00, "false_rate": 0.000},
    "mist": {"alpha": 0.005, "friction": 0.70, "noise_std": 0.01, "drop_scale": 0.15, "false_rate": 0.002},
    "light_fog": {"alpha": 0.01, "friction": 0.60, "noise_std": 0.02, "drop_scale": 0.25, "false_rate": 0.004},
    "medium_fog": {"alpha": 0.03, "friction": 0.45, "noise_std": 0.04, "drop_scale": 0.45, "false_rate": 0.008},
    "heavy_fog": {"alpha": 0.06, "friction": 0.32, "noise_std": 0.07, "drop_scale": 0.70, "false_rate": 0.015},
    "dense_fog": {"alpha": 0.10, "friction": 0.24, "noise_std": 0.10, "drop_scale": 0.85, "false_rate": 0.025},
    "extreme_fog": {"alpha": 0.20, "friction": 0.20, "noise_std": 0.14, "drop_scale": 1.00, "false_rate": 0.040},
}


def weather_preset(name: str) -> dict[str, float]:
    if name not in WEATHER_PRESETS:
        known = ", ".join(WEATHER_PRESETS)
        raise ValueError(f"Unknown weather preset '{name}'. Known presets: {known}")
    return dict(WEATHER_PRESETS[name])


@dataclass
class WeatherConfig:
    enabled: bool = True
    preset: str = "medium_fog"
    alpha: float | None = None
    min_friction: float = 0.20
    intensity: float = 120.0
    noise_variant: str = "v4"
    max_points: int = 60000
    publish_raw: bool = True
    fog_repo: str = "/home/yuan/WeatherGuard-Go2/LiDAR_fog_sim"

    @classmethod
    def from_cfg(cls, cfg) -> "WeatherConfig":
        weather = getattr(cfg, "weather", None)
        if weather is None:
            return cls(enabled=False)
        return cls(
            enabled=bool(getattr(weather, "enabled", True)),
            preset=str(getattr(weather, "preset", "medium_fog")),
            alpha=None if getattr(weather, "alpha", None) is None else float(weather.alpha),
            min_friction=float(getattr(weather, "min_friction", 0.20)),
            intensity=float(getattr(weather, "intensity", 120.0)),
            noise_variant=str(getattr(weather, "noise_variant", "v4")),
            max_points=int(getattr(weather, "max_points", 60000)),
            publish_raw=bool(getattr(weather, "publish_raw", True)),
            fog_repo=str(getattr(weather, "fog_repo", "/home/yuan/WeatherGuard-Go2/LiDAR_fog_sim")),
        )

    @property
    def params(self) -> dict[str, float]:
        params = weather_preset(self.preset)
        if self.alpha is not None:
            params["alpha"] = self.alpha
        params["friction"] = max(self.min_friction, float(params["friction"]))
        return params


class RealtimeFogSimulator:
    """Vectorized realtime approximation of LiDAR_fog_sim/fog_simulation.py.

    It uses the same alpha lookup tables and hard/soft fog structure, but avoids
    Python per-point loops so it can run inside the Isaac ROS2 bridge.
    """

    def __init__(self, config: WeatherConfig):
        self.config = config
        self.rng = np.random.default_rng(seed=42)
        self.lookup_alpha = None
        self.lookup_distance = None
        self.lookup_integral = None
        if self.config.enabled and self.config.params["alpha"] > 0.0:
            self._load_lookup()

    def _load_lookup(self) -> None:
        root = Path(self.config.fog_repo)
        table_dir = root / "integral_lookup_tables" / "original"
        alphas = []
        for file in table_dir.glob("integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_*.pickle"):
            alphas.append(float(file.stem.split("_")[-1]))
        if not alphas:
            raise FileNotFoundError(f"No fog lookup tables found in {table_dir}")
        target_alpha = self.config.params["alpha"]
        self.lookup_alpha = min(alphas, key=lambda value: abs(value - target_alpha))
        filename = table_dir / f"integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_{self.lookup_alpha}.pickle"
        with filename.open("rb") as file:
            table = pickle.load(file)

        max_idx = 2000
        self.lookup_distance = np.zeros(max_idx + 1, dtype=np.float32)
        self.lookup_integral = np.zeros(max_idx + 1, dtype=np.float32)
        for idx in range(max_idx + 1):
            key = round(idx * 0.1, 1)
            distance, integral = table[float(str(key))]
            self.lookup_distance[idx] = distance
            self.lookup_integral[idx] = integral

    def apply(self, xyz: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if xyz.size == 0:
            return xyz.reshape(0, 4).astype(np.float32), {"fog_returns": 0, "dropped": 0, "false_points": 0}

        points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if not self.config.enabled or self.config.params["alpha"] <= 0.0:
            intensity = np.full((points.shape[0], 1), 255.0, dtype=np.float32)
            return np.hstack((points, intensity)), {"fog_returns": 0, "dropped": 0, "false_points": 0}

        if points.shape[0] > self.config.max_points:
            keep_ids = self.rng.choice(points.shape[0], size=self.config.max_points, replace=False)
            points = points[keep_ids]

        params = self.config.params
        alpha = float(params["alpha"])
        ranges = np.linalg.norm(points[:, :3], axis=1)
        valid = ranges > 1e-3
        points = points[valid]
        ranges = ranges[valid]
        if points.size == 0:
            return np.zeros((0, 4), dtype=np.float32), {"fog_returns": 0, "dropped": 0, "false_points": 0}

        base_intensity = np.full(points.shape[0], self.config.intensity, dtype=np.float32)
        hard_intensity = np.exp(-2.0 * alpha * ranges) * base_intensity

        idx = np.clip(np.round(ranges * 10.0).astype(np.int32), 0, 2000)
        fog_distance = self.lookup_distance[idx]
        fog_integral = self.lookup_integral[idx]

        mor = math.log(20.0) / alpha
        beta = 0.046 / mor
        beta_0 = 0.000001 / math.pi
        fog_response = fog_integral * base_intensity * (ranges ** 2) * beta / beta_0
        fog_response = np.minimum(fog_response, 255.0)
        fog_mask = fog_response > hard_intensity

        degraded = points.copy()
        if np.any(fog_mask):
            scale = np.divide(fog_distance[fog_mask], ranges[fog_mask], out=np.ones(np.count_nonzero(fog_mask)), where=ranges[fog_mask] > 1e-3)
            degraded[fog_mask] *= scale[:, None]
            if self.config.noise_variant == "v4":
                additive = 10.0 * self.rng.beta(a=2, b=20, size=np.count_nonzero(fog_mask))
                noise_factor = (fog_distance[fog_mask] + additive) / np.maximum(fog_distance[fog_mask], 1e-3)
                degraded[fog_mask] *= noise_factor[:, None]

        radial_noise = self.rng.normal(0.0, float(params["noise_std"]), size=degraded.shape[0]).astype(np.float32)
        unit = degraded / np.maximum(np.linalg.norm(degraded, axis=1, keepdims=True), 1e-3)
        degraded = degraded + unit * radial_noise[:, None]

        drop_prob = np.clip(1.0 - np.exp(-alpha * ranges * float(params["drop_scale"])), 0.0, 0.85)
        keep = self.rng.random(degraded.shape[0]) > drop_prob
        dropped = int(np.count_nonzero(~keep))
        degraded = degraded[keep]
        intensity = np.where(fog_mask, fog_response, hard_intensity)[keep]

        false_count = int(min(4000, degraded.shape[0] * float(params["false_rate"])))
        if false_count > 0:
            false_range = self.rng.uniform(0.5, 8.0, size=false_count).astype(np.float32)
            false_yaw = self.rng.uniform(-math.pi, math.pi, size=false_count).astype(np.float32)
            false_pitch = self.rng.uniform(-0.25, 0.25, size=false_count).astype(np.float32)
            false_points = np.column_stack(
                (
                    false_range * np.cos(false_pitch) * np.cos(false_yaw),
                    false_range * np.cos(false_pitch) * np.sin(false_yaw),
                    false_range * np.sin(false_pitch),
                )
            ).astype(np.float32)
            false_intensity = self.rng.uniform(5.0, 70.0, size=false_count).astype(np.float32)
            degraded = np.vstack((degraded, false_points))
            intensity = np.concatenate((intensity, false_intensity))

        out = np.column_stack((degraded, intensity.astype(np.float32))).astype(np.float32)
        info = {
            "fog_returns": int(np.count_nonzero(fog_mask)),
            "dropped": dropped,
            "false_points": false_count,
            "alpha": alpha,
            "lookup_alpha": float(self.lookup_alpha or alpha),
        }
        return out, info


class LocalElevationMapper:
    def __init__(
        self,
        resolution: float = 0.20,
        forward_range: float = 10.0,
        lateral_range: float = 8.0,
        min_points: int = 3,
        variance_scale: float = 0.05,
        obstacle_height: float = 0.22,
        obstacle_variance: float = 0.04,
    ):
        self.resolution = resolution
        self.forward_range = forward_range
        self.lateral_range = lateral_range
        self.min_points = min_points
        self.variance_scale = variance_scale
        self.obstacle_height = obstacle_height
        self.obstacle_variance = obstacle_variance
        self.width = int(math.ceil(forward_range / resolution))
        self.height = int(math.ceil(lateral_range / resolution))

    def build(self, xyzi: np.ndarray, frame_id: str, stamp, fog_alpha: float = 0.0) -> tuple[OccupancyGrid, OccupancyGrid, OccupancyGrid]:
        points = np.asarray(xyzi, dtype=np.float32).reshape(-1, xyzi.shape[-1])[:, :3]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        mask = (x >= 0.0) & (x < self.forward_range) & (np.abs(y) < self.lateral_range * 0.5)
        x, y, z = x[mask], y[mask], z[mask]

        count = np.zeros((self.height, self.width), dtype=np.float32)
        z_sum = np.zeros_like(count)
        z2_sum = np.zeros_like(count)
        if x.size:
            ix = np.floor(x / self.resolution).astype(np.int32)
            iy = np.floor((y + self.lateral_range * 0.5) / self.resolution).astype(np.int32)
            valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
            np.add.at(count, (iy[valid], ix[valid]), 1.0)
            np.add.at(z_sum, (iy[valid], ix[valid]), z[valid])
            np.add.at(z2_sum, (iy[valid], ix[valid]), z[valid] ** 2)

        mean = np.divide(z_sum, count, out=np.zeros_like(z_sum), where=count > 0)
        var = np.divide(z2_sum, count, out=np.zeros_like(z2_sum), where=count > 0) - mean ** 2
        var = np.maximum(var, 0.0)

        count_conf = np.clip(count / float(self.min_points), 0.0, 1.0)
        var_conf = np.exp(-var / max(self.variance_scale, 1e-4))
        fog_conf = max(0.25, 1.0 - 2.0 * fog_alpha)
        confidence = count_conf * var_conf * fog_conf
        confidence[count <= 0] = -1

        obstacle = np.zeros_like(count)
        obstacle[(count >= self.min_points) & ((mean > self.obstacle_height) | (var > self.obstacle_variance))] = 100.0

        variance_score = np.clip(var / max(self.obstacle_variance, 1e-4), 0.0, 1.0) * 100.0
        variance_score[count <= 0] = -1

        return (
            self._grid(confidence, frame_id, stamp, scale=100.0),
            self._grid(variance_score, frame_id, stamp, scale=1.0),
            self._grid(obstacle, frame_id, stamp, scale=1.0),
        )

    def _grid(self, values: np.ndarray, frame_id: str, stamp, scale: float) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.info.resolution = float(self.resolution)
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = -self.lateral_range * 0.5
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        data = values.copy()
        known = data >= 0
        data[known] = np.clip(data[known] * scale, 0, 100)
        msg.data = data.astype(np.int8).reshape(-1).tolist()
        return msg
