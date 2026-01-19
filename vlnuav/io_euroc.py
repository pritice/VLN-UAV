import csv
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class CameraInfo:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray
    t_bs: np.ndarray


@dataclass
class FrameData:
    timestamp: int
    cam0_path: pathlib.Path
    cam1_path: pathlib.Path
    t_wb: np.ndarray


def _load_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_sensor_info(sensor_yaml: pathlib.Path) -> CameraInfo:
    data = _load_yaml(sensor_yaml)
    intr = data.get("intrinsics", data.get("camera_matrix", {}).get("data"))
    if intr is None:
        raise ValueError(f"Missing intrinsics in {sensor_yaml}")
    if isinstance(intr, list) and len(intr) >= 4:
        fx, fy, cx, cy = intr[:4]
    else:
        raise ValueError(f"Unsupported intrinsics format in {sensor_yaml}")
    distortion_data = data.get("distortion_coefficients", [0, 0, 0, 0])
    if isinstance(distortion_data, dict):
        distortion_data = distortion_data.get("data", [0, 0, 0, 0])
    distortion = np.array(distortion_data, dtype=float)
    t_bs = np.array(data.get("T_BS"), dtype=float)
    if t_bs.shape != (4, 4):
        raise ValueError(f"Invalid T_BS in {sensor_yaml}")
    return CameraInfo(fx=fx, fy=fy, cx=cx, cy=cy, distortion=distortion, t_bs=t_bs)


def _load_groundtruth(csv_path: pathlib.Path) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            timestamp = int(row[0])
            px, py, pz = map(float, row[1:4])
            qw, qx, qy, qz = map(float, row[4:8])
            t_wb = _pose_matrix(px, py, pz, qw, qx, qy, qz)
            poses[timestamp] = t_wb
    return poses


def _pose_matrix(px: float, py: float, pz: float, qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    r = quat_to_rot(qw, qx, qy, qz)
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = [px, py, pz]
    return t


def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _list_images(path: pathlib.Path) -> Dict[int, pathlib.Path]:
    images = {}
    for img in sorted(path.glob("*.png")):
        try:
            ts = int(img.stem)
        except ValueError:
            continue
        images[ts] = img
    return images


def load_euroc_sequence(root: pathlib.Path, max_frames: Optional[int] = None, pose_tolerance_ns: int = 2_000_000) -> Tuple[List[FrameData], CameraInfo, CameraInfo]:
    cam0_dir = root / "cam0"
    cam1_dir = root / "cam1"
    cam0_info = _load_sensor_info(cam0_dir / "sensor.yaml")
    cam1_info = _load_sensor_info(cam1_dir / "sensor.yaml")
    cam0_images = _list_images(cam0_dir / "data")
    cam1_images = _list_images(cam1_dir / "data")
    gt_path = root / "state_groundtruth_estimate0" / "data.csv"
    gt_poses = _load_groundtruth(gt_path)

    common_timestamps = sorted(set(cam0_images.keys()) & set(cam1_images.keys()))
    frames: List[FrameData] = []
    gt_keys = np.array(sorted(gt_poses.keys()), dtype=np.int64)
    for ts in common_timestamps:
        if gt_keys.size == 0:
            break
        idx = int(np.searchsorted(gt_keys, ts))
        candidate_indices = [idx - 1, idx]
        best_pose = None
        best_dt = None
        for cand in candidate_indices:
            if cand < 0 or cand >= gt_keys.size:
                continue
            gt_ts = int(gt_keys[cand])
            dt = abs(gt_ts - ts)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_pose = gt_poses[gt_ts]
        if best_dt is None or best_dt > pose_tolerance_ns:
            continue
        frames.append(FrameData(timestamp=ts, cam0_path=cam0_images[ts], cam1_path=cam1_images[ts], t_wb=best_pose))
        if max_frames and len(frames) >= max_frames:
            break
    return frames, cam0_info, cam1_info
