import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TartanFrame:
    index: int
    image_path: pathlib.Path
    depth_path: pathlib.Path
    seg_path: Optional[pathlib.Path]
    t_wb: np.ndarray


def _load_pose_file(path: pathlib.Path) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) < 7:
                continue
            px, py, pz, qx, qy, qz, qw = vals[:7]
            poses.append(_pose_matrix(px, py, pz, qw, qx, qy, qz))
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


def load_tartanair_sequence(root: pathlib.Path, max_frames: Optional[int] = None) -> Tuple[List[TartanFrame], Tuple[int, int]]:
    image_dir = root / "image_left"
    depth_dir = root / "depth_left"
    seg_dir = root / "seg_left"
    pose_path = root / "pose_left.txt"

    images = sorted(image_dir.glob("*.png"))
    depths = sorted(depth_dir.glob("*.png"))
    segs = sorted(seg_dir.glob("*.png")) if seg_dir.exists() else []
    poses = _load_pose_file(pose_path)

    count = min(len(images), len(depths), len(poses))
    frames: List[TartanFrame] = []
    for idx in range(count):
        seg_path = segs[idx] if idx < len(segs) else None
        frames.append(
            TartanFrame(
                index=idx,
                image_path=images[idx],
                depth_path=depths[idx],
                seg_path=seg_path,
                t_wb=poses[idx],
            )
        )
        if max_frames and len(frames) >= max_frames:
            break
    if not frames:
        raise FileNotFoundError(f"No frames found in {root}")
    return frames, _infer_image_shape(images[0])


def _infer_image_shape(image_path: pathlib.Path) -> Tuple[int, int]:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read {image_path}")
    h, w = img.shape[:2]
    return h, w
