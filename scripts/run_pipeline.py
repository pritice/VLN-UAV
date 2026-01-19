import argparse
import json
import pathlib
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vlnuav import bev as bev_lib
from vlnuav import io_euroc, io_tartanair
from vlnuav.visualize import save_bev_images


def parse_ids(text: Optional[str]) -> List[int]:
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def load_label_map(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return {int(k): v for k, v in json.load(handle).items()}


def compute_stereo_depth(left: np.ndarray, right: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=9)
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0
    disparity[disparity <= 0.1] = np.nan
    depth = (fx * baseline) / disparity
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def load_depth_tartanair(depth_path: pathlib.Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.max() > 255:
        depth *= 1.0 / 1000.0
    else:
        depth *= 1.0
    return depth


def load_segmentation(seg_path: Optional[pathlib.Path]) -> Optional[np.ndarray]:
    if seg_path is None:
        return None
    seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None
    if seg.ndim == 3:
        seg = seg[:, :, 0]
    return seg.astype(np.int32)


def build_map_from_euroc(frames: List[io_euroc.FrameData], cam0: io_euroc.CameraInfo, grid: bev_lib.GridConfig) -> bev_lib.BEVMap:
    bev = bev_lib.init_bev(grid)
    baseline = abs(cam0.t_bs[0, 3]) if cam0.t_bs is not None else 0.1
    for frame in frames:
        img_l = cv2.imread(str(frame.cam0_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(frame.cam1_path), cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            continue
        depth = compute_stereo_depth(img_l, img_r, cam0.fx, baseline)
        points_c, heights = bev_lib.points_from_depth(depth, (cam0.fx, cam0.fy, cam0.cx, cam0.cy))
        t_wc = frame.t_wb @ cam0.t_bs
        points_w = bev_lib.transform_points(points_c, t_wc)
        valid = (points_w[:, 2] >= grid.min_height) & (points_w[:, 2] <= grid.max_height)
        bev_lib.update_bev(bev, points_w[valid], points_w[valid][:, 2])
    return bev


def build_map_from_tartan(frames: List[io_tartanair.TartanFrame], intrinsics: Tuple[float, float, float, float], grid: bev_lib.GridConfig) -> bev_lib.BEVMap:
    bev = bev_lib.init_bev(grid)
    fx, fy, cx, cy = intrinsics
    for frame in frames:
        depth = load_depth_tartanair(frame.depth_path)
        seg = load_segmentation(frame.seg_path)
        points_c, heights = bev_lib.points_from_depth(depth, (fx, fy, cx, cy))
        t_wc = frame.t_wb
        points_w = bev_lib.transform_points(points_c, t_wc)
        valid = (points_w[:, 2] >= grid.min_height) & (points_w[:, 2] <= grid.max_height)
        labels = seg.reshape(-1) if seg is not None else None
        if labels is not None:
            labels = labels[depth.reshape(-1) > 0]
        bev_lib.update_bev(bev, points_w[valid], points_w[valid][:, 2], labels=labels)
    return bev


def main() -> None:
    parser = argparse.ArgumentParser(description="VLN-UAV 2.5D BEV pipeline")
    parser.add_argument("--dataset", choices=["euroc", "tartanair"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--grid-size-x", type=float, default=20.0)
    parser.add_argument("--grid-size-y", type=float, default=20.0)
    parser.add_argument("--grid-res", type=float, default=0.1)
    parser.add_argument("--min-height", type=float, default=-2.0)
    parser.add_argument("--max-height", type=float, default=2.0)
    parser.add_argument("--things-ids", default="")
    parser.add_argument("--stuff-ids", default="")
    parser.add_argument("--label-map", default=None)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = bev_lib.GridConfig(
        size_x=args.grid_size_x,
        size_y=args.grid_size_y,
        resolution=args.grid_res,
        min_height=args.min_height,
        max_height=args.max_height,
    )

    if args.dataset == "euroc":
        frames, cam0, cam1 = io_euroc.load_euroc_sequence(pathlib.Path(args.data_root), max_frames=args.max_frames)
        bev = build_map_from_euroc(frames, cam0, grid)
        intrinsics = (cam0.fx, cam0.fy, cam0.cx, cam0.cy)
    else:
        frames, (h, w) = io_tartanair.load_tartanair_sequence(pathlib.Path(args.data_root), max_frames=args.max_frames)
        intrinsics = (w / 2.0, w / 2.0, w / 2.0, h / 2.0)
        bev = build_map_from_tartan(frames, intrinsics, grid)

    things_ids = parse_ids(args.things_ids)
    stuff_ids = parse_ids(args.stuff_ids)
    masks = bev_lib.semantic_masks(bev.semantic, things_ids, stuff_ids) if bev.semantic is not None else None

    save_bev_images(output_dir, bev, masks)
    np.save(output_dir / "bev_occupancy.npy", bev.occupancy)
    np.save(output_dir / "bev_height_max.npy", bev.height_max)
    if bev.semantic is not None:
        np.save(output_dir / "bev_semantic.npy", bev.semantic)
    metadata = {
        "dataset": args.dataset,
        "intrinsics": intrinsics,
        "grid": grid.__dict__,
        "frames": len(frames),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
