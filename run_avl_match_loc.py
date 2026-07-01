#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Match + localization AnyVisLoc runner based on avl_data.py.

This is a run_avl.py variant for testing pixel matchers and PnP localization
without image retrieval:
  - reads samples through avl_data.AnyVisLocNPZDataset;
  - loads PNG/NPY references through avl_utils.build_anyvisloc_reference_view;
  - crops the reference patch from the ground-truth view center;
  - initializes matchers through avl_utils.matching_init;
  - matches through avl_utils.run_pixel_match_anyvisloc;
  - converts reference patch matches back to dataset-local XYZ through DSM;
  - localizes through avl_utils.estimate_drone_pose_anyvisloc.

Compared with run_avl.py, this skips retrieval entirely. Compared with the old
AnyVisLoc_open_npz_new.py, it uses the new avl_data data-loading path.
"""

import argparse
import csv
import json
import math
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from avl_data import AnyVisLocNPZDataset
from avl_utils import (
    build_anyvisloc_reference_view,
    draw_pnp_inlier_matches_sampled,
    ensure_dir,
    estimate_drone_pose_anyvisloc,
    matching_init,
    normalize_reference_mode,
    run_pixel_match_anyvisloc,
    sample_dsm_local,
    save_json,
    tensor_rgb_to_bgr_uint8,
    to_numpy,
    to_python_str,
)


def parse_scenes(scenes):
    if scenes is None or len(scenes) == 0:
        return None
    if len(scenes) == 1 and "," in scenes[0]:
        return [x.strip() for x in scenes[0].split(",") if x.strip()]
    return scenes


def parse_matching_methods(values):
    if values is None or len(values) == 0:
        return None
    if len(values) == 1 and "," in values[0]:
        return [x.strip() for x in values[0].split(",") if x.strip()]
    return [str(x).strip() for x in values if str(x).strip()]


def get_parse():
    parser = argparse.ArgumentParser(description="AnyVisLoc match+PnP test runner")
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--yaml", default="config_selectable_matchers.yaml", type=str)
    parser.add_argument("--save_dir", default="./Result/AnyVisLoc_match_loc", type=str)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--reference_mode",
        default="aerial",
        choices=["aerial", "satellite"],
        help="Reference source from Lxx_reference.json.",
    )
    parser.add_argument(
        "--matching_methods",
        nargs="*",
        default=None,
        help="Override MATCHING_METHODS in yaml. Example: --matching_methods SP_LG ALIKED_LG",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--pose_priori",
        default="yp",
        choices=["yp", "p", "unknown"],
        help="yp: view-center shift + yaw-rotated patch; p: view-center shift only; unknown: no shift/no yaw.",
    )
    parser.add_argument("--PnP_method", default="P3P", type=str, help="P3P, AP3P, EPNP")
    parser.add_argument("--resize_ratio", default=0.4, type=float)
    parser.add_argument(
        "--patch_scale",
        default=1.0,
        type=float,
        help="Physical reference patch scale. Same default as current run_avl.py.",
    )
    parser.add_argument("--pre_crop_scale", default=2.0, type=float)
    parser.add_argument("--min_patch_size_m", default=0.0, type=float)
    parser.add_argument("--max_patch_size_m", default=0.0, type=float)
    parser.add_argument("--max_patch_pixels", default=0, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_every", default=20, type=int)
    parser.add_argument("--debug_raise", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Save UAV/ref patch/PnP inlier visualization.")
    parser.add_argument("--draw_pnp_inlier_ratio", default=0.1, type=float)
    parser.add_argument("--draw_pnp_inlier_seed", default=0, type=int)
    parser.add_argument("--save_match_arrays", action="store_true", help="Save raw match arrays to matches.npz.")
    parser.add_argument("--min_matches", default=5, type=int)
    parser.add_argument("--pnp_reproj_error", default=8.0, type=float)
    parser.add_argument("--pnp_iterations", default=2000, type=int)
    parser.add_argument("--pnp_confidence", default=0.999, type=float)
    parser.add_argument("--pnp_zero_distortion", action="store_true")
    parser.add_argument(
        "--selectable_code_dir",
        default=str(Path(__file__).resolve().parent / "Matching_Models" / "Sparse_matchers"),
        type=str,
        help="Directory containing selectable_sparse_matcher.py and matcher dependencies.",
    )
    parser.add_argument("--selectable_module", default="selectable_sparse_matcher", type=str)
    parser.add_argument("--match_keypoints", default=3000, type=int)
    parser.add_argument(
        "--gim_lg_ckpt",
        default=str(
            Path(__file__).resolve().parent
            / "Matching_Models"
            / "Sparse_matchers"
            / "weights"
            / "gim_lightglue_100h.ckpt"
        ),
        type=str,
    )
    parser.add_argument(
        "--minima_lg_ckpt",
        default=str(
            Path(__file__).resolve().parent
            / "Matching_Models"
            / "Sparse_matchers"
            / "weights"
            / "minima_lightglue.pth"
        ),
        type=str,
    )
    opt = parser.parse_args()
    opt.scenes = parse_scenes(opt.scenes)
    opt.matching_methods = parse_matching_methods(opt.matching_methods)
    opt.reference_mode = normalize_reference_mode(opt.reference_mode)
    opt.draw_pnp_inlier_ratio = max(0.0, min(1.0, float(opt.draw_pnp_inlier_ratio)))
    return opt


def save_csv(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_pickle(path, **kwargs):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(kwargs, f)


def safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def make_true_pos(sample):
    image_size = to_numpy(sample.get("image_size", np.array([0, 0]))).astype(np.int32).reshape(-1)
    if image_size.size >= 2 and int(image_size[0]) > 0:
        h, w = int(image_size[0]), int(image_size[1])
    else:
        img = sample["image"]
        h, w = int(img.shape[-2]), int(img.shape[-1])

    xyz = to_numpy(sample["xyz"]).astype(np.float32).reshape(3)
    euler = to_numpy(sample["euler_deg"]).astype(np.float32).reshape(3)
    k = to_numpy(sample["K"]).astype(np.float32)
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
        "width": int(w),
        "height": int(h),
        "K": k,
    }


def compute_view_center_xy(true_pos, use_shift=True):
    """Compute dataset-local view center from ground-truth pose."""
    x = float(true_pos["x"])
    y = float(true_pos["y"])
    z = float(true_pos["z"])
    if not use_shift:
        return x, y

    pitch = float(true_pos["pitch"])
    yaw = float(true_pos["yaw"])
    tan_v = np.tan(pitch / 180.0 * np.pi)
    if (not np.isfinite(tan_v)) or abs(float(tan_v)) < 1e-6:
        return x, y

    yaw_rad = yaw / 180.0 * np.pi
    dx = z * float(tan_v) * np.sin(yaw_rad)
    dy = -z * float(tan_v) * np.cos(yaw_rad)
    if (not np.isfinite(dx)) or (not np.isfinite(dy)):
        return x, y
    return x + float(dx), y + float(dy)


def local_to_pixel(x, y, map_resolution, map_origin_local):
    res = np.asarray(map_resolution, dtype=np.float32).reshape(-1)
    origin = np.asarray(map_origin_local, dtype=np.float32).reshape(-1)
    col = (float(x) - float(origin[0])) / max(float(res[0]), 1e-12)
    row = (float(y) - float(origin[1])) / max(float(res[1]), 1e-12)
    return float(col), float(row)


def pixel_to_local(col, row, map_resolution, map_origin_local):
    res = np.asarray(map_resolution, dtype=np.float32).reshape(-1)
    origin = np.asarray(map_origin_local, dtype=np.float32).reshape(-1)
    x = float(col) * float(res[0]) + float(origin[0])
    y = float(row) * float(res[1]) + float(origin[1])
    return float(x), float(y)


def rotate_image_keep_all(image, angle_deg, interp=cv2.INTER_LINEAR):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    mat[0, 2] += new_w / 2.0 - center[0]
    mat[1, 2] += new_h / 2.0 - center[1]
    out = cv2.warpAffine(image, mat, (new_w, new_h), flags=interp, borderValue=(0, 0, 0))
    return out, mat


def even_ceil(value, min_value=2):
    out = max(int(min_value), int(math.ceil(float(value))))
    if out % 2:
        out += 1
    return out


def estimate_drone_resolution(true_pos):
    """Meters per UAV pixel, matching avl_utils.resolution_size_anyvisloc."""
    k = np.asarray(true_pos["K"], dtype=np.float32)
    fx = max(float(k[0, 0]) if k.size >= 9 else 1.0, 1e-6)
    z = max(float(true_pos["z"]), 1e-6)
    pitch = float(true_pos["pitch"])
    denom = abs(float(np.sin(np.pi * (90.0 + pitch) / 180.0)))
    denom = max(denom, 1e-6)
    return float((z / denom) / fx)


def estimate_patch_size_px(true_pos, map_resolution, opt):
    drone_resolution = estimate_drone_resolution(true_pos)
    min_size = min(int(true_pos["width"]), int(true_pos["height"]))
    view_m = float(min_size) * drone_resolution * float(opt.patch_scale)

    min_patch_size_m = float(opt.min_patch_size_m or 0.0)
    max_patch_size_m = float(opt.max_patch_size_m or 0.0)
    if min_patch_size_m > 0:
        view_m = max(view_m, min_patch_size_m)
    if max_patch_size_m > 0:
        view_m = min(view_m, max_patch_size_m)

    res = np.asarray(map_resolution, dtype=np.float32).reshape(-1)
    patch_w = even_ceil(view_m / max(float(res[0]), 1e-12))
    patch_h = even_ceil(view_m / max(float(res[1]), 1e-12))

    max_px = int(opt.max_patch_pixels or 0)
    if max_px > 0:
        patch_w = min(patch_w, max_px)
        patch_h = min(patch_h, max_px)
        if patch_w % 2:
            patch_w = max(2, patch_w - 1)
        if patch_h % 2:
            patch_h = max(2, patch_h - 1)

    return int(max(2, patch_w)), int(max(2, patch_h)), float(view_m), float(drone_resolution)


def crop_gt_reference_patch(ref_img, true_pos, view, opt):
    """Crop a reference patch centered by ground truth.

    pose_priori=yp follows the old matching+localization test style:
    crop a larger area, yaw-rotate it, then center-crop the final patch.
    """
    map_resolution = np.asarray(view["map_resolution"], dtype=np.float32).reshape(2)
    map_origin_local = np.asarray(view["map_origin_local"], dtype=np.float32).reshape(2)
    center_x, center_y = compute_view_center_xy(
        true_pos,
        use_shift=(opt.pose_priori != "unknown"),
    )
    center_col, center_row = local_to_pixel(center_x, center_y, map_resolution, map_origin_local)
    final_w, final_h, view_m, drone_resolution = estimate_patch_size_px(true_pos, map_resolution, opt)

    h, w = ref_img.shape[:2]
    cxi = int(round(center_col))
    cyi = int(round(center_row))

    if opt.pose_priori == "yp":
        pre_scale = max(1.0, float(opt.pre_crop_scale))
        crop_w = int(final_w * pre_scale)
        crop_h = int(final_h * pre_scale)
    else:
        crop_w = final_w
        crop_h = final_h

    x1 = max(0, cxi - crop_w // 2)
    y1 = max(0, cyi - crop_h // 2)
    x2 = min(w, cxi + crop_w // 2)
    y2 = min(h, cyi + crop_h // 2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid GT crop center=({center_col:.1f},{center_row:.1f}), ref_size=({w},{h})"
        )

    pre_patch = ref_img[y1:y2, x1:x2].copy()
    if opt.pose_priori == "yp":
        rotated, rotation_mat = rotate_image_keep_all(pre_patch, float(true_pos["yaw"]))
        rh, rw = rotated.shape[:2]
        rcx, rcy = rw // 2, rh // 2
        rx1 = max(0, rcx - final_w // 2)
        ry1 = max(0, rcy - final_h // 2)
        rx2 = min(rw, rcx + final_w // 2)
        ry2 = min(rh, rcy + final_h // 2)
        if rx2 <= rx1 or ry2 <= ry1:
            raise ValueError("Invalid rotated GT crop")
        ref_patch = rotated[ry1:ry2, rx1:rx2].copy()
        final_crop = [int(rx1), int(ry1), int(rx2), int(ry2)]
    else:
        rotation_mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        ref_patch = pre_patch
        final_crop = [0, 0, int(ref_patch.shape[1]), int(ref_patch.shape[0])]

    info = {
        "rotation_matrix": np.asarray(rotation_mat, dtype=np.float32),
        "pre_crop": [int(x1), int(y1), int(x2), int(y2)],
        "final_crop": final_crop,
        "map_resolution": map_resolution,
        "map_origin_local": map_origin_local,
        "center_x_dataset_local": float(center_x),
        "center_y_dataset_local": float(center_y),
        "center_col": float(center_col),
        "center_row": float(center_row),
        "target_patch_size_m": float(view_m),
        "drone_resolution_m_per_px": float(drone_resolution),
        "patch_width_px": int(ref_patch.shape[1]),
        "patch_height_px": int(ref_patch.shape[0]),
        "patch_width_m": float(ref_patch.shape[1]) * float(map_resolution[0]),
        "patch_height_m": float(ref_patch.shape[0]) * float(map_resolution[1]),
    }
    return ref_patch, info


def ref_patch_pts_to_local(ref_pts_patch, crop_info):
    """Convert original ref-patch pixel xy to dataset-local xy."""
    ref_pts_patch = np.asarray(ref_pts_patch, dtype=np.float32).reshape(-1, 2)
    rx1, ry1, _, _ = crop_info["final_crop"]
    x1, y1, _, _ = crop_info["pre_crop"]
    mat = np.asarray(crop_info["rotation_matrix"], dtype=np.float32)
    mat_inv = cv2.invertAffineTransform(mat)

    xs = []
    ys = []
    for point in ref_pts_patch:
        xr = float(rx1) + float(point[0])
        yr = float(ry1) + float(point[1])
        xp = mat_inv[0, 0] * xr + mat_inv[0, 1] * yr + mat_inv[0, 2]
        yp = mat_inv[1, 0] * xr + mat_inv[1, 1] * yr + mat_inv[1, 2]
        col = float(x1) + float(xp)
        row = float(y1) + float(yp)
        x, y = pixel_to_local(
            col,
            row,
            crop_info["map_resolution"],
            crop_info["map_origin_local"],
        )
        xs.append(x)
        ys.append(y)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def resize_for_matching(uav_bgr, ref_patch, true_pos, map_resolution, opt):
    resize_ratio = float(opt.resize_ratio)
    drone_resolution = estimate_drone_resolution(true_pos)
    ref_res_scalar = float(np.mean(np.asarray(map_resolution, dtype=np.float32).reshape(-1)))
    fine_scale = drone_resolution / max(ref_res_scalar, 1e-12)

    if resize_ratio < 1:
        uav_match = cv2.resize(uav_bgr, None, fx=resize_ratio, fy=resize_ratio)
        uav_scale = resize_ratio
    else:
        uav_match = uav_bgr
        uav_scale = 1.0

    ref_resize = uav_scale / max(float(fine_scale), 1e-12)
    ref_match = cv2.resize(ref_patch, None, fx=ref_resize, fy=ref_resize)
    return uav_match, ref_match, float(fine_scale), float(ref_resize), float(uav_scale)


def compute_error_m(pred_xyz, true_pos):
    if pred_xyz is None or pred_xyz.get("X") is None or pred_xyz.get("Y") is None:
        return None, None, None
    dx = float(pred_xyz["X"]) - float(true_pos["x"])
    dy = float(pred_xyz["Y"]) - float(true_pos["y"])
    return float(np.sqrt(dx * dx + dy * dy)), float(dx), float(dy)


def process_one_sample(sample, method_dict, matching_method, opt):
    sample_id = to_python_str(sample["sample_id"])
    scene_name = to_python_str(sample["scene_name"])
    save_dir = Path(opt._combo_dir) / scene_name / sample_id
    ensure_dir(save_dir)

    result_json = save_dir / "result.json"
    if opt.skip_existing and result_json.exists():
        with result_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    true_pos = make_true_pos(sample)
    k = to_numpy(sample["K"]).astype(np.float32)
    dist = to_numpy(sample["dist"]).astype(np.float32)
    uav_bgr = tensor_rgb_to_bgr_uint8(sample["image"])

    view = build_anyvisloc_reference_view(sample, opt.reference_mode)
    ref_map = view["map"]
    dsm = view["dsm"]
    dsm_resolution = view["dsm_resolution"]
    dsm_origin_local = view["dsm_origin_local"]
    map_resolution = view["map_resolution"]

    ref_patch, crop_info = crop_gt_reference_patch(ref_map, true_pos, view, opt)
    uav_match, ref_match, fine_scale, ref_resize, uav_scale = resize_for_matching(
        uav_bgr,
        ref_patch,
        true_pos,
        map_resolution,
        opt,
    )

    t_match0 = time.time()
    uav_pts_match, ref_pts_match = run_pixel_match_anyvisloc(
        uav_match,
        ref_match,
        method_dict,
        str(save_dir),
        "/match_raw.jpg",
        need_ransac=False,
        show_matches=False,
    )
    match_time = time.time() - t_match0

    uav_pts_match = np.asarray(uav_pts_match, dtype=np.float32).reshape(-1, 2)
    ref_pts_match = np.asarray(ref_pts_match, dtype=np.float32).reshape(-1, 2)
    num_matches = int(min(len(uav_pts_match), len(ref_pts_match)))
    uav_pts_match = uav_pts_match[:num_matches]
    ref_pts_match = ref_pts_match[:num_matches]

    uav_pts_original = uav_pts_match / max(float(uav_scale), 1e-12)
    ref_pts_patch = ref_pts_match / max(float(ref_resize), 1e-12)
    local_x, local_y = ref_patch_pts_to_local(ref_pts_patch, crop_info)
    dsm_z, valid_dsm = sample_dsm_local(
        dsm,
        local_x,
        local_y,
        dsm_resolution,
        dsm_origin_local,
    )

    valid = (
        valid_dsm
        & np.isfinite(uav_pts_original).all(axis=1)
        & np.isfinite(local_x)
        & np.isfinite(local_y)
        & np.isfinite(dsm_z)
    )
    pnp_input_count = int(np.sum(valid))

    pred_xyz = None
    cam_angle = None
    inliers = 0
    inlier_indices = np.zeros((0, 1), dtype=np.int32)
    pnp_time = 0.0

    if num_matches >= int(opt.min_matches) and pnp_input_count >= 4:
        dist_coeffs = None if opt.pnp_zero_distortion else dist
        t_pnp0 = time.time()
        pred_xyz, cam_angle, inliers, inlier_indices = estimate_drone_pose_anyvisloc(
            uav_pts_original[valid],
            k,
            local_x[valid],
            local_y[valid],
            dsm_z[valid],
            dist_coeffs=dist_coeffs,
            pnp_method=opt.PnP_method,
            reprojection_error=opt.pnp_reproj_error,
            iterations_count=opt.pnp_iterations,
            confidence=opt.pnp_confidence,
        )
        pnp_time = time.time() - t_pnp0

    pred_error, err_x, err_y = compute_error_m(pred_xyz, true_pos)
    pnp_success = pred_error is not None and int(inliers) > 0

    if opt.visualize:
        cv2.imwrite(str(save_dir / "uav_match_input.jpg"), uav_match)
        cv2.imwrite(str(save_dir / "ref_patch_original.jpg"), ref_patch)
        cv2.imwrite(str(save_dir / "ref_match_input.jpg"), ref_match)
        if pnp_input_count > 0:
            draw_pnp_inlier_matches_sampled(
                uav_pts_match[valid],
                ref_pts_match[valid],
                inlier_indices,
                uav_match,
                ref_match,
                str(save_dir / f"pnp_inliers_{int(inliers)}.jpg"),
                input_pnp_count=pnp_input_count,
                localization_error_m=pred_error,
                sample_ratio=opt.draw_pnp_inlier_ratio,
                seed=int(opt.draw_pnp_inlier_seed),
            )

    if opt.save_match_arrays:
        np.savez_compressed(
            str(save_dir / "matches_and_pnp_inputs.npz"),
            uav_pts_match=uav_pts_match,
            ref_pts_match=ref_pts_match,
            uav_pts_original=uav_pts_original,
            ref_pts_patch=ref_pts_patch,
            local_x=local_x,
            local_y=local_y,
            dsm_z=dsm_z,
            valid=valid,
            inlier_indices=np.asarray(inlier_indices, dtype=np.int32),
        )

    result = {
        "scene_name": scene_name,
        "sample_id": sample_id,
        "npz_path": to_python_str(sample.get("npz_path", "")),
        "reference_mode": opt.reference_mode,
        "matching_method": matching_method,
        "pose_priori": opt.pose_priori,
        "device": opt.device,
        "match_keypoints": int(opt.match_keypoints),
        "resize_ratio": float(opt.resize_ratio),
        "uav_match_scale": float(uav_scale),
        "fine_scale": float(fine_scale),
        "ref_resize_factor": float(ref_resize),
        "num_matches": int(num_matches),
        "num_pnp_input": int(pnp_input_count),
        "num_inliers": int(inliers),
        "inlier_ratio": float(inliers / max(pnp_input_count, 1)) if pnp_input_count > 0 else None,
        "pnp_success": bool(pnp_success),
        "pred_error_m": safe_float(pred_error),
        "pred_error_x_m": safe_float(err_x),
        "pred_error_y_m": safe_float(err_y),
        "pred_x": safe_float(None if pred_xyz is None else pred_xyz.get("X")),
        "pred_y": safe_float(None if pred_xyz is None else pred_xyz.get("Y")),
        "pred_z": safe_float(None if pred_xyz is None else pred_xyz.get("Z")),
        "true_x": float(true_pos["x"]),
        "true_y": float(true_pos["y"]),
        "true_z": float(true_pos["z"]),
        "roll_deg": float(true_pos["roll"]),
        "pitch_deg": float(true_pos["pitch"]),
        "yaw_deg": float(true_pos["yaw"]),
        "match_time_s": float(match_time),
        "pnp_time_s": float(pnp_time),
        "total_time_s": float(match_time + pnp_time),
        "uav_input_width": int(uav_bgr.shape[1]),
        "uav_input_height": int(uav_bgr.shape[0]),
        "uav_match_width": int(uav_match.shape[1]),
        "uav_match_height": int(uav_match.shape[0]),
        "ref_map_width": int(ref_map.shape[1]),
        "ref_map_height": int(ref_map.shape[0]),
        "ref_patch_width": int(ref_patch.shape[1]),
        "ref_patch_height": int(ref_patch.shape[0]),
        "ref_match_width": int(ref_match.shape[1]),
        "ref_match_height": int(ref_match.shape[0]),
        "map_resolution_x_m": float(np.asarray(map_resolution).reshape(-1)[0]),
        "map_resolution_y_m": float(np.asarray(map_resolution).reshape(-1)[1]),
        "dsm_resolution_x_m": float(np.asarray(dsm_resolution).reshape(-1)[0]),
        "dsm_resolution_y_m": float(np.asarray(dsm_resolution).reshape(-1)[1]),
        "crop_center_x_dataset_local": float(crop_info["center_x_dataset_local"]),
        "crop_center_y_dataset_local": float(crop_info["center_y_dataset_local"]),
        "crop_center_col": float(crop_info["center_col"]),
        "crop_center_row": float(crop_info["center_row"]),
        "target_patch_size_m": float(crop_info["target_patch_size_m"]),
        "patch_width_m": float(crop_info["patch_width_m"]),
        "patch_height_m": float(crop_info["patch_height_m"]),
        "pre_crop": crop_info["pre_crop"],
        "final_crop": crop_info["final_crop"],
        "result_dir": str(save_dir),
    }

    save_json(result_json, result)
    save_pickle(
        save_dir / f"VG_match_loc_{sample_id}.pkl",
        result=result,
        pred_xyz=pred_xyz,
        cam_angle=None if cam_angle is None else np.asarray(cam_angle).tolist(),
        inlier_indices=np.asarray(inlier_indices, dtype=np.int32).tolist(),
    )
    return result


def threshold_ratio(rows, threshold):
    ok = 0
    total = len(rows)
    for row in rows:
        err = row.get("pred_error_m")
        if row.get("pnp_success") and err is not None and float(err) <= float(threshold):
            ok += 1
    return float(ok / total) if total > 0 else 0.0


def summarize(rows, failed_rows, opt, extra=None):
    errors = [
        float(r["pred_error_m"])
        for r in rows
        if r.get("pnp_success") and r.get("pred_error_m") is not None
    ]
    match_counts = [int(r["num_matches"]) for r in rows if "num_matches" in r]
    inlier_counts = [int(r["num_inliers"]) for r in rows if "num_inliers" in r]

    summary = {
        "dataset_root": str(opt.dataset_root),
        "scenes": opt.scenes,
        "reference_mode": opt.reference_mode,
        "pose_priori": opt.pose_priori,
        "num_results": int(len(rows)),
        "num_failed_pipeline": int(len(failed_rows)),
        "pnp_success_count": int(len(errors)),
        "pnp_success_ratio": float(len(errors) / max(len(rows), 1)) if rows else 0.0,
        "mean_error_m": safe_float(np.mean(errors)) if errors else None,
        "median_error_m": safe_float(np.median(errors)) if errors else None,
        "mean_matches": safe_float(np.mean(match_counts)) if match_counts else None,
        "median_matches": safe_float(np.median(match_counts)) if match_counts else None,
        "mean_inliers": safe_float(np.mean(inlier_counts)) if inlier_counts else None,
        "median_inliers": safe_float(np.median(inlier_counts)) if inlier_counts else None,
        "T1": threshold_ratio(rows, 1.0),
        "T3": threshold_ratio(rows, 3.0),
        "T5": threshold_ratio(rows, 5.0),
        "T10": threshold_ratio(rows, 10.0),
        "T20": threshold_ratio(rows, 20.0),
    }
    if extra:
        summary.update(extra)
    return summary


def flush_outputs(out_dir, rows, failed_rows, summary):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    save_json(out_dir / "all_results.json", rows)
    save_csv(rows, out_dir / "all_results.csv")
    save_json(out_dir / "failed.json", failed_rows)
    save_json(out_dir / "summary.json", summary)


def main():
    opt = get_parse()
    with open(opt.yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["DEVICE"] = opt.device

    matching_methods = opt.matching_methods or list(config.get("MATCHING_METHODS", []))
    if not matching_methods:
        raise ValueError("No matching method configured. Set MATCHING_METHODS in yaml or use --matching_methods.")

    dataset = AnyVisLocNPZDataset(
        root=opt.dataset_root,
        scenes=opt.scenes,
        load_reference=False,
        cache_reference=True,
        recursive=True,
    )
    indices = list(range(len(dataset)))
    if opt.limit and opt.limit > 0:
        indices = indices[: int(opt.limit)]

    scene_count = "all" if opt.scenes is None else f"{len(opt.scenes)}scenes"
    run_tag = f"{datetime.now().strftime('%Y%m%d-%H%M')}_{scene_count}"
    run_root = Path(opt.save_dir) / run_tag / opt.reference_mode
    ensure_dir(run_root)

    all_rows = []
    all_failed = []
    pbar = tqdm(total=len(indices) * len(matching_methods), desc="Matching & Localization", unit="img")

    for matching_method in matching_methods:
        method_dict = {"matching_method": matching_method}
        method_dict = matching_init(method_dict, opt, config)

        combo_name = f"{opt.reference_mode}-{matching_method}-{opt.pose_priori}"
        combo_dir = run_root / combo_name
        opt._combo_dir = str(combo_dir)
        ensure_dir(combo_dir)

        combo_rows = []
        combo_failed = []
        processed = 0

        for idx in indices:
            sample = dataset[idx]
            scene_name = to_python_str(sample.get("scene_name", ""))
            sample_id = to_python_str(sample.get("sample_id", ""))
            try:
                row = process_one_sample(sample, method_dict, matching_method, opt)
                combo_rows.append(row)
                all_rows.append(row)
                msg = (
                    f"{scene_name}/{sample_id} {matching_method}: "
                    f"matches={row['num_matches']} pnp={row['num_pnp_input']} "
                    f"inliers={row['num_inliers']} "
                    f"err={row['pred_error_m'] if row['pred_error_m'] is not None else 'failed'}"
                )
                tqdm.write(msg)
            except Exception as exc:
                err = {
                    "scene_name": scene_name,
                    "sample_id": sample_id,
                    "npz_path": to_python_str(sample.get("npz_path", "")),
                    "reference_mode": opt.reference_mode,
                    "matching_method": matching_method,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                combo_failed.append(err)
                all_failed.append(err)
                tqdm.write(f"[Error] {scene_name}/{sample_id} {matching_method}: {exc}")
                if opt.debug_raise:
                    raise
            finally:
                processed += 1
                pbar.update(1)

            if opt.save_every > 0 and processed % int(opt.save_every) == 0:
                combo_summary = summarize(
                    combo_rows,
                    combo_failed,
                    opt,
                    extra={"matching_method": matching_method, "combo_name": combo_name},
                )
                flush_outputs(combo_dir, combo_rows, combo_failed, combo_summary)

        combo_summary = summarize(
            combo_rows,
            combo_failed,
            opt,
            extra={"matching_method": matching_method, "combo_name": combo_name},
        )
        flush_outputs(combo_dir, combo_rows, combo_failed, combo_summary)

    pbar.close()

    global_summary = summarize(
        all_rows,
        all_failed,
        opt,
        extra={"matching_methods": matching_methods},
    )
    flush_outputs(run_root, all_rows, all_failed, global_summary)
    print(json.dumps(global_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
