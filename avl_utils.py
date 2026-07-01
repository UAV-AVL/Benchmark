#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities that adapt the original UAV-Visual-Localization baseline to the
AnyVisLoc open NPZ format.

This file keeps the minimal retrieval/matching initialization needed by the
demo and replaces the data/coordinate layer:
  - no metadata JSON lookup
  - no Region_params/* UTM yaml dependency
  - PnP output is dataset-local XYZ, not lon/lat
"""

from pathlib import Path
import math
import time
import json
import csv
import sys
import hashlib
import re
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image

from Matching_Models.RoMa.demo.Roma_match import Roma_Init, Roma_match
from Retrieval_Models.feature_extract import extract_features
from Retrieval_Models.multi_model_loader import get_Model


def save_data(filename, **kwargs):
    """Save keyword payload to a pickle file."""
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(kwargs, file)


def dumpRotateImage(img, degree):
    """Rotate an image around its center and return the rotated image plus affine matrix."""
    radians = degree / 180 * np.pi
    height, width = img.shape[:2]
    height_new = int(width * abs(np.sin(radians)) + height * abs(np.cos(radians)))
    width_new = int(height * abs(np.sin(radians)) + width * abs(np.cos(radians)))
    mat_rotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    mat_rotation[0, 2] += (width_new - width) // 2
    mat_rotation[1, 2] += (height_new - height) // 2
    img_rotation = cv2.warpAffine(
        img,
        mat_rotation,
        (width_new, height_new),
        borderValue=(0, 0, 0),
    )
    return img_rotation, mat_rotation


def rotvector2rot(rotvector):
    """Convert OpenCV Rodrigues rotation vector to a 3x3 rotation matrix."""
    return cv2.Rodrigues(rotvector)[0]


def rot_to_euler(dcm, lim=None):
    """Convert a direction cosine matrix to the baseline camera Euler convention."""
    r11 = -dcm[1][0]
    r12 = dcm[1][1]
    r21 = dcm[1][2]
    r31 = -dcm[0][2]
    r32 = dcm[2][2]
    r11a = dcm[0][1]
    r12a = dcm[0][0]

    r1 = np.arctan2(r11, r12)
    r21 = np.clip(r21, -1, 1)
    r2 = np.arcsin(r21)
    r3 = np.arctan2(r31, r32)

    if lim == "zeror3":
        for i in np.where(np.abs(r21) == 1.0)[0]:
            r1[i] = np.arctan2(r11a[i], r12a[i])
            r3[i] = 0

    return np.array([-r1 - np.pi, -r2, r3 + np.pi]) * 180 / np.pi


def _baseline_matching_init(method_dict):
    """Initialize the RoMa matcher used by the baseline path."""
    if method_dict["matching_method"] != "Roma":
        raise ValueError(f"Unsupported baseline matching method: {method_dict['matching_method']!r}")
    method_dict["matching_model"] = Roma_Init()
    return method_dict

def _baseline_retrieval_init(method_dict, config):
    """Initialize retrieval model and retrieval parameters."""
    if method_dict["retrieval_method"] not in config["RETRIEVAL_METHODS"]:
        raise ValueError(
            f"Invalid retrieval method {method_dict['retrieval_method']!r}; "
            f"expected one of {config['RETRIEVAL_METHODS']!r}"
        )

    method_dict["retrieval_model"], method_dict["img_transform"] = get_Model(
        method_dict["retrieval_method"]
    )
    method_dict["retrieval_model"].to(config["DEVICE"])
    method_dict["retrieval_cover"] = config["RETRIEVAL_COVER"]
    method_dict["retrieval_topn"] = config["RETRIEVAL_TOPN"]
    method_dict["retrieval_img_name"] = config["RETRIEVAL_IMG"]
    method_dict["retrieval_feat_norm"] = config["RETRIEVAL_FEATURE_NORM"]
    return method_dict


def compute_block_mid_wo_black(image, block_size, step_size):
    """Return valid reference block centers, filtering blocks with too much black area."""
    x_size, y_size, _ = image.shape
    small_img = cv2.resize(
        image,
        (int(y_size / 10), int(x_size / 10)),
        interpolation=cv2.INTER_NEAREST,
    )

    num_blocks_x = len(range(0, x_size, step_size[0]))
    num_blocks_y = len(range(0, y_size, step_size[1]))
    mids = []
    total_pixels = block_size[0] * block_size[1] / 100

    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            start_x = i * step_size[0]
            start_y = j * step_size[1]
            start_x = min(start_x, x_size - block_size[0] - 1)
            start_y = min(start_y, y_size - block_size[1] - 1)
            end_x = start_x + block_size[0]
            end_y = start_y + block_size[1]

            block = small_img[
                int(start_x / 10): int(end_x / 10),
                int(start_y / 10): int(end_y / 10),
            ]
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2

            count_above_threshold = np.sum(block[:, :, 0] > 0)
            if count_above_threshold >= total_pixels / 5 * 2:
                mids.append([mid_x, mid_y])

    return np.array(mids)



def _call_with_clean_argv(func, *args, **kwargs):
    """Call original project initializers without leaking this script's CLI args.

    Some retrieval backbones construct their own argparse parser inside
    get_Model(...). If they see Baseline_anyvisloc_npz.py arguments such as
    --dataset_root or --scenes, they abort with "unrecognized arguments".
    This wrapper temporarily exposes only argv[0].
    """
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        return func(*args, **kwargs)
    finally:
        sys.argv = old_argv


def retrieval_init(method_dict, config):
    return _call_with_clean_argv(_baseline_retrieval_init, method_dict, config)


def _normalize_selectable_matching_method(name):
    """Normalize matching method names used by config.yaml.

    Supported names:
      Roma
      SP_LG / SP+LG / sp_lg
      SP_LG_GIM
      SP_LG_MINIMA
      ALIKED_LG / ALIKED+LG / aliked_lg
      DISK_LG / DISK+LG / disk_lg
    """
    raw = str(name or "").strip()
    key = raw.lower().replace("-", "_").replace("+", "_")
    key = re.sub(r"[^a-z0-9_]+", "_", key).strip("_")

    aliases = {
        "roma": ("roma", "roma"),
        "sp_lg": ("sp", "lg"),
        "sp_lg_ori": ("sp", "lg"),
        "sp_lg_paper": ("sp", "lg"),
        "superpoint_lg": ("sp", "lg"),
        "sp_lightglue": ("sp", "lg"),
        "superpoint_lightglue": ("sp", "lg"),
        "sp_lg_gim": ("sp", "lg_gim"),
        "superpoint_lg_gim": ("sp", "lg_gim"),
        "sp_gim_lg": ("sp", "lg_gim"),
        "gim_sp_lg": ("sp", "lg_gim"),
        "sp_lg_minima": ("sp", "lg_minima"),
        "superpoint_lg_minima": ("sp", "lg_minima"),
        "sp_minima_lg": ("sp", "lg_minima"),
        "minima_sp_lg": ("sp", "lg_minima"),
        "aliked_lg": ("aliked", "lg"),
        "aliked_lightglue": ("aliked", "lg"),
        "disk_lg": ("disk", "lg"),
        "disk_lightglue": ("disk", "lg"),
    }

    if key not in aliases:
        supported = [
            "Roma",
            "SP_LG",
            "SP_LG_GIM",
            "SP_LG_MINIMA",
            "ALIKED_LG",
            "DISK_LG",
        ]
        raise ValueError(f"Unsupported matching method: {raw!r}. Supported: {supported}")
    return aliases[key]


def _selectable_code_dir(opt=None):
    default_dir = "/"
    if opt is None:
        return default_dir
    return str(getattr(opt, "selectable_code_dir", default_dir) or default_dir)


def _load_selectable_module(opt=None):
    """Lazy-load selectable_sparse_matcher and its dependencies."""
    import importlib

    from pathlib import Path

    code_dir = Path(_selectable_code_dir(opt)).resolve()
    package_parent = str(code_dir.parent)

    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    module_name = str(
        getattr(opt, "selectable_module", "selectable_sparse_matcher")
        if opt is not None else "selectable_sparse_matcher"
    )

    if "." not in module_name:
        module_name = f"{code_dir.name}.{module_name}"
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"Failed to import {module_name!r} from {code_dir!r}. "
            "Check --selectable_code_dir and make sure selectable_sparse_matcher.py, "
            "aliked_feature.py, and disk_feature.py are on that path."
        ) from exc


def _make_selectable_args(opt=None, config=None):
    """Small args namespace expected by selectable matcher helper functions."""
    import argparse

    if config is None:
        config = {}
    device = None
    if opt is not None:
        device = getattr(opt, "device", None)
    if device is None:
        device = config.get("DEVICE", "cuda")

    args = argparse.Namespace()
    args.device = device
    args.match_keypoints = int(getattr(opt, "match_keypoints", 1024) if opt is not None else 1024)

    args.gim_lg_ckpt = getattr(opt, "gim_lg_ckpt", None) if opt is not None else None
    args.minima_lg_ckpt = getattr(opt, "minima_lg_ckpt", None) if opt is not None else None
    return args


def matching_init(method_dict, opt=None, config=None):
    """Initialize Roma or the selectable local feature matchers.

    For old config files using MATCHING_METHODS: [Roma], this preserves the
    original behavior. For selectable methods, this initializes only the
    extractor/matcher pair and stores it in method_dict["matching_model"].
    """
    feature_type, matcher_type = _normalize_selectable_matching_method(method_dict.get("matching_method", "Roma"))

    if feature_type == "roma":
        return _call_with_clean_argv(_baseline_matching_init, method_dict)

    selectable = _load_selectable_module(opt)
    args = _make_selectable_args(opt, config)
    device = args.device
    max_k = int(args.match_keypoints)

    if feature_type == "sp" and matcher_type == "lg":
        extractor, matcher = selectable.SP_LG_ORI_Init(
            device=device,
            max_num_keypoints=max_k ,
        )
    elif feature_type == "sp" and matcher_type == "lg_gim":
        extractor, matcher = selectable.SP_LG_GIM_Init(
            device=device,
            checkpoint_path=args.gim_lg_ckpt,
            max_num_keypoints=max_k ,
        )
    elif feature_type == "sp" and matcher_type == "lg_minima":
        extractor, matcher = selectable.SP_LG_MINIMA_Init(
            device=device,
            checkpoint_path=args.minima_lg_ckpt,
            max_num_keypoints=max_k ,
        )
    elif feature_type == "aliked" and matcher_type == "lg":
        extractor, matcher = selectable.ALIKED_LG_Init(
            device=device,
            max_num_keypoints=max_k ,
        )
    elif feature_type == "disk" and matcher_type == "lg":
        extractor, matcher = selectable.DISK_LG_Init(
            device=device,
            max_num_keypoints=max_k ,
        )
    else:
        raise ValueError(f"Unsupported selectable matcher pair: {feature_type}+{matcher_type}")

    method_dict["matching_model"] = {
        "kind": "selectable",
        "feature_type": feature_type,
        "matcher_type": matcher_type,
        "extractor": extractor,
        "matcher": matcher,
        "module": selectable,
        "args": args,
    }
    if feature_type == "sp" and matcher_type == "lg":
        method_dict["matching_method"] = "SP_LG"
    elif feature_type == "sp" and matcher_type == "lg_gim":
        method_dict["matching_method"] = "SP_LG_GIM"
    elif feature_type == "sp" and matcher_type == "lg_minima":
        method_dict["matching_method"] = "SP_LG_MINIMA"
    else:
        method_dict["matching_method"] = f"{feature_type.upper()}_{matcher_type.upper()}"
    return method_dict


def _ensure_numpy_points(pts0, pts1):
    if torch.is_tensor(pts0):
        pts0 = pts0.detach().cpu().numpy()
    else:
        pts0 = np.asarray(pts0, dtype=np.float32)

    if torch.is_tensor(pts1):
        pts1 = pts1.detach().cpu().numpy()
    else:
        pts1 = np.asarray(pts1, dtype=np.float32)

    pts0 = pts0.reshape(-1, 2).astype(np.float32) if pts0.size else np.zeros((0, 2), dtype=np.float32)
    pts1 = pts1.reshape(-1, 2).astype(np.float32) if pts1.size else np.zeros((0, 2), dtype=np.float32)
    return pts0, pts1


def _selectable_extract_features(image_bgr, model):
    """Direct feature extraction through selectable_sparse_matcher.py."""
    selectable = model["module"]
    feature_type = model["feature_type"]
    matcher_type = model["matcher_type"]
    extractor = model["extractor"]
    args = model["args"]
    device = args.device

    if feature_type == "sp":
        if matcher_type == "lg":
            return selectable.SP_LG_ORI_extract(image_bgr, extractor, device)
        if matcher_type == "lg_gim":
            return selectable.SP_LG_GIM_extract(image_bgr, extractor, device)
        if matcher_type == "lg_minima":
            return selectable.SP_LG_MINIMA_extract(image_bgr, extractor, device)

    if feature_type == "aliked" and matcher_type == "lg":
        return selectable.ALIKED_LG_extract(image_bgr, extractor, device)

    if feature_type == "disk" and matcher_type == "lg":
        return selectable.DISK_LG_extract(image_bgr, extractor, device)

    raise ValueError(f"Unsupported feature extraction pair: {feature_type}+{matcher_type}")


def _selectable_match_features(feat_uav, feat_ref, model):
    """Direct sparse matching."""
    selectable = model["module"]
    feature_type = model["feature_type"]
    matcher_type = model["matcher_type"]
    extractor = model["extractor"]
    matcher = model["matcher"]
    args = model["args"]

    if feature_type == "sp" and matcher_type == "lg":
        pts0, pts1 = selectable.SP_LG_ORI_match(feat_uav, feat_ref, matcher)
        return _ensure_numpy_points(pts0, pts1)

    if feature_type == "sp" and matcher_type == "lg_gim":
        pts0, pts1 = selectable.SP_LG_GIM_match(feat_uav, feat_ref, matcher)
        return _ensure_numpy_points(pts0, pts1)

    if feature_type == "sp" and matcher_type == "lg_minima":
        pts0, pts1 = selectable.SP_LG_MINIMA_match(feat_uav, feat_ref, matcher)
        return _ensure_numpy_points(pts0, pts1)

    if feature_type == "aliked" and matcher_type == "lg":
        pts0, pts1 = selectable.ALIKED_LG_match(feat_uav, feat_ref, matcher)
        return _ensure_numpy_points(pts0, pts1)

    if feature_type == "disk" and matcher_type == "lg":
        pts0, pts1 = selectable.DISK_LG_match(feat_uav, feat_ref, matcher)
        return _ensure_numpy_points(pts0, pts1)

    raise ValueError(f"Unsupported matching pair: {feature_type}+{matcher_type}")


def run_pixel_match_anyvisloc(
    uav_img,
    ref_img,
    method_dict,
    save_path,
    ransac_name,
    need_ransac=False,
    show_matches=False,
):
    """Run the configured pixel-level matcher and return UAV/ref xy points.

    Roma keeps the original call path. Selectable matchers use the direct
    init/extract/match functions from selectable_sparse_matcher.py.
    """
    matching_method = method_dict.get("matching_method", "Roma")
    feature_type, matcher_type = _normalize_selectable_matching_method(matching_method)

    if feature_type == "roma":
        Roma_model = method_dict["matching_model"]
        return Roma_match(
            uav_img,
            ref_img,
            Roma_model,
            save_path,
            ransac_name,
            need_ransac=need_ransac,
            show_matches=show_matches,
        )

    model = method_dict["matching_model"]
    feat_uav = _selectable_extract_features(uav_img, model)
    feat_ref = _selectable_extract_features(ref_img, model)
    return _selectable_match_features(feat_uav, feat_ref, model)

_REF_JSON_CACHE = {}
_REF_ARRAY_CACHE = {}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_python_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)


def normalize_uint8(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.percentile(x[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def ensure_bgr_uint8(img):
    """Reference maps are stored/read in BGR order for OpenCV."""
    img = np.asarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(normalize_uint8(img), cv2.COLOR_GRAY2BGR)
    if img.ndim != 3:
        raise ValueError(f"map image must be [H,W,3] or [H,W], got {img.shape}")
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return normalize_uint8(img)


def dsm_to_2d(dsm):
    dsm = np.asarray(dsm)
    if dsm.ndim == 3:
        dsm = dsm[:, :, 0]
    if dsm.ndim != 2:
        raise ValueError(f"DSM must be [H,W], got {dsm.shape}")
    return dsm.astype(np.float32, copy=False)


def as_vec2(x, name, default=None):
    if x is None:
        if default is None:
            raise KeyError(f"missing {name}")
        x = default
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must have at least 2 values, got {arr}")
    return arr[:2].astype(np.float32)


def tensor_rgb_to_bgr_uint8(x):
    """AnyVisLocNPZDataset returns RGB tensor [3,H,W] in [0,1]."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 1)
        x = (x * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def crop_center_from_bgr(img_bgr, width, height):
    """Return center crop as RGB PIL image for retrieval transforms."""
    h0, w0 = img_bgr.shape[:2]
    width = int(min(width, w0))
    height = int(min(height, h0))
    cx, cy = w0 // 2, h0 // 2
    left = max(0, int(cx - width // 2))
    top = max(0, int(cy - height // 2))
    right = min(w0, int(cx + width // 2))
    bottom = min(h0, int(cy + height // 2))
    crop_bgr = img_bgr[top:bottom, left:right]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def _resolve_ref_json_path(sample, ref):
    json_path = None
    if isinstance(ref, dict):
        json_path = ref.get("ref_json_path", None) or ref.get("ref_path", None)

    sample_npz = sample.get("npz_path", None)
    if sample_npz is not None and json_path is None:
        p = Path(to_python_str(sample_npz))
        scene_id = int(sample["scene_id"])
        json_path = p.parent / f"L{scene_id:02d}_reference.json"

    return None if json_path is None else Path(to_python_str(json_path))


def _load_reference_json(sample, ref):
    json_path = _resolve_ref_json_path(sample, ref)
    if json_path is None:
        return None, ""
    if not json_path.exists():
        return None, str(json_path)

    key = str(json_path)
    if key not in _REF_JSON_CACHE:
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["__base_dir__"] = str(json_path.parent)
        meta["__ref_json_path__"] = str(json_path)
        _REF_JSON_CACHE[key] = meta
    return _REF_JSON_CACHE[key], str(json_path)


def _json_path_to_abs(meta, value):
    if value is None:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return Path(meta.get("__base_dir__", ".")) / p


def _load_ref_array(path, kind):
    path = Path(path)
    key = (str(path), kind)
    if key in _REF_ARRAY_CACHE:
        return _REF_ARRAY_CACHE[key]
    if kind == "map":
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Cannot read reference map PNG: {path}")
    elif kind == "dsm":
        # mmap keeps large DSMs cheap to open. If later code needs a real ndarray,
        # NumPy will load pages on demand.
        arr = np.load(str(path), mmap_mode="r")
    elif kind == "mask":
        arr = np.load(str(path), mmap_mode="r")
    else:
        raise ValueError(f"Unknown reference array kind: {kind}")
    _REF_ARRAY_CACHE[key] = arr
    return arr

def normalize_reference_mode(mode="aerial"):
    """Normalize reference mode. The CLI only exposes aerial/satellite."""
    m = str(mode or "aerial").lower()
    if m in ("aerial", "ortho", "high"):
        return "aerial"
    if m in ("satellite", "sat", "low"):
        return "satellite"
    raise ValueError(f"Unsupported reference_mode: {mode!r}. Use 'aerial' or 'satellite'.")


def _reference_view_from_json(sample, ref, reference_mode):
    meta, ref_path = _load_reference_json(sample, ref)
    if meta is None:
        return None

    modes = meta.get("modes", {})
    if reference_mode not in modes:
        raise KeyError(f"Reference JSON {ref_path} has no mode {reference_mode!r}. Available={list(modes.keys())}")
    mode = modes[reference_mode]

    map_path = _json_path_to_abs(meta, mode.get("map_path"))
    dsm_path = _json_path_to_abs(meta, mode.get("dsm_path"))
    if map_path is None or dsm_path is None:
        raise KeyError(f"Reference JSON {ref_path} mode {reference_mode!r} missing map_path/dsm_path")

    map_img = _load_ref_array(map_path, "map")
    dsm = _load_ref_array(dsm_path, "dsm")

    map_origin = as_vec2(mode.get("map_origin_local", [0.0, 0.0]), f"{reference_mode}_map_origin_local", [0.0, 0.0])
    dsm_origin_rel = as_vec2(mode.get("dsm_origin_local", [0.0, 0.0]), f"{reference_mode}_dsm_origin_local", [0.0, 0.0])
    dsm_origin_dataset = (map_origin + dsm_origin_rel).astype(np.float32)

    valid_mask = None

    return {
        "reference_mode": reference_mode,
        "map": ensure_bgr_uint8(map_img),
        "dsm": dsm_to_2d(dsm),
        "dsm_valid_mask": valid_mask,
        "map_resolution": as_vec2(mode.get("map_resolution"), f"{reference_mode}_map_resolution"),
        "dsm_resolution": as_vec2(mode.get("dsm_resolution"), f"{reference_mode}_dsm_resolution"),
        "map_origin_local": map_origin,
        "dsm_origin_local": dsm_origin_dataset,
        "dsm_origin_relative_to_map": dsm_origin_rel,
        "map_field": mode.get("map_field", f"{reference_mode}_map"),
        "dsm_field": mode.get("dsm_field", f"{reference_mode}_dsm"),
        "ref_path": ref_path,
        "ref_json_path": ref_path,
    }


def build_anyvisloc_reference_view(sample, reference_mode="aerial"):
    reference_mode = normalize_reference_mode(reference_mode)
    ref = sample.get("reference", {})
    json_view = _reference_view_from_json(sample, ref, reference_mode)
    if json_view is None:
        json_path = _resolve_ref_json_path(sample, ref)
        raise FileNotFoundError(
            f"Reference JSON not found. Expected: {json_path}. "
            "This short-version runner requires Lxx_reference.json plus PNG/NPY reference files."
        )
    return json_view

def anyvisloc_sample_to_truepos(sample):
    """Build a small metadata dict compatible with the baseline naming style."""
    image_size = to_numpy(sample.get("image_size", np.array([0, 0]))).astype(np.int32).reshape(-1)
    if image_size.size >= 2 and image_size[0] > 0:
        h, w = int(image_size[0]), int(image_size[1])
    else:
        img = sample["image"]
        h, w = int(img.shape[-2]), int(img.shape[-1])
    xyz = to_numpy(sample["xyz"]).astype(np.float32).reshape(3)
    euler = to_numpy(sample["euler_deg"]).astype(np.float32).reshape(3)
    K = to_numpy(sample["K"]).astype(np.float32)
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
        "rel_alt": float(xyz[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "yaw": float(euler[2]),
        "width": int(w),
        "height": int(h),
        "K": K,
        "sample_id": to_python_str(sample.get("sample_id", "")),
        "scene_name": to_python_str(sample.get("scene_name", "")),
        "npz_path": to_python_str(sample.get("npz_path", "")),
    }


def _even_ceil(v, min_value=2):
    out = int(math.ceil(float(v)))
    out = max(int(min_value), out)
    if out % 2:
        out += 1
    return out


def resolution_size_anyvisloc(data, opt):
    """Estimate UAV ground sampling distance from K, altitude, and pitch.

    This follows the same geometry used by the previous working
    AnyVisLoc_open_npz_new.py implementation:

        denom = abs(sin(90 deg + pitch))
        ground_dist = z / denom
        drone_resolution = ground_dist / fx

    Patch scaling is intentionally NOT folded into this return value, because
    ``drone_resolution`` is also used as the RoMa fine-scale factor. Patch
    scaling is applied later when computing the retrieval patch size.
    """
    w = int(data["width"])
    h = int(data["height"])
    K = np.asarray(data["K"], dtype=np.float32)
    fx = max(float(K[0, 0]) if K.size >= 9 else 1.0, 1e-6)
    z = max(float(data["rel_alt"]), 1e-6)

    if getattr(opt, "pose_priori", "yp") == "unknown":
        denom = 1.0
    else:
        pitch = float(data.get("pitch", 0.0))
        denom = abs(float(np.sin(np.pi * (90.0 + pitch) / 180.0)))
        denom = max(denom, 1e-6)

    ground_dist = z / denom
    drone_resolution = ground_dist / fx
    min_size = min(w, h)
    return float(drone_resolution), np.array([min_size, min_size], dtype=np.int32)


def _apply_affine_xy(x, y, mat):
    """Apply an OpenCV-style affine/homography matrix to one x/y pixel point."""
    M = np.asarray(mat, dtype=np.float32)
    if M.size == 6 and M.shape != (2, 3):
        M = M.reshape(2, 3)
    p = np.array([float(x), float(y), 1.0], dtype=np.float32)
    if M.shape == (2, 3):
        out = M @ p
        return float(out[0]), float(out[1])
    if M.shape == (3, 3):
        out = M @ p
        denom = float(out[2]) if abs(float(out[2])) > 1e-12 else 1.0
        return float(out[0] / denom), float(out[1] / denom)
    raise ValueError(f"matRotation must be 2x3 or 3x3, got {M.shape}")


def view_center_anyvisloc(data, map_resolution, map_origin_local, matRotation, use_shift=True):
    """Estimate UAV view center in the current reference-map pixel frame.

    This intentionally matches the previous working
    AnyVisLoc_open_npz_new.py::compute_view_center_xy() formula:

        dx = z * tan(pitch) * sin(yaw)
        dy = -z * tan(pitch) * cos(yaw)

    The shifted dataset-local point is then converted to selected-map col/row
    pixels and, if the map has already been yaw-rotated globally, transformed
    by the same affine matrix.
    """
    x = float(data["x"])
    y = float(data["y"])
    z = float(data["rel_alt"])

    if use_shift:
        pitch = float(data.get("pitch", 0.0))
        yaw = float(data.get("yaw", 0.0))
        tan_v = np.tan(pitch / 180.0 * np.pi)
        if np.isfinite(tan_v) and abs(float(tan_v)) >= 1e-6:
            yaw_rad = yaw / 180.0 * np.pi
            dx = z * float(tan_v) * np.sin(yaw_rad)
            dy = -z * float(tan_v) * np.cos(yaw_rad)
            if np.isfinite(dx) and np.isfinite(dy):
                x += float(dx)
                y += float(dy)

    res = as_vec2(map_resolution, "map_resolution")
    origin = as_vec2(map_origin_local, "map_origin_local", [0.0, 0.0])
    col = (x - float(origin[0])) / max(float(res[0]), 1e-12)
    row = (y - float(origin[1])) / max(float(res[1]), 1e-12)

    return _apply_affine_xy(col, row, matRotation)


def _sanitize_filename_token(text):
    text = to_python_str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "unknown"


def _circled_number(n):
    """Return circled number labels for visualization titles."""
    circled = {
        1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
        6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
        11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮",
        16: "⑯", 17: "⑰", 18: "⑱", 19: "⑲", 20: "⑳",
    }
    return circled.get(int(n), str(n))


def _draw_red_cross_if_inside(img_bgr, center_col, center_row, col0, row0):
    """Draw a red cross at the true view-center if it falls inside this patch.

    ``center_col``/``center_row`` and ``col0``/``row0`` must be expressed in
    the same reference-map pixel frame. In yaw-prior mode this is the rotated
    map frame; otherwise it is the original selected reference-map frame.
    """
    img_bgr = np.asarray(img_bgr)
    if img_bgr.size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    x_f = float(center_col) - float(col0)
    y_f = float(center_row) - float(row0)
    if not (np.isfinite(x_f) and np.isfinite(y_f)):
        return img_bgr

    x = int(round(x_f))
    y = int(round(y_f))
    if not (0 <= x < w and 0 <= y < h):
        return img_bgr

    out = img_bgr.copy()
    marker_size = max(12, int(round(min(w, h) * 0.070)))
    thickness = max(2, int(round(min(w, h) * 0.008)))
    cv2.drawMarker(
        out,
        (x, y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    return out


def _draw_cross_at_image_center(img, color, marker_ratio=0.070, thickness_ratio=0.008):
    """Draw a crosshair at the geometric center of an image.

    The color tuple is interpreted in the image's native channel order. For
    BGR images use red=(0,0,255), green=(0,255,0). For RGB images use
    red=(255,0,0), green=(0,255,0).
    """
    img = np.asarray(img)
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    x = int(round((w - 1) * 0.5))
    y = int(round((h - 1) * 0.5))
    marker_size = max(12, int(round(min(w, h) * float(marker_ratio))))
    thickness = max(2, int(round(min(w, h) * float(thickness_ratio))))
    cv2.drawMarker(
        out,
        (x, y),
        tuple(int(c) for c in color),
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    return out


def _make_ref_feature_cache_meta(
    *,
    opt,
    method_dict,
    config,
    ref_image,
    block_size,
    step_size,
    mids,
):
    """Build strict metadata so cached gallery features are reused only safely."""
    mids_arr = np.ascontiguousarray(np.asarray(mids, dtype=np.float32).reshape(-1, 2))
    mids_hash = hashlib.sha1(mids_arr.tobytes()).hexdigest()[:16]
    return {
        "cache_version": "anyvisloc_npz_v5_ref_features",
        "scene_name": _sanitize_filename_token(getattr(opt, "_current_scene_name", "")),
        "reference_mode": _sanitize_filename_token(getattr(opt, "reference_mode", "")),
        "retrieval_method": _sanitize_filename_token(method_dict.get("retrieval_method", "")),
        "retrieval_size": int(config.get("RETRIEVAL_SIZE", 384)),
        "retrieval_cover": int(method_dict.get("retrieval_cover", -1)),
        "ref_shape": [int(x) for x in np.asarray(ref_image).shape[:2]],
        "block_size": [int(block_size[0]), int(block_size[1])],
        "step_size": [int(step_size[0]), int(step_size[1])],
        "num_blocks": int(mids_arr.shape[0]),
        "mids_hash": mids_hash,
    }


def _default_ref_feature_cache_root(save_path, opt):
    user_cache_dir = getattr(opt, "ref_feature_cache_dir", None)
    if user_cache_dir:
        return Path(user_cache_dir)
    save_path = Path(save_path)
    # save_path is normally combo_dir / scene_name / sample_id.
    # Put cache under combo_dir so different UAV samples in the same combo can share it.
    try:
        return save_path.parents[1] / "_ref_feature_cache"
    except Exception:
        return save_path.parent / "_ref_feature_cache"


def _make_ref_feature_cache_path(save_path, opt, meta):
    cache_root = _default_ref_feature_cache_root(save_path, opt)
    scene = _sanitize_filename_token(meta.get("scene_name", "scene"))
    reference_mode = _sanitize_filename_token(meta.get("reference_mode", "ref"))
    retrieval_method = _sanitize_filename_token(meta.get("retrieval_method", "retrieval"))
    num_blocks = int(meta.get("num_blocks", 0))
    block_h, block_w = meta.get("block_size", [0, 0])
    suffix = _sanitize_filename_token(meta.get("mids_hash", "hash"))
    filename = f"{scene}__{reference_mode}__{retrieval_method}__blocks{num_blocks}__{block_h}x{block_w}__{suffix}.pt"
    return cache_root / filename


def _load_cached_ref_features(cache_path, expected_meta):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(str(cache_path), map_location="cpu")
        if not isinstance(payload, dict):
            return None
        if payload.get("meta", None) != expected_meta:
            return None
        gf = payload.get("gf", None)
        if gf is None:
            return None
        if not torch.is_tensor(gf):
            gf = torch.as_tensor(gf)
        return gf.detach().cpu()
    except Exception as exc:
        print(f"[v6 patched cache warning] failed to load reference features: {cache_path} ({exc})")
        return None


def _save_cached_ref_features(cache_path, meta, gf):
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)
    try:
        payload = {
            "meta": meta,
            "gf": gf.detach().cpu() if torch.is_tensor(gf) else torch.as_tensor(gf),
            "saved_at_unix": float(time.time()),
        }
        torch.save(payload, str(cache_path))
        meta_path = cache_path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[v6 patched cache save] saved reference features: {cache_path}")
    except Exception as exc:
        print(f"[v6 patched cache warning] failed to save reference features: {cache_path} ({exc})")


def _extract_query_feature_with_dummy_gallery(
    retrieval_size,
    config,
    method_dict,
    MID,
    block_size,
    ref_image,
    UAV_img,
    img_transform,
    retrieval_model,
):
    """Reuse the original extractor but pass only one dummy gallery block to get qf."""
    dummy_mid = np.asarray(MID, dtype=np.float32).reshape(-1, 2)[:1]
    _, qf = extract_features(
        retrieval_size,
        config,
        method_dict,
        dummy_mid,
        block_size,
        ref_image,
        UAV_img,
        img_transform,
        retrieval_model,
    )
    return qf


def retrieval_all_anyvisloc(
    ref_image,
    uav_image_bgr,
    uav_data,
    map_resolution,
    map_origin_local,
    matRotation,
    save_path,
    opt,
    config,
    method_dict,
):
    """Image-level retrieval for AnyVisLoc NPZ samples.

    v5 changes:
      - when opt.pose_priori == "p", reference/gallery features are cached by
        scene, reference mode, retrieval method, block size, step size, and block layout;
      - retrieval visualization separates UAV and reference results with a dashed line;
      - retrieval result titles use circled numbers;
      - each retrieved reference crop draws the true view-center as a red cross
        when that point lies inside the crop.
    """
    ensure_dir(save_path)
    retrieval_size = int(config.get("RETRIEVAL_SIZE", 384))
    map_res = as_vec2(map_resolution, "map_resolution")

    # Default behavior: use pitch/yaw/altitude to estimate the UAV view center
    # unless the whole pose prior is explicitly disabled.
    center_col, center_row = view_center_anyvisloc(
        uav_data,
        map_res,
        map_origin_local,
        matRotation,
        use_shift=(getattr(opt, "pose_priori", "yp") != "unknown"),
    )

    cover = method_dict["retrieval_cover"]
    drone_resolution, drone_size = resolution_size_anyvisloc(uav_data, opt)

    # RoMa fine scaling is scalar. AnyVisLoc maps are normally square-resolution;
    # use the mean as a safe fallback for slightly anisotropic maps.
    ref_res_scalar = float(np.mean(map_res))
    finescale = drone_resolution / max(ref_res_scalar, 1e-12)

    patch_scale = float(getattr(opt, "patch_scale", 1.5))
    view_m = float(drone_size[0]) * float(drone_resolution) * patch_scale
    min_patch_size_m = float(getattr(opt, "min_patch_size_m", 0.0) or 0.0)
    max_patch_size_m = float(getattr(opt, "max_patch_size_m", 0.0) or 0.0)
    if min_patch_size_m > 0:
        view_m = max(view_m, min_patch_size_m)
    if max_patch_size_m > 0:
        view_m = min(view_m, max_patch_size_m)
    patch_h = _even_ceil(view_m / max(float(map_res[1]), 1e-12))
    patch_w = _even_ceil(view_m / max(float(map_res[0]), 1e-12))

    # Avoid invalid negative starts in the original sliding-window helper.
    max_h = max(2, int(ref_image.shape[0]) - 2)
    max_w = max(2, int(ref_image.shape[1]) - 2)
    patch_h = min(patch_h, max_h)
    patch_w = min(patch_w, max_w)
    if patch_h % 2:
        patch_h = max(2, patch_h - 1)
    if patch_w % 2:
        patch_w = max(2, patch_w - 1)

    block_size = [patch_h, patch_w]
    step_size = [
        max(1, int(patch_h * (100 - cover) / 100)),
        max(1, int(patch_w * (100 - cover) / 100)),
    ]

    MID = compute_block_mid_wo_black(ref_image, block_size, step_size)
    if MID.size == 0:
        raise RuntimeError("No valid retrieval gallery blocks. Check map content, patch size, or retrieval cover.")
    mids = MID.reshape(-1, 2)

    UAV_image = crop_center_from_bgr(uav_image_bgr, int(drone_size[0]), int(drone_size[1]))
    retrieval_model = method_dict["retrieval_model"]
    img_transform = method_dict["img_transform"]

    retrieval_t0 = time.time()
    UAV_image = UAV_image.resize((retrieval_size, retrieval_size))
    UAV_img = img_transform(UAV_image)

    use_ref_cache = (
        getattr(opt, "pose_priori", "yp") == "p"
        and not bool(getattr(opt, "disable_ref_feature_cache", False))
    )
    cache_path = None
    cache_meta = None
    gf = None

    if use_ref_cache:
        cache_meta = _make_ref_feature_cache_meta(
            opt=opt,
            method_dict=method_dict,
            config=config,
            ref_image=ref_image,
            block_size=block_size,
            step_size=step_size,
            mids=mids,
        )
        cache_path = _make_ref_feature_cache_path(save_path, opt, cache_meta)
        gf = _load_cached_ref_features(cache_path, cache_meta)

    if gf is not None:
        print(f"[v6 patched cache hit] loaded reference features: {cache_path}")
        qf = _extract_query_feature_with_dummy_gallery(
            retrieval_size,
            config,
            method_dict,
            MID,
            block_size,
            ref_image,
            UAV_img,
            img_transform,
            retrieval_model,
        )
        gf = gf.to(qf.device)
    else:
        gf, qf = extract_features(
            retrieval_size,
            config,
            method_dict,
            MID,
            block_size,
            ref_image,
            UAV_img,
            img_transform,
            retrieval_model,
        )
        if use_ref_cache and cache_path is not None and cache_meta is not None:
            _save_cached_ref_features(cache_path, cache_meta, gf)

    retrieval_t1 = time.time()

    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().detach().cpu().numpy()
    order = np.argsort(score)[::-1]
    retrieval_time_cost = (retrieval_t1 - retrieval_t0) / (len(MID) + 1)

    row_starts, col_starts, PDE_list = [], [], []
    tile_physical_width = max(float(map_res[0]) * float(patch_w), 1e-12)
    for i in range(len(order)):
        mid_row, mid_col = mids[order[i], :]
        row0 = max(0, int(mid_row - patch_h / 2))
        col0 = max(0, int(mid_col - patch_w / 2))
        row_starts.append(row0)
        col_starts.append(col0)
        dx_m = (center_col - mid_col) * float(map_res[0])
        dy_m = (center_row - mid_row) * float(map_res[1])
        d_i = float(np.hypot(dx_m, dy_m))
        PDE_list.append(float(d_i / tile_physical_width))

    if bool(getattr(opt, "visualize", False)):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D

        show_n = int(config.get("RETRIEVAL_IMG_NUM", 5))
        retrieval_img_name = method_dict["retrieval_img_name"]
        show_n_eff = min(len(mids), show_n)

        fig = plt.figure(figsize=(10, 2.8))
        axes = []

        ax0 = plt.subplot(1, show_n + 1, 1)
        axes.append(ax0)
        ax0.axis("off")
        uav_vis = _draw_cross_at_image_center(np.asarray(UAV_image), color=(255, 0, 0))
        ax0.imshow(uav_vis)
        ax0.set_title("UAV")

        for i in range(show_n_eff):
            ax = plt.subplot(1, show_n + 1, i + 2)
            axes.append(ax)
            ax.axis("off")

            mid_row, mid_col = mids[order[i], :]
            row0 = max(0, int(mid_row - patch_h / 2))
            col0 = max(0, int(mid_col - patch_w / 2))
            img = ref_image[row0:row0 + int(patch_h), col0:col0 + int(patch_w)]
            # Red cross: estimated/true UAV view center if it falls inside this retrieved patch.
            img = _draw_red_cross_if_inside(img, center_col, center_row, col0, row0)
            # Green cross: geometric center of each retrieved patch, always visible.
            img = _draw_cross_at_image_center(img, color=(0, 255, 0))

            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{_circled_number(i + 1)} R:{PDE_list[i]:.3f}")

        plt.ioff()
        plt.tight_layout()

        # Dashed visual separator between UAV image and retrieval results.
        if len(axes) >= 2:
            fig.canvas.draw()
            b0 = axes[0].get_position()
            b1 = axes[1].get_position()
            x_sep = (b0.x1 + b1.x0) * 0.5
            y0 = min(ax.get_position().y0 for ax in axes)
            y1 = max(ax.get_position().y1 for ax in axes)
            fig.add_artist(
                Line2D(
                    [x_sep, x_sep],
                    [y0, y1],
                    transform=fig.transFigure,
                    linestyle="--",
                    linewidth=1.2,
                    color="0.35",
                    alpha=0.9,
                )
            )

        plt.savefig(str(Path(save_path) / retrieval_img_name.lstrip("/")), dpi=160)
        plt.close(fig)

    return order, row_starts, col_starts, PDE_list, patch_h, patch_w, finescale, retrieval_time_cost


def sample_dsm_local(dsm, xs, ys, dsm_resolution, dsm_origin_local=(0.0, 0.0)):
    dsm = dsm_to_2d(dsm)
    h, w = dsm.shape[:2]
    res = as_vec2(dsm_resolution, "dsm_resolution")
    origin = as_vec2(dsm_origin_local, "dsm_origin_local", [0.0, 0.0])
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    cols = np.round((xs - float(origin[0])) / max(float(res[0]), 1e-12)).astype(np.int64)
    rows = np.round((ys - float(origin[1])) / max(float(res[1]), 1e-12)).astype(np.int64)
    valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    z = np.full(xs.shape, np.nan, dtype=np.float32)
    z[valid] = dsm[rows[valid], cols[valid]]
    valid = valid & np.isfinite(z)
    return z, valid


def draw_pnp_inlier_matches_sampled(
    Sen_pts_valid,
    Ref_pts_valid,
    inlier_indices,
    uav_image,
    ref_image,
    save_path,
    input_pnp_count,
    localization_error_m=None,
    sample_ratio=0.1,
    seed=0,
    match_color=(0, 255, 0),
    point_color=(0, 0, 255),
):
    """Draw sampled PnP RANSAC inlier matches without hollow keypoint circles.

    Lines use ``match_color``. Endpoints are drawn manually as small red filled
    dots, with radius only slightly larger than the line thickness.
    """
    Sen_pts_valid = np.asarray(Sen_pts_valid, dtype=np.float32)
    Ref_pts_valid = np.asarray(Ref_pts_valid, dtype=np.float32)

    if inlier_indices is None:
        return

    inlier_indices = np.asarray(inlier_indices).reshape(-1).astype(np.int64)
    inlier_indices = inlier_indices[
        (inlier_indices >= 0)
        & (inlier_indices < len(Sen_pts_valid))
        & (inlier_indices < len(Ref_pts_valid))
    ]

    input_pnp_count = int(input_pnp_count)
    inlier_count = int(len(inlier_indices))
    if input_pnp_count <= 0 or inlier_count <= 0:
        return

    ratio = float(sample_ratio)
    if not np.isfinite(ratio):
        ratio = 0.1
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0:
        return

    draw_count = int(math.ceil(inlier_count * ratio))
    draw_count = max(1, min(draw_count, inlier_count))

    rng = np.random.default_rng(int(seed))
    draw_indices = rng.choice(inlier_indices, size=draw_count, replace=False)

    sen_draw = Sen_pts_valid[draw_indices]
    ref_draw = Ref_pts_valid[draw_indices]

    left = ensure_bgr_uint8(uav_image)
    right = ensure_bgr_uint8(ref_image)
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    vis_h = max(h_left, h_right)
    vis_w = w_left + w_right
    vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    vis[:h_left, :w_left] = left
    vis[:h_right, w_left:w_left + w_right] = right

    line_color = tuple(int(c) for c in match_color)
    dot_color = tuple(int(c) for c in point_color)
    line_thickness = max(1, int(round(min(vis_w, vis_h) * 0.0015)))
    point_radius = max(line_thickness + 1, int(round(line_thickness * 1.6)))

    for p_left, p_right in zip(sen_draw, ref_draw):
        x1 = int(round(float(p_left[0])))
        y1 = int(round(float(p_left[1])))
        x2 = int(round(float(p_right[0]))) + w_left
        y2 = int(round(float(p_right[1])))
        if not (0 <= x1 < w_left and 0 <= y1 < h_left and w_left <= x2 < vis_w and 0 <= y2 < h_right):
            continue
        cv2.line(vis, (x1, y1), (x2, y2), line_color, line_thickness, cv2.LINE_AA)
        cv2.circle(vis, (x1, y1), point_radius, dot_color, -1, cv2.LINE_AA)
        cv2.circle(vis, (x2, y2), point_radius, dot_color, -1, cv2.LINE_AA)

    if localization_error_m is None or not np.isfinite(float(localization_error_m)):
        err_text = "Err: N/A"
    else:
        err_text = f"Err: {float(localization_error_m):.1f} m"

    lines = [
        f"PnP: {input_pnp_count}",
        f"In: {inlier_count}",
        err_text,
    ]

    x1, y1 = 4, 4
    max_box_w = max(1, vis_w - x1)
    max_box_h = max(1, min(vis_h - y1, int(round(vis_h * 0.24))))
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(4, int(round(min(vis_w, vis_h) * 0.006)))
    line_gap = max(2, int(round(min(vis_w, vis_h) * 0.004)))

    font_scale = 0.10
    thickness = 1
    sizes = []
    baselines = []
    for fs in np.linspace(0.90, 0.10, 100):
        candidate_thickness = max(1, int(round(float(fs) * 1.4)))
        metrics = [cv2.getTextSize(line, font, float(fs), candidate_thickness) for line in lines]
        candidate_sizes = [m[0] for m in metrics]
        candidate_baselines = [m[1] for m in metrics]
        text_w = max(w for w, _ in candidate_sizes)
        text_h = sum(h + b for (_, h), b in zip(candidate_sizes, candidate_baselines)) + line_gap * (len(lines) - 1)
        if text_w + 2 * margin <= max_box_w and text_h + 2 * margin <= max_box_h:
            font_scale = float(fs)
            thickness = candidate_thickness
            sizes = candidate_sizes
            baselines = candidate_baselines
            break
    if not sizes:
        metrics = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
        sizes = [m[0] for m in metrics]
        baselines = [m[1] for m in metrics]

    text_w = max(w for w, _ in sizes)
    text_h = sum(h + b for (_, h), b in zip(sizes, baselines)) + line_gap * (len(lines) - 1)
    box_w = min(max_box_w, text_w + 2 * margin)
    box_h = min(vis_h - y1, text_h + 2 * margin)
    x2 = min(vis_w - 1, x1 + box_w)
    y2 = min(vis_h - 1, y1 + box_h)

    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    y = y1 + margin
    for line, (_, th), baseline in zip(lines, sizes, baselines):
        y += th
        cv2.putText(
            vis,
            line,
            (x1 + margin, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += baseline + line_gap

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    cv2.imwrite(str(save_path), vis)


def _pnp_flag(name):
    name = str(name or "P3P").upper()
    if name == "EPNP":
        return cv2.SOLVEPNP_EPNP
    if name == "AP3P":
        return cv2.SOLVEPNP_AP3P
    return cv2.SOLVEPNP_P3P


def estimate_drone_pose_anyvisloc(
    match_points,
    K,
    local_x,
    local_y,
    dsm_z,
    dist_coeffs=None,
    pnp_method="P3P",
    reprojection_error=8.0,
    iterations_count=2000,
    confidence=0.999,
):
    """PnP in AnyVisLoc dataset-local coordinates.

    Dataset local y points downward. For OpenCV PnP we solve in a conventional
    x-right/y-up/z-up frame by negating y, then convert the camera center back.
    """
    image_points = np.asarray(match_points, dtype=np.float32).reshape(-1, 2)
    object_points_dataset = np.column_stack([local_x, local_y, dsm_z]).astype(np.float32)
    valid = np.isfinite(image_points).all(axis=1) & np.isfinite(object_points_dataset).all(axis=1)
    image_points = image_points[valid]
    object_points_dataset = object_points_dataset[valid]
    if len(image_points) < 4:
        return None, None, 0, np.zeros((0, 1), dtype=np.int32)

    object_points_pnp = object_points_dataset.copy()
    object_points_pnp[:, 1] *= -1.0

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1, 1)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points_pnp,
        image_points,
        np.asarray(K, dtype=np.float32),
        dist_coeffs,
        flags=_pnp_flag(pnp_method),
        reprojectionError=float(reprojection_error),
        iterationsCount=int(iterations_count),
        confidence=float(confidence),
    )
    if (not ok) or inliers is None or len(inliers) == 0:
        return None, None, 0, np.zeros((0, 1), dtype=np.int32)

    R = rotvector2rot(rvec)
    center_pnp = (-R.T @ tvec).reshape(3)
    center_dataset = center_pnp.copy()
    center_dataset[1] *= -1.0
    XYZ = {
        "X": float(center_dataset[0]),
        "Y": float(center_dataset[1]),
        "Z": float(center_dataset[2]),
        # Compatibility aliases; they are local coordinates, not lon/lat.
        "L": float(center_dataset[0]),
        "B": float(center_dataset[1]),
        "H": float(center_dataset[2]),
    }

    Rx_90 = np.array([[1, 0, 0], [0, np.cos(-np.pi / 2), np.sin(-np.pi / 2)], [0, -np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
    cam_angle = rot_to_euler(Rx_90 @ R)
    return XYZ, cam_angle, int(len(inliers)), inliers


def Match2Pos_all_anyvisloc(
    opt,
    config,
    uav_img0,
    finescale,
    K,
    ref_image,
    dsm_image,
    row_start_list,
    col_start_list,
    patch_h,
    patch_w,
    save_path,
    method_dict,
    matRotation,
    map_resolution,
    map_origin_local,
    dsm_resolution,
    dsm_origin_local,
    dist=None,
    truePos=None,
):
    """Pixel-level matching + PnP for AnyVisLoc local coordinates."""
    ensure_dir(save_path)

    reverseMatRotation = cv2.invertAffineTransform(matRotation) if opt.pose_priori == "yp" else None
    resize_ratio = float(opt.resize_ratio)
    map_res = as_vec2(map_resolution, "map_resolution")
    map_origin = as_vec2(map_origin_local, "map_origin_local", [0.0, 0.0])

    if resize_ratio < 1:
        uav_img = cv2.resize(uav_img0, None, fx=resize_ratio, fy=resize_ratio)
    else:
        uav_img = uav_img0

    XYZ_list, inliers_list, time_list = [], [], []
    if not isinstance(row_start_list, list):
        row_start_list = [row_start_list]
        col_start_list = [col_start_list]

    if opt.strategy == "Top1":
        top_n = 1
    elif opt.strategy == "Topn_opt":
        top_n = min(int(method_dict["retrieval_topn"]), len(row_start_list))
    else:
        top_n = len(row_start_list)

    for index in range(top_n):
        ransac_name = f"/top{index + 1}_ransac.jpg"
        row0 = int(row_start_list[index])
        col0 = int(col_start_list[index])

        fineRef = ref_image[row0:row0 + int(patch_h), col0:col0 + int(patch_w)]
        if fineRef.size == 0:
            XYZ_list.append({"X": None, "Y": None, "Z": None, "L": None, "B": None, "H": None})
            inliers_list.append(0)
            time_list.append([0.0, 0.0])
            continue

        fine_scale = resize_ratio / max(float(finescale), 1e-12)
        fineRef = cv2.resize(fineRef, None, fx=fine_scale, fy=fine_scale)

        match_time_start = time.time()
        Sen_pts, Ref_pts = run_pixel_match_anyvisloc(
            uav_img,
            fineRef,
            method_dict,
            save_path,
            ransac_name,
            need_ransac=False,
            show_matches=False,
        )
        match_time_end = time.time()
        single_match_time = match_time_end - match_time_start

        if len(Ref_pts) >= 5:
            Sen_pts_arr = np.asarray(Sen_pts, dtype=np.float32)
            Ref_pts_arr = np.asarray(Ref_pts, dtype=np.float32)

            # Ref_pts are patch-local image coordinates in x/y order, i.e. col/row.
            refCoordinate = (
                Ref_pts_arr / resize_ratio * finescale
                + np.array([col0, row0], dtype=np.float32)
            )
            if reverseMatRotation is not None:
                refCoordinate1 = np.hstack([refCoordinate, np.ones((refCoordinate.shape[0], 1), dtype=np.float32)])
                refCoordinate = refCoordinate1 @ reverseMatRotation.T

            cols = refCoordinate[:, 0]
            rows = refCoordinate[:, 1]
            local_x = cols * float(map_res[0]) + float(map_origin[0])
            local_y = rows * float(map_res[1]) + float(map_origin[1])
            DSM, valid_dsm = sample_dsm_local(dsm_image, local_x, local_y, dsm_resolution, dsm_origin_local)

            match_points = Sen_pts_arr / resize_ratio
            valid = (
                valid_dsm
                & np.isfinite(match_points).all(axis=1)
                & np.isfinite(local_x)
                & np.isfinite(local_y)
                & np.isfinite(DSM)
            )

            # These are the actual correspondences sent to PnP.
            match_points = match_points[valid]
            local_x = local_x[valid]
            local_y = local_y[valid]
            DSM = DSM[valid]

            # These keep the same indexing as the PnP input and are used only for visualization.
            # inliers_all returned by solvePnPRansac indexes these filtered arrays.
            Sen_pts_valid = Sen_pts_arr[valid]
            Ref_pts_valid = Ref_pts_arr[valid]
            input_pnp_count = int(len(match_points))

            # Default behavior: use the NPZ camera distortion coefficients in PnP.
            XYZ, _, inliers, inliers_all = estimate_drone_pose_anyvisloc(
                match_points,
                K,
                local_x,
                local_y,
                DSM,
                dist_coeffs=dist,
                pnp_method=getattr(opt, "PnP_method", "P3P"),
                reprojection_error=getattr(opt, "pnp_reproj_error", 8.0),
                iterations_count=getattr(opt, "pnp_iterations", 2000),
                confidence=getattr(opt, "pnp_confidence", 0.999),
            )

            candidate_error_m = None
            if XYZ is not None and truePos is not None and XYZ.get("X") is not None and XYZ.get("Y") is not None:
                dx = float(XYZ["X"]) - float(truePos["x"])
                dy = float(XYZ["Y"]) - float(truePos["y"])
                candidate_error_m = float(np.sqrt(dx * dx + dy * dy))

            if XYZ is None:
                XYZ = {"X": None, "Y": None, "Z": None, "L": None, "B": None, "H": None}
                inliers = 0

            if bool(getattr(opt, "visualize", False)) and inliers > 0:
                draw_ratio = float(getattr(opt, "draw_pnp_inlier_ratio", 0.1))
                draw_seed = int(getattr(opt, "draw_pnp_inlier_seed", 0)) + int(index)

                draw_pnp_inlier_matches_sampled(
                    Sen_pts_valid,
                    Ref_pts_valid,
                    inliers_all,
                    uav_img,
                    fineRef,
                    str(Path(save_path) / f"top{index + 1}_pnp_inliers_{inliers}.jpg"),
                    input_pnp_count=input_pnp_count,
                    localization_error_m=candidate_error_m,
                    sample_ratio=draw_ratio,
                    seed=draw_seed,
                    match_color=(0, 255, 0),
                )
        else:
            XYZ = {"X": None, "Y": None, "Z": None, "L": None, "B": None, "H": None}
            inliers = 0

        pnp_time_end = time.time()
        time_list.append([single_match_time, pnp_time_end - match_time_end])
        XYZ_list.append(XYZ)
        inliers_list.append(int(inliers))

    match_time = [t[0] for t in time_list]
    pnp_time = [t[1] for t in time_list]
    return XYZ_list, inliers_list, match_time, pnp_time


def pos2error_anyvisloc(truePos, XYZ_list, inliers_list):
    """Choose the Top-N candidate with max PnP inliers and compute local XY error in meters.

    Invalid PnP results return pred_error=None instead of the old 10000.0 sentinel,
    so summary statistics can exclude failed localizations cleanly.
    """
    if len(XYZ_list) == 0:
        return {"x": None, "y": None, "z": None}, None, [], False, None

    inliers_array = np.asarray(inliers_list, dtype=np.int64)
    best_index = int(np.argmax(inliers_array)) if inliers_array.size else 0

    location_error_list = []
    for pred in XYZ_list:
        if pred.get("X") is not None and pred.get("Y") is not None:
            dx = float(pred["X"]) - float(truePos["x"])
            dy = float(pred["Y"]) - float(truePos["y"])
            location_error_list.append(float(np.sqrt(dx * dx + dy * dy)))
        else:
            location_error_list.append(None)

    best = XYZ_list[best_index]
    pred_loc = {"x": best.get("X"), "y": best.get("Y"), "z": best.get("Z")}

    pred_error = location_error_list[best_index]
    pnp_success = (
        pred_loc["x"] is not None
        and pred_loc["y"] is not None
        and pred_error is not None
        and best_index < len(inliers_list)
        and int(inliers_list[best_index]) > 0
    )
    return pred_loc, pred_error, location_error_list, bool(pnp_success), best_index


def save_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
