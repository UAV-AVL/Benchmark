#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse matcher entry points used by run_avl.py.

Supported methods:
  - SP_LG: original SuperPoint + LightGlue implementation
  - SP_LG_GIM: GIM SuperPoint + LightGlue checkpoint
  - SP_LG_MINIMA: Minima LightGlue checkpoint
  - ALIKED_LG
  - DISK_LG

The module intentionally contains no experiment CLI, RANSAC, PnP, dataset
parsing, or visualization code.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from .LightGlue_main.lightglue import LightGlue, SuperPoint
from .LightGlue_main.lightglue.utils import rbd

from .aliked_feature import ALIKED_LG_Init, ALIKED_LG_extract, ALIKED_LG_match
from .disk_feature import DISK_LG_Init, DISK_LG_extract, DISK_LG_match

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

__all__ = [
    "SP_LG_ORI_Init",
    "SP_LG_ORI_extract",
    "SP_LG_ORI_match",
    "SP_LG_GIM_Init",
    "SP_LG_GIM_extract",
    "SP_LG_GIM_match",
    "SP_LG_MINIMA_Init",
    "SP_LG_MINIMA_extract",
    "SP_LG_MINIMA_match",
    "ALIKED_LG_Init",
    "ALIKED_LG_extract",
    "ALIKED_LG_match",
    "DISK_LG_Init",
    "DISK_LG_extract",
    "DISK_LG_match",
]


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _check_weight(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{name} checkpoint not found: {path}. "
            "Put the weight file there or pass the corresponding command-line path."
        )
    return path


def _bgr_to_rgb_chw(image_bgr, device):
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim == 2:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().to(device)


def _bgr_to_gray_bchw(image_bgr, device):
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    gray = gray.astype(np.float32) / 255.0
    return torch.from_numpy(gray)[None, None].contiguous().to(device)


def _empty_matches():
    empty = np.zeros((0, 2), dtype=np.float32)
    return empty, empty.copy()


def _lightglue_match(feats0, feats1, matcher):
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
    matches = matches01["matches"]
    if matches.numel() == 0:
        return _empty_matches()
    pts0 = feats0["keypoints"][matches[..., 0]]
    pts1 = feats1["keypoints"][matches[..., 1]]
    return (
        pts0.detach().cpu().numpy().astype(np.float32),
        pts1.detach().cpu().numpy().astype(np.float32),
    )


def SP_LG_ORI_Init(device="cuda", max_num_keypoints=4096):
    """Original SuperPoint + LightGlue paper implementation."""
    device = _normalize_device(device)
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


@torch.no_grad()
def SP_LG_ORI_extract(image_bgr, extractor, device=None):
    if device is None:
        device = next(extractor.parameters()).device
    image_tensor = _bgr_to_rgb_chw(image_bgr, _normalize_device(device))
    return extractor.extract(image_tensor)


@torch.no_grad()
def SP_LG_ORI_match(feats0, feats1, matcher):
    return _lightglue_match(feats0, feats1, matcher)


def SP_LG_MINIMA_Init(device="cuda", checkpoint_path=None, max_num_keypoints=4096):
    """SuperPoint + Minima LightGlue weights."""
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).resolve().parent / "weights" / "minima_lightglue.pth"
    checkpoint_path = _check_weight(checkpoint_path, "Minima LightGlue")

    device = _normalize_device(device)
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": max_num_keypoints,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }
    lg_conf = {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "filter_threshold": 0.1,
        "weights": None,
    }
    extractor = SuperPoint(**sp_conf).eval().to(device)
    matcher = LightGlue(features="superpoint", **lg_conf).eval().to(device)

    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    for i in range(lg_conf["n_layers"]):
        state_dict = {
            k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn"): v
            for k, v in state_dict.items()
        }
        state_dict = {
            k.replace(f"cross_attn.{i}", f"transformers.{i}.cross_attn"): v
            for k, v in state_dict.items()
        }
    matcher.load_state_dict(state_dict, strict=False)
    return extractor, matcher


@torch.no_grad()
def SP_LG_MINIMA_extract(image_bgr, extractor, device=None):
    return SP_LG_ORI_extract(image_bgr, extractor, device=device)


@torch.no_grad()
def SP_LG_MINIMA_match(feats0, feats1, matcher):
    return _lightglue_match(feats0, feats1, matcher)


def SP_LG_GIM_Init(device="cuda", checkpoint_path=None, max_num_keypoints=4096):
    """GIM SuperPoint + LightGlue checkpoint.

    Expected checkpoint format follows GIM/GlueFactory naming:
    SuperPoint keys are prefixed by ``superpoint.`` and LightGlue keys by
    ``model.``.
    """
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).resolve().parent / "weights" / "gim_lightglue_100h.ckpt"
    checkpoint_path = _check_weight(checkpoint_path, "GIM LightGlue")

    from gluefactory.superpoint import SuperPoint as GIMSuperPoint
    from gluefactory.models.matchers.lightglue import LightGlue as GIMLightGlue

    device = _normalize_device(device)
    extractor = GIMSuperPoint({
        "max_num_keypoints": max_num_keypoints,
        "force_num_keypoints": True,
        "detection_threshold": 0.0,
        "nms_radius": 3,
        "trainable": False,
    })
    matcher = GIMLightGlue({
        "filter_threshold": 0.1,
        "flash": False,
        "checkpointed": True,
    })

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    sp_state = {}
    lg_state = {}
    for key, value in state_dict.items():
        if key.startswith("superpoint."):
            sp_state[key.replace("superpoint.", "", 1)] = value
        elif key.startswith("model."):
            lg_state[key.replace("model.", "", 1)] = value

    if not sp_state:
        raise KeyError(f"No 'superpoint.' keys found in GIM checkpoint: {checkpoint_path}")
    if not lg_state:
        raise KeyError(f"No 'model.' keys found in GIM checkpoint: {checkpoint_path}")

    extractor.load_state_dict(sp_state, strict=False)
    matcher.load_state_dict(lg_state, strict=False)
    extractor = extractor.eval().to(device)
    matcher = matcher.eval().to(device)
    return extractor, matcher


@torch.no_grad()
def SP_LG_GIM_extract(image_bgr, extractor, device=None):
    if device is None:
        device = next(extractor.parameters()).device
    device = _normalize_device(device)
    image_tensor = _bgr_to_gray_bchw(image_bgr, device)
    size = torch.tensor(image_tensor.shape[-2:][::-1], device=device)[None]
    pred = extractor({"image": image_tensor, "image_size": size})
    return {
        "image_tensor": image_tensor,
        "size": size,
        "pred": pred,
    }


def _gim_pair_pred(feat0, feat1, matcher):
    pred = {}
    pred.update({f"{k}0": v for k, v in feat0["pred"].items()})
    pred.update({f"{k}1": v for k, v in feat1["pred"].items()})

    data = {
        "image0": feat0["image_tensor"],
        "image1": feat1["image_tensor"],
        "gray0": feat0["image_tensor"],
        "gray1": feat1["image_tensor"],
        "size0": feat0["size"],
        "size1": feat1["size"],
        "resize0": feat0["size"],
        "resize1": feat1["size"],
    }
    pred.update(matcher({**pred, **data}))
    return pred


@torch.no_grad()
def SP_LG_GIM_match(feat0, feat1, matcher):
    pred = _gim_pair_pred(feat0, feat1, matcher)
    matches = pred["matches"][0]
    if matches.numel() == 0:
        return _empty_matches()
    kpts0 = pred["keypoints0"][0]
    kpts1 = pred["keypoints1"][0]
    pts0 = kpts0[matches[..., 0]]
    pts1 = kpts1[matches[..., 1]]
    return (
        pts0.detach().cpu().numpy().astype(np.float32),
        pts1.detach().cpu().numpy().astype(np.float32),
    )
