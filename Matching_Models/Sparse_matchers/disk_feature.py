#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal DISK+LightGlue wrappers used by run_avl.py."""

import cv2
import numpy as np
import torch

from .LightGlue_main.lightglue import DISK, LightGlue
from .LightGlue_main.lightglue.utils import rbd


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _bgr_to_rgb_torch(image_bgr, device):
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim == 2:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().to(device)


def DISK_LG_Init(device="cuda", max_num_keypoints=4096):
    device = _normalize_device(device)
    extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)
    return extractor, matcher


@torch.no_grad()
def DISK_LG_extract(image_bgr, extractor, device=None):
    if device is None:
        device = next(extractor.parameters()).device
    image_tensor = _bgr_to_rgb_torch(image_bgr, _normalize_device(device))
    return extractor.extract(image_tensor)


@torch.no_grad()
def DISK_LG_match(feats0, feats1, matcher):
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    matches = matches01["matches"]
    if matches.numel() == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty.copy()

    pts0 = feats0["keypoints"][matches[..., 0]]
    pts1 = feats1["keypoints"][matches[..., 1]]
    return (
        pts0.detach().cpu().numpy().astype(np.float32),
        pts1.detach().cpu().numpy().astype(np.float32),
    )
