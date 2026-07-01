#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrieval-only AnyVisLoc runner.

This is a slim run_avl.py variant:
  - uses avl_data.AnyVisLocNPZDataset;
  - loads PNG/NPY references through avl_utils.build_anyvisloc_reference_view;
  - initializes retrieval methods through avl_utils.retrieval_init;
  - calls avl_utils.retrieval_all_anyvisloc;
  - does not initialize matchers;
  - does not run pixel matching, RANSAC, or PnP;
  - reports only Recall@K and PDM@K for retrieval.
"""

import argparse
import csv
import json
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from avl_data import AnyVisLocNPZDataset
from avl_utils import (
    anyvisloc_sample_to_truepos,
    build_anyvisloc_reference_view,
    dumpRotateImage,
    ensure_dir,
    normalize_reference_mode,
    retrieval_all_anyvisloc,
    retrieval_init,
    save_json,
    tensor_rgb_to_bgr_uint8,
)

warnings.filterwarnings("ignore")


def parse_scenes(scenes):
    if scenes is None or len(scenes) == 0:
        return None
    if len(scenes) == 1 and "," in scenes[0]:
        return [x.strip() for x in scenes[0].split(",") if x.strip()]
    return scenes


def parse_retrieval_methods(values):
    if values is None or len(values) == 0:
        return None
    if len(values) == 1 and "," in values[0]:
        return [x.strip() for x in values[0].split(",") if x.strip()]
    return [str(x).strip() for x in values if str(x).strip()]


def parse_retrieval_ks(values, fallback=None):
    if fallback is None:
        fallback = [1, 3, 5]

    raw_values = fallback if values is None or len(values) == 0 else values
    ks = []
    for value in raw_values:
        if isinstance(value, str) and "," in value:
            parts = [x.strip() for x in value.split(",") if x.strip()]
        else:
            parts = [value]
        for part in parts:
            k = max(1, int(part))
            ks.append(k)

    if not ks:
        ks = [1, 3, 5]
    return sorted(set(ks))


def get_parse():
    parser = argparse.ArgumentParser(description="AnyVisLoc retrieval-only baseline")
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--yaml", default="config_selectable_matchers.yaml", type=str)
    parser.add_argument("--save_dir", default="./Result/AnyVisLoc_retrieval_only", type=str)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--reference_mode",
        default="aerial",
        choices=["aerial", "satellite"],
        help="Reference source from Lxx_reference.json.",
    )
    parser.add_argument(
        "--retrieval_methods",
        nargs="*",
        default=None,
        help="Override RETRIEVAL_METHODS in yaml. Example: --retrieval_methods CAMP",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--pose_priori",
        default="yp",
        choices=["yp", "p", "unknown"],
        help="Use pose prior. yp rotates map by yaw; p uses shifted view center without yaw rotation.",
    )
    parser.add_argument("--patch_scale", default=1.0, type=float)
    parser.add_argument("--min_patch_size_m", default=0.0, type=float)
    parser.add_argument("--max_patch_size_m", default=0.0, type=float)
    parser.add_argument(
        "--retrieval_ks",
        nargs="*",
        default=None,
        help="K values for Recall@K and PDM@K. Example: --retrieval_ks 1 3 5",
    )
    parser.add_argument(
        "--retrieval_k",
        default=None,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pdm_lambda", default=6.0, type=float)
    parser.add_argument("--pdm_alpha", default=0.9, type=float)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--save_every", default=20, type=int)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Save retrieval visualization images.")
    parser.add_argument("--debug_raise", action="store_true")
    parser.add_argument(
        "--ref_feature_cache_dir",
        default=None,
        type=str,
        help="Directory for cached reference/gallery retrieval features used when --pose_priori p.",
    )
    parser.add_argument(
        "--disable_ref_feature_cache",
        action="store_true",
        help="Disable reference/gallery feature cache even when --pose_priori p.",
    )

    opt = parser.parse_args()
    opt.scenes = parse_scenes(opt.scenes)
    opt.retrieval_methods = parse_retrieval_methods(opt.retrieval_methods)
    opt.reference_mode = normalize_reference_mode(opt.reference_mode)
    fallback_ks = [opt.retrieval_k] if opt.retrieval_k is not None else [1, 3, 5]
    opt.retrieval_ks = parse_retrieval_ks(opt.retrieval_ks, fallback=fallback_ks)
    opt.retrieval_k = int(opt.retrieval_ks[-1])
    return opt


def save_csv(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*[row.keys() for row in rows]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)


def safe_float(x):
    if x is None:
        return None
    try:
        value = float(x)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def retrieval_metrics_from_ratios(ratios, k=5, pdm_lambda=6.0, pdm_alpha=0.9):
    """Compute per-query PDM@K and Recall@K from retrieval ratios.

    ratios/PDE are ordered by retrieval rank. The nearest tile to the
    ground-truth view center has the smallest ratio, so its one-based index is
    the ground-truth rank.
    """
    ratios = np.asarray(ratios, dtype=np.float64).reshape(-1)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return None, None, None

    k_eff = min(max(1, int(k)), int(ratios.size))
    top_ratios = ratios[:k_eff]
    weights = np.arange(k_eff, 0, -1, dtype=np.float64)

    logits = np.clip(float(pdm_lambda) * (top_ratios - float(pdm_alpha)), -60.0, 60.0)
    retrieval_scores = 1.0 / (1.0 + np.exp(logits))
    pdm_at_k = float(np.sum(weights * retrieval_scores) / np.sum(weights))

    gt_rank = int(np.argmin(ratios)) + 1
    recall_at_k = float(gt_rank <= k_eff)
    return pdm_at_k, recall_at_k, gt_rank


def retrieval_metrics_for_ks(ratios, ks, pdm_lambda=6.0, pdm_alpha=0.9):
    metrics = {}
    retrieval_gt_rank = None

    for k in ks:
        k = int(k)
        pdm_at_k, recall_at_k, gt_rank = retrieval_metrics_from_ratios(
            ratios,
            k=k,
            pdm_lambda=pdm_lambda,
            pdm_alpha=pdm_alpha,
        )
        metrics[f"pdm_at_{k}"] = safe_float(pdm_at_k)
        metrics[f"recall_at_{k}"] = safe_float(recall_at_k)
        metrics[f"PDM@{k}"] = safe_float(pdm_at_k)
        metrics[f"Recall@{k}"] = safe_float(recall_at_k)
        if retrieval_gt_rank is None and gt_rank is not None:
            retrieval_gt_rank = int(gt_rank)

    return metrics, retrieval_gt_rank


def summarize_retrieval(rows, failed_rows, opt, extra=None):
    attempted = len(rows) + len(failed_rows)
    denominator = max(attempted, 1)
    retrieval_times = [
        float(v)
        for v in (safe_float(r.get("retrieval_time_s")) for r in rows)
        if v is not None
    ]

    ks = [int(k) for k in opt.retrieval_ks]
    primary_k = int(opt.retrieval_k)

    summary = {
        "dataset_root": str(opt.dataset_root),
        "scenes": opt.scenes,
        "reference_mode": opt.reference_mode,
        "pose_priori": opt.pose_priori,
        "retrieval_ks": ks,
        "retrieval_k": primary_k,
        "num_attempted": int(attempted),
        "num_success": int(len(rows)),
        "num_failed": int(len(failed_rows)),
        "mean_retrieval_time_s": float(np.mean(retrieval_times)) if retrieval_times else None,
    }

    # Failed retrieval rows contribute 0 to aggregate metrics.
    for k in ks:
        pdm_values = [
            float(v)
            for v in (safe_float(r.get(f"PDM@{k}")) for r in rows)
            if v is not None
        ]
        recall_values = [
            float(v)
            for v in (safe_float(r.get(f"Recall@{k}")) for r in rows)
            if v is not None
        ]
        pdm = float(np.sum(pdm_values) / denominator) if attempted else None
        recall = float(np.sum(recall_values) / denominator) if attempted else None
        summary[f"metric_valid_rows@{k}"] = int(min(len(pdm_values), len(recall_values)))
        summary[f"pdm_at_{k}"] = pdm
        summary[f"recall_at_{k}"] = recall
        summary[f"PDM@{k}"] = pdm
        summary[f"Recall@{k}"] = recall

    summary["metric_valid_rows"] = summary.get(f"metric_valid_rows@{primary_k}", 0)
    summary["pdm_at_k"] = summary.get(f"PDM@{primary_k}")
    summary["recall_at_k"] = summary.get(f"Recall@{primary_k}")

    if extra:
        summary.update(extra)
    return summary


def flush_outputs(out_dir, rows, failed_rows, summary):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "all_results.json", rows)
    save_json(out_dir / "failed.json", failed_rows)
    save_csv(rows, out_dir / "all_results.csv")


def process_one_sample(sample, opt, config, method_dict, retrieval_method, combo_dir):
    sample_id = as_text(sample["sample_id"])
    scene_name = as_text(sample["scene_name"])
    save_path = Path(combo_dir) / scene_name / sample_id
    ensure_dir(save_path)

    result_json = save_path / "retrieval_result.json"
    if opt.skip_existing and result_json.exists():
        with result_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    opt._current_scene_name = scene_name
    opt._current_sample_id = sample_id

    true_pos = anyvisloc_sample_to_truepos(sample)
    uav_image = tensor_rgb_to_bgr_uint8(sample["image"])

    view = build_anyvisloc_reference_view(sample, opt.reference_mode)
    ref_map0 = view["map"]
    map_resolution = view["map_resolution"]
    map_origin_local = view["map_origin_local"]

    if opt.pose_priori == "yp":
        ref_map, mat_rotation = dumpRotateImage(ref_map0, true_pos["yaw"])
    else:
        ref_map = ref_map0
        mat_rotation = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    t0 = time.time()
    (
        ir_order,
        row_starts,
        col_starts,
        pde_list,
        patch_h,
        patch_w,
        fine_scale,
        retrieval_time,
    ) = retrieval_all_anyvisloc(
        ref_map,
        uav_image,
        true_pos,
        map_resolution,
        map_origin_local,
        mat_rotation,
        str(save_path),
        opt,
        config,
        method_dict,
    )
    total_time = time.time() - t0

    metrics_by_k, retrieval_gt_rank = retrieval_metrics_for_ks(
        pde_list,
        ks=opt.retrieval_ks,
        pdm_lambda=opt.pdm_lambda,
        pdm_alpha=opt.pdm_alpha,
    )
    primary_k = int(opt.retrieval_k)

    result = {
        "scene_name": scene_name,
        "sample_id": sample_id,
        "npz_path": as_text(sample.get("npz_path", "")),
        "reference_mode": opt.reference_mode,
        "retrieval_method": retrieval_method,
        "pose_priori": opt.pose_priori,
        "retrieval_ks": [int(k) for k in opt.retrieval_ks],
        "retrieval_k": primary_k,
        "pdm_lambda": float(opt.pdm_lambda),
        "pdm_alpha": float(opt.pdm_alpha),
        "pdm_at_k": metrics_by_k.get(f"PDM@{primary_k}"),
        "recall_at_k": metrics_by_k.get(f"Recall@{primary_k}"),
        "retrieval_gt_rank": None if retrieval_gt_rank is None else int(retrieval_gt_rank),
        "num_gallery_blocks": int(len(pde_list)),
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "fine_scale": float(fine_scale),
        "retrieval_time_s": float(retrieval_time),
        "total_time_s": float(total_time),
        "best_pde": safe_float(np.min(pde_list) if len(pde_list) else None),
        "top1_pde": safe_float(pde_list[0] if len(pde_list) else None),
        "result_dir": str(save_path),
    }
    result.update(metrics_by_k)

    save_json(result_json, result)
    return result


def main():
    opt = get_parse()
    with open(opt.yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["DEVICE"] = opt.device

    retrieval_methods = opt.retrieval_methods or list(config.get("RETRIEVAL_METHODS", []))
    if not retrieval_methods:
        raise ValueError("No retrieval method configured. Set RETRIEVAL_METHODS in yaml or use --retrieval_methods.")

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
    pbar = tqdm(total=len(indices) * len(retrieval_methods), desc="Retrieval Evaluation", unit="img")

    for retrieval_method in retrieval_methods:
        method_dict = {"retrieval_method": retrieval_method}
        method_dict = retrieval_init(method_dict, config)

        combo_name = f"{opt.reference_mode}-{retrieval_method}-{opt.pose_priori}"
        combo_dir = run_root / combo_name
        ensure_dir(combo_dir)

        combo_rows = []
        combo_failed = []
        processed = 0

        for idx in indices:
            sample = dataset[idx]
            scene_name = as_text(sample.get("scene_name", ""))
            sample_id = as_text(sample.get("sample_id", ""))
            try:
                row = process_one_sample(sample, opt, config, method_dict, retrieval_method, combo_dir)
                combo_rows.append(row)
                all_rows.append(row)
                metric_msg = " ".join(
                    f"Recall@{int(k)}={row.get(f'Recall@{int(k)}')} "
                    f"PDM@{int(k)}={row.get(f'PDM@{int(k)}')}"
                    for k in opt.retrieval_ks
                )
                tqdm.write(
                    f"{scene_name}/{sample_id} {retrieval_method}: "
                    f"{metric_msg}"
                )
            except Exception as exc:
                err = {
                    "scene_name": scene_name,
                    "sample_id": sample_id,
                    "npz_path": as_text(sample.get("npz_path", "")),
                    "reference_mode": opt.reference_mode,
                    "retrieval_method": retrieval_method,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                combo_failed.append(err)
                all_failed.append(err)
                tqdm.write(f"[Error] {scene_name}/{sample_id} {retrieval_method}: {exc}")
                if opt.debug_raise:
                    raise
            finally:
                processed += 1
                pbar.update(1)

            if opt.save_every > 0 and processed % int(opt.save_every) == 0:
                combo_summary = summarize_retrieval(
                    combo_rows,
                    combo_failed,
                    opt,
                    extra={"retrieval_method": retrieval_method, "combo_name": combo_name},
                )
                flush_outputs(combo_dir, combo_rows, combo_failed, combo_summary)

        combo_summary = summarize_retrieval(
            combo_rows,
            combo_failed,
            opt,
            extra={"retrieval_method": retrieval_method, "combo_name": combo_name},
        )
        flush_outputs(combo_dir, combo_rows, combo_failed, combo_summary)

    pbar.close()

    global_summary = summarize_retrieval(
        all_rows,
        all_failed,
        opt,
        extra={"retrieval_methods": retrieval_methods},
    )
    flush_outputs(run_root, all_rows, all_failed, global_summary)
    print(json.dumps(global_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
