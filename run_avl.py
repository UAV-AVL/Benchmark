#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the original retrieve -> match -> PnP baseline on prepared AnyVisLoc NPZ scenes.
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import time

import traceback
import warnings

import yaml
import numpy as np
from tqdm import tqdm

from avl_data import AnyVisLocNPZDataset

from avl_utils import (
    ensure_dir,
    save_data,
    save_json,
    save_csv,
    tensor_rgb_to_bgr_uint8,
    normalize_reference_mode,
    build_anyvisloc_reference_view,
    anyvisloc_sample_to_truepos,
    dumpRotateImage,
    retrieval_init,
    matching_init,
    retrieval_all_anyvisloc,
    Match2Pos_all_anyvisloc,
    pos2error_anyvisloc,
)

warnings.filterwarnings("ignore")


def parse_scenes(scenes):
    if scenes is None or len(scenes) == 0:
        return None
    if len(scenes) == 1 and "," in scenes[0]:
        return [x.strip() for x in scenes[0].split(",") if x.strip()]
    return scenes


def get_parse():
    parser = argparse.ArgumentParser(description="AnyVisLoc NPZ retrieve-match-PnP baseline")
    parser.add_argument("--dataset_root", required=True, type=str, help="Prepared AnyVisLoc NPZ root")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names, or omit for all scenes")
    parser.add_argument(
        "--reference_mode",
        default="aerial",
        choices=["aerial", "satellite"],
        help="Reference source from Lxx_reference.json: aerial or satellite.",
    )

    parser.add_argument("--yaml", default="config.yaml", type=str, help="Global baseline config yaml")
    parser.add_argument("--save_dir", default="./Result/AnyVisLoc", type=str, help="Output directory")
    parser.add_argument("--device", default="cuda", type=str, help="Inference device")
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Save retrieval and PnP match visualizations.",
    )
    visualization.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Do not save retrieval or PnP match visualizations.",
    )
    parser.set_defaults(visualize=False)
    parser.add_argument(
        "--pose_priori",
        default="yp",
        type=str,
        choices=["yp", "p", "unknown"],
        help="Use pose prior. yp rotates map by yaw and uses pitch/yaw/altitude for view-center shift.",
    )
    parser.add_argument("--strategy", default="Topn_opt", type=str, help="Top1; Topn_opt")
    parser.add_argument("--PnP_method", default="P3P", type=str, help="P3P, AP3P, EPNP")
    parser.add_argument("--resize_ratio", default=0.4, type=float, help="Resize UAV/reference patch before matching")
    parser.add_argument(
        "--selectable_code_dir",
        default=str(Path(__file__).resolve().parent / "Matching_Models" / "Sparse_matchers"),
        type=str,
        help="Directory containing selectable_sparse_matcher.py and its matcher dependencies.",
    )
    parser.add_argument(
        "--selectable_module",
        default="selectable_sparse_matcher",
        type=str,
        help="Python module name that provides selectable matcher init/extract/match functions.",
    )
    parser.add_argument(
        "--match_keypoints",
        default=3000,
        type=int,
        help="Base max keypoints for sparse matchers.",
    )
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
        help="Local checkpoint path for SP_LG_GIM.",
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
        help="Local checkpoint path for SP_LG_MINIMA.",
    )
    parser.add_argument(
        "--patch_scale",
        default=1.0,
        type=float,
        help="Physical retrieval patch scale. Default matches AnyVisLoc_open_npz_new.py.",
    )
    parser.add_argument(
        "--min_patch_size_m",
        default=0.0,
        type=float,
        help="Minimum physical retrieval patch size in meters. 0 disables this clamp.",
    )
    parser.add_argument(
        "--max_patch_size_m",
        default=0.0,
        type=float,
        help="Maximum physical retrieval patch size in meters. 0 disables this clamp.",
    )

    parser.add_argument("--limit", default=0, type=int, help="Only process first N samples. 0 means all")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose pkl result already exists, and read those pkl files back into summary.",
    )
    parser.add_argument(
        "--save_every",
        default=20,
        type=int,
        help="Flush combo CSV/JSON every N processed samples. 0 disables periodic full-file flushing.",
    )
    parser.add_argument(
        "--debug_raise",
        action="store_true",
        help="Re-raise the first per-sample exception after saving its traceback.",
    )
    parser.add_argument(
        "--draw_pnp_inlier_ratio",
        default=0.1,
        type=float,
        help="Randomly draw this ratio of PnP RANSAC inlier matches. 0.1 means 10%%.",
    )
    parser.add_argument(
        "--draw_pnp_inlier_seed",
        default=0,
        type=int,
        help="Random seed for sampled PnP inlier match visualization.",
    )
    parser.add_argument(
        "--ref_feature_cache_dir",
        default=None,
        type=str,
        help="Directory for cached reference/gallery retrieval features used when --pose_priori p. Default: <combo_dir>/_ref_feature_cache",
    )
    parser.add_argument(
        "--disable_ref_feature_cache",
        action="store_true",
        help="Disable reference/gallery feature cache even when --pose_priori p.",
    )
    parser.add_argument(
        "--success_thresholds",
        nargs="+",
        default=[1, 3, 5, 10, 20],
        type=float,
        help="Meter thresholds used for scene/global success-rate statistics and curves.",
    )
    parser.add_argument("--pnp_reproj_error", default=8.0, type=float)
    parser.add_argument("--pnp_iterations", default=2000, type=int)
    parser.add_argument("--pnp_confidence", default=0.999, type=float)
    parser.add_argument("--retrieval_k", default=5, type=int, help="K used by PDM@K and Recall@K")
    parser.add_argument("--pdm_lambda", default=6.0, type=float, help="Lambda in the PDM@K score")
    parser.add_argument("--pdm_alpha", default=0.9, type=float, help="Alpha in the PDM@K score")

    opt = parser.parse_args()
    opt.scenes = parse_scenes(opt.scenes)
    opt.reference_mode = normalize_reference_mode(opt.reference_mode)
    opt.draw_pnp_inlier_ratio = max(0.0, min(1.0, float(opt.draw_pnp_inlier_ratio)))
    opt.success_thresholds = sorted({float(x) for x in opt.success_thresholds if float(x) > 0})
    if not opt.success_thresholds:
        opt.success_thresholds = [1.0, 3.0, 5.0, 10.0, 20.0]
    opt.retrieval_k = max(1, int(opt.retrieval_k))
    return opt


def _as_text(x):
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


def _safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _retrieval_metrics_from_ratios(ratios, k=5, pdm_lambda=6.0, pdm_alpha=0.9):
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

    # Ratios are stored in retrieval-result order. The nearest gallery tile is
    # the ground truth, so its one-based position is its rank in this list.
    gt_rank = int(np.argmin(ratios)) + 1
    recall_at_k = float(gt_rank <= k_eff)
    return pdm_at_k, recall_at_k, gt_rank


def _append_jsonl(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _reset_jsonl(path):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("", encoding="utf-8")


def _load_pickle_dict(path):
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in pkl, got {type(obj)}")
    return obj


def _make_result_row(
    *,
    scene_name,
    sample_id,
    npz_path,
    reference_mode,
    retrieval_method,
    matching_method,
    pose_priori,
    truePos,
    pred_loc,
    pred_error,
    pnp_success,
    best_inliers,
    num_retrieval_candidates,
    retrieval_time,
    total_time,
    retrieval_k,
    pdm_at_k,
    recall_at_k,
    retrieval_gt_rank,
    result_pkl,
    source,
):
    pred_loc = pred_loc or {}
    return {
        "scene_name": _as_text(scene_name),
        "sample_id": _as_text(sample_id),
        "npz_path": _as_text(npz_path),
        "reference_mode": reference_mode,
        "retrieval_method": retrieval_method,
        "matching_method": matching_method,
        "pose_priori": pose_priori,
        "gt_x": _safe_float(truePos.get("x")),
        "gt_y": _safe_float(truePos.get("y")),
        "gt_z": _safe_float(truePos.get("z")),
        "pred_x": _safe_float(pred_loc.get("x")),
        "pred_y": _safe_float(pred_loc.get("y")),
        "pred_z": _safe_float(pred_loc.get("z")),
        "pred_error_m": _safe_float(pred_error) if pnp_success else None,
        "pnp_success": bool(pnp_success),
        "best_inliers": _safe_int(best_inliers),
        "num_retrieval_candidates": _safe_int(num_retrieval_candidates),
        "retrieval_time": _safe_float(retrieval_time),
        "total_time": _safe_float(total_time),
        "retrieval_k": _safe_int(retrieval_k),
        "pdm_at_k": _safe_float(pdm_at_k),
        "recall_at_k": _safe_float(recall_at_k),
        "retrieval_gt_rank": _safe_int(retrieval_gt_rank, default=None),
        "result_pkl": str(result_pkl),
        "source": source,
    }


def _make_row_from_existing_pkl(
    *,
    pkl_path,
    sample,
    reference_mode,
    retrieval_method,
    matching_method,
    pose_priori,
    retrieval_k,
    pdm_lambda,
    pdm_alpha,
):
    data = _load_pickle_dict(pkl_path)

    truePos = data.get("truePos")
    if truePos is None:
        truePos = anyvisloc_sample_to_truepos(sample)

    pred_loc = data.get("pred_loc", {"x": None, "y": None, "z": None})
    pred_error = data.get("pred_error", None)
    inliers_list = data.get("inliers", [])
    best_inliers = max(inliers_list) if len(inliers_list) else 0

    # New pkl files save pnp_success explicitly. Older pkl files are inferred.
    pnp_success = data.get("pnp_success", None)
    if pnp_success is None:
        pnp_success = (
            pred_loc is not None
            and pred_loc.get("x") is not None
            and pred_loc.get("y") is not None
            and _safe_float(pred_error) is not None
            and int(best_inliers) > 0
        )
        # Backward compatibility with the old invalid sentinel.
        if _safe_float(pred_error) is not None and float(pred_error) >= 9999.0:
            pnp_success = False

    row_starts = data.get("row_starts", data.get("refLocX", []))
    num_retrieval_candidates = len(row_starts) if hasattr(row_starts, "__len__") else 0
    pdm_at_k = data.get("pdm_at_k")
    recall_at_k = data.get("recall_at_k")
    retrieval_gt_rank = data.get("retrieval_gt_rank")
    metric_config_matches = (
        _safe_int(data.get("retrieval_k"), 0) == int(retrieval_k)
        and _safe_float(data.get("pdm_lambda")) == float(pdm_lambda)
        and _safe_float(data.get("pdm_alpha")) == float(pdm_alpha)
    )
    if (
        pdm_at_k is None
        or recall_at_k is None
        or retrieval_gt_rank is None
        or not metric_config_matches
    ):
        raise ValueError(
            "Existing result predates the complete PDM@K/Recall@K implementation; recomputation is required."
        )

    return _make_result_row(
        scene_name=data.get("scene_name", sample.get("scene_name", "")),
        sample_id=data.get("sample_id", sample.get("sample_id", "")),
        npz_path=data.get("npz_path", sample.get("npz_path", "")),
        reference_mode=reference_mode,
        retrieval_method=retrieval_method,
        matching_method=matching_method,
        pose_priori=pose_priori,
        truePos=truePos,
        pred_loc=pred_loc,
        pred_error=pred_error,
        pnp_success=bool(pnp_success),
        best_inliers=best_inliers,
        num_retrieval_candidates=num_retrieval_candidates,
        retrieval_time=data.get("retrieval_time", None),
        total_time=data.get("total_time", None),
        retrieval_k=retrieval_k,
        pdm_at_k=pdm_at_k,
        recall_at_k=recall_at_k,
        retrieval_gt_rank=retrieval_gt_rank,
        result_pkl=pkl_path,
        source="existing_pkl",
    )


def _make_failure_row(
    *,
    sample,
    scene_name,
    sample_id,
    reference_mode,
    retrieval_method,
    matching_method,
    error,
    tb,
):
    return {
        "scene_name": _as_text(scene_name),
        "sample_id": _as_text(sample_id),
        "npz_path": _as_text(sample.get("npz_path", "")),
        "reference_mode": reference_mode,
        "retrieval_method": retrieval_method,
        "matching_method": matching_method,
        "error": repr(error),
        "traceback": tb,
    }


def _format_threshold_for_key(threshold):
    threshold = float(threshold)
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def _valid_error_array(rows):
    values = []
    for r in rows:
        if not bool(r.get("pnp_success", False)):
            continue
        err = _safe_float(r.get("pred_error_m"))
        if err is not None:
            values.append(float(err))
    return np.asarray(values, dtype=np.float64)


def _success_rate_dict(rows, failed_rows, thresholds):
    attempted = len(rows) + len(failed_rows)
    denom = max(attempted, 1)
    errors = _valid_error_array(rows)
    out = {}
    for t in thresholds:
        key = _format_threshold_for_key(t)
        out[f"success_rate_{key}m"] = float(np.sum(errors <= float(t)) / denom) if attempted else 0.0
        out[f"recall_{key}m"] = out[f"success_rate_{key}m"]
    return out


def _retrieval_metric_summary(rows, failed_rows):
    attempted = len(rows) + len(failed_rows)
    pdm_values = [
        float(v)
        for v in (_safe_float(r.get("pdm_at_k")) for r in rows)
        if v is not None
    ]
    recall_values = [
        float(v)
        for v in (_safe_float(r.get("recall_at_k")) for r in rows)
        if v is not None
    ]
    k_values = [_safe_int(r.get("retrieval_k"), 0) for r in rows]
    retrieval_k = max(k_values) if k_values else 0
    denominator = max(attempted, 1)

    # Retrieval failures contribute zero, matching evaluation over all queries.
    pdm = float(np.sum(pdm_values) / denominator) if attempted else None
    recall = float(np.sum(recall_values) / denominator) if attempted else None
    key_suffix = str(retrieval_k) if retrieval_k > 0 else "K"
    return {
        "retrieval_k": int(retrieval_k),
        "retrieval_metric_valid_rows": int(min(len(pdm_values), len(recall_values))),
        "pdm_at_k": pdm,
        "recall_at_k": recall,
        f"PDM@{key_suffix}": pdm,
        f"Recall@{key_suffix}": recall,
    }


def _compute_summary(rows, failed_rows, *, dataset_root, scenes, reference_mode, num_samples_requested, thresholds=None, extra=None):
    thresholds = thresholds or [1.0, 3.0, 5.0, 10.0, 20.0]
    attempted = len(rows) + len(failed_rows)
    errors = _valid_error_array(rows)

    num_pnp_success = int(errors.size)
    num_pnp_failed = len(rows) - num_pnp_success
    num_exception_failed = len(failed_rows)
    failure_count = num_pnp_failed + num_exception_failed
    denom = max(attempted, 1)

    summary = {
        "dataset_root": dataset_root,
        "scenes": scenes,
        "reference_mode": reference_mode,
        "num_samples_requested": int(num_samples_requested),
        "num_attempted_rows": int(attempted),
        "num_result_rows": int(len(rows)),
        "num_exception_failed_rows": int(num_exception_failed),
        "num_pnp_success_rows": int(num_pnp_success),
        "num_pnp_failed_rows": int(num_pnp_failed),
        "failure_rate": float(failure_count / denom),
        "mean_error_m_success_only": float(np.mean(errors)) if errors.size else None,
        "median_error_m_success_only": float(np.median(errors)) if errors.size else None,
        "success_thresholds_m": [float(x) for x in thresholds],
    }
    summary.update(_success_rate_dict(rows, failed_rows, thresholds))
    summary.update(_retrieval_metric_summary(rows, failed_rows))
    if extra:
        summary.update(extra)
    return summary


def _plot_success_curve(rows, failed_rows, thresholds, out_png, title):
    """Plot localization success rate versus error threshold.

    PnP failures and exception failures are counted as localization failures
    in the denominator.
    """
    out_png = Path(out_png)
    ensure_dir(out_png.parent)

    attempted = len(rows) + len(failed_rows)
    denom = max(attempted, 1)
    errors = _valid_error_array(rows)
    rates = [float(np.sum(errors <= float(t)) / denom * 100.0) if attempted else 0.0 for t in thresholds]

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(7.0, 4.5))
    ax = plt.gca()
    ax.plot(thresholds, rates, marker="o", linewidth=2)
    ax.set_xlabel("Error threshold (m)")
    ax.set_ylabel("Localization success rate (%)")
    ax.set_title(title)
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)

    for x, y in zip(thresholds, rates):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    footer = f"Attempted: {attempted}, PnP success: {len(errors)}, failed: {attempted - len(errors)}"
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)

    return {
        "thresholds_m": [float(x) for x in thresholds],
        "success_rates_percent": rates,
        "attempted": int(attempted),
        "pnp_success": int(len(errors)),
        "failed": int(attempted - len(errors)),
        "plot": str(out_png),
    }


def _save_scene_and_overall_stats(out_dir, rows, failed_rows, thresholds, title_prefix):
    """Save per-scene and overall success-rate tables and curves."""
    out_dir = Path(out_dir)
    stats_dir = out_dir / "scene_success_stats"
    ensure_dir(stats_dir)

    scene_names = sorted(
        set([_as_text(r.get("scene_name", "")) for r in rows])
        | set([_as_text(r.get("scene_name", "")) for r in failed_rows])
    )

    scene_stats = []
    for scene_name in scene_names:
        scene_rows = [r for r in rows if _as_text(r.get("scene_name", "")) == scene_name]
        scene_failed = [r for r in failed_rows if _as_text(r.get("scene_name", "")) == scene_name]

        scene_summary = _compute_summary(
            scene_rows,
            scene_failed,
            dataset_root="",
            scenes=[scene_name],
            reference_mode="",
            num_samples_requested=len(scene_rows) + len(scene_failed),
            thresholds=thresholds,
            extra={"scene_name": scene_name},
        )

        plot_info = _plot_success_curve(
            scene_rows,
            scene_failed,
            thresholds,
            stats_dir / scene_name / "success_curve.png",
            title=f"{title_prefix} | {scene_name}",
        )
        scene_summary.update({
            "success_curve_png": plot_info["plot"],
            "success_rates_percent": plot_info["success_rates_percent"],
        })
        scene_stats.append(scene_summary)

        save_json(stats_dir / scene_name / "summary.json", scene_summary)

    overall_plot_info = _plot_success_curve(
        rows,
        failed_rows,
        thresholds,
        out_dir / "success_curve_overall.png",
        title=f"{title_prefix} | Overall",
    )

    save_json(stats_dir / "scene_success_stats.json", scene_stats)
    save_csv(scene_stats, stats_dir / "scene_success_stats.csv")
    save_json(out_dir / "success_curve_overall.json", overall_plot_info)

    pdm_scene_values = [
        _safe_float(item.get("pdm_at_k"))
        for item in scene_stats
        if _safe_float(item.get("pdm_at_k")) is not None
    ]
    recall_scene_values = [
        _safe_float(item.get("recall_at_k"))
        for item in scene_stats
        if _safe_float(item.get("recall_at_k")) is not None
    ]
    retrieval_k = max([_safe_int(item.get("retrieval_k"), 0) for item in scene_stats] or [0])
    key_suffix = str(retrieval_k) if retrieval_k > 0 else "K"
    pdm_norm = float(np.mean(pdm_scene_values)) if pdm_scene_values else None
    recall_norm = float(np.mean(recall_scene_values)) if recall_scene_values else None
    retrieval_metrics = {
        "retrieval_k": int(retrieval_k),
        "num_regions": int(len(scene_stats)),
        "normalization": "equal mean over regions; each region is the mean over its UAV queries",
        "pdm_at_k_norm": pdm_norm,
        "recall_at_k_norm": recall_norm,
        f"PDM@{key_suffix}_norm": pdm_norm,
        f"Recall@{key_suffix}_norm": recall_norm,
        "per_scene": [
            {
                "scene_name": item.get("scene_name"),
                "num_queries": item.get("num_attempted_rows"),
                "pdm_at_k": item.get("pdm_at_k"),
                "recall_at_k": item.get("recall_at_k"),
            }
            for item in scene_stats
        ],
    }
    save_json(out_dir / "retrieval_metrics.json", retrieval_metrics)
    save_csv(retrieval_metrics["per_scene"], out_dir / "retrieval_metrics_by_scene.csv")

    return {
        "scene_stats_dir": str(stats_dir),
        "overall_success_curve_png": overall_plot_info["plot"],
        "scene_count": len(scene_names),
        "pdm_at_k_norm": pdm_norm,
        "recall_at_k_norm": recall_norm,
        f"PDM@{key_suffix}_norm": pdm_norm,
        f"Recall@{key_suffix}_norm": recall_norm,
        "retrieval_metrics_json": str(out_dir / "retrieval_metrics.json"),
    }


def _flush_outputs(out_dir, rows, failed_rows, summary):
    out_dir = Path(out_dir)
    save_json(out_dir / "all_results.json", rows)
    save_csv(rows, out_dir / "all_results.csv")
    save_json(out_dir / "failed.json", failed_rows)
    save_json(out_dir / "summary.json", summary)


def main():
    opt = get_parse()
    with open(opt.yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["DEVICE"] = opt.device

    all_retrieval = config["RETRIEVAL_METHODS"]
    all_matching = config["MATCHING_METHODS"]

    dataset = AnyVisLocNPZDataset(
        root=opt.dataset_root,
        scenes=opt.scenes,
        # Reference maps are now PNG/NPY + JSON and should be loaded lazily
        # by build_anyvisloc_reference_view(), not inside dataset[idx].
        load_reference=False,
        cache_reference=True,
        recursive=True,
    )
    indices = list(range(len(dataset)))
    if opt.limit and opt.limit > 0:
        indices = indices[:opt.limit]

    scene_count = "all" if opt.scenes is None else f"{len(opt.scenes)}scenes"
    run_tag = f"{datetime.now().strftime('%Y%m%d-%H%M')}_{scene_count}"
    run_root = Path(opt.save_dir) / run_tag / opt.reference_mode

    ensure_dir(run_root)

    global_results_jsonl = run_root / "all_results.jsonl"
    global_failed_jsonl = run_root / "failed.jsonl"
    _reset_jsonl(global_results_jsonl)
    _reset_jsonl(global_failed_jsonl)

    global_rows = []
    global_failed_rows = []
    total_tasks = len(indices) * len(all_retrieval) * len(all_matching)
    overall_progress = tqdm(total=total_tasks, desc="Overall Progress", unit="img", position=0)
    retrieval_progress = tqdm(total=total_tasks, desc="Retrieval Progress", unit="img", position=1)

    for retrieval_method in all_retrieval:
        method_dict = {"retrieval_method": retrieval_method}
        method_dict = retrieval_init(method_dict, config)

        for matching_method in all_matching:
            method_dict["matching_method"] = matching_method
            method_dict = matching_init(method_dict, opt, config)
            combo_name = f"{opt.reference_mode}-{retrieval_method}-{matching_method}-{opt.pose_priori}"
            combo_dir = run_root / combo_name
            ensure_dir(combo_dir)

            combo_results_jsonl = combo_dir / "all_results.jsonl"
            combo_failed_jsonl = combo_dir / "failed.jsonl"
            _reset_jsonl(combo_results_jsonl)
            _reset_jsonl(combo_failed_jsonl)

            combo_rows = []
            combo_failed_rows = []
            processed_in_combo = 0

            for idx in indices:
                sample = dataset[idx]
                sample_id = _as_text(sample["sample_id"])
                scene_name = _as_text(sample["scene_name"])
                retrieval_finished = False
                save_path = combo_dir / scene_name / sample_id
                ensure_dir(save_path)

                pkl_path = save_path / f"VG_data_{sample_id}.pkl"

                if opt.skip_existing and pkl_path.exists():
                    try:
                        row = _make_row_from_existing_pkl(
                            pkl_path=pkl_path,
                            sample=sample,
                            reference_mode=opt.reference_mode,
                            retrieval_method=retrieval_method,
                            matching_method=matching_method,
                            pose_priori=opt.pose_priori,
                            retrieval_k=opt.retrieval_k,
                            pdm_lambda=opt.pdm_lambda,
                            pdm_alpha=opt.pdm_alpha,
                        )
                        combo_rows.append(row)
                        global_rows.append(row)
                        _append_jsonl(combo_results_jsonl, row)
                        _append_jsonl(global_results_jsonl, row)
                        processed_in_combo += 1
                        retrieval_progress.update(1)
                        overall_progress.update(1)

                        if opt.save_every > 0 and processed_in_combo % opt.save_every == 0:
                            combo_summary = _compute_summary(
                                combo_rows,
                                combo_failed_rows,
                                dataset_root=opt.dataset_root,
                                scenes=opt.scenes,
                                reference_mode=opt.reference_mode,
                                num_samples_requested=len(indices),
                                thresholds=opt.success_thresholds,
                                extra={
                                    "combo_name": combo_name,
                                    "retrieval_method": retrieval_method,
                                    "matching_method": matching_method,
                                    "pose_priori": opt.pose_priori,
                                },
                            )
                            _flush_outputs(combo_dir, combo_rows, combo_failed_rows, combo_summary)
                        continue
                    except Exception:
                        pass

                try:
                    # Used only to build readable and safe v5 reference-feature cache names.
                    opt._current_scene_name = scene_name
                    opt._current_sample_id = sample_id

                    truePos = anyvisloc_sample_to_truepos(sample)
                    K = sample["K"].detach().cpu().numpy().astype(np.float32)
                    dist = sample["dist"].detach().cpu().numpy().astype(np.float32)
                    uav_image = tensor_rgb_to_bgr_uint8(sample["image"])

                    view = build_anyvisloc_reference_view(sample, opt.reference_mode)
                    ref_map0 = view["map"]
                    dsm_map0 = view["dsm"]
                    map_resolution = view["map_resolution"]
                    dsm_resolution = view["dsm_resolution"]
                    map_origin_local = view["map_origin_local"]
                    dsm_origin_local = view["dsm_origin_local"]

                    # Rotate the selected reference map by yaw, matching the original baseline behavior.
                    if opt.pose_priori == "yp":
                        ref_map, matRotation = dumpRotateImage(ref_map0, truePos["yaw"])
                    else:
                        ref_map = ref_map0
                        matRotation = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

                    t0 = time.time()

                    (
                        IR_order,
                        row_starts,
                        col_starts,
                        PDE_list,
                        patch_h,
                        patch_w,
                        fineScale,
                        retrieval_time,
                    ) = retrieval_all_anyvisloc(
                        ref_map,
                        uav_image,
                        truePos,
                        map_resolution,
                        map_origin_local,
                        matRotation,
                        str(save_path),
                        opt,
                        config,
                        method_dict,
                    )
                    retrieval_finished = True
                    retrieval_progress.update(1)
                    pdm_at_k, recall_at_k, retrieval_gt_rank = _retrieval_metrics_from_ratios(
                        PDE_list,
                        k=opt.retrieval_k,
                        pdm_lambda=opt.pdm_lambda,
                        pdm_alpha=opt.pdm_alpha,
                    )

                    XYZ_list, inliers_list, match_time, pnp_time = Match2Pos_all_anyvisloc(
                        opt,
                        config,
                        uav_image,
                        fineScale,
                        K,
                        ref_map,
                        dsm_map0,
                        row_starts,
                        col_starts,
                        patch_h,
                        patch_w,
                        str(save_path),
                        method_dict,
                        matRotation,
                        map_resolution,
                        map_origin_local,
                        dsm_resolution,
                        dsm_origin_local,
                        dist=dist,
                        truePos=truePos,
                    )

                    (
                        pred_loc,
                        pred_error,
                        location_error_list,
                        pnp_success,
                        best_index,
                    ) = pos2error_anyvisloc(truePos, XYZ_list, inliers_list)

                    total_time = time.time() - t0
                    best_inliers = int(max(inliers_list) if len(inliers_list) else 0)
                    if pnp_success:
                        tqdm.write(f"Localization Error {scene_name}/{sample_id}: {pred_error:.1f} m")
                    else:
                        tqdm.write(f"Localization Error {scene_name}/{sample_id}: PnP failed")

                    save_data(
                        str(pkl_path),
                        opt=opt,
                        sample_id=sample_id,
                        scene_name=scene_name,
                        npz_path=_as_text(sample.get("npz_path", "")),
                        reference_info={k: v for k, v in view.items() if k not in ("map", "dsm")},
                        truePos=truePos,
                        pred_loc=pred_loc,
                        pred_error=pred_error,
                        pnp_success=bool(pnp_success),
                        best_index=best_index,
                        row_starts=row_starts,
                        col_starts=col_starts,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        IR_order=IR_order,
                        PDE=PDE_list,
                        R_i=PDE_list,
                        retrieval_k=opt.retrieval_k,
                        pdm_lambda=opt.pdm_lambda,
                        pdm_alpha=opt.pdm_alpha,
                        pdm_at_k=pdm_at_k,
                        recall_at_k=recall_at_k,
                        retrieval_gt_rank=retrieval_gt_rank,
                        inliers=inliers_list,
                        XYZ_list=XYZ_list,
                        location_error_list=location_error_list,
                        retrieval_time=retrieval_time,
                        match_time=match_time,
                        pnp_time=pnp_time,
                        total_time=total_time,
                    )

                    row = _make_result_row(
                        scene_name=scene_name,
                        sample_id=sample_id,
                        npz_path=sample.get("npz_path", ""),
                        reference_mode=opt.reference_mode,
                        retrieval_method=retrieval_method,
                        matching_method=matching_method,
                        pose_priori=opt.pose_priori,
                        truePos=truePos,
                        pred_loc=pred_loc,
                        pred_error=pred_error,
                        pnp_success=bool(pnp_success),
                        best_inliers=best_inliers,
                        num_retrieval_candidates=len(row_starts),
                        retrieval_time=retrieval_time,
                        total_time=total_time,
                        retrieval_k=opt.retrieval_k,
                        pdm_at_k=pdm_at_k,
                        recall_at_k=recall_at_k,
                        retrieval_gt_rank=retrieval_gt_rank,
                        result_pkl=pkl_path,
                        source="computed",
                    )

                    combo_rows.append(row)
                    global_rows.append(row)
                    _append_jsonl(combo_results_jsonl, row)
                    _append_jsonl(global_results_jsonl, row)

                except Exception as e:
                    tb = traceback.format_exc()
                    err = _make_failure_row(
                        sample=sample,
                        scene_name=scene_name,
                        sample_id=sample_id,
                        reference_mode=opt.reference_mode,
                        retrieval_method=retrieval_method,
                        matching_method=matching_method,
                        error=e,
                        tb=tb,
                    )
                    combo_failed_rows.append(err)
                    global_failed_rows.append(err)
                    _append_jsonl(combo_failed_jsonl, err)
                    _append_jsonl(global_failed_jsonl, err)

                    tqdm.write(f"Localization Error {scene_name}/{sample_id}: failed")
                    if opt.debug_raise:
                        raise

                if not retrieval_finished:
                    retrieval_progress.update(1)
                processed_in_combo += 1
                overall_progress.update(1)
                if opt.save_every > 0 and processed_in_combo % opt.save_every == 0:
                    combo_summary = _compute_summary(
                        combo_rows,
                        combo_failed_rows,
                        dataset_root=opt.dataset_root,
                        scenes=opt.scenes,
                        reference_mode=opt.reference_mode,
                        num_samples_requested=len(indices),
                        extra={
                            "combo_name": combo_name,
                            "retrieval_method": retrieval_method,
                            "matching_method": matching_method,
                            "pose_priori": opt.pose_priori,
                        },
                    )
                    _flush_outputs(combo_dir, combo_rows, combo_failed_rows, combo_summary)

            combo_summary = _compute_summary(
                combo_rows,
                combo_failed_rows,
                dataset_root=opt.dataset_root,
                scenes=opt.scenes,
                reference_mode=opt.reference_mode,
                num_samples_requested=len(indices),
                thresholds=opt.success_thresholds,
                extra={
                    "combo_name": combo_name,
                    "retrieval_method": retrieval_method,
                    "matching_method": matching_method,
                    "pose_priori": opt.pose_priori,
                },
            )
            _flush_outputs(combo_dir, combo_rows, combo_failed_rows, combo_summary)
            scene_stats_info = _save_scene_and_overall_stats(
                combo_dir,
                combo_rows,
                combo_failed_rows,
                opt.success_thresholds,
                title_prefix=combo_name,
            )
            combo_summary.update(scene_stats_info)
            save_json(combo_dir / "summary.json", combo_summary)

    global_summary = _compute_summary(
        global_rows,
        global_failed_rows,
        dataset_root=opt.dataset_root,
        scenes=opt.scenes,
        reference_mode=opt.reference_mode,
        num_samples_requested=len(indices),
        thresholds=opt.success_thresholds,
        extra={
            "num_retrieval_methods": len(all_retrieval),
            "num_matching_methods": len(all_matching),
            "num_combos": len(all_retrieval) * len(all_matching),
        },
    )

    global_scene_stats_info = _save_scene_and_overall_stats(
        run_root,
        global_rows,
        global_failed_rows,
        opt.success_thresholds,
        title_prefix=f"{opt.reference_mode}-global",
    )
    global_summary.update(global_scene_stats_info)

    save_json(run_root / "summary.json", global_summary)
    save_csv(global_rows, run_root / "all_results.csv")
    save_json(run_root / "all_results.json", global_rows)
    save_json(run_root / "failed.json", global_failed_rows)
    retrieval_progress.close()
    overall_progress.close()


if __name__ == "__main__":
    main()
