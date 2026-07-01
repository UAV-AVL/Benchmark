#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast dataloader for the sanitized AnyVisLoc NPZ format.

Expected new reference layout per scene folder:
  Scene_01/
    L01_reference.json
    aerial_map.png
    aerial_dsm.npy
    satellite_map.png
    satellite_dsm.npy
    L01_0001.npz
    L01_0002.npz

"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _as_string(x) -> str:
    if isinstance(x, str):
        return x
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


def _normalize_scenes(scenes: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if scenes is None:
        return None
    if isinstance(scenes, str):
        if len(scenes.strip()) == 0:
            return None
        return [x.strip() for x in scenes.split(",") if x.strip()]
    return [str(x).strip() for x in scenes if str(x).strip()]


def _read_image_from_relpath(sample_path: Path, relpath: str) -> np.ndarray:
    img_path = sample_path.parent / relpath
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image_relpath: {img_path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class AnyVisLocNPZDataset(Dataset):
    """
    Each item returns:
      image             FloatTensor [3,H,W], RGB normalized to [0,1]
      K                 FloatTensor [3,3], intrinsics after UAV downsampling
      K_original        FloatTensor [3,3], fallback equals K if absent
      dist              FloatTensor [5]
      uav_downsample    float
      original_size     IntTensor [2], fallback equals image_size if absent
      image_size        IntTensor [2], [H, W]
      pose_c2w          FloatTensor [4,4]
      pose_w2c          FloatTensor [4,4]
      xyz               FloatTensor [3]
      euler_deg         FloatTensor [3]
      reference         dict with ref_json_path/ref_path only by default
      sample_id         str
      scene_name        str, normally Scene_XX
      scene_id          int
      npz_path          str
    """

    def __init__(
        self,
        root: str,
        scenes: Optional[Union[str, Sequence[str]]] = None,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        load_reference: bool = False,
        cache_reference: bool = True,
        recursive: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.transform = transform
        self.load_reference = bool(load_reference)
        self.cache_reference = bool(cache_reference)
        self.recursive = bool(recursive)
        self._ref_cache: Dict[str, Dict] = {}

        scene_list = _normalize_scenes(scenes)
        wanted = set(scene_list) if scene_list is not None else None

        self.samples: List[Path] = []
        self.sample_scene: List[str] = []
        self.scene_dirs: Dict[str, Path] = {}

        if recursive:
            # First find scene folders through the lightweight reference JSON.
            # This avoids scanning and probing every sample's parents.
            scene_dirs = []
            seen = set()
            for ref_json in sorted(self.root.rglob("L??_reference.json")):
                scene_dir = ref_json.parent
                scene_name = scene_dir.name
                if wanted is not None and scene_name not in wanted:
                    continue
                key = str(scene_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    scene_dirs.append(scene_dir)

            # Fallback for old NPZ reference layout or partially converted data.
            if not scene_dirs:
                for ref_npz in sorted(self.root.rglob("L??_RRF.npz")):
                    scene_dir = ref_npz.parent
                    scene_name = scene_dir.name
                    if wanted is not None and scene_name not in wanted:
                        continue
                    key = str(scene_dir.resolve())
                    if key not in seen:
                        seen.add(key)
                        scene_dirs.append(scene_dir)
        else:
            scene_dirs = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
            if wanted is not None:
                scene_dirs = [p for p in scene_dirs if p.name in wanted]

        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            sample_files = sorted(scene_dir.glob("L??_????.npz"))
            for p in sample_files:
                self.samples.append(p)
                self.sample_scene.append(scene_name)
            if sample_files:
                self.scene_dirs[scene_name] = scene_dir

        # Last fallback: direct sample scan, useful for legacy data.
        if not self.samples:
            candidate_files = sorted(self.root.rglob("L??_????.npz")) if recursive else []
            for p in candidate_files:
                scene_dir = p.parent
                scene_name = scene_dir.name
                if wanted is not None and scene_name not in wanted:
                    continue
                self.samples.append(p)
                self.sample_scene.append(scene_name)
                self.scene_dirs[scene_name] = scene_dir

        if not self.samples:
            msg = (
                f"No sample NPZ files found under {self.root}. Expected files like "
                "/root/Scene_01/L01_0001.npz with /root/Scene_01/L01_reference.json."
            )
            if scene_list is not None:
                msg += f" Scene filter was: {scene_list}."
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def _to_image_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        if self.transform is not None:
            return self.transform(image_rgb)
        image = image_rgb.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()

    def _reference_stub(self, scene_name: str, scene_id: int) -> Dict:
        scene_dir = self.scene_dirs.get(scene_name, self.root / scene_name)
        ref_json = scene_dir / f"L{scene_id:02d}_reference.json"
        ref_npz = scene_dir / f"L{scene_id:02d}_RRF.npz"
        return {
            "ref_json_path": str(ref_json),
            # ref_path is kept as a compatibility alias used by older helpers.
            "ref_path": str(ref_json if ref_json.exists() else ref_npz),
            "scene_dir": str(scene_dir),
        }

    def _load_reference(self, scene_name: str, scene_id: int) -> Dict:
        # Keep this lightweight. The heavy PNG/NPY arrays are loaded by
        # build_anyvisloc_reference_view() only for the requested reference mode.
        if scene_name in self._ref_cache:
            return self._ref_cache[scene_name]
        out = self._reference_stub(scene_name, scene_id)
        if self.cache_reference:
            self._ref_cache[scene_name] = out
        return out

    def __getitem__(self, idx: int) -> Dict:
        path = self.samples[idx]
        scene_name = self.sample_scene[idx]
        s = _load_npz(path)

        scene_id = int(s["scene_id"])
        image = s["image"]
        if image.size == 0:
            relpath = _as_string(s.get("image_relpath", ""))
            image = _read_image_from_relpath(path, relpath)

        image_size_np = (
            s["image_size"].astype(np.int32)
            if "image_size" in s
            else s["downsampled_size"].astype(np.int32)
            if "downsampled_size" in s
            else np.asarray(image.shape[:2], dtype=np.int32)
        )
        image_size = torch.from_numpy(image_size_np)
        original_size = (
            torch.from_numpy(s["original_size"].astype(np.int32))
            if "original_size" in s
            else image_size.clone()
        )
        uav_downsample = (
            float(s["uav_downsample"])
            if "uav_downsample" in s
            else float(s["downsample_factor"])
            if "downsample_factor" in s
            else 1.0
        )

        K = s["K"].astype(np.float32)
        item = {
            "image": self._to_image_tensor(image),
            "K": torch.from_numpy(K),
            "dist": torch.from_numpy(s["dist"].astype(np.float32)),
            "uav_downsample": uav_downsample,
            "original_size": original_size,
            "image_size": image_size,
            "pose_c2w": torch.from_numpy(s["pose_c2w"].astype(np.float32)),
            "pose_w2c": torch.from_numpy(s["pose_w2c"].astype(np.float32)),
            "xyz": torch.from_numpy(s["xyz"].astype(np.float32)),
            "euler_deg": torch.from_numpy(s["euler_deg"].astype(np.float32)),
            "sample_id": _as_string(s["sample_id"]),
            "scene_id": scene_id,
            "scene_name": scene_name,
            "npz_path": str(path),
        }

        # Always provide a lightweight reference pointer. load_reference only
        # controls whether this pointer is cached, not whether large arrays are read.
        item["reference"] = self._load_reference(scene_name, scene_id) if self.load_reference else self._reference_stub(scene_name, scene_id)
        return item


def make_dataloader(
    root: str,
    scenes: Optional[Union[str, Sequence[str]]] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    dataset = AnyVisLocNPZDataset(root=root, scenes=scenes, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--load-reference", action="store_true")
    args = parser.parse_args()

    ds = AnyVisLocNPZDataset(args.root, scenes=args.scenes, load_reference=args.load_reference)
    print(f"Loaded {len(ds)} samples")
    item = ds[0]
    for k, v in item.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, type(v), v)
