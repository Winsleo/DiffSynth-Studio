"""在 train_a2v.py 启动时 monkey-patch UnifiedDataset,让 cache mode 一次性预加载所有 .pth 到内存。

用法:在调 train_a2v.py 之前 PYTHONPATH 加这个文件路径,或者在 train_a2v.py 里 import。

原理:
- 替换 UnifiedDataset.search_for_cached_data_files → 把 pth 路径存到 self.cached_data(原有)
- 替换 UnifiedDataset.__getitem__:在第一次调用时全部 torch.load 到 self.cached_data_tensors,然后 [data_id] 直接索引
- 避免 250 MB pth lazy load 慢

大 cache 模式 (>A2V_PRELOAD_MAX_GB GB):跳过 preload,fallback 到原始 lazy LoadTorchPickle
  - Phase 1 (50 sample × 290 MB ≈ 14 GB) → preload OK
  - Phase 2 (8404 sample × 290 MB ≈ 2.4 TB) → 单 node 1 TB RAM 不够,必须 skip
"""

from __future__ import annotations

import os
import torch


# 默认阈值:超过这个总大小就 skip preload,fallback 到原始 lazy LoadTorchPickle
# (单 node ~1 TB RAM,8 GPU 共享 → 安全阈值 100 GB)
_DEFAULT_MAX_GB = float(os.environ.get("A2V_PRELOAD_MAX_GB", "100"))
# 强制 skip(1)/强制 preload(0)/自动按大小判断(默认)
_DISABLE = os.environ.get("A2V_PRELOAD_DISABLE", "0") == "1"
_FORCE = os.environ.get("A2V_PRELOAD_FORCE", "0") == "1"


def _estimate_total_size_gb(paths):
    """采样前 5 个 pth 估算平均大小,然后乘总文件数。返回估算的 GB 数。"""
    if not paths:
        return 0.0
    sample = paths[: min(5, len(paths))]
    sizes = [os.path.getsize(p) for p in sample]
    avg = sum(sizes) / len(sizes)
    return (avg * len(paths)) / (1024 ** 3)


def install_preload_patch() -> None:
    """Monkey-patch UnifiedDataset + cached_data_operator(LoadTorchPickle)。"""
    from diffsynth.core.data.unified_dataset import UnifiedDataset
    from diffsynth.core.data.operators import LoadTorchPickle

    original_search = UnifiedDataset.search_for_cached_data_files
    original_getitem = UnifiedDataset.__getitem__

    def patched_search(self, path):
        original_search(self, path)
        # 在 search 时就估算大小,后续 __getitem__ 用得上
        try:
            self._preload_estimated_gb = _estimate_total_size_gb(self.cached_data)
        except Exception:
            self._preload_estimated_gb = 0.0

    def patched_getitem(self, data_id):
        if not self.load_from_cache:
            return original_getitem(self, data_id)
        # 决定是否 preload
        if _DISABLE:
            return original_getitem(self, data_id)
        estimated_gb = getattr(self, "_preload_estimated_gb", 0.0)
        if not _FORCE and estimated_gb > _DEFAULT_MAX_GB:
            # 大 cache:跳过 preload,直接 lazy(避免 OOM)
            if not getattr(self, "_preload_warned", False):
                self._preload_warned = True
                print(f"[preload_patch] SKIPPED: estimated cache={estimated_gb:.1f} GB > "
                      f"A2V_PRELOAD_MAX_GB={_DEFAULT_MAX_GB:.0f} (8 GPU × {estimated_gb:.1f} GB = {estimated_gb * 8:.0f} GB)"
                      f" → fallback to lazy LoadTorchPickle (slow first epoch).", flush=True)
            return original_getitem(self, data_id)
        # 正常 preload
        if not hasattr(self, "_cached_tensors"):
            n = len(self.cached_data)
            print(f"[preload_patch] loading {n} pth files (~{estimated_gb:.1f} GB) into RAM ...", flush=True)
            tensors = []
            for i, p in enumerate(self.cached_data):
                tensors.append(torch.load(p, map_location="cpu", weights_only=False))
                if (i + 1) % 20 == 0:
                    print(f"  [preload_patch] {i+1}/{n}", flush=True)
            self._cached_tensors = tensors
            print(f"[preload_patch] done. RAM preloaded.", flush=True)
        return self._cached_tensors[data_id % len(self._cached_tensors)]

    UnifiedDataset.search_for_cached_data_files = patched_search
    UnifiedDataset.__getitem__ = patched_getitem
    print(f"[preload_patch] installed (max_gb={_DEFAULT_MAX_GB:.0f}, force={_FORCE}, disable={_DISABLE}).", flush=True)
