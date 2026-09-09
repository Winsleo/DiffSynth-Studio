"""在 train_a2v.py 启动时 monkey-patch UnifiedDataset,让 cache mode 一次性预加载所有 .pth 到内存。

用法:在调 train_a2v.py 之前 PYTHONPATH 加这个文件路径,或者在 train_a2v.py 里 import。

原理:
- 替换 UnifiedDataset.search_for_cached_data_files → 把 pth 路径存到 self.cached_data(原有)
- 替换 UnifiedDataset.__getitem__:在第一次调用时全部 torch.load 到 self.cached_data_tensors,然后 [data_id] 直接索引
- 避免 250 MB pth lazy load 慢
"""

from __future__ import annotations

import os
import torch


def install_preload_patch() -> None:
    """Monkey-patch UnifiedDataset + cached_data_operator(LoadTorchPickle)。"""
    from diffsynth.core.data.unified_dataset import UnifiedDataset
    from diffsynth.core.data.operators import LoadTorchPickle

    original_search = UnifiedDataset.search_for_cached_data_files
    original_getitem = UnifiedDataset.__getitem__

    def patched_search(self, path):
        original_search(self, path)

    def patched_getitem(self, data_id):
        if not self.load_from_cache:
            return original_getitem(self, data_id)
        # Lazy preload all pth to RAM on first call
        if not hasattr(self, "_cached_tensors"):
            print(f"[preload_patch] loading {len(self.cached_data)} pth files into RAM ...", flush=True)
            tensors = []
            for i, p in enumerate(self.cached_data):
                tensors.append(torch.load(p, map_location="cpu", weights_only=False))
                if (i + 1) % 20 == 0:
                    print(f"  [preload_patch] {i+1}/{len(self.cached_data)}", flush=True)
            self._cached_tensors = tensors
            print(f"[preload_patch] done. RAM preloaded.", flush=True)
        return self._cached_tensors[data_id % len(self._cached_tensors)]

    UnifiedDataset.search_for_cached_data_files = patched_search
    UnifiedDataset.__getitem__ = patched_getitem
    print("[preload_patch] installed.", flush=True)