from collections import Counter
import numpy as np
import torch

def build_tempered_weighted_sampler(ds, alpha: float = 0.5):
    ys = [y for _, y in ds.samples]
    c = Counter(ys)

    # enforce contiguous label space [0..num_classes-1]
    n = ds.num_classes
    for i in range(n):
        if i not in c:
            raise RuntimeError(f"Sampler: class {i} has 0 samples (num_classes={n}).")

    counts = {i: int(c[i]) for i in range(n)}

    counts_arr = np.array([counts[i] for i in range(n)], dtype=np.float64)
    w_class = (counts_arr.max() / counts_arr) ** float(alpha)
    w_sample = np.array([w_class[y] for y in ys], dtype=np.float64)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(w_sample).double(),
        num_samples=len(ys),
        replacement=True,
    )
    return sampler, counts