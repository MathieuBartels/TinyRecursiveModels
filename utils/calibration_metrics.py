"""Calibration metrics for binary (solved / not-solved) TRM evaluation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

ECE_NUM_BINS = 15


def append_calibration_sums(
    metrics: dict[str, torch.Tensor],
    *,
    valid: torch.Tensor,
    conf: torch.Tensor,
    correct: torch.Tensor,
) -> None:
    """Add reducible calibration sums to *metrics* (eval aggregation divides by count).

    conf: per-puzzle P(solved), typically sigmoid(q_halt).
    correct: per-puzzle 0/1 exact correctness.
    """
    if not valid.any():
        metrics.setdefault("brier_sum", torch.zeros((), device=conf.device))
        metrics.setdefault("fair_crps_sum", torch.zeros((), device=conf.device))
        for i in range(ECE_NUM_BINS):
            metrics.setdefault(f"ece_{i}_n", torch.zeros((), device=conf.device))
            metrics.setdefault(f"ece_{i}_conf", torch.zeros((), device=conf.device))
            metrics.setdefault(f"ece_{i}_corr", torch.zeros((), device=conf.device))
        return

    conf_v = conf[valid].float()
    correct_v = correct[valid].float()
    # 2-class Brier: ||[1-p, p] - [1-y, y]||^2 = 2(p - y)^2
    brier = 2.0 * (conf_v - correct_v) ** 2
    metrics["brier_sum"] = metrics.get("brier_sum", torch.zeros((), device=conf.device)) + brier.sum()
    metrics["fair_crps_sum"] = metrics.get("fair_crps_sum", torch.zeros((), device=conf.device)) + (
        (conf_v - correct_v).abs().sum()
    )

    # ECE bins (confidence = max class probability for binary)
    confidence = torch.maximum(conf_v, 1.0 - conf_v)
    bin_edges = torch.linspace(0.0, 1.0, ECE_NUM_BINS + 1, device=conf.device)
    for i in range(ECE_NUM_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidence > lo) & (confidence <= hi)
        n = in_bin.sum()
        metrics[f"ece_{i}_n"] = metrics.get(f"ece_{i}_n", torch.zeros((), device=conf.device)) + n
        metrics[f"ece_{i}_conf"] = metrics.get(
            f"ece_{i}_conf", torch.zeros((), device=conf.device)
        ) + conf_v[in_bin].sum()
        metrics[f"ece_{i}_corr"] = metrics.get(
            f"ece_{i}_corr", torch.zeros((), device=conf.device)
        ) + correct_v[in_bin].sum()


def ece_from_bin_sums(m: dict[str, float], total_n: float) -> float:
    """Compute ECE from accumulated per-bin counts and sums."""
    if total_n <= 0:
        return 0.0
    ece = 0.0
    for i in range(ECE_NUM_BINS):
        n = m.pop(f"ece_{i}_n", 0.0)
        if n <= 0:
            m.pop(f"ece_{i}_conf", None)
            m.pop(f"ece_{i}_corr", None)
            continue
        conf_sum = m.pop(f"ece_{i}_conf", 0.0)
        corr_sum = m.pop(f"ece_{i}_corr", 0.0)
        avg_conf = conf_sum / n
        avg_corr = corr_sum / n
        ece += (n / total_n) * abs(avg_conf - avg_corr)
    return float(ece)


def postprocess_eval_set_metrics(m: dict[str, float]) -> dict[str, float]:
    """Turn raw eval sums into rates; add brier / ece / fair_crps."""
    count = max(m.pop("count", 0.0), 1.0)
    total_n = count

    brier = m.pop("brier_sum", 0.0) / count
    fair_crps = m.pop("fair_crps_sum", 0.0) / count
    ece = ece_from_bin_sums(m, total_n)

    out = {k: v / count for k, v in m.items()}
    out["brier"] = brier
    out["ece"] = ece
    out["fair_crps"] = fair_crps
    return out


def compute_spearman_rho(conf: np.ndarray, correct: np.ndarray) -> float:
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.size == 0:
        return 0.0
    confidence = np.maximum(conf, 1.0 - conf)
    rho, _ = spearmanr(confidence, correct)
    if rho is None or np.isnan(rho):
        return 0.0
    return float(rho)


def gather_vectors(
    local: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> np.ndarray:
    """All-gather 1-D tensors; return concatenated array on rank 0."""
    if world_size == 1:
        return local.detach().cpu().numpy()

    import torch.distributed as dist

    n = torch.tensor([local.numel()], device=local.device, dtype=torch.long)
    sizes = [torch.zeros_like(n) for _ in range(world_size)]
    dist.all_gather(sizes, n)
    max_len = max(int(s.item()) for s in sizes)

    def _pad(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == max_len:
            return x.reshape(-1).float()
        pad = torch.zeros(max_len - x.numel(), device=x.device, dtype=torch.float32)
        return torch.cat([x.reshape(-1).float(), pad])

    padded = _pad(local)
    gathered = [torch.zeros(max_len, device=padded.device) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if rank != 0:
        return np.array([])

    parts = []
    for r in range(world_size):
        length = int(sizes[r].item())
        if length > 0:
            parts.append(gathered[r][:length].cpu().numpy())
    return np.concatenate(parts) if parts else np.array([])
