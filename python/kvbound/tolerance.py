"""Error metrics used to compare engine output against golden tensors.

All metrics take (actual, expected) and compute in float64 so the metric adds no error of its own.
"""

import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def rel_rms(actual: Tensor, expected: Tensor) -> float:
    """RMS of the error divided by RMS of the expected tensor."""
    a, e = actual.double(), expected.double()
    return ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()


def cosine(actual: Tensor, expected: Tensor) -> float:
    return F.cosine_similarity(actual.double().flatten(), expected.double().flatten(), dim=0).item()


def max_abs(actual: Tensor, expected: Tensor) -> float:
    return (actual.double() - expected.double()).abs().max().item()


def top1_agreement(actual_logits: Tensor, expected_logits: Tensor) -> float:
    """Fraction of positions whose argmax token matches (teacher-forced comparison)."""
    return (actual_logits.argmax(-1) == expected_logits.argmax(-1)).double().mean().item()


def mean_kl(actual_logits: Tensor, expected_logits: Tensor) -> float:
    """Mean over positions of KL(expected || actual) between next-token distributions, in nats."""
    log_p = F.log_softmax(expected_logits.double(), dim=-1)
    log_q = F.log_softmax(actual_logits.double(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()
