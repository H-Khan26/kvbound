"""Predicts prefill and decode latency bounds from a model, device, precision, batch and context.

Usage: python -m kvbound.perfmodel --model llama-3.2-1b --device rtx3060ti --batch 1 --context 1024
"""

import argparse

from kvbound.models import load_spec
from kvbound.perfmodel import hardware
from kvbound.perfmodel.model import Arch, OpCost, Precision, kv_bytes_per_token, predict


def _table(title: str, costs: list[OpCost]) -> str:
    total = sum(c.seconds for c in costs)
    lines = [
        f"{title}: {total * 1e3:.2f} ms",
        f"  {'op':<10} {'ms':>8} {'share':>6} {'MB':>9} {'ops/byte':>9}  bound by",
    ]
    for c in costs:
        lines.append(
            f"  {c.name:<10} {c.seconds * 1e3:>8.3f} {c.seconds / total:>6.1%} "
            f"{c.bytes / 1e6:>9.1f} {c.intensity:>9.1f}  {c.bound_by}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="name from configs/models/")
    parser.add_argument("--device", required=True, help="name from configs/devices/")
    parser.add_argument("--ceilings", choices=["measured", "spec"], default="measured")
    parser.add_argument("--weights", choices=["fp16", "int8_pc", "int4_g"], default="fp16")
    parser.add_argument("--kv", choices=["fp16", "int8", "fp8"], default="fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument(
        "--context", type=int, default=None, help="tokens cached at the decode step"
    )
    args = parser.parse_args()
    context = args.context if args.context is not None else args.prompt_len

    arch = Arch.from_table(load_spec(args.model).arch)
    hw = hardware.load(args.device, args.ceilings)
    precision = Precision.named(args.weights, args.kv, arch.head_dim)
    p = predict(arch, hw, precision, args.batch, args.prompt_len, context)

    print(
        f"{args.model} on {hw.device} ({hw.source} ceilings: {hw.dram_gbs:.0f} GB/s, "
        f"{hw.peak_tops[precision.compute]:.1f} e12 ops/s {precision.compute})"
    )
    print(
        f"weights {args.weights} {p.weights_gb:.2f} GB | KV {args.kv} "
        f"{kv_bytes_per_token(arch, precision.kv_bytes) / 1024:.0f} KiB/token, "
        f"{p.kv_gb:.3f} GB for batch {args.batch} at context {context}\n"
    )
    print(_table(f"prefill bound (TTFT), {args.batch} x {args.prompt_len} tokens", p.prefill_costs))
    print()
    print(
        _table(f"decode step bound (TPOT), batch {args.batch}, context {context}", p.decode_costs)
    )
    print(f"\ndecode ceiling: {p.decode_tok_s:.0f} tokens/s")


if __name__ == "__main__":
    main()
