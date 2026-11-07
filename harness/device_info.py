"""Device facts and an L2 bandwidth sweep for the local GPU."""
import json
from pathlib import Path

import torch


def get_device_facts() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "compute_capability": (p.major, p.minor),
        "sm_count": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "smem_per_block_optin_bytes": p.shared_memory_per_block_optin,
        "smem_per_sm_bytes": p.shared_memory_per_multiprocessor,
        "total_mem_bytes": p.total_memory,
    }


def l2_bandwidth_sweep(sizes_mb: list[int], iters: int = 50) -> list[tuple[int, float]]:
    """Repeatedly sum a bf16 buffer of each size; return (MB, GB/s)."""
    out = []
    for mb in sizes_mb:
        nbytes = mb * 2**20
        # 2D buffer + row-wise reduce gives enough blocks to saturate the GPU
        x = torch.ones(nbytes // 2 // 4096, 4096, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            x.sum(dim=1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            x.sum(dim=1)
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1e3 / iters
        out.append((mb, nbytes / sec / 1e9))
        del x
    return out


if __name__ == "__main__":
    facts = get_device_facts()
    facts["l2_sweep_gbps"] = [
        {"mb": mb, "gbps": gbps}
        for mb, gbps in l2_bandwidth_sweep([4, 8, 16, 32, 64, 128, 256, 512])
    ]
    path = Path("results/device.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))
