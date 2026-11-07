"""I1: the static guard rejects ways to skip Triton or game the timing (delegation to torch/FA,
data-keyed output caches, side streams, CUDA graphs)."""
from pathlib import Path

import pytest

from driver.guard import kernel_violation

REPO = Path(__file__).resolve().parent.parent
SEED = (REPO / "kernel" / "attention.py").read_text()
SMOKE = (REPO / "tests" / "fixtures" / "kernels" / "smoke_exp1.py").read_text()


@pytest.mark.parametrize("snippet", [
    "def f(q, k, v):\n    return torch.nn.functional.scaled_dot_product_attention(q, k, v)\n",
    "import torch.nn.functional as F\n",
    "from torch.nn import functional as F\n",
    "from torch import nn\n",
    "def f(q, k, v):\n    return torch.ops.aten._flash_attention_forward(q, k, v)\n",
    "def f(q):\n    return getattr(torch, 'ops')\n",
    "import flash_attn\n",
    "_OUT = {}\ndef f(q, o):\n    _OUT[(q.data_ptr(), 1)] = o\n",
    "def f(q):\n    return q._version\n",
    "s = torch.cuda.Stream()\n",
    "def f():\n    with torch.cuda.stream(None):\n        pass\n",
    "from torch.cuda import Stream\n",
    "g = torch.cuda.CUDAGraph()\n",
    "def f():\n    with torch.cuda.graph(None):\n        pass\n",
    "def f(q, k):\n    return torch.matmul(q, k)\n",
    "def f(q, k):\n    return q.matmul(k)\n",
    "def f(q, k):\n    return torch.bmm(q, k)\n",
    "def f(q, k):\n    return torch.einsum('ij,jk->ik', q, k)\n",
    "def f(s):\n    return torch.softmax(s, -1)\n",
    "def f(s):\n    return s.softmax(-1)\n",
    "def f(a, q, k):\n    return torch.baddbmm(a, q, k)\n",
    "def f(q, k):\n    return torch.mm(q, k)\n",
    "def f(q, k):\n    return q @ k\n",
    "def f(q, k):\n    q @= k\n",
    "fn = torch.compile(lambda x: x)\n",
    "_C2 = {}\ndef f(q, o):\n    _C2[(id(q), q[0, 0, 0, 0].item())] = o\n",
    "def f(q):\n    return q.flatten()[:4].tolist()\n",
    "def f(q):\n    return q.cpu()\n",
    "def f(q):\n    return q.numpy()\n",
    "def f():\n    triton.knobs.runtime.launch_enter_hook({'name': 'x'})\n",
    "def f(h):\n    return h.launch_enter_hook\n",
])
def test_guard_rejects_non_triton_and_gaming(snippet):
    assert kernel_violation(SEED + "\n" + snippet) is not None


@pytest.mark.parametrize("src", [SEED, SMOKE])
def test_guard_accepts_seed_and_smoke_kernels(src):
    assert kernel_violation(src) is None


@pytest.mark.parametrize("snippet", [
    "V = triton.__version__\n",
    "def f(q):\n    return torch.empty_like(q).view(1, -1).transpose(0, 1).contiguous()\n",
    "def f(q):\n    return q.new_zeros(4), torch.zeros(3, device=q.device), q.stride(0)\n",
])
def test_guard_still_accepts_allocation_and_views(snippet):
    assert kernel_violation(SEED + "\n" + snippet) is None


def test_guard_rules_list_every_i1_ban():
    from driver.guard import BANNED_ATTRS, BANNED_TOKENS, GUARD_RULES
    flat = " ".join(GUARD_RULES.split())
    for name in [*BANNED_ATTRS, *BANNED_TOKENS]:
        assert f"`{name}`" in flat or name in ("exec(", "eval(", "__import__"), name
    assert "`@`" in flat
