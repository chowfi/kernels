import pytest

from harness.device_info import get_device_facts, l2_bandwidth_sweep


@pytest.mark.gpu
def test_device_is_5090():
    f = get_device_facts()
    assert f["compute_capability"] == (12, 0)
    assert f["sm_count"] == 170
    assert 80 * 2**20 <= f["l2_bytes"] <= 130 * 2**20
    assert f["smem_per_block_optin_bytes"] >= 96 * 1024


@pytest.mark.gpu
def test_l2_sweep_shows_cliff():
    pts = dict(l2_bandwidth_sweep([16, 512]))
    assert pts[16] > 1.3 * pts[512]  # in-L2 reads beat DRAM reads
