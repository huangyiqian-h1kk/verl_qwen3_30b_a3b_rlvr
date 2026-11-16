import pytest
from d4_mopd.profile_report import memory_summary


def samples(path, count=8):
    path.write_text("".join(f"2026/11/16 12:00:00.000, {i}, GPU-{i}, 100, 143000, 0\n"
                            for i in range(count)) +
                    "2026/11/16 12:00:00.250, 2, GPU-2, 120000, 143000, 99\n")


def test_all_physical_gpus_report_sampled_peaks_without_adding_process_allocations(tmp_path):
    samples(tmp_path / "gpu_memory_0.csv")
    result = memory_summary(tmp_path, 1)
    assert result["sample_interval_ms"] == 250
    assert len(result["nodes"][0]["devices"]) == 8
    gpu = result["nodes"][0]["devices"][2]
    assert gpu["peak_sampled_used_MiB"] == 120000
    assert gpu["samples"] == 2


def test_missing_node_or_gpu_cannot_be_capacity_evidence(tmp_path):
    samples(tmp_path / "gpu_memory_0.csv", 7)
    with pytest.raises(RuntimeError, match="all eight"):
        memory_summary(tmp_path, 1)
    with pytest.raises(RuntimeError, match="part of the allocation"):
        memory_summary(tmp_path, 2)


def test_a_dead_sampler_cannot_certify_completion_but_remains_readable_offline(tmp_path):
    import os
    path = tmp_path / "gpu_memory_0.csv"
    samples(path)
    os.utime(path, (1, 1))
    with pytest.raises(RuntimeError, match="sampling stopped"):
        memory_summary(tmp_path, 1, max_age_seconds=60)
    assert len(memory_summary(tmp_path, 1)["nodes"]) == 1
