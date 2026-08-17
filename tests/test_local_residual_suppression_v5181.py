import numpy as np

from crcv52.local_residual_suppression import (
    LocalResidualSuppressionConfig,
    local_residual_suppress,
)


def test_removed_pixels_stay_inside_frozen_base():
    base = np.zeros((16, 16), dtype=bool)
    base[4:12, 7] = True
    base[6:8, 8:10] = True
    auth = np.ones((16, 16), dtype=np.float32)
    auth[6:8, 8:10] = 0.0

    refined, removed = local_residual_suppress(base, auth)

    assert np.all(~removed | base)
    assert np.all(~refined | base)
    assert removed.any()


def test_high_authenticity_crack_is_preserved():
    base = np.zeros((12, 12), dtype=bool)
    base[2:10, 6] = True
    auth = np.ones((12, 12), dtype=np.float32)

    refined, removed = local_residual_suppress(base, auth)

    assert np.array_equal(refined, base)
    assert not removed.any()


def test_region_cap_blocks_large_deletion():
    base = np.ones((20, 20), dtype=bool)
    auth = np.zeros((20, 20), dtype=np.float32)
    cfg = LocalResidualSuppressionConfig(
        authenticity_threshold=0.03,
        max_region_fraction=0.01,
        max_total_remove_fraction=0.03,
    )

    refined, removed = local_residual_suppress(base, auth, cfg)

    assert np.array_equal(refined, base)
    assert not removed.any()


def test_runtime_has_no_gt_argument():
    base = np.zeros((8, 8), dtype=bool)
    auth = np.ones((8, 8), dtype=np.float32)
    try:
        local_residual_suppress(base, auth, gt=np.zeros_like(base))
    except TypeError:
        pass
    else:
        raise AssertionError("runtime API unexpectedly accepted GT")
