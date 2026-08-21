import numpy as np

from crcv52.action_targets import build_action_targets, encode_actions
from crcv52.counterfactual_errors import simulate_corruption
from crcv52.removal_policy import build_removal_features, sample_keep_remove_training
from crcv52.safety_projection import SafetyProjectionConfig, project_remove
from crcv_q1.action_metrics import correction_metrics
from crcv_q1.remove_qualification import qualify_remove_policy


def crack_mask(h=64, w=64):
    g = np.zeros((h,w), bool)
    for y in range(8, h-8):
        x = 18 + y//5
        g[y, x:x+2] = True
    return g


def test_action_targets_are_exact_and_disjoint():
    gt = crack_mask()
    base = gt.copy(); base[20:24, 22:24] = False; base[30,40:45] = True
    t = build_action_targets(base, gt)
    assert np.array_equal(t['keep'], base & gt)
    assert np.array_equal(t['remove'], base & ~gt)
    assert np.array_equal(t['add'], gt & ~base)
    enc = encode_actions(base, gt)
    assert set(np.unique(enc)).issubset({0,1,2,3})


def test_counterfactual_operators_create_known_actions():
    gt = crack_mask()
    for op in ['gap_delete','endpoint_truncate','boundary_erode','width_dilate','side_spur','isolated_blob']:
        corrupted, meta = simulate_corruption(gt, op, seed=1337)
        assert corrupted.shape == gt.shape
        t = build_action_targets(corrupted, gt)
        assert t['add'].any() or t['remove'].any() or meta['status'].startswith('NO_OP')


def test_removal_features_are_gt_free_and_finite():
    gt = crack_mask()
    base, _ = simulate_corruption(gt, 'side_spur', seed=1)
    image = np.repeat((~gt)[...,None], 3, axis=2).astype(np.float32)
    prob = base.astype(np.float32)*0.8 + 0.1
    X, names = build_removal_features(image, prob, base)
    assert X.shape == (*base.shape, len(names))
    assert np.isfinite(X).all()
    Xt, y, names2 = sample_keep_remove_training(image, prob, base, gt, max_per_class=100)
    assert Xt.shape[1] == len(names2)
    assert set(np.unique(y)).issubset({0,1})


def test_safety_projection_never_removes_protected_core_or_outside_base():
    gt = crack_mask()
    base, _ = simulate_corruption(gt, 'side_spur', seed=2)
    score = np.zeros(base.shape, np.float32)
    score[base] = 1.0
    refined, removed, info = project_remove(base, score, 0.5, SafetyProjectionConfig(core_protection_radius=1))
    assert info['status'] == 'PASS'
    assert np.all(~removed | base)
    assert np.all(~refined | base)
    assert removed.sum() <= info['total_budget']


def test_long_parallel_boundary_strip_is_protected():
    base = np.zeros((64,64), bool)
    base[8:56, 30:34] = True
    score = np.zeros(base.shape, np.float32)
    score[8:56, 33] = 1.0  # long shell along one crack edge
    refined, removed, info = project_remove(
        base, score, 0.5,
        SafetyProjectionConfig(core_protection_radius=1,
                               boundary_strip_min_skeleton_length=6,
                               boundary_strip_max_median_distance=2.5)
    )
    assert not removed.any()
    assert np.array_equal(refined, base)
    assert info['protected_parallel_boundary_regions'] >= 1


def test_safety_projection_is_gt_free_api():
    gt = crack_mask(); score = np.zeros(gt.shape, np.float32)
    try:
        project_remove(gt, score, 0.5, gt_mask=gt)
    except TypeError:
        pass
    else:
        raise AssertionError('runtime safety API unexpectedly accepted GT')


def test_tcrr_fprr_capture_wrong_and_correct_removal():
    gt = crack_mask(); base = gt.copy(); base[30,40:45] = True
    good = base.copy(); good[30,40:45] = False
    bad = base.copy(); bad[20:24,22:24] = False
    mg = correction_metrics(base, good, gt)
    mb = correction_metrics(base, bad, gt)
    assert mg['fprr'] > 0 and mg['tcrr'] == 0
    assert mb['tcrr'] > 0


def test_remove_qualification_fails_closed_on_boundary_damage():
    bad = qualify_remove_policy({
        'delta_dice': 0.01,
        'delta_recall': -0.006,
        'tcrr': 0.008,
        'fprr': 0.05,
    })
    assert bad['status'] == 'NO_OP'
    good = qualify_remove_policy({
        'delta_dice': 0.002,
        'delta_recall': -0.0005,
        'tcrr': 0.001,
        'fprr': 0.02,
    })
    assert good['status'] == 'ACTIVE'
