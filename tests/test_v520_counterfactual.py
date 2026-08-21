import numpy as np

from crcv52.action_targets import build_action_targets, encode_actions
from crcv52.counterfactual_errors import simulate_corruption
from crcv52.removal_policy import build_removal_features, sample_keep_remove_training
from crcv52.safety_projection import SafetyProjectionConfig, project_remove
from crcv_q1.action_metrics import correction_metrics


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
