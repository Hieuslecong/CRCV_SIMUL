import numpy as np
from scipy import ndimage as ndi

from crcv52.action_targets import build_action_targets, encode_actions
from crcv52.counterfactual_errors import simulate_corruption
from crcv52.removal_policy import build_removal_features, sample_keep_remove_training
from crcv52.component_policy import ComponentRemovalConfig, component_remove_mask
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
    gt = crack_mask(); base = gt.copy(); base[20:24, 22:24] = False; base[30,40:45] = True
    t = build_action_targets(base, gt)
    assert np.array_equal(t['keep'], base & gt)
    assert np.array_equal(t['remove'], base & ~gt)
    assert np.array_equal(t['add'], gt & ~base)
    assert set(np.unique(encode_actions(base, gt))).issubset({0,1,2,3})


def test_counterfactual_operators_create_known_actions():
    gt = crack_mask()
    for op in ['gap_delete','endpoint_truncate','boundary_erode','width_dilate','side_spur','isolated_blob']:
        corrupted, meta = simulate_corruption(gt, op, seed=1337)
        t = build_action_targets(corrupted, gt)
        assert corrupted.shape == gt.shape
        assert t['add'].any() or t['remove'].any() or meta['status'].startswith('NO_OP')


def test_side_spur_does_not_dilate_entire_gt_boundary():
    gt = np.zeros((64,64), bool); gt[8:56, 30:32] = True
    out, _ = simulate_corruption(gt, 'side_spur', seed=7)
    added = out & ~gt
    assert 0 < int(added.sum()) < 70
    assert np.array_equal(out & gt, gt)


def test_false_bridge_does_not_dilate_both_gt_components():
    gt = np.zeros((64,64), bool); gt[10:25,15:17] = True; gt[35:50,45:47] = True
    out, meta = simulate_corruption(gt, 'false_bridge', seed=3)
    assert meta['status'] == 'APPLIED'
    added = out & ~gt
    assert added.any()
    halo = np.zeros_like(gt); halo[10:25,13:14] = True; halo[35:50,48:49] = True
    assert int((added & halo).sum()) <= 2


def test_removal_features_are_gt_free_finite_and_topology_aware():
    gt = crack_mask(); base, _ = simulate_corruption(gt, 'side_spur', seed=1)
    image = np.repeat((~gt)[...,None], 3, axis=2).astype(np.float32)
    prob = base.astype(np.float32)*0.8 + 0.1
    X, names = build_removal_features(image, prob, base)
    assert X.shape == (*base.shape, len(names)); assert np.isfinite(X).all()
    assert 'normalized_x' not in names and 'normalized_y' not in names
    assert 'distance_to_endpoint_norm' in names and 'distance_to_junction_norm' in names


def test_remove_training_ignores_one_pixel_outside_gt_boundary():
    gt = np.zeros((32,32), bool); gt[8:24,15:17] = True
    base = ndi.binary_dilation(gt, iterations=1)
    image = np.repeat((~gt)[...,None],3,axis=2).astype(np.float32); prob = base.astype(np.float32)*0.9
    _, y, _ = sample_keep_remove_training(image, prob, base, gt, max_keep=1000, max_remove=1000,
                                           remove_exclusion_radius=1, seed=1)
    assert int((y == 1).sum()) == 0


def test_keep_heavy_sampling_does_not_force_class_balance():
    gt = crack_mask(); base = gt.copy(); base[2:5,2:5] = True
    image = np.zeros((*gt.shape,3), np.float32); prob = base.astype(np.float32)*0.9
    _, y, _ = sample_keep_remove_training(image, prob, base, gt, max_keep=500, max_remove=100,
                                           require_detached_remove=True, seed=2)
    assert int((y == 0).sum()) > int((y == 1).sum())


def test_component_policy_removes_whole_small_component_only():
    base = np.zeros((32,32), bool); base[5:25,15:17] = True; base[2:4,2:4] = True
    prob = np.zeros(base.shape, np.float32); prob[5:25,15:17] = 0.95; prob[2:4,2:4] = 0.55
    rm = component_remove_mask(base, prob, ComponentRemovalConfig(max_area_fraction=.01,max_mean_probability=.8,max_pixels=8))
    assert rm[2:4,2:4].all()
    assert not rm[5:25,15:17].any()


def test_error_profile_calibrates_counterfactual_config_from_fit_like_errors():
    from crcv52.error_profile import profile_base_error, merge_error_profiles, calibrate_counterfactual_config
    gt = crack_mask(); base = gt.copy(); base[20,40:48] = True; base[45:48,5:8] = True; base[30:33,24:26] = False
    cfg = calibrate_counterfactual_config(merge_error_profiles([profile_base_error(base, gt)]))
    assert 3 <= cfg.spur_length <= 16
    assert 1 <= cfg.blob_radius <= 5
    assert 1 <= cfg.dilation_radius <= 3
    assert 0.025 <= cfg.gap_fraction <= 0.12


def test_safety_projection_never_removes_protected_core_or_outside_base():
    gt = crack_mask(); base, _ = simulate_corruption(gt, 'side_spur', seed=2)
    score = np.zeros(base.shape, np.float32); score[base] = 1.0
    refined, removed, info = project_remove(base, score, 0.5, SafetyProjectionConfig(core_protection_radius=1))
    assert info['status'] == 'PASS'; assert np.all(~removed | base); assert np.all(~refined | base)
    assert removed.sum() <= info['total_budget']


def test_long_parallel_boundary_strip_is_protected():
    base = np.zeros((64,64), bool); base[8:56,30:34] = True
    score = np.zeros(base.shape, np.float32); score[8:56,33] = 1.0
    refined, removed, info = project_remove(base, score, 0.5,
        SafetyProjectionConfig(core_protection_radius=1,boundary_strip_min_skeleton_length=6,
                               boundary_strip_max_median_distance=2.5))
    assert not removed.any(); assert np.array_equal(refined, base); assert info['protected_parallel_boundary_regions'] >= 1


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
    good = base.copy(); good[30,40:45] = False; bad = base.copy(); bad[20:24,22:24] = False
    mg = correction_metrics(base, good, gt); mb = correction_metrics(base, bad, gt)
    assert mg['fprr'] > 0 and mg['tcrr'] == 0; assert mb['tcrr'] > 0


def test_remove_qualification_fails_closed_on_boundary_damage():
    bad = qualify_remove_policy({'delta_dice':0.01,'delta_recall':-0.006,'tcrr':0.008,'fprr':0.05})
    good = qualify_remove_policy({'delta_dice':0.002,'delta_recall':-0.0005,'tcrr':0.001,'fprr':0.02})
    assert bad['status'] == 'NO_OP'; assert good['status'] == 'ACTIVE'
