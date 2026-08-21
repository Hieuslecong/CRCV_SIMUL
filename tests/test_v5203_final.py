import copy
import numpy as np
import pytest
from scipy import ndimage as ndi

from crcv52.action_targets import build_action_targets, encode_actions
from crcv52.counterfactual_errors import simulate_corruption
from crcv52.region_supervision import (
    build_remove_supervision, distance_ratio_to_reference,
    RemoveSupervisionConfig, KEEP, REMOVE, IGNORE,
)
from crcv52.removal_policy import build_removal_features, sample_keep_remove_training
from crcv52.component_policy import ComponentRemovalConfig, component_remove_mask
from crcv52.safety_projection import SafetyProjectionConfig, project_remove
from crcv52.error_profile import profile_base_error, merge_error_profiles, calibrate_counterfactual_config
from crcv52.operating_point import OperatingPoint, select_conservative_pixel_threshold, select_policy_family
from crcv52.v5203_policy_training import V5203PolicyTrainingConfig, build_training_matrix, train_pixel_remove_policy
from crcv_q1.action_metrics import correction_metrics, binary_segmentation_metrics
from crcv_q1.remove_qualification import qualify_remove_policy
from crcv_q1.multiseed_gate import assess_multiseed_smoke


def crack_mask(h=64, w=64):
    g = np.zeros((h,w), bool)
    for y in range(8, h-8):
        x = min(w-4, 18 + y//5)
        g[y, x:x+2] = True
    return g


def simple_record(name='a', h=48, w=48):
    gt = np.zeros((h,w), bool); gt[8:h-8, w//2:w//2+2] = True
    image = np.repeat((~gt)[...,None],3,axis=2).astype(np.float32)
    prob = gt.astype(np.float32)*0.82 + 0.08
    prob[3:6,3:6] = 0.72
    return {'name':name,'image':image,'gt':gt}, prob


def test_action_targets_exact_partition_and_2d_contract():
    gt = crack_mask(); base = gt.copy(); base[20:24,22:24] = False; base[30,40:45] = True
    t = build_action_targets(base, gt)
    assert np.array_equal(t['keep'], base & gt)
    assert np.array_equal(t['remove'], base & ~gt)
    assert np.array_equal(t['add'], gt & ~base)
    assert np.array_equal(t['keep'] | t['remove'], base)
    assert np.array_equal(t['keep'] | t['add'], gt)
    assert set(np.unique(encode_actions(base,gt))).issubset({0,1,2,3})
    with pytest.raises(ValueError): build_action_targets(base[...,None], gt[...,None])


def test_additive_corruptions_never_modify_gt_pixels():
    gt = np.zeros((64,64), bool); gt[8:28,14:16] = True; gt[38:56,44:46] = True
    for op in ['width_dilate','side_spur','isolated_blob','false_bridge']:
        out, meta = simulate_corruption(gt, op, seed=7)
        if meta['status'].startswith('NO_OP'):
            continue
        assert np.array_equal(out & gt, gt)


def test_side_spur_is_primitive_only_not_global_halo():
    gt = np.zeros((64,64), bool); gt[8:56,30:32] = True
    out,_ = simulate_corruption(gt,'side_spur',seed=7)
    added = out & ~gt
    assert 0 < added.sum() < 80


def test_region_supervision_default_ignores_attached_fp_and_distal_mode_is_explicit():
    gt = np.zeros((64,64), bool); gt[10:54,30:32] = True
    base = gt.copy(); base[30,32:45] = True
    sup = build_remove_supervision(base, gt)
    assert np.all(sup[base & gt] == KEEP)
    assert np.all(sup[base & ~gt] == IGNORE)
    distal = build_remove_supervision(base, gt, RemoveSupervisionConfig(mode='distal_pixel'))
    assert distal[30,32] == IGNORE
    assert np.any(distal[30,38:45] == REMOVE)


def test_region_supervision_true_negative_all_fp_is_remove():
    gt=np.zeros((32,32),bool); base=np.zeros_like(gt); base[4:8,4:8]=True
    sup=build_remove_supervision(base,gt)
    assert np.all(sup[base] == REMOVE)


def test_distance_ratio_is_approximately_scale_equivalent():
    gt=np.zeros((32,32),bool); gt[8:24,15:17]=True
    gt2=ndi.zoom(gt.astype(np.uint8),2,order=0).astype(bool)
    r1=distance_ratio_to_reference(gt); r2=distance_ratio_to_reference(gt2)
    assert abs(float(r1[16,20]) - float(r2[32,40])) < 0.75


def test_feature_contract_gt_free_finite_no_xy_and_scale_aware():
    rec,prob=simple_record(); base=prob>=.5
    X,names=build_removal_features(rec['image'],prob,base)
    assert X.shape == (*base.shape,len(names)); assert np.isfinite(X).all()
    assert 'normalized_x' not in names and 'normalized_y' not in names
    assert 'component_bbox_diagonal_norm' in names
    assert 'radial_position_ratio' in names


def test_explicit_distal_sampling_is_keep_heavy_and_has_remove_examples():
    gt=np.zeros((64,64),bool); gt[10:54,30:32]=True
    base=gt.copy(); base[30,32:50]=True
    image=np.zeros((*gt.shape,3),np.float32); prob=base.astype(np.float32)*.8
    X,y,_=sample_keep_remove_training(
        image,prob,base,gt,max_keep=300,max_remove=100,
        supervision_mode='distal_pixel',seed=3,
    )
    assert len(X)==len(y)
    assert (y==0).sum() > (y==1).sum() > 0


def test_legacy_removal_sampling_keywords_are_still_accepted():
    gt=np.zeros((32,32),bool); gt[8:24,15:17]=True
    base=ndi.binary_dilation(gt,iterations=1)
    image=np.zeros((*gt.shape,3),np.float32); prob=base.astype(np.float32)*.9
    _,y,_=sample_keep_remove_training(
        image,prob,base,gt,max_keep=100,max_remove=100,
        remove_exclusion_radius=1,require_detached_remove=True,seed=2,
    )
    assert int((y==1).sum()) == 0


def test_component_policy_removes_compact_blob_but_protects_long_thin_component():
    base=np.zeros((64,64),bool); base[5:55,30:31]=True; base[3:6,3:6]=True
    prob=np.zeros(base.shape,np.float32); prob[base]=.55
    cfg=ComponentRemovalConfig(
        max_area_fraction=.01,max_mean_probability=.8,
        max_skeleton_length_fraction=.08,max_bbox_diagonal_fraction=.08,max_elongation=4,
        max_total_remove_fraction=.05,max_foreground_remove_fraction=.5,
    )
    rm=component_remove_mask(base,prob,cfg)
    assert rm[3:6,3:6].all()
    assert not rm[5:55,30:31].any()


def test_component_policy_scale_equivalence():
    base=np.zeros((32,32),bool); base[8:24,15:17]=True; base[2:4,2:4]=True
    prob=np.zeros(base.shape,np.float32); prob[8:24,15:17]=.95; prob[2:4,2:4]=.55
    base2=ndi.zoom(base.astype(np.uint8),2,order=0).astype(bool); prob2=ndi.zoom(prob,2,order=0)
    cfg=ComponentRemovalConfig(
        max_area_fraction=.01,max_mean_probability=.8,
        max_skeleton_length_fraction=.05,max_bbox_diagonal_fraction=.08,max_elongation=4,
        max_total_remove_fraction=.05,max_foreground_remove_fraction=.5,
    )
    r1=component_remove_mask(base,prob,cfg); r2=component_remove_mask(base2,prob2,cfg)
    assert r1[2:4,2:4].all() and r2[4:8,4:8].all()
    assert not r1[8:24,15:17].any() and not r2[16:48,30:34].any()


def test_component_policy_legacy_max_pixels_is_accepted():
    base=np.zeros((32,32),bool); base[5:25,15:17]=True; base[2:4,2:4]=True
    prob=np.zeros(base.shape,np.float32); prob[5:25,15:17]=.95; prob[2:4,2:4]=.55
    rm=component_remove_mask(base,prob,ComponentRemovalConfig(
        max_area_fraction=.01,max_mean_probability=.8,max_pixels=8,
        max_total_remove_fraction=.05,max_foreground_remove_fraction=.5,
        max_skeleton_length_fraction=.1,max_bbox_diagonal_fraction=.1,
    ))
    assert rm[2:4,2:4].all()


def test_safety_projection_never_removes_skeleton_or_outside_base():
    base=np.zeros((64,64),bool); base[8:56,30:35]=True
    score=np.zeros(base.shape,np.float32); score[base]=1
    refined,removed,info=project_remove(base,score,.5)
    assert np.all(~removed | base)
    assert info['removed_pixels'] <= info['total_budget']
    assert correction_metrics(base,refined,base)['base_skeleton_removed'] == 0


def test_parallel_boundary_strip_protected_at_two_scales():
    for scale in [1,2]:
        h=64*scale; base=np.zeros((h,h),bool); base[8*scale:56*scale,30*scale:35*scale]=True
        score=np.zeros(base.shape,np.float32); score[8*scale:56*scale,34*scale]=1
        refined,removed,info=project_remove(base,score,.5)
        assert not removed.any()
        assert np.array_equal(refined,base)
        assert info['protected_parallel_boundary_regions'] >= 1 or info['protected_pixels'] > 0


def test_safety_projection_legacy_overrides_still_work():
    base=np.zeros((64,64),bool); base[8:56,30:34]=True
    score=np.zeros(base.shape,np.float32); score[8:56,33]=1
    cfg=SafetyProjectionConfig(
        core_protection_radius=1,boundary_strip_min_skeleton_length=6,
        boundary_strip_max_median_distance=2.5,
    )
    refined,removed,_=project_remove(base,score,.5,cfg)
    assert np.array_equal(refined,base); assert not removed.any()


def test_safety_projection_rejects_gt_argument():
    base=crack_mask(); score=np.zeros(base.shape,np.float32)
    with pytest.raises(TypeError): project_remove(base,score,.5,gt_mask=base)


def test_error_profile_is_type_and_scale_aware():
    gt=np.zeros((64,64),bool); gt[10:54,30:32]=True
    base=gt.copy(); base[30,32:48]=True; base[4:7,4:7]=True
    p=profile_base_error(base,gt); m=merge_error_profiles([p]); cfg=calibrate_counterfactual_config(m)
    assert all('error_type' in c for c in p['fp_components'])
    assert 'fp_components_by_type' in m
    assert cfg.spur_length >= 2 and cfg.blob_radius >= 1


def test_operating_point_is_safety_first_inside_feasible_set():
    vals={
        .9:{'delta_dice':.001,'delta_recall':-.001,'delta_crack_iou':.001,'tcrr':.0001,'fprr':.01,'removed_pixels':10},
        .95:{'delta_dice':.0005,'delta_recall':0.,'delta_crack_iou':.0004,'tcrr':0.,'fprr':.002,'removed_pixels':2},
    }
    op=select_conservative_pixel_threshold(lambda t:vals[t],[.9,.95])
    assert op.parameters == .95


def test_operating_point_never_selects_unsafe_effect():
    vals={
        .8:{'delta_dice':.02,'delta_recall':-.01,'delta_crack_iou':.02,'tcrr':.01,'fprr':.2,'removed_pixels':100},
        .95:{'delta_dice':.0002,'delta_recall':0.,'delta_crack_iou':.0002,'tcrr':0.,'fprr':.002,'removed_pixels':2},
    }
    op=select_conservative_pixel_threshold(lambda t:vals[t],[.8,.95])
    assert op.parameters == .95


def test_policy_family_selection_prefers_lower_tcrr():
    a=OperatingPoint('pixel',.9,{'delta_dice':.001,'delta_recall':0,'delta_crack_iou':.001,'tcrr':.001,'fprr':.03},.001)
    b=OperatingPoint('component',{}, {'delta_dice':.002,'delta_recall':0,'delta_crack_iou':.002,'tcrr':0,'fprr':.01},.002)
    assert select_policy_family(a,b).family == 'component'


def test_qualification_fails_closed_and_noop_is_not_active():
    bad=qualify_remove_policy({'delta_dice':.01,'delta_recall':-.01,'tcrr':.01,'fprr':.2,'removed_pixels':100})
    noop=qualify_remove_policy({'delta_dice':0,'delta_recall':0,'tcrr':0,'fprr':0,'removed_pixels':0})
    good=qualify_remove_policy({'delta_dice':.001,'delta_recall':-.001,'tcrr':.001,'fprr':.01,'removed_pixels':10})
    assert bad['status']=='NO_OP' and noop['status']=='NO_OP' and good['status']=='ACTIVE'


def test_training_matrix_is_order_invariant_by_record_name():
    r1,p1=simple_record('b'); r2,p2=simple_record('a'); probs={'a':p2,'b':p1}
    cfg=V5203PolicyTrainingConfig(n_estimators=10,n_jobs=1)
    X1,y1,n1,m1=build_training_matrix([r1,r2],probs,.5,seed=11,config=cfg)
    X2,y2,n2,m2=build_training_matrix([r2,r1],probs,.5,seed=11,config=cfg)
    assert n1==n2 and m1['record_names']==['a','b']==m2['record_names']
    assert np.array_equal(y1,y2); assert np.allclose(X1,X2)


def test_training_rejects_duplicate_names_and_missing_probabilities():
    r,p=simple_record('a')
    with pytest.raises(ValueError): build_training_matrix([r,copy.deepcopy(r)],{'a':p},.5)
    with pytest.raises(KeyError): build_training_matrix([r],{},.5)


def test_lightgbm_training_is_reproducible_with_same_seed():
    r1,p1=simple_record('a'); r2,p2=simple_record('b')
    cfg=V5203PolicyTrainingConfig(
        n_estimators=12,n_jobs=1,synthetic_max_keep=30,synthetic_max_remove=20,
        natural_max_keep=80,natural_max_remove=40,
    )
    c1,m1=train_pixel_remove_policy([r1,r2],{'a':p1,'b':p2},.5,seed=21,config=cfg)
    c2,m2=train_pixel_remove_policy([r2,r1],{'a':p1,'b':p2},.5,seed=21,config=cfg)
    X,_,_,_=build_training_matrix([r1,r2],{'a':p1,'b':p2},.5,seed=21,config=cfg)
    assert np.allclose(c1.predict_proba(X),c2.predict_proba(X),atol=0,rtol=0)
    assert m1['feature_schema_sha256']==m2['feature_schema_sha256']


def test_binary_metrics_2d_contract():
    g=crack_mask(); m=binary_segmentation_metrics(g,g); assert m['dice']==1.0
    with pytest.raises(ValueError): binary_segmentation_metrics(g[...,None],g[...,None])


def test_multiseed_gate_blocks_when_one_seed_is_not_robust():
    def fake(seed, active):
        models={}
        for i in range(5):
            models[str(i)]={
                'qualification':{'status':'ACTIVE' if i<active else 'NO_OP'},
                'deployed':{'delta_dice':.001 if i<active else 0.0,'tcrr':.001 if i<active else 0.0},
            }
        return {'protocol':{'seed':seed},'models':models}
    result=assess_multiseed_smoke([fake(1337,4),fake(2027,2),fake(31415,4)])
    assert result['status']=='BLOCKED'
    assert any('seed 2027' in x for x in result['failures'])
