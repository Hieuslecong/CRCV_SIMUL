import inspect
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from crcv_core.actions import action_targets
from crcv_core.features import build_features
from crcv_core.policy import TrainingConfig,build_training_matrices,train,add_candidate,select_asymmetric_operating_point
from crcv_core.runtime import refine,action_scores
from crcv_core.safety import SafetyConfig,project_add,project_remove


def crack(s=64):
    g=np.zeros((s,s),bool)
    for y in range(max(3,s//8),s-max(3,s//8)):
        x=s//3+y//8; g[y,max(0,x-1):min(s,x+2)]=True
    return g

def rec(name='a',s=64,source='real'):
    g=crack(s); im=np.repeat((~g)[...,None],3,axis=2).astype(np.float32); p=g.astype(np.float32)*.82+.08
    p[2:5,2:5]=.75
    yy=s//2; xx=s//3+yy//8
    p[yy:yy+3,max(0,xx-1):min(s,xx+2)]=.35
    y2=s//3; x2=s//3+y2//8
    p[y2:y2+2,min(s-1,x2+3):min(s,x2+4)]=.35
    return {'name':name,'source':source,'image':im,'gt':g},p

# REVIEW 1 — contracts and leakage

def test_exact_action_partitions():
    r,p=rec(); b=p>=.5; t=action_targets(b,r['gt'])
    assert np.array_equal(t['keep'],b&r['gt'])
    assert np.array_equal(t['remove'],b&~r['gt'])
    assert np.array_equal(t['add'],r['gt']&~b)
    assert np.array_equal(t['keep']|t['remove'],b)
    assert np.array_equal(t['keep']|t['add'],r['gt'])

def test_action_shapes_fail_closed():
    try: action_targets(np.zeros((2,2,1)),np.zeros((2,2,1)))
    except ValueError: pass
    else: raise AssertionError

def test_runtime_api_has_no_gt():
    sig=inspect.signature(refine)
    assert 'gt' not in sig.parameters and 'gt_mask' not in sig.parameters

def test_no_generator_module_in_core():
    root=Path(__file__).resolve().parents[1]/'crcv_core'
    mods=sorted(p.name for p in root.glob('*.py') if p.name!='__init__.py')
    assert mods==['actions.py','features.py','policy.py','runtime.py','safety.py']
    text='\n'.join(p.read_text() for p in root.glob('*.py'))
    assert 'simulate(' not in text and 'counterfactual' not in text.lower()

def test_coordinate_only_structural_record_is_rejected():
    bad={'name':'S:trajectory','source':'provided_xy','xy':np.array([[.1,.2],[.2,.3]],np.float32)}
    try: build_training_matrices([bad],{'S:trajectory':np.zeros((64,64),np.float32)},.5,1)
    except (KeyError,ValueError): pass
    else: raise AssertionError

# REVIEW 2 — numerical and scale behavior

def test_features_are_nine_gt_free_finite():
    r,p=rec(); X,n=build_features(r['image'],p,p>=.5)
    assert X.shape[-1]==9 and 'blackhat' in n and np.isfinite(X).all()
    assert not any(x in n for x in ('x','y','normalized_x','normalized_y'))

def test_features_finite_128_256():
    for s in (128,256):
        r,p=rec(s=s); X,_=build_features(r['image'],p,p>=.5); assert np.isfinite(X).all()

def test_blackhat_highlights_dark_line_more_than_flat_area():
    s=64; im=np.ones((s,s,3),np.float32); im[:,31:33]=.1; p=np.zeros((s,s),np.float32); b=np.zeros((s,s),bool); b[:,30:34]=True
    X,n=build_features(im,p,b); bh=X[...,n.index('blackhat')]
    assert float(bh[:,31:33].mean())>float(bh[:,5:10].mean())

def test_add_candidate_outside_base_only():
    r,p=rec(); b=p>=.5; X,n=build_features(r['image'],p,b); c=add_candidate(p,b,X,n,.5,TrainingConfig())
    assert not np.any(c&b)

def test_bad_feature_input_rejected():
    r,p=rec(); p=p.copy(); p[0,0]=np.nan
    try: build_features(r['image'],p,p>=.5)
    except ValueError: pass
    else: raise AssertionError

# REVIEW 3 — training determinism and data semantics

def test_training_matrix_order_invariant():
    a,pa=rec('a'); b,pb=rec('b'); probs={'a':pa,'b':pb}
    A1,R1,m1=build_training_matrices([a,b],probs,.5,42)
    A2,R2,m2=build_training_matrices([b,a],probs,.5,42)
    assert np.array_equal(A1[0],A2[0]) and np.array_equal(A1[1],A2[1])
    assert np.array_equal(R1[0],R2[0]) and np.array_equal(R1[1],R2[1])
    assert m1['training_matrix_sha256']==m2['training_matrix_sha256']

def test_duplicate_names_rejected():
    a,pa=rec('a'); b,pb=rec('a')
    try: build_training_matrices([a,b],{'a':pa},.5)
    except ValueError: pass
    else: raise AssertionError

def test_missing_probability_rejected():
    a,pa=rec('a')
    try: build_training_matrices([a],{},.5)
    except KeyError: pass
    else: raise AssertionError

def test_both_heads_have_two_classes():
    a,pa=rec('a'); A,R,m=build_training_matrices([a],{'a':pa},.5)
    assert set(np.unique(A[1]))=={0,1}; assert set(np.unique(R[1]))=={0,1}

def test_training_deterministic():
    a,pa=rec('a'); h1,m1=train([a],{'a':pa},.5,7); h2,m2=train([a],{'a':pa},.5,7)
    X,_=build_features(a['image'],pa,pa>=.5); z=X.reshape(-1,X.shape[-1])
    assert np.array_equal(h1['add'].booster_.predict(z),h2['add'].booster_.predict(z))
    assert np.array_equal(h1['remove'].booster_.predict(z),h2['remove'].booster_.predict(z))
    assert m1['training_matrix_sha256']==m2['training_matrix_sha256']

def test_metadata_binds_sources_and_hashes():
    a,pa=rec('a',source='real'); _,_,m=build_training_matrices([a],{'a':pa},.5,9)
    assert m['sources']=={'real':1}; assert len(m['feature_schema_sha256'])==64 and len(m['training_matrix_sha256'])==64

# REVIEW 4 — bidirectional safety/runtime

def test_remove_never_outside_base_or_on_skeleton():
    b=np.zeros((64,64),bool); b[8:56,30:34]=True; s=np.ones(b.shape,np.float32); rm,_=project_remove(b,s,.5)
    assert not np.any(rm&~b); assert not np.any(rm&skeletonize(b))

def test_add_never_inside_base_or_outside_candidate():
    b=np.zeros((64,64),bool); b[10:50,30:33]=True; c=np.zeros_like(b); c[20:30,33:35]=True; s=np.ones(b.shape,np.float32); add,_=project_add(b,c,s,.5)
    assert not np.any(add&b); assert not np.any(add&~c)

def test_add_requires_connectivity_to_base():
    b=np.zeros((32,32),bool); b[10:20,15:17]=True; c=np.zeros_like(b); c[2:4,2:4]=True; s=np.ones(b.shape,np.float32); add,_=project_add(b,c,s,.5)
    assert not add.any()

def test_remove_budget_respected():
    b=np.zeros((64,64),bool); b[5:59,15:49]=True; s=np.ones(b.shape,np.float32); rm,info=project_remove(b,s,.5,SafetyConfig(max_total_remove_fraction=.01,max_foreground_remove_fraction=.05)); assert int(rm.sum())<=info['budget']

def test_runtime_does_not_mutate_inputs():
    r,p=rec(); h,_=train([r],{'a':p},.5,11); im=r['image'].copy(); pp=p.copy(); refine(im,pp,.5,h,.5,.8,qualified=True); assert np.array_equal(im,r['image']) and np.array_equal(pp,p)

def test_runtime_combines_add_keep_remove():
    r,p=rec(); h,_=train([r],{'a':p},.5,12); out,info=refine(r['image'],p,.5,h,.1,.1,SafetyConfig(core_radius_fraction=.5),qualified=True)
    b=info['base']; assert np.array_equal(out,(b&~info['remove'])|info['add'])

# REVIEW 5 — compactness and interface stability

def test_core_is_compact():
    root=Path(__file__).resolve().parents[1]/'crcv_core'; loc=sum(len(p.read_text().splitlines()) for p in root.glob('*.py'))
    assert loc<=260

def test_training_config_small_surface():
    names=set(TrainingConfig.__dataclass_fields__)
    assert len(names)<=14 and {'add_margin','add_radial_max','add_max_positive','add_max_negative','remove_max_keep','remove_max_positive'}<=names

def test_action_scores_shape():
    r,p=rec(); h,_=train([r],{'a':p},.5,13); a,rm,_,_=action_scores(h,r['image'],p,p>=.5); assert a.shape==p.shape and rm.shape==p.shape


def test_final_core_has_no_s_or_c_training_path():
    root=Path(__file__).resolve().parents[1]/'crcv_core'
    text='\n'.join(p.read_text().lower() for p in root.glob('*.py'))
    assert 'counterfactual' not in text and 'provided_sim' not in text and 'simulate(' not in text


def test_add_budget_trim_preserves_connectivity_to_base():
    b=np.zeros((40,40),bool); b[20,5:10]=True
    cand=np.zeros_like(b); cand[20,10:30]=True; cand[5:8,30:33]=True
    score=np.zeros(b.shape,np.float32); score[cand]=.8; score[5:8,30:33]=1.0
    add,_=project_add(b,cand,score,.5,SafetyConfig(max_add_foreground_fraction=.4))
    lab,_=ndi.label(b|add,structure=np.ones((3,3),bool)); good=set(np.unique(lab[b]))
    assert all(int(x) in good for x in np.unique(lab[add]))


def test_remove_uses_eight_connectivity_for_diagonal_crack():
    b=np.zeros((24,24),bool)
    for i in range(5,19): b[i,i]=True
    assert ndi.label(b,structure=np.ones((3,3),bool))[1]==1
    score=np.zeros(b.shape,np.float32); score[b]=1.0
    rm,_=project_remove(b,score,.5,SafetyConfig(core_radius_fraction=0,min_radius_norm=0,preserve_component_count=True,max_total_remove_fraction=.5,max_foreground_remove_fraction=.5))
    trial=b&~rm
    assert ndi.label(trial,structure=np.ones((3,3),bool))[1]<=1


def test_embedded_core_version_is_frozen():
    import crcv_core
    assert crcv_core.__version__=='1.1.2'


# REVIEW 6 — action-asymmetric calibration

def test_aac_preserves_most_conservative_safe_remove_then_optimizes_balance():
    rows=[
        (.012,.020,.80,.90,{'id':'conservative_add'}),
        (.018,.025,.20,.90,{'id':'better_balance'}),
        (.020,.030,.10,.85,{'id':'less_conservative_remove'}),
    ]
    out=select_asymmetric_operating_point(rows,.01)
    assert out[3]==.90
    assert out[2]==.20
    assert out[4]['id']=='better_balance'

def test_aac_falls_back_to_best_balanced_cal_point_when_target_unattainable():
    rows=[
        (.004,.015,.70,.95,{'id':'a'}),
        (.009,.012,.30,.80,{'id':'b'}),
        (.008,.030,.20,.98,{'id':'c'}),
    ]
    out=select_asymmetric_operating_point(rows,.01)
    assert out[4]['id']=='b'
