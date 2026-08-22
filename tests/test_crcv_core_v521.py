import inspect
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from crcv_core.actions import action_targets
from crcv_core.features import build_features
from crcv_core.policy import (
    TrainingConfig,
    add_candidate,
    build_training_matrices,
    select_asymmetric_operating_point,
    train,
)
from crcv_core.runtime import refine
from crcv_core.safety import SafetyConfig, project_add, project_remove


def _record(name="a", size=64):
    gt=np.zeros((size,size),bool)
    for y in range(size//8,size-size//8):
        x=size//3+y//8; gt[y,max(0,x-1):min(size,x+2)]=True
    image=np.repeat((~gt)[...,None],3,axis=2).astype(np.float32)
    probability=gt.astype(np.float32)*.82+.08
    probability[2:5,2:5]=.75
    y=size//2; x=size//3+y//8
    probability[y:y+3,max(0,x-1):min(size,x+2)]=.35
    y=size//3; x=size//3+y//8
    probability[y:y+2,min(size-1,x+3):min(size,x+4)]=.35
    return {"name":name,"source":"real","image":image,"gt":gt},probability


def test_v521_exact_actions_and_runtime_is_gt_free():
    rec,p=_record(); b=p>=.5; target=action_targets(b,rec["gt"])
    assert np.array_equal(target["keep"],b&rec["gt"])
    assert np.array_equal(target["remove"],b&~rec["gt"])
    assert np.array_equal(target["add"],rec["gt"]&~b)
    assert "gt" not in inspect.signature(refine).parameters
    assert "gt_mask" not in inspect.signature(refine).parameters


def test_v521_core_surface_is_five_scientific_modules():
    root=Path(__file__).resolve().parents[1]/"crcv_core"
    modules=sorted(p.name for p in root.glob("*.py") if p.name!="__init__.py")
    assert modules==["actions.py","features.py","policy.py","runtime.py","safety.py"]
    assert sum(len(p.read_text().splitlines()) for p in root.glob("*.py"))<=260


def test_v521_features_are_compact_gt_free_and_finite():
    rec,p=_record(); X,names=build_features(rec["image"],p,p>=.5)
    assert X.shape[-1]==9 and np.isfinite(X).all()
    assert "blackhat" in names and "radial_ratio" in names
    assert not any(name in names for name in ("x","y","normalized_x","normalized_y"))


def test_v521_training_is_record_order_invariant():
    a,pa=_record("a"); b,pb=_record("b"); probs={"a":pa,"b":pb}
    A1,R1,m1=build_training_matrices([a,b],probs,.5,42)
    A2,R2,m2=build_training_matrices([b,a],probs,.5,42)
    assert np.array_equal(A1[0],A2[0]) and np.array_equal(A1[1],A2[1])
    assert np.array_equal(R1[0],R2[0]) and np.array_equal(R1[1],R2[1])
    assert m1["training_matrix_sha256"]==m2["training_matrix_sha256"]


def test_v521_add_remove_safety_invariants():
    base=np.zeros((64,64),bool); base[8:56,30:34]=True
    score=np.ones(base.shape,np.float32)
    removed,_=project_remove(base,score,.5)
    assert not np.any(removed&~base)
    assert not np.any(removed&skeletonize(base))
    cand=np.zeros_like(base); cand[20:30,34:36]=True
    added,_=project_add(base,cand,score,.5)
    assert not np.any(added&base) and not np.any(added&~cand)


def test_v521_add_requires_connection_and_remove_budget():
    base=np.zeros((40,40),bool); base[20,5:10]=True
    isolated=np.zeros_like(base); isolated[2:4,2:4]=True
    score=np.ones(base.shape,np.float32)
    added,_=project_add(base,isolated,score,.5)
    assert not added.any()
    thick=np.zeros((64,64),bool); thick[5:59,15:49]=True
    removed,info=project_remove(
        thick,np.ones(thick.shape,np.float32),.5,
        SafetyConfig(max_total_remove_fraction=.01,max_foreground_remove_fraction=.05),
    )
    assert int(removed.sum())<=info["budget"]


def test_v521_aac_is_action_asymmetric():
    rows=[
        (.012,.020,.80,.90,{"id":"conservative_add"}),
        (.018,.025,.20,.90,{"id":"better_balance"}),
        (.020,.030,.10,.85,{"id":"less_conservative_remove"}),
    ]
    out=select_asymmetric_operating_point(rows,.01)
    assert out[3]==.90 and out[2]==.20 and out[4]["id"]=="better_balance"


def test_v521_runtime_combination_and_external_gt_mutation_invariance():
    rec,p=_record("runtime"); heads,_=train([rec],{"runtime":p},.5,77)
    out1,info1=refine(rec["image"],p,.5,heads,.2,.9)
    mutated_gt=~rec["gt"]
    assert mutated_gt.shape==rec["gt"].shape
    out2,info2=refine(rec["image"],p,.5,heads,.2,.9)
    assert np.array_equal(out1,(info1["base"]&~info1["remove"])|info1["add"])
    assert np.array_equal(out1,out2)
    assert np.array_equal(info1["add"],info2["add"])
    assert np.array_equal(info1["remove"],info2["remove"])


def test_v521_embedded_core_provenance_version():
    import crcv_core
    assert crcv_core.__version__=="1.1.0"
