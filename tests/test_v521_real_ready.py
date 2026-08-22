from pathlib import Path
import csv
import hashlib
import cv2
import lightgbm
import numpy as np
import pytest

from crcv_core.features import build_features
from crcv_core.policy import TrainingConfig, build_training_matrices, train
from crcv_core.runtime import action_scores, refine
from crcv_core.safety import SafetyConfig, project_add
from scripts.train_v521_real import read_manifest, run, load_sample, qualify_val


def sample(name="a", s=48):
    gt=np.zeros((s,s),bool); gt[8:s-8,s//2:s//2+2]=True
    im=np.ones((s,s,3),np.float32); im[gt]=.1
    p=gt.astype(np.float32)*.82+.08; p[3:6,3:6]=.72
    p[s//2:s//2+3,s//2:s//2+2]=.35
    p[s//3:s//3+2,s//2+2:s//2+3]=.35
    return {"name":name,"source":"real","image":im,"gt":gt},p


def test_feature_contract_rejects_unscaled_image_and_probability():
    r,p=sample(); b=p>=.5
    with pytest.raises(ValueError): build_features(r["image"]*255,p,b)
    q=p.copy(); q[0,0]=1.1
    with pytest.raises(ValueError): build_features(r["image"],q,q>=.5)


def test_training_accepts_whitespace_name_only_after_canonical_strip():
    r,p=sample("  a  ")
    A,R,m=build_training_matrices([r],{"a":p},.5,7)
    assert len(A[1]) and len(R[1]) and m["method"]=="CRCV-V5.21"


def test_bad_learner_or_sample_config_fails_before_lightgbm():
    r,p=sample()
    with pytest.raises(ValueError): build_training_matrices([r],{"a":p},.5,config=TrainingConfig(n_estimators=0))
    with pytest.raises(ValueError): build_training_matrices([r],{"a":p},.5,config=TrainingConfig(add_max_positive=0))


def test_zero_add_budget_is_exact_noop():
    b=np.zeros((32,32),bool); b[16,6:12]=True
    cand=np.zeros_like(b); cand[16,12:20]=True; score=np.ones(b.shape,np.float32)
    add,info=project_add(b,cand,score,.5,SafetyConfig(max_add_foreground_fraction=0))
    assert not add.any() and info["budget"]==0 and info["status"]=="NO_OP_BUDGET"


def test_runtime_is_fail_closed_until_val_qualification():
    r,p=sample(); b=p>=.5
    out,info=refine(r["image"],p,.5,None,.1,.1)
    assert np.array_equal(out,b) and info["status"]=="NO_OP_UNQUALIFIED" and not info["add"].any() and not info["remove"].any()


def test_booster_reload_is_runtime_compatible(tmp_path):
    r,p=sample(); heads,_=train([r],{"a":p},.5,11)
    path=tmp_path/"add.txt"; heads["add"].booster_.save_model(str(path))
    reloaded={"add":lightgbm.Booster(model_file=str(path)),"remove":heads["remove"]}
    b=p>=.5
    a1,rm1,_,_=action_scores(heads,r["image"],p,b)
    a2,rm2,_,_=action_scores(reloaded,r["image"],p,b)
    assert np.allclose(a1,a2) and np.allclose(rm1,rm2)


def _write_manifest(root: Path, include_bad_split=False, leak=False, base_sha="a"*64):
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split_i,split in enumerate(("fit","cal","val")):
        for j in range(2):
            name=f"{split}_{j}"; r,p=sample(name,40)
            r["image"][0,split_i*3+j,0]=.2+.1*j
            ip=root/f"{name}.png"; mp=root/f"{name}_mask.png"; pp=root/f"{name}.npy"
            cv2.imwrite(str(ip),cv2.cvtColor((r["image"]*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mp),r["gt"].astype(np.uint8)*255); np.save(pp,p)
            lineage="shared" if leak and j==0 else f"{split}_{j}"
            rows.append({"name":name,"split":split,"source":"real","lineage":lineage,"image":ip.name,"mask":mp.name,"probability":pp.name,"base_artifact_sha256":base_sha})
    if include_bad_split:
        rows[-1]["split"]="test"
    path=root/"manifest.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return path


def test_val_topology_regression_fails_closed():
    m={"delta_precision":.01,"delta_recall":.01,"delta_dice":.01,"delta_cldice":-.0001,"tcrr":0.,"max_image_tcrr":0.}
    ok,fail=qualify_val(m)
    assert not ok and "delta_cldice < 0" in fail


def test_manifest_rejects_test_or_lineage_leakage(tmp_path):
    with pytest.raises(ValueError): read_manifest(_write_manifest(tmp_path/"a",include_bad_split=True))


def test_manifest_rejects_cross_split_lineage(tmp_path):
    root=tmp_path/"b"; root.mkdir()
    with pytest.raises(ValueError): read_manifest(_write_manifest(root,leak=True))


def test_manifest_rejects_exact_duplicate_image_even_with_new_name(tmp_path):
    root=tmp_path/"dup"; manifest=_write_manifest(root)
    rows=list(csv.DictReader(manifest.open()))
    rows[1]["image"]=rows[0]["image"]
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError): read_manifest(manifest)


def test_real_runner_fit_cal_val_and_saved_boosters(tmp_path):
    root=tmp_path/"data"; root.mkdir(); base=root/"base.ckpt"; base.write_bytes(b"frozen-base-smoke")
    base_sha=hashlib.sha256(base.read_bytes()).hexdigest(); manifest=_write_manifest(root,base_sha=base_sha)
    out=tmp_path/"out"; meta=run(manifest,base,.5,out,seed=17,target_gain=0.0,git_commit="abcdef1",dataset="D",backbone="unet",resolution=128)
    assert meta["status"] in {"ACTIVE","NO_OP_VAL","NO_OP_CAL"}
    assert meta["sample_counts"]=={"cal":2,"fit":2,"val":2}
    assert len(meta["dataset_content_sha256"])==64 and len(meta["config_sha256"])==64
    assert all(len(x)==64 for x in meta["artifact_sha256"].values())
    assert (out/"add_model.txt").is_file() and (out/"remove_model.txt").is_file() and (out/"run.json").is_file()
    rows=read_manifest(manifest); row=next(x for x in rows if x["split"]=="val"); rec,p=load_sample(row); b=p>=.5
    heads={"add":lightgbm.Booster(model_file=str(out/"add_model.txt")),"remove":lightgbm.Booster(model_file=str(out/"remove_model.txt"))}
    a,r,_,_=action_scores(heads,rec["image"],p,b)
    assert a.shape==p.shape==r.shape and np.isfinite(a).all() and np.isfinite(r).all()


def test_runtime_truthy_nonbool_qualification_is_fail_closed():
    r,p=sample(); heads,_=train([r],{"a":p},.5,11); b=p>=.5
    out,info=refine(r["image"],p,.5,heads,.1,.98,qualified="NO_OP_VAL")
    assert np.array_equal(out,b) and info["status"]=="NO_OP_UNQUALIFIED"


def test_manifest_binds_probability_cache_to_base_artifact(tmp_path):
    root=tmp_path/"bind"; root.mkdir(); base=root/"base.ckpt"; base.write_bytes(b"correct-base")
    manifest=_write_manifest(root,base_sha="0"*64)
    with pytest.raises(ValueError,match="probability provenance"):
        run(manifest,base,.5,tmp_path/"out",seed=1,target_gain=0.0,git_commit="abcdef1",dataset="D",backbone="unet",resolution=128)


def test_runner_requires_provenance_identity(tmp_path):
    root=tmp_path/"identity"; root.mkdir(); base=root/"base.ckpt"; base.write_bytes(b"base")
    sha=hashlib.sha256(base.read_bytes()).hexdigest(); manifest=_write_manifest(root,base_sha=sha)
    with pytest.raises(ValueError,match="git_commit"):
        run(manifest,base,.5,tmp_path/"o1",git_commit="bad",dataset="D",backbone="unet",resolution=128)
    with pytest.raises(ValueError,match="dataset and backbone"):
        run(manifest,base,.5,tmp_path/"o2",git_commit="abcdef1",dataset="",backbone="unet",resolution=128)
    with pytest.raises(ValueError,match="resolution"):
        run(manifest,base,.5,tmp_path/"o3",git_commit="abcdef1",dataset="D",backbone="unet",resolution=0)
    with pytest.raises(ValueError,match="target_gain"):
        run(manifest,base,.5,tmp_path/"o4",target_gain=float("nan"),git_commit="abcdef1",dataset="D",backbone="unet",resolution=128)
