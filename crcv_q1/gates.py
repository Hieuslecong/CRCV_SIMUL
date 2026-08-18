from __future__ import annotations
import json, math
from pathlib import Path
from .protocol import Q1Protocol

REQUIRED_COMPARATORS={"base","morphology","crcv_v5181"}

def _finite(x): return isinstance(x,(int,float)) and math.isfinite(float(x))

def assess(payload:dict, protocol:Q1Protocol|None=None):
    p=(protocol or Q1Protocol()).validate(); failures=[]; warnings=[]
    datasets=payload.get("datasets",[]); backbones=payload.get("backbones",[]); resolutions=payload.get("resolutions",[])
    full_seeds=payload.get("full_training_seeds",[]); routes=payload.get("cross_dataset_routes",[]); comps=set(payload.get("comparators",[]))
    if len(set(datasets))<p.min_datasets: failures.append(f"datasets {len(set(datasets))} < {p.min_datasets}")
    if len(set(backbones))<p.min_backbones: failures.append(f"backbones {len(set(backbones))} < {p.min_backbones}")
    if not set(p.resolutions).issubset(set(resolutions)): failures.append(f"missing required resolutions {set(p.resolutions)-set(resolutions)}")
    if len(set(full_seeds))<len(p.full_seeds): failures.append("fewer than 3 full end-to-end training seeds")
    if len(routes)<p.min_cross_dataset_routes: failures.append("insufficient cross-dataset routes")
    if p.require_lobo and not payload.get("lobo",{}).get("completed",False): failures.append("LOBO unseen-backbone validation missing")
    if p.require_official_backbone and not payload.get("official_backbone",False): failures.append("official backbone baseline missing")
    if not REQUIRED_COMPARATORS.issubset(comps): failures.append(f"missing comparators {REQUIRED_COMPARATORS-comps}")
    lat=payload.get("latency",{})
    if p.require_cpu_latency and not lat.get("cpu",False): failures.append("CPU latency missing")
    if p.require_edge_latency and not lat.get("edge",False): failures.append("edge latency missing")
    if p.require_final_sealed_until_freeze and payload.get("final_test_state") not in {"SEALED","FROZEN_ONE_SHOT_DONE"}: failures.append("invalid final-test state")
    agg=payload.get("aggregate",{})
    if not _finite(agg.get("mean_delta_f1")): failures.append("aggregate mean_delta_f1 missing")
    elif agg["mean_delta_f1"]<p.mean_f1_gain_floor: failures.append("mean ΔF1 below floor")
    if not _finite(agg.get("mean_delta_miou")): failures.append("aggregate mean_delta_mIoU missing")
    elif agg["mean_delta_miou"]<p.mean_miou_gain_floor: failures.append("mean ΔmIoU below floor")
    if _finite(agg.get("positive_pair_rate")) and agg["positive_pair_rate"]<p.pair_positive_rate_floor: failures.append("positive backbone×dataset pair rate below floor")
    elif not _finite(agg.get("positive_pair_rate")): failures.append("positive_pair_rate missing")
    worst=agg.get("worst_delta_f1")
    if not _finite(worst): failures.append("worst_delta_f1 missing")
    elif worst<p.catastrophic_f1_floor: failures.append("catastrophic negative ΔF1 pair")
    stats=payload.get("statistics",{})
    for metric in ("f1","miou"):
        s=stats.get(metric,{})
        if not (_finite(s.get("ci_low")) and _finite(s.get("holm_p"))): failures.append(f"paired statistics missing for {metric}")
        else:
            if p.bootstrap_ci_must_exclude_zero and s["ci_low"]<=0: failures.append(f"{metric} bootstrap CI crosses zero")
            if s["holm_p"]>=p.corrected_p_floor: failures.append(f"{metric} Holm-corrected p not significant")
    ext=payload.get("external",{})
    if int(ext.get("datasets_completed",0))<p.min_external_datasets: failures.append("independent external dataset missing")
    status="Q1_READY" if not failures else "BLOCKED"
    return {"status":status,"failures":failures,"warnings":warnings,"n_failures":len(failures)}

def assess_file(path):
    data=json.loads(Path(path).read_text()); return assess(data)
