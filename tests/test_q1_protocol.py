import hashlib
import json
from pathlib import Path

from crcv_q1.gates import assess
from crcv_q1.protocol import Q1Protocol, hash_protocol
from crcv_q1.split_guard import audit_rows
from crcv_q1.stats import (
    cluster_bootstrap_ci,
    cluster_permutation_p,
    holm_bonferroni,
)


def _artifact(tmp_path: Path, name: str, content: str = "evidence") -> dict:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {
        "path": str(p.relative_to(tmp_path)),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "kind": "test",
    }


def _run(tmp_path: Path, dataset: str, backbone: str, seed: int, resolution: int) -> dict:
    artifact = _artifact(
        tmp_path,
        f"runs/{dataset}-{backbone}-{seed}-{resolution}.json",
        json.dumps({"ok": True}),
    )
    digest = "a" * 64
    return {
        "experiment_id": f"{dataset}-{backbone}-{seed}-{resolution}",
        "git_commit": "abcdef1234567890",
        "dataset_manifest_sha256": digest,
        "split_manifest_sha256": digest,
        "config_sha256": digest,
        "base_artifact_sha256": digest,
        "probability_provenance_bound": True,
        "seed": seed,
        "backbone": backbone,
        "dataset": dataset,
        "resolution": resolution,
        "method": "crcv_v521",
        "artifacts": [artifact],
    }


def good_payload(tmp_path: Path) -> dict:
    backbones = ["unet", "deeplabv3p", "fastscnn", "bisenet", "dsunet"]
    datasets = ["a", "b", "c"]
    runs = []
    for i, backbone in enumerate(backbones):
        runs.append(_run(tmp_path, datasets[i % 3], backbone, [1337, 2027, 31415][i % 3], 128 if i % 2 == 0 else 256))
    runs.extend([
        _run(tmp_path, "b", "unet", 2027, 256),
        _run(tmp_path, "c", "deeplabv3p", 31415, 128),
    ])
    shared = _artifact(tmp_path, "evidence/audit.json")
    return {
        "proposed_method": "crcv_v521",
        "datasets": datasets,
        "backbones": backbones,
        "reference_backbones": ["unet", "deeplabv3p"],
        "resolutions": [128, 256],
        "full_training_seeds": [1337, 2027, 31415],
        "cross_dataset_routes": ["a+b->c", "a+c->b"],
        "comparators": ["base", "morphology", "stripcuts", "crcv_v5181"],
        "published_refiner": "stripcuts",
        "historical_exposure_audit": {"status": "PASS", "artifacts": [shared]},
        "final_test_state": "SEALED_FRESH_EXTERNAL",
        "external": {"datasets_completed": 1, "fresh_unexposed": True, "artifacts": [shared]},
        "lobo": {"held_out_backbones": backbones, "artifacts": [shared]},
        "runtime_gt_mutation_test": {"status": "PASS", "artifacts": [shared]},
        "run_records": runs,
        "aggregate": {
            "mean_delta_dice": 0.015,
            "mean_delta_crack_iou": 0.008,
            "positive_pair_rate": 0.9,
            "worst_delta_dice": -0.001,
        },
        "statistics": {
            "dice": {"ci_low": 0.003, "holm_p": 0.01, "method": "paired_cluster_bootstrap"},
            "crack_iou": {"ci_low": 0.002, "holm_p": 0.02, "method": "paired_cluster_bootstrap"},
        },
        "latency": {"cpu": {"median_ms": 12.5, "artifacts": [shared]}},
        "claims": {"edge": False, "topology_aware": False},
    }


def test_protocol_hash_stable():
    p = Q1Protocol().validate()
    assert hash_protocol(p) == hash_protocol(p)


def test_cluster_statistics_positive():
    base = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6]
    refined = [0.23, 0.32, 0.43, 0.53, 0.58, 0.63]
    clusters = ["p1", "p1", "p2", "p2", "p3", "p3"]
    ci = cluster_bootstrap_ci(base, refined, clusters, n_boot=1000)
    pv = cluster_permutation_p(base, refined, clusters, n_perm=2000)
    assert ci["ci_low"] > 0
    assert ci["n_clusters"] == 3
    assert 0 <= pv["p"] <= 1


def test_holm_bounds():
    adjusted = holm_bonferroni([0.01, 0.02])
    assert all(0 <= x <= 1 for x in adjusted)


def test_good_evidence_package_passes(tmp_path):
    assert assess(good_payload(tmp_path), root=tmp_path)["status"] == "EVIDENCE_COMPLETE"


def test_published_refiner_is_hard_gate(tmp_path):
    payload = good_payload(tmp_path)
    payload["comparators"].remove("stripcuts")
    payload["published_refiner"] = ""
    result = assess(payload, root=tmp_path)
    assert result["status"] == "BLOCKED"
    assert any("published refinement comparator" in x for x in result["failures"])


def test_self_declared_lobo_cannot_pass(tmp_path):
    payload = good_payload(tmp_path)
    payload["lobo"] = {"completed": True, "artifacts": payload["lobo"]["artifacts"]}
    assert assess(payload, root=tmp_path)["status"] == "BLOCKED"


def test_artifact_hash_mismatch_blocks(tmp_path):
    payload = good_payload(tmp_path)
    payload["external"]["artifacts"][0]["sha256"] = "0" * 64
    assert assess(payload, root=tmp_path)["status"] == "BLOCKED"


def test_historical_exposure_guard_rejects_final_sample():
    rows = [
        {"sample_id": "a", "split": "train", "lineage_id": "p1", "source_dataset": "D", "historically_exposed": "true"},
        {"sample_id": "b", "split": "final_external", "lineage_id": "p2", "source_dataset": "E", "historically_exposed": "true"},
    ]
    result = audit_rows(rows)
    assert result["status"] == "FAIL"
    assert any("historically exposed" in x for x in result["failures"])


def test_legacy_only_run_records_cannot_satisfy_v521_gate(tmp_path):
    payload=good_payload(tmp_path)
    for run in payload["run_records"]: run["method"]="crcv_v5181"
    result=assess(payload,root=tmp_path)
    assert result["status"]=="BLOCKED"
    assert any("proposed method" in x or "proposed-method" in x for x in result["failures"])


def test_declared_proposed_method_is_frozen(tmp_path):
    payload=good_payload(tmp_path); payload["proposed_method"]="crcv_v5181"
    result=assess(payload,root=tmp_path)
    assert result["status"]=="BLOCKED" and any("proposed_method" in x for x in result["failures"])
