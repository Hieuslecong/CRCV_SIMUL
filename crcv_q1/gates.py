from __future__ import annotations

import json
import math
from pathlib import Path

from .evidence import validate_artifacts, validate_run_record
from .protocol import Q1Protocol


BASE_COMPARATORS = {"base", "morphology", "crcv_v5181"}
PUBLISHED_REFINER_ALIASES = {"stripcuts", "minimum_strip_cuts", "published_refiner"}


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _norm_set(values):
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _artifact_failures(section: dict, root: str | Path, label: str) -> list[str]:
    refs = section.get("artifacts", []) if isinstance(section, dict) else []
    return [f"{label}: {x}" for x in validate_artifacts(refs, root=root)]


def assess(payload: dict, protocol: Q1Protocol | None = None, root: str | Path = "."):
    """Assess a publication evidence package.

    The gate is fail-closed and evidence-backed. Boolean declarations are not
    sufficient for provenance, LOBO, external-test integrity, statistics or
    latency. Passing means EVIDENCE_COMPLETE under the frozen protocol; it does
    not mean a Q1 journal will accept the paper.
    """
    p = (protocol or Q1Protocol()).validate()
    failures: list[str] = []
    warnings: list[str] = []

    datasets = _norm_set(payload.get("datasets", []))
    backbones = _norm_set(payload.get("backbones", []))
    reference_backbones = _norm_set(payload.get("reference_backbones", []))
    resolutions = set(payload.get("resolutions", []))
    full_seeds = set(payload.get("full_training_seeds", []))
    routes = payload.get("cross_dataset_routes", [])
    comparators = _norm_set(payload.get("comparators", []))

    if len(datasets) < p.min_datasets:
        failures.append(f"datasets {len(datasets)} < {p.min_datasets}")
    if len(backbones) < p.min_backbones:
        failures.append(f"backbones {len(backbones)} < {p.min_backbones}")
    if len(reference_backbones) < p.min_reference_backbones:
        failures.append(
            f"reference backbones {len(reference_backbones)} < {p.min_reference_backbones}"
        )
    if not reference_backbones.issubset(backbones):
        failures.append("reference_backbones must be included in backbones")
    if not set(p.resolutions).issubset(resolutions):
        failures.append(f"missing required resolutions {set(p.resolutions) - resolutions}")
    if full_seeds != set(p.full_seeds):
        failures.append(
            f"full training seeds must equal frozen seeds {list(p.full_seeds)}; got {sorted(full_seeds)}"
        )
    if len(routes) < p.min_cross_dataset_routes:
        failures.append("insufficient cross-dataset routes")

    if not BASE_COMPARATORS.issubset(comparators):
        failures.append(f"missing core comparators {BASE_COMPARATORS - comparators}")
    if p.require_published_refiner:
        declared = str(payload.get("published_refiner", "")).strip().lower()
        has_alias = bool(comparators & PUBLISHED_REFINER_ALIASES)
        if not declared and not has_alias:
            failures.append("published refinement comparator missing")
        if declared and declared not in comparators:
            failures.append("published_refiner must also be listed in comparators")

    # Historical exposure / final-test integrity must be backed by an audit artifact.
    exposure = payload.get("historical_exposure_audit", {})
    if p.require_historical_exposure_guard:
        if exposure.get("status") != "PASS":
            failures.append("historical exposure audit missing or failed")
        failures.extend(_artifact_failures(exposure, root, "historical exposure audit"))

    final_state = payload.get("final_test_state")
    if p.require_final_sealed_until_freeze and final_state not in {
        "SEALED_FRESH_EXTERNAL",
        "FROZEN_ONE_SHOT_DONE",
    }:
        failures.append("final test must be fresh-external sealed or one-shot completed")

    external = payload.get("external", {})
    if int(external.get("datasets_completed", 0)) < p.min_external_datasets:
        failures.append("independent external dataset missing")
    if not external.get("fresh_unexposed", False):
        failures.append("external test is not certified fresh/unexposed")
    failures.extend(_artifact_failures(external, root, "external evidence"))

    # LOBO requires every declared backbone to appear as a held-out architecture.
    lobo = payload.get("lobo", {})
    if p.require_lobo:
        held_out = _norm_set(lobo.get("held_out_backbones", []))
        missing = backbones - held_out
        if missing:
            failures.append(f"LOBO missing held-out backbones {sorted(missing)}")
        failures.extend(_artifact_failures(lobo, root, "LOBO evidence"))

    # End-to-end GT independence is a mutation-test artifact, not a flag.
    gt_test = payload.get("runtime_gt_mutation_test", {})
    if p.require_runtime_gt_mutation_test:
        if gt_test.get("status") != "PASS":
            failures.append("full-pipeline GT mutation test missing or failed")
        failures.extend(_artifact_failures(gt_test, root, "GT mutation test"))

    # Every run record must bind code/data/config/results by immutable hashes.
    run_records = payload.get("run_records", [])
    if p.require_artifact_provenance:
        if not run_records:
            failures.append("run provenance records missing")
        for i, record in enumerate(run_records):
            failures.extend(
                f"run[{i}]: {msg}" for msg in validate_run_record(record, root=root)
            )

        observed_datasets = _norm_set(r.get("dataset") for r in run_records)
        observed_backbones = _norm_set(r.get("backbone") for r in run_records)
        observed_resolutions = {r.get("resolution") for r in run_records}
        observed_seeds = {r.get("seed") for r in run_records}
        if datasets - observed_datasets:
            failures.append(f"provenance missing datasets {sorted(datasets - observed_datasets)}")
        if backbones - observed_backbones:
            failures.append(f"provenance missing backbones {sorted(backbones - observed_backbones)}")
        if set(p.resolutions) - observed_resolutions:
            failures.append(
                f"provenance missing resolutions {sorted(set(p.resolutions) - observed_resolutions)}"
            )
        if set(p.full_seeds) - observed_seeds:
            failures.append(f"provenance missing full seeds {sorted(set(p.full_seeds) - observed_seeds)}")

    aggregate = payload.get("aggregate", {})
    for key, floor, label in (
        ("mean_delta_dice", p.mean_dice_gain_floor, "mean ΔDice"),
        ("mean_delta_crack_iou", p.mean_crack_iou_gain_floor, "mean ΔCrackIoU"),
    ):
        value = aggregate.get(key)
        if not _finite(value):
            failures.append(f"aggregate {key} missing")
        elif value < floor:
            failures.append(f"{label} below protocol floor")

    positive_rate = aggregate.get("positive_pair_rate")
    if not _finite(positive_rate):
        failures.append("positive_pair_rate missing")
    elif positive_rate < p.pair_positive_rate_floor:
        failures.append("positive backbone×dataset pair rate below floor")

    worst = aggregate.get("worst_delta_dice")
    if not _finite(worst):
        failures.append("worst_delta_dice missing")
    elif worst < p.catastrophic_dice_floor:
        failures.append("catastrophic negative ΔDice pair")

    stats = payload.get("statistics", {})
    for metric in p.primary_metrics:
        s = stats.get(metric, {})
        if not (_finite(s.get("ci_low")) and _finite(s.get("holm_p"))):
            failures.append(f"paired statistics missing for {metric}")
            continue
        if p.require_cluster_statistics and not str(s.get("method", "")).startswith(
            "paired_cluster"
        ):
            failures.append(f"{metric} statistics are not cluster/lineage-aware")
        if p.bootstrap_ci_must_exclude_zero and s["ci_low"] <= 0:
            failures.append(f"{metric} bootstrap CI crosses zero")
        if s["holm_p"] >= p.corrected_p_floor:
            failures.append(f"{metric} Holm-corrected p not significant")

    latency = payload.get("latency", {})
    if p.require_cpu_latency:
        cpu = latency.get("cpu", {})
        if not isinstance(cpu, dict) or not _finite(cpu.get("median_ms")):
            failures.append("CPU latency measurement missing")
        else:
            failures.extend(_artifact_failures(cpu, root, "CPU latency"))

    if payload.get("claims", {}).get("edge", False) and p.require_edge_latency_if_claimed:
        edge = latency.get("edge", {})
        if not isinstance(edge, dict) or not _finite(edge.get("median_ms")):
            failures.append("edge latency required by edge claim")
        else:
            failures.extend(_artifact_failures(edge, root, "edge latency"))

    if payload.get("claims", {}).get("topology_aware", False):
        structural = payload.get("structural_metrics", {})
        if not any(k in structural for k in ("cldice", "cts", "fragmentation", "false_bridge_rate")):
            failures.append("topology-aware claim lacks a structural/topology metric")

    status = "EVIDENCE_COMPLETE" if not failures else "BLOCKED"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "n_failures": len(failures),
        "protocol_version": p.version,
    }


def assess_file(path, protocol: Q1Protocol | None = None, root: str | Path | None = None):
    p = Path(path)
    data = json.loads(p.read_text())
    return assess(data, protocol=protocol, root=p.parent if root is None else root)
