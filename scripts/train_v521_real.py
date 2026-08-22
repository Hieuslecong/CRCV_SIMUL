from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import lightgbm
import numpy as np
from skimage.morphology import skeletonize

from crcv_core.features import build_features
from crcv_core.policy import TrainingConfig, add_candidate, select_asymmetric_operating_point, train
from crcv_core.runtime import action_scores
from crcv_core.safety import SafetyConfig, project_add, project_remove

ALLOWED_SPLITS = {"fit", "cal", "val"}
ADD_THRESHOLDS = tuple(float(x) for x in np.linspace(.10, .90, 17))
REMOVE_THRESHOLDS = tuple(float(x) for x in np.linspace(.35, .98, 15))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"name", "split", "source", "lineage", "image", "mask", "probability"}
    if not rows:
        raise ValueError("empty manifest")
    names, lineage_split, image_hash_owner = set(), {}, {}
    out = []
    for i, row in enumerate(rows):
        missing = [k for k in required if not str(row.get(k, "")).strip()]
        if missing:
            raise ValueError(f"manifest row {i}: missing {','.join(sorted(missing))}")
        name = str(row["name"]).strip()
        split = str(row["split"]).strip().lower()
        lineage = str(row["lineage"]).strip()
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"manifest row {i}: split {split!r} is not FIT/CAL/VAL")
        if name in names:
            raise ValueError(f"duplicate sample name: {name}")
        names.add(name)
        previous = lineage_split.setdefault(lineage, split)
        if previous != split:
            raise ValueError(f"lineage leakage: {lineage} in {previous} and {split}")
        root = path.parent
        image = _resolve(root, str(row["image"]).strip())
        mask = _resolve(root, str(row["mask"]).strip())
        probability = _resolve(root, str(row["probability"]).strip())
        for p in (image, mask, probability):
            if not p.is_file():
                raise FileNotFoundError(p)
        image_sha = sha256_file(image); mask_sha = sha256_file(mask); prob_sha = sha256_file(probability)
        old_owner = image_hash_owner.setdefault(image_sha, name)
        if old_owner != name:
            raise ValueError(f"exact duplicate image: {name} duplicates {old_owner}")
        out.append({
            "name": name,
            "split": split,
            "source": str(row["source"]).strip(),
            "lineage": lineage,
            "image": image,
            "mask": mask,
            "probability": probability,
            "image_sha256": image_sha, "mask_sha256": mask_sha, "probability_sha256": prob_sha,
        })
    if {r["split"] for r in out} != ALLOWED_SPLITS:
        raise ValueError("manifest must contain non-empty FIT, CAL and VAL splits")
    return sorted(out, key=lambda r: (r["split"], r["name"]))


def load_sample(row: dict) -> tuple[dict, np.ndarray]:
    bgr = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(row["mask"]), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None:
        raise ValueError(f"unreadable image/mask: {row['name']}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt = mask > 127
    prob = np.asarray(np.load(row["probability"], allow_pickle=False), np.float32)
    if gt.shape != image.shape[:2] or prob.shape != gt.shape:
        raise ValueError(f"registration/shape mismatch: {row['name']}")
    if not np.isfinite(prob).all() or (prob.size and (float(prob.min()) < 0 or float(prob.max()) > 1)):
        raise ValueError(f"invalid probability range: {row['name']}")
    return {"name": row["name"], "source": row["source"], "image": image, "gt": gt}, prob


def load_splits(rows: list[dict]):
    records = {k: [] for k in ALLOWED_SPLITS}
    probs = {k: {} for k in ALLOWED_SPLITS}
    for row in rows:
        rec, prob = load_sample(row)
        records[row["split"]].append(rec)
        probs[row["split"]][rec["name"]] = prob
    return records, probs


def _confusion(pred, gt):
    p = np.asarray(pred, bool); g = np.asarray(gt, bool)
    return np.array([(p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()], np.int64)


def _seg(c):
    tp, fp, fn, tn = map(float, c); eps = 1e-12
    precision = tp / (tp + fp + eps); recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps); iou = tp / (tp + fp + fn + eps)
    return {"precision": precision, "recall": recall, "dice": dice, "crack_iou": iou}


def _cldice(pred, gt):
    p = np.asarray(pred, bool); g = np.asarray(gt, bool); eps = 1e-12
    if not p.any() and not g.any():
        return 1.0
    sp = skeletonize(p); sg = skeletonize(g)
    tprec = float((sp & g).sum()) / (float(sp.sum()) + eps)
    tsens = float((sg & p).sum()) / (float(sg.sum()) + eps)
    return 2 * tprec * tsens / (tprec + tsens + eps)


def prepare(records, probabilities, base_threshold, heads, tcfg):
    out = []
    for rec in records:
        p = probabilities[rec["name"]]; b = p >= base_threshold
        a, r, X, names = action_scores(heads, rec["image"], p, b)
        cand = add_candidate(p, b, X, names, base_threshold, tcfg)
        out.append((rec, b, cand, a, r))
    return out


def projections(prepared, safety):
    adds = {t: [project_add(b, cand, a, t, safety)[0] for rec, b, cand, a, r in prepared] for t in ADD_THRESHOLDS}
    rems = {t: [project_remove(b, r, t, safety)[0] for rec, b, cand, a, r in prepared] for t in REMOVE_THRESHOLDS}
    return adds, rems


def evaluate(prepared, adds, rems, add_tau, remove_tau, topology=False):
    c0 = np.zeros(4, np.int64); c1 = np.zeros(4, np.int64)
    true_removed = fp_removed = add_tp = add_total = base_fn = mutated = 0
    max_image_tcrr = 0.0; base_cl, refined_cl = [], []
    for i, (rec, b, cand, a, r) in enumerate(prepared):
        rm = rems[remove_tau][i]; add = adds[add_tau][i]; q = (b & ~rm) | add; gt = rec["gt"]
        c0 += _confusion(b, gt); c1 += _confusion(q, gt)
        tr = int((rm & gt).sum()); fr = int((rm & ~gt).sum())
        true_removed += tr; fp_removed += fr; add_tp += int((add & gt).sum()); add_total += int(add.sum()); base_fn += int((~b & gt).sum())
        mutated += int(bool(rm.any() or add.any())); max_image_tcrr = max(max_image_tcrr, tr / max(int((b & gt).sum()), 1))
        if topology:
            base_cl.append(_cldice(b, gt)); refined_cl.append(_cldice(q, gt))
    bm = _seg(c0); m = _seg(c1)
    result = {
        "base": bm, "refined": m,
        "delta_precision": m["precision"] - bm["precision"],
        "delta_recall": m["recall"] - bm["recall"],
        "delta_dice": m["dice"] - bm["dice"],
        "delta_crack_iou": m["crack_iou"] - bm["crack_iou"],
        "tcrr": true_removed / max(int(c0[0]), 1),
        "max_image_tcrr": max_image_tcrr,
        "fprr": fp_removed / max(int(c0[1]), 1),
        "add_precision": add_tp / max(add_total, 1),
        "add_recall": add_tp / max(base_fn, 1),
        "mutated_images": mutated,
    }
    if topology:
        result.update({
            "base_cldice": float(np.mean(base_cl)),
            "refined_cldice": float(np.mean(refined_cl)),
            "delta_cldice": float(np.mean(refined_cl) - np.mean(base_cl)),
        })
    return result


def qualify_val(metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    for key in ("delta_precision", "delta_recall", "delta_dice", "delta_cldice"):
        if float(metrics[key]) < 0:
            failures.append(f"{key} < 0")
    if float(metrics["tcrr"]) > .003:
        failures.append("tcrr > 0.003")
    if float(metrics["max_image_tcrr"]) > .02:
        failures.append("max_image_tcrr > 0.02")
    return not failures, failures


def run(manifest, base_artifact, base_threshold, out_dir, seed=1337, target_gain=.01):
    if not np.isfinite(base_threshold) or not 0 <= float(base_threshold) <= 1:
        raise ValueError("base_threshold must be in [0,1]")
    rows = read_manifest(manifest); records, probs = load_splits(rows)
    tcfg = TrainingConfig(); safety = SafetyConfig()
    heads, train_meta = train(records["fit"], probs["fit"], float(base_threshold), int(seed), tcfg)
    cal = prepare(records["cal"], probs["cal"], float(base_threshold), heads, tcfg)
    adds, rems = projections(cal, safety); candidates = []
    for at in ADD_THRESHOLDS:
        for rt in REMOVE_THRESHOLDS:
            m = evaluate(cal, adds, rems, at, rt, topology=False)
            if m["tcrr"] <= .003 and m["max_image_tcrr"] <= .02:
                candidates.append((min(m["delta_precision"], m["delta_recall"], m["delta_dice"]), m["delta_dice"], at, rt, m))
    selected = select_asymmetric_operating_point(candidates, target_gain) if candidates else None
    status, failures, cal_metrics, val_metrics, add_tau, remove_tau = "NO_OP_CAL", ["no safe CAL operating point"], None, None, None, None
    if selected is not None:
        _, _, add_tau, remove_tau, cal_metrics = selected
        val = prepare(records["val"], probs["val"], float(base_threshold), heads, tcfg)
        va, vr = projections(val, safety)
        val_metrics = evaluate(val, va, vr, add_tau, remove_tau, topology=True)
        ok, failures = qualify_val(val_metrics); status = "ACTIVE" if ok else "NO_OP_VAL"

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    add_model=out / "add_model.txt"; remove_model=out / "remove_model.txt"
    heads["add"].booster_.save_model(str(add_model)); heads["remove"].booster_.save_model(str(remove_model))
    content_index=[{k:(str(r[k]) if isinstance(r[k],Path) else r[k]) for k in ("name","split","source","lineage","image_sha256","mask_sha256","probability_sha256")} for r in rows]
    config_payload={"seed":int(seed),"base_threshold":float(base_threshold),"target_gain":float(target_gain),"training_config":tcfg.__dict__,"safety_config":safety.__dict__,"threshold_grid":{"add":ADD_THRESHOLDS,"remove":REMOVE_THRESHOLDS}}
    meta = {
        "method": "CRCV-V5.21", "repository_version": "5.21.0.dev2", "core_version": "1.1.1",
        "status": status, "failures": failures, "seed": int(seed), "base_threshold": float(base_threshold),
        "add_threshold": add_tau, "remove_threshold": remove_tau, "target_gain": float(target_gain),
        "training": train_meta, "cal": cal_metrics, "val": val_metrics,
        "manifest_sha256": sha256_file(manifest), "dataset_content_sha256": sha256_json(content_index), "config_sha256": sha256_json(config_payload),
        "base_artifact_sha256": sha256_file(base_artifact), "artifact_sha256": {"add_model":sha256_file(add_model),"remove_model":sha256_file(remove_model)},
        "sample_counts": {k: len(records[k]) for k in sorted(records)},
        "lineages": {k: len({r["lineage"] for r in rows if r["split"] == k}) for k in sorted(ALLOWED_SPLITS)},
        "safety_config": safety.__dict__, "threshold_grid": {"add": ADD_THRESHOLDS, "remove": REMOVE_THRESHOLDS},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "lightgbm": lightgbm.__version__},
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    p = argparse.ArgumentParser(description="Train/calibrate CRCV V5.21 from frozen real-data Base probabilities; never opens TEST/final data.")
    p.add_argument("--manifest", required=True); p.add_argument("--base-artifact", required=True)
    p.add_argument("--base-threshold", required=True, type=float); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1337); p.add_argument("--target-gain", type=float, default=.01)
    a = p.parse_args(); result = run(a.manifest, a.base_artifact, a.base_threshold, a.out, a.seed, a.target_gain)
    print(json.dumps({k: result[k] for k in ("status", "failures", "add_threshold", "remove_threshold", "sample_counts")}, indent=2))


if __name__ == "__main__":
    main()
