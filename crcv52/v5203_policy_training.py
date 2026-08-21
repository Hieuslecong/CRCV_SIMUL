from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
from lightgbm import LGBMClassifier

from .counterfactual_errors import CounterfactualConfig, simulate_corruption
from .removal_policy import sample_keep_remove_training


@dataclass(frozen=True)
class V5203PolicyTrainingConfig:
    natural_max_keep: int = 1400
    natural_max_remove: int = 450
    natural_boundary_keep_fraction: float = 0.60
    natural_remove_min_distance_ratio: float = 1.25
    synthetic_max_keep: int = 260
    synthetic_max_remove: int = 120
    synthetic_boundary_keep_fraction: float = 0.70
    synthetic_remove_min_distance_ratio: float = 1.25
    synthetic_probability_lift_min: float = 0.01
    synthetic_probability_lift_max: float = 0.12
    counterfactual_operators: tuple[str, ...] = (
        "width_dilate", "side_spur", "isolated_blob", "false_bridge"
    )
    n_estimators: int = 160
    learning_rate: float = 0.04
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 30
    reg_lambda: float = 0.5
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    n_jobs: int = 1
    deterministic: bool = True


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return int(value % (2**31 - 1))


def _record_name(record: dict) -> str:
    name = str(record.get("name", "")).strip()
    if not name:
        raise ValueError("every FIT record requires a stable non-empty name")
    return name


def _validate_record(record: dict, probability: np.ndarray) -> None:
    name = _record_name(record)
    image = np.asarray(record.get("image"), np.float32)
    gt = np.asarray(record.get("gt"), bool)
    prob = np.asarray(probability, np.float32)
    if image.ndim != 3 or gt.ndim != 2 or image.shape[:2] != gt.shape or prob.shape != gt.shape:
        raise ValueError(f"bad FIT shapes for {name}")
    if not np.isfinite(image).all() or not np.isfinite(prob).all():
        raise ValueError(f"non-finite FIT data for {name}")


def counterfactual_training_rows(record, real_probability, base_threshold: float,
                                 seed: int = 1337,
                                 simulator_config: CounterfactualConfig | None = None,
                                 config: V5203PolicyTrainingConfig | None = None):
    """Build canonical V5.20.3 rows from one FIT record.

    Randomness is keyed by record name and operator, not by record-list position,
    so reordering the FIT manifest cannot silently change the training corpus.
    """
    cfg = config or V5203PolicyTrainingConfig()
    sim_cfg = simulator_config or CounterfactualConfig()
    name = _record_name(record)
    image = np.asarray(record["image"], np.float32)
    gt = np.asarray(record["gt"], bool)
    real_probability = np.asarray(real_probability, np.float32)
    _validate_record(record, real_probability)
    if not np.isfinite(float(base_threshold)) or not 0 <= float(base_threshold) <= 1:
        raise ValueError("base_threshold must be finite and in [0,1]")
    real_base = real_probability >= float(base_threshold)
    Xs, ys, feature_names = [], [], None
    source_counts = {"natural": 0, "synthetic": 0}

    nat_seed = _stable_seed(seed, name, "natural")
    X, y, feature_names = sample_keep_remove_training(
        image, real_probability, real_base, gt,
        max_keep=cfg.natural_max_keep,
        max_remove=cfg.natural_max_remove,
        boundary_keep_fraction=cfg.natural_boundary_keep_fraction,
        remove_min_distance_ratio=cfg.natural_remove_min_distance_ratio,
        seed=nat_seed,
    )
    if len(y):
        Xs.append(X); ys.append(y); source_counts["natural"] += int(len(y))

    for op in cfg.counterfactual_operators:
        op_seed = _stable_seed(seed, name, op, "geometry")
        corr, _ = simulate_corruption(gt, op, seed=op_seed, config=sim_cfg)
        added = corr & ~gt
        if not added.any():
            continue
        cbase = real_base | added
        cprob = real_probability.copy()
        rng = np.random.default_rng(_stable_seed(seed, name, op, "probability"))
        lift = np.clip(
            base_threshold + rng.uniform(
                cfg.synthetic_probability_lift_min,
                cfg.synthetic_probability_lift_max,
                size=int(added.sum()),
            ),
            0, 0.995,
        ).astype(np.float32)
        cprob[added] = np.maximum(cprob[added], lift)
        X, y, names2 = sample_keep_remove_training(
            image, cprob, cbase, gt,
            max_keep=cfg.synthetic_max_keep,
            max_remove=cfg.synthetic_max_remove,
            boundary_keep_fraction=cfg.synthetic_boundary_keep_fraction,
            remove_min_distance_ratio=cfg.synthetic_remove_min_distance_ratio,
            seed=_stable_seed(seed, name, op, "sampling"),
        )
        if feature_names is None:
            feature_names = names2
        elif names2 != feature_names:
            raise AssertionError("feature schema changed within one training build")
        if len(y):
            Xs.append(X); ys.append(y); source_counts["synthetic"] += int(len(y))
    return Xs, ys, feature_names, source_counts


def build_training_matrix(fit_records, fit_probabilities: dict, base_threshold: float,
                          seed: int = 1337,
                          simulator_config: CounterfactualConfig | None = None,
                          config: V5203PolicyTrainingConfig | None = None):
    cfg = config or V5203PolicyTrainingConfig()
    records = sorted(list(fit_records), key=_record_name)
    names_seen = [_record_name(r) for r in records]
    if len(names_seen) != len(set(names_seen)):
        raise ValueError("duplicate FIT record names are not allowed")
    missing = [n for n in names_seen if n not in fit_probabilities]
    if missing:
        raise KeyError(f"missing FIT probabilities for: {missing[:3]}")
    Xs, ys, feature_names = [], [], None
    source_counts = {"natural": 0, "synthetic": 0}
    for record in records:
        n = _record_name(record)
        a, b, fnames, counts = counterfactual_training_rows(
            record, fit_probabilities[n], base_threshold, seed,
            simulator_config, cfg,
        )
        if feature_names is None:
            feature_names = fnames
        elif fnames != feature_names:
            raise AssertionError("feature schema changed across FIT records")
        Xs += a; ys += b
        for k, v in counts.items():
            source_counts[k] += int(v)
    if not Xs:
        raise ValueError("no policy training rows generated")
    X = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int8, copy=False)
    if len(np.unique(y)) < 2:
        raise ValueError("KEEP/REMOVE training requires both classes")
    schema_json = json.dumps(feature_names, separators=(",", ":"), ensure_ascii=True)
    feature_schema_sha256 = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
    return X, y, feature_names, {
        "n": int(len(y)),
        "keep": int((y == 0).sum()),
        "remove": int((y == 1).sum()),
        "remove_fraction": float((y == 1).mean()),
        "source_counts": source_counts,
        "feature_names": feature_names,
        "feature_schema_sha256": feature_schema_sha256,
        "record_names": names_seen,
    }


def train_pixel_remove_policy(fit_records, fit_probabilities: dict, base_threshold: float,
                              seed: int = 1337,
                              simulator_config: CounterfactualConfig | None = None,
                              config: V5203PolicyTrainingConfig | None = None):
    """Canonical reproducible V5.20.3 pixel-candidate learner."""
    cfg = config or V5203PolicyTrainingConfig()
    X, y, _, metadata = build_training_matrix(
        fit_records, fit_probabilities, base_threshold, seed,
        simulator_config, cfg,
    )
    kwargs = dict(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        reg_lambda=cfg.reg_lambda,
        subsample=cfg.subsample,
        subsample_freq=1 if cfg.subsample < 1.0 else 0,
        colsample_bytree=cfg.colsample_bytree,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        n_jobs=cfg.n_jobs,
        verbosity=-1,
    )
    if cfg.deterministic:
        kwargs.update(deterministic=True, force_col_wise=True)
    clf = LGBMClassifier(**kwargs)
    clf.fit(X, y)
    metadata.update({
        "seed": int(seed),
        "base_threshold": float(base_threshold),
        "training_config": asdict(cfg),
        "simulator_config": asdict(simulator_config or CounterfactualConfig()),
        "learner": "LGBMClassifier",
    })
    return clf, metadata
