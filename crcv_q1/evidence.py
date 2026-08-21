from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    kind: str


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_artifact_ref(ref: dict, root: str | Path = ".") -> list[str]:
    failures: list[str] = []
    path = ref.get("path")
    expected = ref.get("sha256")
    kind = ref.get("kind")
    if not isinstance(path, str) or not path:
        return ["artifact.path missing"]
    if not isinstance(expected, str) or len(expected) != 64:
        failures.append(f"artifact {path}: invalid sha256")
    if not isinstance(kind, str) or not kind:
        failures.append(f"artifact {path}: kind missing")

    p = Path(root) / path
    if not p.is_file():
        failures.append(f"artifact {path}: file missing")
        return failures
    if isinstance(expected, str) and len(expected) == 64:
        actual = sha256_file(p)
        if actual.lower() != expected.lower():
            failures.append(f"artifact {path}: sha256 mismatch")
    return failures


def validate_artifacts(refs: Iterable[dict], root: str | Path = ".") -> list[str]:
    failures: list[str] = []
    refs = list(refs)
    if not refs:
        return ["artifact evidence missing"]
    for ref in refs:
        failures.extend(validate_artifact_ref(ref, root=root))
    return failures


def load_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("evidence JSON must contain an object")
    return data


def require_fields(data: dict, fields: Iterable[str], prefix: str = "") -> list[str]:
    failures: list[str] = []
    for field in fields:
        if field not in data or data[field] in (None, "", [], {}):
            failures.append(f"{prefix}{field} missing")
    return failures


def validate_run_record(record: dict, root: str | Path = ".") -> list[str]:
    """Validate provenance for one executable experiment run.

    A run is not considered evidence merely because it is named in a payload.
    It must bind code, data, configuration, checkpoints and outputs by hashes.
    """
    failures = require_fields(
        record,
        (
            "experiment_id",
            "git_commit",
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "config_sha256",
            "seed",
            "backbone",
            "dataset",
            "resolution",
            "method",
            "artifacts",
        ),
        prefix="run.",
    )
    commit = record.get("git_commit")
    if isinstance(commit, str) and len(commit) < 7:
        failures.append("run.git_commit is not a valid commit identifier")
    for key in ("dataset_manifest_sha256", "split_manifest_sha256", "config_sha256"):
        value = record.get(key)
        if isinstance(value, str) and len(value) != 64:
            failures.append(f"run.{key} must be a SHA256 digest")
    resolution = record.get("resolution")
    if not isinstance(resolution, int) or resolution <= 0:
        failures.append("run.resolution must be a positive integer")
    seed = record.get("seed")
    if not isinstance(seed, int):
        failures.append("run.seed must be an integer")
    failures.extend(validate_artifacts(record.get("artifacts", []), root=root))
    return failures
