"""
Dataset loading and splitting for GDPval tasks.

Loads the ``openai/gdpval`` dataset from HuggingFace, downloads reference files,
and splits tasks into train/eval sets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GDPvalTask:
    """A single GDPval task with metadata and rubric."""

    task_id: str
    prompt: str
    occupation: str
    sector: str
    reference_files: list[Path] = field(default_factory=list)
    rubric_json: list[dict] = field(default_factory=list)
    rubric_pretty: str = ""
    total_points: int = 0


def _format_rubric(rubric_json: list[dict]) -> str:
    """Format rubric criteria into a human-readable string."""
    lines = []
    for i, criterion in enumerate(rubric_json, 1):
        name = criterion.get("criterion", criterion.get("name", f"Criterion {i}"))
        points = criterion.get("points", 1)
        description = criterion.get("description", "")
        lines.append(f"{i}. [{points}pt] {name}: {description}")
    return "\n".join(lines)


def _download_reference_files(
    task_id: str,
    uris: list[str],
    cache_dir: Path,
) -> list[Path]:
    """Download reference files from HuggingFace URIs to local cache.

    Returns list of local file paths.
    """
    if not uris:
        return []

    from huggingface_hub import hf_hub_download

    ref_dir = cache_dir / "reference_files" / task_id
    ref_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for uri in uris:
        # URIs are like "hf://datasets/openai/gdpval/reference_files/task_id/filename"
        # Extract the filename from the URI (URL-decode %20 etc.)
        filename = unquote(uri.rsplit("/", 1)[-1])
        local_path = ref_dir / filename

        if local_path.exists():
            local_paths.append(local_path)
            continue

        # Parse hf:// URI to get repo path
        # Format: hf://datasets/<org>/<repo>[@<revision>]/<path_in_repo>
        if not uri.startswith("hf://datasets/"):
            raise ValueError(
                f"gdpval: unexpected reference file URI format: {uri!r}. "
                f"Expected 'hf://datasets/<org>/<repo>/...'"
            )
        parts = uri.removeprefix("hf://datasets/").split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"gdpval: malformed URI, need at least org/repo/path: {uri!r}")

        # Strip optional @revision from repo name (e.g. "gdpval@main" -> "gdpval")
        repo_name = parts[1].split("@")[0]
        repo_id = f"{parts[0]}/{repo_name}"
        revision = parts[1].split("@")[1] if "@" in parts[1] else None
        # URL-decode the path (HF URIs may have %20 etc. which hf_hub_download
        # would double-encode to %2520)
        path_in_repo = unquote(parts[2])

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                revision=revision,
                local_dir=str(cache_dir / "hf_download"),
            )
            import shutil

            shutil.copy2(downloaded, local_path)
            local_paths.append(local_path)
        except Exception as e:
            logger.warning("gdpval: failed to download reference file %s: %s", uri, e)

    return local_paths


def load_gdpval_dataset(cache_dir: str | Path) -> list[GDPvalTask]:
    """Load the GDPval dataset from HuggingFace.

    Args:
        cache_dir: Directory for caching downloaded data and reference files.

    Returns:
        List of GDPvalTask objects sorted by task_id.
    """
    from datasets import load_dataset

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("gdpval: loading dataset from openai/gdpval...")
    ds = load_dataset("openai/gdpval", split="train", cache_dir=str(cache_dir / "hf_cache"))

    tasks = []
    for row in ds:
        task_id = row["task_id"]

        # Parse rubric JSON
        rubric_raw = row.get("rubric_json", "[]")
        rubric_json = json.loads(rubric_raw) if isinstance(rubric_raw, str) else rubric_raw

        # Download reference files
        ref_uris = row.get("reference_file_hf_uris") or []
        if isinstance(ref_uris, str):
            ref_uris = json.loads(ref_uris) if ref_uris.strip() else []
        reference_files = _download_reference_files(task_id, ref_uris, cache_dir)

        total_points = sum(c.get("points", 1) for c in rubric_json)

        task = GDPvalTask(
            task_id=task_id,
            prompt=row["prompt"],
            occupation=row.get("occupation", ""),
            sector=row.get("sector", ""),
            reference_files=reference_files,
            rubric_json=rubric_json,
            rubric_pretty=_format_rubric(rubric_json),
            total_points=total_points,
        )
        tasks.append(task)

    tasks.sort(key=lambda t: t.task_id)
    logger.info("gdpval: loaded %d tasks", len(tasks))
    return tasks


def split_tasks(
    tasks: list[GDPvalTask],
    num_train: int = 180,
    num_eval: int = 40,
    seed: int = 0,
) -> dict[str, list[GDPvalTask]]:
    """Deterministically split tasks into train/eval sets.

    Args:
        tasks: List of GDPvalTask objects.
        num_train: Number of training tasks.
        num_eval: Number of eval tasks.
        seed: Random seed for reproducibility.

    Returns:
        ``{"train": [...], "eval": [...]}``
    """
    total = num_train + num_eval
    if total > len(tasks):
        logger.warning(
            "gdpval: requested %d tasks (train=%d, eval=%d) but only %d available. "
            "Adjusting counts.",
            total,
            num_train,
            num_eval,
            len(tasks),
        )
        num_eval = min(num_eval, len(tasks))
        num_train = len(tasks) - num_eval

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(tasks)).tolist()

    train_indices = indices[:num_train]
    eval_indices = indices[num_train : num_train + num_eval]

    return {
        "train": [tasks[i] for i in sorted(train_indices)],
        "eval": [tasks[i] for i in sorted(eval_indices)],
    }
