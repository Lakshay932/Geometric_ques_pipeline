"""Run a batch of fold_punch records through generate -> verify -> repair
and persist them to the dataset store (TRD Section 3.1, step 7).

Usage:
    python scripts/run_batch.py --count 100 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common.logging import configure_logging, get_logger
from providers.factory import get_llm_provider, get_vlm_provider
from schemas.record import ExamStyle
from verify.graph import run_fold_punch_pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")


def run_batch(count: int, seed: int, data_dir: str = DEFAULT_DATA_DIR) -> dict:
    configure_logging()
    log = get_logger("run_batch")

    vlm_provider = get_vlm_provider()
    llm_provider = get_llm_provider()
    log.info(
        "batch_start",
        count=count,
        seed=seed,
        vlm_provider=vlm_provider.name,
        llm_provider=llm_provider.name,
    )

    rng = np.random.default_rng(seed)
    records_path = os.path.join(data_dir, "records", "fold_punch.jsonl")
    flagged_path = os.path.join(data_dir, "records", "flagged_review.jsonl")
    os.makedirs(os.path.dirname(records_path), exist_ok=True)

    verified_count = 0
    flagged_count = 0
    start = time.time()

    with open(records_path, "a") as records_file, open(flagged_path, "a") as flagged_file:
        for i in range(count):
            record, flagged_entry = run_fold_punch_pipeline(
                rng, vlm_provider, llm_provider, data_dir, exam_style=ExamStyle.GENERIC
            )
            if record is not None:
                verified_count += 1
                records_file.write(record.model_dump_json() + "\n")
            else:
                flagged_count += 1
                flagged_file.write(json.dumps(flagged_entry, default=str) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == count:
                log.info(
                    "batch_progress",
                    completed=i + 1,
                    verified=verified_count,
                    flagged=flagged_count,
                )

    elapsed = time.time() - start
    total = verified_count + flagged_count
    agreement_rate = verified_count / total if total else 0.0
    log.info(
        "batch_complete",
        verified=verified_count,
        flagged=flagged_count,
        agreement_rate=round(agreement_rate, 4),
        elapsed_seconds=round(elapsed, 2),
        verified_per_hour=round(verified_count / elapsed * 3600, 1) if elapsed > 0 else None,
    )
    return {
        "verified": verified_count,
        "flagged": flagged_count,
        "agreement_rate": agreement_rate,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fold_punch generation batch")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_batch(args.count, args.seed)


if __name__ == "__main__":
    main()
