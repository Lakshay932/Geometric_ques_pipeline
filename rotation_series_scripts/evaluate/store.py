from __future__ import annotations

import json
import os

from .dedup import hamming_distance

DEFAULT_STORE_FILENAME = "dedup_store.jsonl"

class DedupStore:

    def __init__(self, path: str):
        self.path = path
        self.seen_param_hashes: set[str] = set()
        self.seen_canonical_hashes: set[str] = set()
        self._phash_entries: list[tuple[int, str]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.seen_param_hashes.add(entry["param_hash"])
                self.seen_canonical_hashes.add(entry["canonical_hash"])
                if entry.get("phash") is not None:
                    self._phash_entries.append((entry["phash"], entry["question_id"]))

    def has_param_hash(self, param_hash: str) -> bool:
        return param_hash in self.seen_param_hashes

    def has_canonical_hash(self, canonical_hash: str) -> bool:
        return canonical_hash in self.seen_canonical_hashes

    def closest_phash_distance(self, phash: int) -> int | None:
        if not self._phash_entries:
            return None
        return min(hamming_distance(phash, stored) for stored, _ in self._phash_entries)

    def record(
        self,
        question_id: str,
        param_hash: str,
        canonical_hash: str,
        phash: int | None = None,
    ) -> None:
        self.seen_param_hashes.add(param_hash)
        self.seen_canonical_hashes.add(canonical_hash)
        if phash is not None:
            self._phash_entries.append((phash, question_id))
        with open(self.path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "param_hash": param_hash,
                        "canonical_hash": canonical_hash,
                        "phash": phash,
                    }
                )
                + "\n"
            )

    def reset(self) -> None:
        self.seen_param_hashes.clear()
        self.seen_canonical_hashes.clear()
        self._phash_entries.clear()
        if os.path.exists(self.path):
            os.remove(self.path)

def default_store_path(data_dir: str) -> str:
    return os.path.join(data_dir, DEFAULT_STORE_FILENAME)
