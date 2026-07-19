# Phase 1 — Complete

Fold & punch generator, end-to-end vertical slice, per `IMPLEMENTATION_PHASES.md`
Phase 1 (matches TRD Phase 1 / Lane 1).

## What was built

### Directory structure (new/changed since Phase 0)
```
engine/fold_punch/
├── geometry.py     # DONE — fold/unfold engine, symmetry-validity check, FR-12 reconstruction
├── sampler.py      # DONE — parameter sampler (uniform; bucket-bias deferred to Phase 3)
└── difficulty.py   # DONE — difficulty 1-5 from fold/punch count + distractor subtlety (FR-11)

distractors/
└── fold_punch.py   # DONE — 5-rule pool, picks 3 distinct tagged distractors per record (FR-5)

render/
└── fold_punch.py   # DONE — Pillow renderer: question strip (fold steps + arrows) + option images

textgen/
└── generator.py    # DONE — stem/option_labels/explanation via LLMProvider; rule->explanation templates (FR-8)

verify/
└── graph.py        # DONE — LangGraph generate->verify->repair loop (Section 7), 4 attempts max, flags on repeated disagreement

scripts/
├── run_batch.py        # DONE — batch runner, persists to data/records/ + data/images/
└── build_golden_set.py # DONE — stratified freeze into tests/golden_set/

tests/
├── unit/           # DONE — 8 new test files, 24 tests (was 4 after Phase 0)
└── golden_set/     # DONE — 80 frozen records, 160 regression assertions
```

### Geometry model (the core design decision this phase)

The folded region is tracked as a Shapely polygon expressed directly in the
*original* unit-square coordinate frame — no separate "folded space"
transform needed, since every fold is a reflection and the visible region
after folding is always literally a subregion of the original square.

A fold is only accepted if the chosen line is an actual **symmetry line**
of the *current* polygon (reflecting the whole polygon across it
reproduces the same polygon, checked directly via Shapely rather than
hand-coded per shape). This one check automatically:
- lets unlimited vertical/horizontal folds chain (a rectangle's bbox
  center lines are always symmetry lines of that rectangle),
- allows a diagonal fold only when the current shape is actually a square,
- and after a diagonal fold, correctly restricts further folds to the one
  remaining symmetry line of the resulting right-isosceles triangle (its
  other diagonal) — rejecting vertical/horizontal/repeat-diagonal folds
  instead of silently producing a physically-impossible fold.

Unfolding a punched hole = reflecting it across each fold line in reverse
order, doubling the point set each step (n folds → up to 2ⁿ holes) —
verified against 6 hand-computed cases (`tests/unit/test_fold_punch_geometry.py`)
before the general sampler was written (TDD, per the phase doc).

### Distractor rules (FR-5)
5-rule pool, 3 sampled per record, each tagged:
`missing_hole`, `extra_hole`, `wrong_symmetry_axis` (skips one fold's
mirror step — the "forgot a layer" mistake), `mirrored_wrong` (flips the
whole pattern), `shifted_hole` (systematic position offset). Subtlety
scores per rule feed difficulty (FR-11).

### Verify/repair loop (LangGraph, TRD Section 7)
`generate → distractors → render → vlm_verify` with a conditional edge:
agree → `text_gen` → done; disagree → loop back to `generate` (resample)
up to 4 total attempts, then → `flag` (written to
`data/records/flagged_review.jsonl`, never silently dropped — Reliability NFR).

### A real bug this phase caught
Initial `FoldStep.diagonal_variant` recorded the *requested* variant
(often `None`, meaning "either"), not which one was actually chosen when
both main and anti diagonal were valid candidates — so re-rendering from
stored params could silently reconstruct a *different* valid fold instead
of the original one. Caught by the FR-12 re-render test
(`tests/unit/test_rerender_from_params.py`) before any data was frozen;
fixed by having `candidate_lines_for_axis` return `(line, variant)` pairs
so the actually-chosen variant gets recorded on the `FoldStep`.

### Params now fully capture the realized question (FR-12)
`Record.params` stores not just the recipe (`axis_sequence`, `punch_count`)
but the realized `fold_steps` (axis, kept_side, diagonal_variant per step)
and exact `punch_points` — so `engine.fold_punch.geometry.reconstruct_geometry(...)`
rebuilds the *exact* original geometry deterministically, with no
dependency on the original rng draw sequence. Proven across 30 seeds and
exercised by every golden-set regression check.

### Batch run results (stub VLM/LLM — no API key wired in yet)
- **750 attempts → 506 verified, 244 flagged** (67.5% agreement).
  Low/expected: `StubVLMProvider` always answers "A" regardless of the
  image, so agreement only happens when the randomly-assigned correct
  letter happens to be "A" (~25%/attempt, ~68% cumulative over 4 tries) —
  per the phase doc, this is fine "at this stage," and will jump once a
  real OpenRouter-hosted VLM is wired in (Phase 1 follow-up / Phase 2).
- **Throughput: ~10,600–11,200 verified records/hour** (CPU-bound
  geometry+render only) — far above the 500/hour NFR floor.
- **Storage: ~954 bytes average per PNG**, max 2.7 KB — far under the
  50 KB NFR target.
- **Difficulty spread across the 506 verified**: {1: 2, 2: 129, 3: 157,
  4: 191, 5: 41} — present across the full 1–5 range (a *balanced*
  distribution is a Phase 2 exit criterion, not Phase 1's).
- Data persisted to `data/records/fold_punch.jsonl` (verified) and
  `data/records/flagged_review.jsonl` (flagged, for human review queue),
  images under `data/images/{question_id}/`, all gitignored.

### Golden set (TRD Section 9)
80 records frozen into `tests/golden_set/`, stratified across difficulty
1–5, with images copied alongside and two companion regression tests
(160 assertions total, all passing):
1. Recomputing geometry from stored `params` still yields the exact same
   `answer_points`.
2. Re-rendering the correct option from that geometry is pixel-identical
   to the frozen image.

This catches silent regressions in either the geometry engine or the
renderer on any future change. "Hand-verified" here means spot-checked
visually (fold diagrams + correct-answer symmetry confirmed by eye across
several records spanning single/multi-fold and diagonal cases) — not VLM
agreement, since the stub VLM has no real judgment to lend.

### Verification (exit criteria met)
- ~500 verified fold_punch records generated and stored — **506** ✓.
- Geometry unit tests pass for fold-axis × fold-count × punch-count
  combinations, including the diagonal-validity edge cases — **24 unit
  tests passing** ✓.
- VLM agreement rate tracked and logged every batch (67.5%, aspirational
  per the phase doc at this stage) ✓.
- Every record traceable: full realized `params` + `vlm_model` +
  `llm_provider` stored per record (auditability) ✓.

## Next
Phase 2 — Mirror/Water Image & Rotation/Series generators, refactoring
the shared transform utilities (reflection, symmetry-line validity,
polygon splitting) out of `engine/fold_punch/geometry.py` into
`engine/common/` as the second family reveals what's actually shared, per
`IMPLEMENTATION_PHASES.md` Phase 2. Also a good point to wire in a real
OpenRouter VLM/LLM model to replace the stub and re-run the batch with a
meaningful agreement rate.
