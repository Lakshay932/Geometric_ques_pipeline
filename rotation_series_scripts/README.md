# Rotation Series — Question Generator

This is the code that generates one question type for a visual-reasoning
question bank: a short sequence of panels shows the same figure rotating
by a fixed step each time, then a "?" mystery panel — the test-taker has
to pick which of 4 options correctly continues the pattern.

This folder is a **trimmed copy** of the real pipeline — only the files
needed to generate this one question type, nothing else. The working
project (with the other two question types, tests, and history) is
unaffected by anything here.

## 1. What the question looks like

Each question has 5 images and some text:

- **1 question image** — a strip of 3 or 4 panels showing the same figure
  rotated a fixed number of degrees further each time, ending in a "?"
  panel.
- **4 option images** — candidate next panels. Exactly one correctly
  continues the rotation; the other three are plausible mistakes (rotated
  the wrong direction, rotated by the wrong angle, reflected instead of
  rotated, repeated a previous panel instead of advancing, skipped a
  step).
- **Stem + explanation text**, and the exact recipe that produced the
  question (the figure's vertex coordinates, the step angle, the
  direction, how many panels).

## 2. Why the answer is *computed*, not guessed

Nothing here asks an AI model "what comes next in this rotation?" and
hopes it's right. The next panel is worked out with real geometry —
`shapely.affinity.rotate`, applied about the figure's own centre point —
not guessed or pattern-matched.

An AI model is only used for the surrounding sentence (question stem and
explanation) — never for deciding what's correct.

## 3. Note: this reuses mirror_water's figure generator on purpose

You'll notice an `engine/mirror_water/` folder inside this bundle even
though this is the rotation-series question type. That's not left-over
clutter — rotation_series questions need the same thing mirror_water
questions need: a small figure that's deliberately **asymmetric** (a
figure with rotational symmetry would make some "wrong" answers
accidentally look identical to the right one). Rather than write that
check twice, this question type reuses mirror_water's figure generator
and its 5 hand-drawn shapes directly. `engine/mirror_water/difficulty.py`
is *not* included — this question type has its own difficulty scoring.

## 4. How one question gets built (the pipeline)

1. **Pick a figure** — either one of 5 hand-drawn shapes, or a freshly
   generated random asymmetric polygon (from `engine/mirror_water/`, see
   §3 above).
2. **Pick rotation parameters** — a step angle, a direction (clockwise or
   counterclockwise), and how many panels (`engine/rotation_series/geometry.py`).
3. **Compute each panel and the correct next one** — real rotation math
   about the figure's centroid, applied step by step.
4. **Score the difficulty** — `engine/rotation_series/difficulty.py` rates
   the question 1–5 from the step angle, sequence length, figure origin,
   and how subtle the wrong answers are.
5. **Build the 3 wrong answers** — `distractors/rotation_series.py`
   applies one of 5 "plausible mistake" rules per wrong option
   (`wrong_direction`, `wrong_step_size`, `reflected_instead`,
   `stale_repeat`, `skipped_a_step`).
6. **Draw the images** — `render/rotation_series.py` draws the panel
   strip, the "?" mystery panel, and all 4 option images.
7. **Check it's actually a good question** — before anything is accepted:
   `evaluate/dedup.py` (is this a duplicate, or near-identical, to a
   question already made?), `evaluate/quality.py` (is the correct answer
   unique among the 4 options, do all 4 images render properly, does the
   answer recompute correctly from its own recipe?).
8. **Write the words** — `textgen/generator.py` fills in stem/explanation
   text (via `providers/`, see the file list below).

`scripts/generate_rotation_series_batch.py` runs all of the above, end to
end, for as many records as you ask for (`--gated` mode turns on the
duplicate/quality checks).

> **Checked for the same visual-ambiguity bug found in mirror_water:**
> mirror_water's generator (this project's other question type) had a bug
> where one wrong-answer rule sometimes produced an option that looked
> identical to the correct one to a human eye (see that folder's README).
> This generator was checked for the same class of bug — 500+ generated
> records swept directly against their rendered images, plus the specific
> case most likely to trigger it (a 3-panel, 120°-step sequence, where a
> "wrong direction" mistake and the real answer can coincide) — and found
> clean: zero near-identical option pairs. This question type's 5 rules
> are all structurally distinct transforms rather than a small perturbation
> of the right answer, which is why it isn't exposed to this bug the way
> mirror_water's `scaled_reflection` rule was.

## 5. Real numbers from a real run

A 500-question batch, generated fresh, from an empty duplicate-check store:

```
475 / 500 accepted — 95.0% survival (25 rejected as duplicates, 0 quality failures)
hand-designed figures: 40/475 (8.4%) — the rest procedurally generated
```

Confirmed this isn't a fluke: the same 500-record generation was run twice
independently against two fresh, empty stores, and both runs landed on
exactly the same 475/25 split.

## 6. Every file, one line each

**The two scripts you actually run:**
- `scripts/generate_rotation_series_batch.py` — the main script; runs the
  full pipeline in §4 to build a batch of questions from scratch.
- `scripts/build_pdf.py` — takes a finished batch and turns it into a
  printable exam-booklet PDF.

**The question-making logic:**
- `distractors/rotation_series.py` — builds the 3 wrong answer options for
  each question.
- `engine/rotation_series/geometry.py` — does the actual rotation math
  that decides the correct next panel.
- `engine/rotation_series/difficulty.py` — works out how hard a question
  is, on a 1–5 scale.
- `render/rotation_series.py` — draws the panel strip and all 4 option
  images.
- `engine/mirror_water/geometry.py` + `engine/mirror_water/figures.py` —
  the shared figure library (5 hand-drawn shapes + random asymmetric
  figure generator), reused from the mirror_water question type rather
  than duplicated — see §3.
- `engine/fold_punch/geometry.py` — shared basic point/line geometry
  types that the figure library above depends on.

**Checking the work before it's accepted:**
- `evaluate/dedup.py` — checks a new question isn't a duplicate of one
  already made.
- `evaluate/quality.py` — double-checks a question is actually correct and
  not broken.
- `evaluate/report.py` — builds a summary report of how a batch turned
  out.
- `evaluate/store.py` — keeps track of every question made so far, so
  duplicates can be caught.

**Shared support code (used by the scripts above):**
- `schemas/record.py` — defines what one finished question's data looks
  like.
- `textgen/generator.py` — writes the question text and answer
  explanations in words.
- `providers/factory.py` — picks which AI text/image provider to use.
- `providers/base.py` — the basic rules every AI provider has to follow.
- `providers/stub.py` — a fake AI provider used when there's no real one
  set up, so things still run.
- `common/config.py` — reads settings (like API keys) the project needs.
- `common/logging.py` — sets up how the program prints its progress/log
  messages.
- `requirements.txt` — the list of outside tools/libraries needed to run
  any of this.
- every `__init__.py` — an empty file with no logic; just tells Python
  "this folder is a package."

## 7. Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=. python scripts/generate_rotation_series_batch.py --count 100 --seed 0 --gated
PYTHONPATH=. python scripts/build_pdf.py \
    --records data/records/rotation_series.jsonl --images-dir data/images --output my_batch.pdf
```

`generate_rotation_series_batch.py` writes `data/records/`, `data/images/`,
and `data/reports/` here as it runs — nothing is pre-populated in this
folder.

## 8. What was deliberately left out

- **The other two question types** this project also builds (paper
  fold-and-punch, mirror/water reflection) — kept out of this folder,
  except for the two mirror_water files this question type genuinely
  depends on (§3).
- **Tests and frozen sample data** — used during development to prove the
  math is right; not needed to run the generator itself.

Verified before trimming, and again after removing all code comments: a
real end-to-end run (records + PDF) on this exact file set, with no import
errors and identical output to the untrimmed version.
