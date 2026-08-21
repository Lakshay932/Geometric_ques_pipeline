# Fold & Punch — Question Generator

## 1. What the question looks like

Each question has 5 images and some text:

- **1 question image** — a strip showing a square/circle/rectangle sheet
  being folded 1–4 times, then a hole (or several) punched through the
  folded stack.
- **4 option images** — what the paper looks like unfolded. Exactly one
  (**A**, **B**, **C**, or **D**) is correct; the other three are
  plausible wrong answers (hole shifted slightly, one hole missing, a
  hole mirrored to the wrong side of a fold line, etc.).
- **Stem + explanation text**, and the full numeric recipe that produced
  the question (fold axes, punch coordinates, which rule built each wrong
  option).

Example of one real generated record's recipe: fold vertically, then
vertically again (a square folded down to a quarter), punch 4 holes at
given coordinates, correct option = A, the three distractors built by the
rules `shifted_hole`, `missing_hole`, `single_hole_wrong_side`.



## 3. How one question gets built (the pipeline)

1. **Pick a recipe** — `engine/fold_punch/sampler.py` randomly chooses how
   many folds, which axes, what starting shape (square/circle/rectangle),
   and how many punches.
2. **Do the fold + punch math** — `engine/fold_punch/geometry.py` folds
   the paper along those axes, places the punch(es), and works out the
   correct unfolded hole pattern. `engine/fold_punch/punches.py` supplies
   the punch shapes themselves (circle, square, triangle, star, slit).
3. **Score the difficulty** — `engine/fold_punch/difficulty.py` rates the
   question 1–5 from the fold/punch count and how subtle the wrong
   answers are.
4. **Build the 3 wrong answers** — `distractors/fold_punch.py` applies one
   of 10 "plausible mistake" rules per wrong option (shifted hole, missing
   hole, extra hole, mirrored to the wrong side, wrong rotation, etc.).
5. **Draw the images** — `render/fold_punch.py` renders the fold-sequence
   strip and all 4 option images.
6. **Check it's actually a good question** — before anything is accepted:
   - `evaluate/dedup.py` + `evaluate/store.py`: is this the same question
     (or a visually near-identical one) as something already generated?
   - `evaluate/quality.py`: is the correct answer actually unique among
     the 4 options, are all 4 images real and non-blank, does the answer
     recompute correctly from its own recipe?
   - Only if all checks pass does the record get kept; otherwise it's
     retried with a fresh recipe.
7. **Write the words** — `textgen/generator.py` fills in the stem and
   explanation text (via `providers/`, see §6).
8. **Record it** — `schemas/record.py` defines the saved JSON shape;
   `evaluate/report.py` summarizes how a whole batch turned out.

`scripts/generate_upgraded_batch.py` runs all of the above, end to end,
for as many records as you ask for.

## 4. Duplicate detection, in one sentence

Two fold_punch questions count as "the same" if their correct answer's
hole pattern matches — even after accounting for the paper's own
symmetry (a pattern and its mirror image are the same underlying
question) — or if their rendered images are visually near-identical. Both
checks run automatically on every candidate before it's accepted.

## 5. Real numbers from a real run

A 500-question batch, generated fresh, from an empty duplicate-check store:

```
500 / 500 accepted — 0 duplicates, 0 quality failures
difficulty spread: {2: 116, 3: 152, 4: 136, 5: 96}
fold counts:       {1: 197, 2: 130, 3: 145, 4: 28}
starting shapes:   {square: 372, circle: 49, rectangle: 79}
punch shapes:      circle, square, triangle, star, slit — all 5 used
all 10 wrong-answer rules were used at least once
```

Exported as a 503-page PDF via `scripts/build_pdf.py` and visually
spot-checked (not just counted) — the fold and punch shapes render
correctly, including on non-square starting paper.

## 6. Every file, one line each

**The two scripts you actually run:**
- `scripts/generate_upgraded_batch.py` — the main script; runs the full
  pipeline in §3 to build a batch of questions from scratch.
- `scripts/build_pdf.py` — takes a finished batch and turns it into a
  printable exam-booklet PDF.

**The question-making logic:**
- `distractors/fold_punch.py` — builds the 3 wrong answer options for
  each question.
- `engine/fold_punch/geometry.py` — does the actual fold / punch / unfold
  math that decides the correct answer.
- `engine/fold_punch/sampler.py` — randomly picks how many folds and
  punches a new question should have.
- `engine/fold_punch/punches.py` — draws the punch shapes (circle, square,
  triangle, star, slit).
- `engine/fold_punch/difficulty.py` — works out how hard a question is, on
  a 1–5 scale.
- `render/fold_punch.py` — draws the actual question and answer images.

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
  set up, so things still run (this is why the sample text above says
  "Stub question stem" — no real text-generation API key was configured
  for this bundle).
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

PYTHONPATH=. python scripts/generate_upgraded_batch.py --count 100 --seed 0
PYTHONPATH=. python scripts/build_pdf.py \
    --records data/records/fold_punch.jsonl --images-dir data/images --output my_batch.pdf
```

`generate_upgraded_batch.py` writes `data/records/`, `data/images/`, and
`data/reports/` here as it runs — nothing is pre-populated in this folder.

## 8. What was deliberately left out

- **The other two question types** this project also builds (mirror/water
  reflection, rotation series) — kept entirely out of this folder. The
  one shared file that needed trimming: the full project's
  `evaluate/quality.py` has two extra functions for those other families
  — removed here, along with their imports, so this folder has zero
  dependency on anything outside fold & punch.
- **An older, alternate generation pipeline** (`verify/graph.py`, not
  included) that used a language model to double-check each answer — it's
  not used by the script here, and it had a bug where its check always
  said "yes" to option A regardless of the real answer. The script in
  this bundle doesn't use it, and produces genuinely randomized correct
  answers.
- **Other, earlier batch-generator scripts** — narrower or superseded
  versions from this family's own development history; the one script
  kept here is the one that actually produced the numbers in §5.
- **Tests and frozen sample data** — used during development to prove the
  math is right; not needed to run the generator itself.

Verified before trimming, and again after removing all code comments:
a real end-to-end run (records + PDF) on this exact file set, with no
import errors and identical output to the untrimmed version.
