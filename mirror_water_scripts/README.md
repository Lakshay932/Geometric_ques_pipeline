# Mirror / Water — Question Generator

This is the code that generates one question type for a visual-reasoning
question bank: a small figure sits next to a dashed mirror line, and the
test-taker has to pick which of 4 options correctly shows the figure
reflected across that line (the "if this were held up to a mirror, or its
reflection in still water, what would it look like?" puzzle).

This folder is a **trimmed copy** of the real pipeline — only the files
needed to generate this one question type, nothing else. The working
project (with the other two question types, tests, and history) is
unaffected by anything here.

## 1. What the question looks like

Each question has 5 images and some text:

- **1 question image** — an asymmetric figure drawn next to a dashed
  mirror line (vertical, horizontal, or diagonal; sometimes off-centre).
- **4 option images** — candidate reflections. Exactly one is correct;
  the other three are plausible mistakes (reflected across the wrong
  axis, shifted off the mirror line, rotated instead of reflected, scaled
  slightly wrong, etc.).
- **Stem + explanation text**, and the exact recipe that produced the
  question (the figure's vertex coordinates, which axis, how far off
  centre).

The figure is always deliberately **asymmetric** — if it looked the same
after reflecting, there'd be no real question to ask. That's checked in
code, not just assumed (§3).

## 2. Why the answer is *computed*, not guessed

Nothing here asks an AI model "what does this look like reflected?" and
hopes it's right. The reflection is done with real 2D geometry — the same
`reflect_polygon` function this project's fold-and-punch question type
already uses for folding paper, reused as-is rather than rewritten,
because reflecting a shape across a line is the same math either way.

An AI model is only used for the surrounding sentence (question stem and
explanation) — never for deciding what's correct.

## 3. How one question gets built (the pipeline)

1. **Pick a figure** — either one of 5 hand-drawn shapes, or a freshly
   generated random asymmetric polygon (`engine/mirror_water/figures.py`)
   — a jittered star-like shape, checked to make sure it isn't
   accidentally symmetric and doesn't cross itself.
2. **Pick a mirror line** — vertical, horizontal, or one of the two
   diagonals, optionally offset from dead centre
   (`engine/mirror_water/geometry.py`).
3. **Compute the real reflection** — the correct answer, via
   `reflect_polygon`.
4. **Score the difficulty** — `engine/mirror_water/difficulty.py` rates
   the question 1–5 from the axis, the figure's origin, and how subtle
   the wrong answers are.
5. **Build the 3 wrong answers** — `distractors/mirror_water.py` applies
   one of 8 "plausible mistake" rules per wrong option (wrong axis,
   rotated instead of reflected, shifted off the mirror line, scaled
   wrong, only half the shape reflected, etc.). Every candidate is also
   test-rendered and compared against the correct answer's rendered image
   — if the two look almost identical to the eye, that candidate is
   thrown out and another is tried. This check exists because of a real
   bug found in this generator (see the box below).
6. **Draw the images** — `render/mirror_water.py`.
7. **Check it's actually a good question** — before anything is accepted:
   `evaluate/dedup.py` (is this a duplicate, or near-identical, to a
   question already made?), `evaluate/quality.py` (is the correct answer
   unique among the 4 options, do all 4 images render properly, does the
   answer recompute correctly from its own recipe?).
8. **Write the words** — `textgen/generator.py` fills in stem/explanation
   text (via `providers/`, see the file list below).

`scripts/generate_mirror_water_batch.py` runs all of the above, end to
end, for as many records as you ask for (`--gated` mode turns on the
duplicate/quality checks).

> **A real bug this generator had, and how it was caught and fixed:** one
> of the 8 wrong-answer rules (a reflection scaled down slightly) used a
> scale factor subtle enough that, on about 1 in 5 draws, the wrong answer
> was visually indistinguishable from the correct one to a human eye —
> even though the two were mathematically different shapes. This was
> found by actually rendering and comparing option images, not just their
> coordinates. Fixed two ways: the scale factor was tuned to a visibly
> different value, and step 5 above (render-and-compare before accepting
> any wrong answer) was added as a permanent safety net for every rule,
> not just this one.

## 4. Real numbers from a real run

A 500-question batch, generated fresh, from an empty duplicate-check store:

```
457 / 500 accepted — 91.4% survival (43 rejected as duplicates, 0 quality failures)
```

Checked directly against the actual rendered images of all 457 questions:
**zero** near-identical option pairs — down from 94 near-identical pairs
found in a diagnostic sweep before the fix in §3 was applied.

## 5. Every file, one line each

**The two scripts you actually run:**
- `scripts/generate_mirror_water_batch.py` — the main script; runs the
  full pipeline in §3 to build a batch of questions from scratch.
- `scripts/build_pdf.py` — takes a finished batch and turns it into a
  printable exam-booklet PDF.

**The question-making logic:**
- `distractors/mirror_water.py` — builds the 3 wrong answer options for
  each question, and checks each one is visually distinct from the
  correct answer before accepting it.
- `engine/mirror_water/geometry.py` — does the actual reflection math
  that decides the correct answer, and defines the 5 hand-drawn figures
  and the mirror-line options.
- `engine/mirror_water/figures.py` — generates fresh random asymmetric
  figures instead of only using the 5 hand-drawn ones.
- `engine/mirror_water/difficulty.py` — works out how hard a question is,
  on a 1–5 scale.
- `render/mirror_water.py` — draws the actual question and answer images.
- `engine/fold_punch/geometry.py` — shared geometry helpers (the actual
  `reflect_polygon` reflection math, plus basic point/line types) —
  originally written for the fold-and-punch question type, reused here
  unchanged rather than rewritten.

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

## 6. Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=. python scripts/generate_mirror_water_batch.py --count 100 --seed 0 --gated
PYTHONPATH=. python scripts/build_pdf.py \
    --records data/records/mirror_water.jsonl --images-dir data/images --output my_batch.pdf
```

`generate_mirror_water_batch.py` writes `data/records/`, `data/images/`,
and `data/reports/` here as it runs — nothing is pre-populated in this
folder.

## 7. What was deliberately left out

- **The other two question types** this project also builds (paper
  fold-and-punch, rotation series) — kept entirely out of this folder.
- **Tests and frozen sample data** — used during development to prove the
  math is right and to catch the bug described in §3; not needed to run
  the generator itself.

Verified before trimming, and again after removing all code comments: a
real end-to-end run (records + PDF) on this exact file set, with no import
errors and identical output to the untrimmed version.
