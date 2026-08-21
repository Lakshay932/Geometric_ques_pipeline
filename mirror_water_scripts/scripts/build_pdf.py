from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "golden_set")
DEFAULT_RECORDS_PATH = os.path.join(GOLDEN_DIR, "fold_punch_golden.jsonl")
DEFAULT_IMAGES_DIR = os.path.join(GOLDEN_DIR, "images")
DEFAULT_OUTPUT_PDF = os.path.join(REPO_ROOT, "exports", "fold_punch_golden_500.pdf")

PAGE_W, PAGE_H = 1000, 1300
MARGIN = 60
OPTION_CELL = 200
OPTION_GAP = 50
ANSWER_KEY_COLUMNS = 5
ANSWER_KEY_ROWS_PER_COLUMN = 40

_FONT_DIR = "/System/Library/Fonts/Supplemental"

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = os.path.join(_FONT_DIR, name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def render_question_page(index: int, total: int, record: dict, images_dir: str) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    title_font = _load_font(28, bold=True)
    body_font = _load_font(18)
    label_font = _load_font(20, bold=True)

    y = MARGIN
    draw.text((MARGIN, y), f"Question {index} of {total}", font=title_font, fill="black")
    y += 42

    for line in _wrap_text(draw, record["text"]["stem"], body_font, PAGE_W - 2 * MARGIN):
        draw.text((MARGIN, y), line, font=body_font, fill="black")
        y += 24
    y += 16

    question_id = record["question_id"]
    q_img = Image.open(os.path.join(images_dir, question_id, "question.png")).convert("RGB")
    max_w = PAGE_W - 2 * MARGIN
    scale = min(max_w / q_img.width, 1.0)
    if scale < 1.0:
        q_img = q_img.resize((int(q_img.width * scale), int(q_img.height * scale)))
    page.paste(q_img, (MARGIN, y))
    y += q_img.height + 34

    draw.text(
        (MARGIN, y), "Which option shows the paper correctly unfolded?", font=body_font, fill="black"
    )
    y += 38

    positions = [("A", 0, 0), ("B", 1, 0), ("C", 0, 1), ("D", 1, 1)]
    for letter, col, row in positions:
        opt_img = Image.open(os.path.join(images_dir, question_id, f"{letter}.png")).convert("RGB")
        x = MARGIN + col * (OPTION_CELL + OPTION_GAP)
        cell_y = y + row * (OPTION_CELL + OPTION_GAP)
        draw.text((x, cell_y), f"{letter}.", font=label_font, fill="black")
        page.paste(opt_img.resize((OPTION_CELL, OPTION_CELL)), (x, cell_y + 28))

    draw.text((PAGE_W - MARGIN - 40, PAGE_H - 40), f"#{index}", font=body_font, fill="gray")
    return page

def render_answer_key_pages(records: list[dict]) -> list[Image.Image]:
    entries = [(i + 1, r["correct_option"]) for i, r in enumerate(records)]
    cols = ANSWER_KEY_COLUMNS
    rows_per_col = ANSWER_KEY_ROWS_PER_COLUMN
    per_page = cols * rows_per_col

    title_font = _load_font(26, bold=True)
    body_font = _load_font(16)
    col_width = (PAGE_W - 2 * MARGIN) // cols

    pages = []
    for start in range(0, len(entries), per_page):
        chunk = entries[start : start + per_page]
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(page)
        draw.text((MARGIN, MARGIN), "Answer Key", font=title_font, fill="black")
        for idx, (qnum, letter) in enumerate(chunk):
            col = idx // rows_per_col
            row = idx % rows_per_col
            x = MARGIN + col * col_width
            yy = MARGIN + 60 + row * 26
            draw.text((x, yy), f"{qnum}: {letter}", font=body_font, fill="black")
        pages.append(page)
    return pages

def main() -> None:
    parser = argparse.ArgumentParser(description="Export fold_punch records as a PDF booklet")
    parser.add_argument("--records", default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PDF)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.records) as f:
        records = [json.loads(line) for line in f]
    records.sort(key=lambda r: r["question_id"])

    total = len(records)
    question_pages = [
        render_question_page(i + 1, total, r, args.images_dir) for i, r in enumerate(records)
    ]
    answer_pages = render_answer_key_pages(records)
    pages = question_pages + answer_pages

    first, rest = pages[0], pages[1:]
    first.save(args.output, save_all=True, append_images=rest)
    print(f"Wrote {len(pages)} pages ({total} questions + {len(answer_pages)} answer-key pages) to {args.output}")

if __name__ == "__main__":
    main()
