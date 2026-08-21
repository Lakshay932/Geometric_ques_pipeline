from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

CONTACT_SHEET_SAMPLE_SIZE = 12
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_THUMB_SIZE = 140
_FONT_DIR = "/System/Library/Fonts/Supplemental"

def build_scorecard(
    *,
    attempted: int,
    verified: int,
    flagged_vlm_disagreement: int,
    rejected_duplicate: int,
    rejected_quality: int,
    difficulty_counts: dict[int, int],
    subtype_difficulty_counts: dict[str, dict[int, int]],
    rule_counts: dict[str, int],
    elapsed_seconds: float,
) -> dict:
    survival_rate = verified / attempted if attempted else 0.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "funnel": {
            "attempted": attempted,
            "verified": verified,
            "flagged_vlm_disagreement": flagged_vlm_disagreement,
            "rejected_duplicate": rejected_duplicate,
            "rejected_quality": rejected_quality,
        },
        "survival_rate": round(survival_rate, 4),
        "difficulty_histogram": {str(k): v for k, v in sorted(difficulty_counts.items())},
        "subtype_difficulty_heatmap": {
            sub: {str(k): v for k, v in sorted(counts.items())}
            for sub, counts in sorted(subtype_difficulty_counts.items())
        },
        "distractor_rule_histogram": dict(sorted(rule_counts.items())),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "verified_per_hour": round(verified / elapsed_seconds * 3600, 1) if elapsed_seconds > 0 else None,
    }

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(os.path.join(_FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()

def _build_contact_sheet(image_paths: list[str], sample_size: int = CONTACT_SHEET_SAMPLE_SIZE) -> Image.Image | None:
    if not image_paths:
        return None
    sample = random.sample(image_paths, min(sample_size, len(image_paths)))
    cols = CONTACT_SHEET_COLUMNS
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * CONTACT_SHEET_THUMB_SIZE, rows * CONTACT_SHEET_THUMB_SIZE),
        "white",
    )
    for i, path in enumerate(sample):
        if not os.path.exists(path):
            continue
        thumb = Image.open(path).convert("RGB").resize((CONTACT_SHEET_THUMB_SIZE, CONTACT_SHEET_THUMB_SIZE))
        x = (i % cols) * CONTACT_SHEET_THUMB_SIZE
        y = (i // cols) * CONTACT_SHEET_THUMB_SIZE
        sheet.paste(thumb, (x, y))
    return sheet

def _render_html(scorecard: dict, contact_sheet_filename: str | None) -> str:
    funnel = scorecard["funnel"]
    funnel_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in funnel.items())

    heatmap = scorecard["subtype_difficulty_heatmap"]
    all_difficulties = sorted({d for counts in heatmap.values() for d in counts})
    heatmap_header = "".join(f"<th>{d}</th>" for d in all_difficulties)
    heatmap_rows = "".join(
        "<tr><td>{}</td>{}</tr>".format(
            sub, "".join(f"<td>{counts.get(d, 0)}</td>" for d in all_difficulties)
        )
        for sub, counts in sorted(heatmap.items())
    )

    rule_rows = "".join(f"<tr><td>{rule}</td><td>{count}</td></tr>" for rule, count in scorecard["distractor_rule_histogram"].items())

    contact_sheet_html = (
        f'<img src="{contact_sheet_filename}" alt="contact sheet" style="max-width:100%">'
        if contact_sheet_filename
        else "<p>(no verified records to sample)</p>"
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Batch Scorecard {scorecard['generated_at']}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
h2 {{ margin-top: 2rem; }}
</style></head>
<body>
<h1>Batch Scorecard</h1>
<p>Generated: {scorecard['generated_at']} &middot; Survival rate: {scorecard['survival_rate'] * 100:.1f}%
&middot; {scorecard.get('verified_per_hour') or 'n/a'} verified/hour</p>

<h2>Funnel (every attempted record accounted for)</h2>
<table><tr><th>Stage</th><th>Count</th></tr>{funnel_rows}</table>

<h2>Difficulty &times; sub-type heatmap</h2>
<table><tr><th>sub_type \\ difficulty</th>{heatmap_header}</tr>{heatmap_rows}</table>

<h2>Distractor-rule usage</h2>
<table><tr><th>Rule</th><th>Count</th></tr>{rule_rows}</table>

<h2>Contact sheet ({CONTACT_SHEET_SAMPLE_SIZE} random verified records)</h2>
{contact_sheet_html}

</body></html>"""

MIN_SURVIVAL_RATE = 0.70
MAX_CELL_FRACTION = 0.15

def _fold_punch_cell_key(record: dict) -> tuple:
    params = record["params"]
    shape = params.get("start_shape", "square")
    return (params["fold_count"], params["punch_count"], shape)

def check_diversity_gate(
    records: list[dict],
    attempted: int,
    verified: int,
    min_survival_rate: float = MIN_SURVIVAL_RATE,
    max_cell_fraction: float = MAX_CELL_FRACTION,
    cell_key_fn=_fold_punch_cell_key,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    survival_rate = verified / attempted if attempted else 1.0
    if attempted and survival_rate < min_survival_rate:
        reasons.append(f"survival rate {survival_rate:.1%} is below the {min_survival_rate:.0%} floor")

    cell_counts: dict[tuple, int] = {}
    for record in records:
        key = cell_key_fn(record)
        cell_counts[key] = cell_counts.get(key, 0) + 1

    total = len(records)
    if total:
        for cell, count in cell_counts.items():
            fraction = count / total
            if fraction > max_cell_fraction:
                reasons.append(f"cell {cell} is {fraction:.1%} of the batch (> {max_cell_fraction:.0%} cap)")

    return not reasons, reasons

def write_report(
    scorecard: dict,
    reports_dir: str,
    verified_image_paths: list[str] | None = None,
    label: str | None = None,
) -> tuple[str, str]:
    os.makedirs(reports_dir, exist_ok=True)
    stamp = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(reports_dir, f"batch_{stamp}.json")
    html_path = os.path.join(reports_dir, f"batch_{stamp}.html")
    contact_sheet_path = os.path.join(reports_dir, f"batch_{stamp}_contact_sheet.png")

    with open(json_path, "w") as f:
        json.dump(scorecard, f, indent=2)

    contact_sheet_filename = None
    sheet = _build_contact_sheet(verified_image_paths or [])
    if sheet is not None:
        sheet.save(contact_sheet_path)
        contact_sheet_filename = os.path.basename(contact_sheet_path)

    with open(html_path, "w") as f:
        f.write(_render_html(scorecard, contact_sheet_filename))

    return json_path, html_path
