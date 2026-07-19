# Phase 0 — Complete

Repo skeleton, environment, and shared data contracts, per `IMPLEMENTATION_PHASES.md` Phase 0.

## What was built

### Directory structure
```
visual-reasoning-engine/
├── engine/           # geometry engines (fold_punch, mirror_water, rotation_series, common) — empty, Phase 1+
├── render/           # image rendering — empty, Phase 1+
├── distractors/       # distractor generation per family — empty, Phase 1+
├── verify/           # VLM verification + LangGraph flow — empty, Phase 1+
├── textgen/          # LLM stem/option/explanation generation — empty, Phase 1+
├── pipeline/         # Prefect batch orchestration — empty, Phase 3+
├── index/            # embedding index builder — empty, Phase 4+
├── api/              # FastAPI retrieval + admin service — empty, Phase 4+
├── schemas/
│   └── record.py     # Pydantic Record schema (TRD Section 5) — DONE
├── providers/         # swappable VLM/LLM interfaces
│   ├── base.py        # VLMProvider, LLMProvider abstract classes — DONE
│   ├── stub.py         # StubVLMProvider, StubLLMProvider (canned output) — DONE
│   └── factory.py     # get_vlm_provider()/get_llm_provider() reads Settings — DONE
├── common/
│   ├── config.py      # Settings (pydantic-settings, env prefix VRE_) — DONE
│   └── logging.py     # structlog JSON logging scaffold — DONE
├── data/
│   ├── records/       # dataset store (gitignored, populated from Phase 1)
│   └── images/         # question/option images (gitignored, populated from Phase 1)
├── tests/
│   ├── unit/          # test_record_schema.py, test_providers.py — DONE, 4 passing
│   ├── golden_set/    # empty, Phase 1+
│   └── load/          # empty, Phase 4+
└── scripts/           # empty
```

### Key files

- **`schemas/record.py`** — single source of truth for a question record. Defines
  `Family` (`fold_punch`/`mirror_water`/`rotation_series`), `ExamStyle`, `Source` enums,
  and the `Record` model (`question_id`, `params`, `image_paths`, `correct_option`,
  `distractor_rules`, `text`, `verification`, etc.) exactly matching TRD Section 5.
  Every later component reads/writes against this model.

- **`providers/base.py`** — two abstract interfaces the rest of the system depends on,
  never on a concrete backend:
  - `VLMProvider.solve(question_image_path, option_image_paths) -> str`
  - `LLMProvider.generate_text(params, correct_option, distractor_rules) -> dict`

- **`providers/stub.py`** — canned-output implementations of both, used until a real
  backend (OpenRouter-hosted model) is wired in during Phase 1.

- **`providers/factory.py`** — `get_vlm_provider()` / `get_llm_provider()` look up the
  provider name from `Settings` and return the matching instance. Adding a real backend
  later means implementing the interface and registering it here — no caller changes.

- **`common/config.py`** — `Settings` (pydantic-settings) reads `VRE_VLM_PROVIDER`,
  `VRE_LLM_PROVIDER`, `VRE_OPENROUTER_API_KEY`, `VRE_OPENROUTER_VLM_MODEL`,
  `VRE_OPENROUTER_LLM_MODEL` from `.env` (see `.env.example`). Defaults to `stub` for
  both providers.

- **`common/logging.py`** — `configure_logging()` + `get_logger(name)`, structured JSON
  logs via `structlog`. Every later phase logs through this instead of ad hoc `print`.

### Environment
- Python **3.13** venv at `.venv/` (system default `python3` is 3.9.6, incompatible with
  the `X | None` union syntax used throughout — used `/opt/anaconda3/bin/python3.13` as
  the venv's base interpreter instead).
- All deps in `requirements.txt` install cleanly: numpy, shapely, pillow, svgwrite,
  pydantic, fastapi, uvicorn, langgraph, langchain, faiss-cpu, prefect, pyarrow (bumped
  to `>=17` for a Python 3.13-compatible wheel), openai (OpenRouter client), httpx,
  python-dotenv, pydantic-settings, structlog, pytest.
- `pytest.ini` sets `pythonpath = .` so tests import top-level packages without manual
  `PYTHONPATH`.

### Verification (exit criteria met)
- `pip install -r requirements.txt` succeeds cleanly in the venv.
- `schemas/record.py` imports and validates a hand-written dummy record
  (`tests/unit/test_record_schema.py`), and correctly rejects an invalid one
  (difficulty out of 1–5 range).
- `common/config.py` loads a VLM/LLM provider name from env without erroring; stub
  providers return canned output (`tests/unit/test_providers.py`).
- `python -m pytest -q` → 4 passed.

### Decisions made this phase
- Env manager: **venv + requirements.txt** (not poetry).
- Vector store target: **FAISS** (deferred to Phase 4, only `faiss-cpu` pinned so far).
- Batch orchestrator target: **Prefect** (deferred to Phase 3).
- VLM/LLM backend target: **OpenRouter** (OpenAI-compatible API, one key for many
  swappable models) — to be wired into `providers/` in Phase 1 once a model choice is
  made.

## Next
Phase 1 — Fold & Punch generator end-to-end (sampler → geometry engine → renderer →
distractor generator → VLM verify via LangGraph → LLM text gen → persist to
`data/records/` + `data/images/`), per `IMPLEMENTATION_PHASES.md`.
