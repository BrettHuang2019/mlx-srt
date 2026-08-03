# mlx-srt

Local macOS CLI: one media file in, bilingual French/Chinese SRT out. One
installed package, one process — model stages pass Python values and on-disk
artifacts, never subprocesses.

## Commands

Always use the existing `.venv`; never create a new environment.

```bash
make test              # or: .venv/bin/pytest      — fast, all models mocked
make lint              # or: .venv/bin/ruff check src tests
make check             # lint + test; run before declaring work done
make test-integration  # opt-in, downloads real models, needs Apple Silicon
bin/mlx-srt clip.mp4   # run the pipeline end to end
```

## Layout

`src/mlx_srt/` — everything. One module per pipeline stage:

| File | Role |
|---|---|
| `cli.py` | Click entry point. `run` plus one standalone command per stage. Bare paths dispatch to `run`. |
| `pipeline.py` | Stage runner, `state.json` writes, resume resolution, cleanup. |
| `config.py` | Frozen dataclasses loaded from `defaults.yaml` plus one optional override file. |
| `audio.py` | ffmpeg → 16 kHz mono s16 WAV. |
| `stt.py` | mlx-audio Whisper → `{"text": ...}`, no timings. |
| `punctuate.py` | `kredor/punctuate-all` token classification, chunked with rollback. |
| `align.py` | mlx-audio Qwen forced aligner → word timestamps. |
| `merge.py` | Timestamps + punctuation → French cues, with split/merge scoring. |
| `translate.py` | mlx-lm batch translation, strict validation, recursive halving. |
| `srt.py` | Shared SRT parse/render. |
| `lock.py` | `fcntl.flock` single-run gate and RAM check. |

Stage order is `audio → stt → punctuate → align → merge → translate`, defined
once in `pipeline.py:STEPS`.

## Rules that bite

- **Read [docs/PIPELINE.md](docs/PIPELINE.md) before touching `pipeline.py`.**
  Resume, artifact placement, and cleanup are contracts with documented
  behaviour and tests, not implementation details.
- `defaults.yaml` is the only source of default values. Stage functions take
  their settings as required parameters — do not reintroduce module-level
  `DEFAULT_*` constants.
- Model imports stay lazy (inside the function). CLI help, config loading,
  resume resolution, and already-complete runs must not load a model.
- New tests mock every model loader. Anything needing a real model gets
  `@pytest.mark.integration`.
- No auto-formatter. `ruff check` is the gate; wrapping is by hand.

## Style

Minimal, readable, standard. Fewest lines and simplest logic that solves the
problem; clarity over cleverness; language idioms over invention. No unused
imports, variables, or comments. Optimize only when measured.

## Out of scope

URL downloads, cookies, scheduling, databases, web APIs, summarization, and
transcript-segment refinement. See [docs/PRD.md](docs/PRD.md).
