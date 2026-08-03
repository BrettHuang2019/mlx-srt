# mlx-srt

Local-first macOS CLI that turns one audio or video file into language-learning
subtitles: French on the first line, Chinese on the second. Everything runs on
Apple Silicon through MLX; the only network access is a one-time model download.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires Python 3.10+, Apple Silicon, and `ffmpeg` on `PATH`
(`brew install ffmpeg`).

## Run

```bash
bin/mlx-srt /path/to/clip.mp4
```

The bilingual `clip.srt` lands next to the input. `bin/mlx-srt` is a shim that
calls the `.venv` entry point, so the macOS Finder Service
(`mlx-srt-service.sh`) and the shell use the same path.

Common options — `--no-translate` for French only, `--keep-artifacts` to keep
stage outputs, `--output DIR`, `--srt-output PATH`, `--config PATH`, `--debug`.

## Develop

```bash
make install   # venv + editable install with test and dev extras
make test      # fast mocked unit tests
make lint      # ruff check
make fix       # ruff check --fix
make check     # lint + test
```

`make test-integration` runs the opt-in tests that download real models and need
Apple Silicon.

## Docs

- [docs/PIPELINE.md](docs/PIPELINE.md) — stages, artifacts, config resolution,
  resume policy, concurrency. Read this before touching `pipeline.py`.
- [docs/PRD.md](docs/PRD.md) — product contract and acceptance criteria.
- [docs/EDGE_CASES.md](docs/EDGE_CASES.md) — operational limits and known risks.
