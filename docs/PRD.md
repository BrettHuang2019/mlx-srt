# MLX-SRT product requirements

## Purpose

MLX-SRT is a local-first macOS command-line tool and Finder Service that converts a selected audio or video file into language-learning subtitles: French on the first line and Chinese on the second.

## User contract

- `mlx-srt <file>` processes one local media file.
- The final `<input_stem>.srt` appears next to the source so the Finder Service can locate it deterministically.
- Processing stays local apart from one-time model downloads.
- Multiple independent invocations serialize to avoid competing for unified memory.
- Interrupted model work resumes from durable stage artifacts.
- URL downloads, cookies, scheduling, databases, APIs, summaries, and transcript-segment refinement are out of scope.

## Functional requirements

1. Normalize every media input to 16 kHz mono signed-16-bit WAV.
2. Transcribe French speech to flat text using mlx-audio Whisper.
3. Restore punctuation using a token-classification model, with a config escape hatch to bypass it.
4. Force-align the transcript to audio and emit plain word timestamp dictionaries.
5. Construct readable French cues using punctuation, timing gaps, length limits, and French-aware split penalties.
6. Translate cue batches with deterministic sampling, contextual preceding cues, strict response validation, retries, and recursive batch splitting.
7. Render valid bilingual SRT with French before Chinese.
8. Expose every stage as a standalone debugging command without corrupting pipeline artifacts.
9. Preserve the final-path, artifact-path, no-translation, cleanup, and explicit-resume contracts documented in `PIPELINE.md`.

## Non-functional requirements

- One src-layout Python package, one virtual environment, one console script, and no internal subprocess stage boundaries.
- Lazy model imports so CLI help, config, resume resolution, and completed runs do not load models.
- Frozen typed configuration loaded once per invocation.
- Packaged defaults and prompt work in editable installs and wheels regardless of current directory.
- Fast unit tests mock every model loader; real model tests use the `integration` marker.
- State writes are atomic and use schema version 2.
- The concurrency lock survives crashes safely through kernel ownership and is never unlinked.

## Acceptance criteria

- `pip install -e '.[test,dev]'` installs the `mlx-srt` entry point and package data.
- Bare-path and explicit `run` CLI forms are equivalent.
- All six stage commands run independently and respect `--output-file`/`--stdout`.
- Pipeline output and artifact placement match the documented matrix.
- Resume tests cover interrupted stages, missing artifacts, incompatible state, both terminal stages, and completed idempotence.
- Unit tests run without model downloads and pass on a non-ML test host.
- A separately invoked integration test produces parseable bilingual SRT on Apple Silicon.

## Known risk

Whole-file STT and alignment have no measured duration ceiling. A duration sweep must be completed before claiming support for long-form recordings; see `EDGE_CASES.md`.
