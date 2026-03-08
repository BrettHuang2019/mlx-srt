# MLX-SRT Pipeline

This document summarizes the pipeline implemented in `src/` and the edge cases covered in `tests/`.

## Entry points

- `src/main.py`
  - Accepts a local audio file, local video file, or URL.
  - Runs resource checks before starting.
  - Auto-resumes when `state.json` already exists in the output directory.
- `src/translation/translate.py`
  - Contains the translation sub-pipeline, state management, batching, validation, and resume logic.

## End-to-end flow

### 1. System gating

Before any media work starts, `check_system_resources()`:

- Looks for another pipeline with `pipeline_info.status == "running"` under `downloads/**/state.json`.
- Waits for that task to finish, with configurable poll interval and timeout.
- Checks available RAM against `config.yaml` minimum.
- Aborts startup if the wait times out, the user interrupts the wait, or RAM is insufficient.

### 2. Input classification

`main.py` splits the flow into three cases:

- URL input
  - Uses `is_url()`.
  - Resolves a default output directory under `downloads/<safe_title>_output`.
- Local audio input
  - Supported extensions: `.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`.
- Local video input
  - Supported extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`.

Unsupported extensions fail immediately.

### 3. URL ingestion

If the input is a URL, `process_url()`:

- Checks `state.json` for a previously completed download matching the same URL.
- Reuses the downloaded file if the state says download completed and the file still exists.
- Otherwise downloads with `yt-dlp` through `download_video_from_url()`.

Download behavior:

- Downloads into the repo `downloads/` directory, not the pipeline output directory.
- Requests best quality up to 1080p, preferring at least 720p.
- Tries Chrome cookies when available.
- Saves download state into the pipeline output folder.
- On download failure, writes failed download state and raises an error.

### 4. Video audio extraction

If the input is a video, `process_video_file()`:

- Reuses `extracted_audio.wav` when resuming and the file already exists.
- Otherwise calls `extract_audio()` to run `ffmpeg`.
- Produces 16 kHz, mono, 16-bit PCM WAV.
- Verifies the output file exists and is non-empty.

If extraction fails, the pipeline updates `state.json` to `failed` when possible.

### 5. Whisper transcription

`process_audio_file()` performs transcription:

- Reuses `00_whisper_transcription.json` when resuming and the file already exists.
- Otherwise calls `transcribe_audio()`.
- `transcribe_audio()`:
  - Requires `mlx_whisper`.
  - Loads Whisper config from `config.yaml`, with defaults if the file is missing.
  - Passes model path, language, and optional initial prompt.
  - Returns only `text` and cleaned `segments`.

Each transcription segment is normalized to:

- `id`
- `start`
- `end`
- `text`

If transcription fails, the pipeline updates `state.json` to `failed` when possible.

### 6. Translation pipeline initialization

`translation_pipeline()`:

- Creates or loads `state.json`.
- Builds a state model with:
  - `pipeline_info`
  - `input_files`
  - `completed_steps`
  - `current_step`
  - step states for `refinement`, `summary`, `preprocessing`, `translation`, `validation`, `merging`, and `final_output`
  - optional `download_info`
- Uses `get_resume_point()` plus file validation to decide which step to restart from.

Resume order is:

1. `refinement`
2. `summary`
3. `preprocessing`
4. `translation`
5. `validation`
6. `merging`
7. `final_output`

The implementation does not actually run a distinct validation step even though the state model includes one.

### 7. Segment refinement

`refine_segments()` transforms raw Whisper segments before translation.

Actual processing order:

1. Remove empty segments.
2. Remove segments that are only punctuation or symbols.
3. Remove leading spaces.
4. Remove duplicate segments whose normalized text appears 4 or more times.
5. Fix spaces before `!` and `?`.
6. Remove repeated words appearing 4 or more times in one segment, keeping the first.
7. Remove repetitive word patterns that look like LLM artifacts.
8. Remove repeated character-level patterns.
9. Merge sentence fragments until a segment ends with `.`, `!`, or `?`.
10. Split long multi-sentence blocks into separate timed segments.

Refinement outputs:

- `01_refined_transcript.json`
- optionally `01_merge_report.txt`
- optionally `01_split_report.txt`

### 8. Summary generation

`summarize()` reads the refined transcript and:

- Uses the transcript `text`, or rebuilds it from segment text if `text` is missing.
- Saves original transcript and extracted full text.
- Generates a Chinese summary with `mlx_lm` if available.
- Falls back to a mock Chinese summary if `mlx_lm` is unavailable or summary generation fails.

Artifacts include:

- `00_original_transcript.json`
- `00_extracted_text.txt`
- `04_summary.txt`
- `04_summary_report.txt`

### 9. Translation preprocessing

The refined segments are prepared for translation by:

1. `filter_out_empty_and_ellipsis_segments()`
   - Removes empty strings and ellipsis-only content such as `...`.
2. `preserve_segment_order()`
   - Sorts by `id`.
3. `regenerate_sequential_ids()`
   - Reassigns IDs sequentially from `1`.
4. `convert_segments_to_translation_format()`
   - Converts each segment into:
     - `index`
     - `fr`

Artifacts include:

- `01_converted_segments.json`
- `02_filtered_segments.json`
- `03_ordered_segments.json`
- `03b_regenerated_ids_segments.json`

### 10. Batch translation

`batch_translate()`:

- Loads translation config from `config.yaml`.
- Requires `mlx_lm`.
- Loads the model once for all batches.
- Splits segments using configured `batch_size`.
- Builds prompts from:
  - transcript summary
  - up to 3 previous segments as context
  - current batch as JSON
- Saves a batch report and raw LLM response per batch.

For each batch it:

1. Generates model output.
2. Sanitizes common Unicode quotes and some JSON mistakes.
3. Parses JSON.
4. Validates structure.
5. Validates item order and index fidelity.
6. Validates that every item has a non-empty `zh`.
7. Validates translation quality with `is_valid_translation()`.

Accepted translation forms are:

- Chinese text that is not identical to the French input.
- Numbers-only output.
- Very short names or expressions that intentionally remain unchanged.
- Non-Chinese text that still differs from the original French.

If a full batch fails validation:

- The batch is marked failed in state.
- The code recursively splits the batch into smaller halves.
- Each half is retried.
- A single-sentence batch is retried `max_retries + 1` times and then fails hard.

Artifacts include:

- `05_translation_context.txt`
- `06_batch_report_<batch_id>.txt`
- `07_llm_response_<batch_id>.json`
- `08_translated_segments.json`
- `09_translation_validation.txt`

### 11. Merge translations back

`merge_translations_back_to_segments()`:

- Maps translations back by `id` or `index`.
- Reattaches `zh` to the refined, regenerated segment structure.
- Rebuilds the top-level transcript text from the refined segment text.

Outputs include:

- `10_merged_segments.json`
- `11_final_translated_transcript.json`

`main.py` also writes a user-facing final transcript:

- local file input: `<input_stem>_translated.json`
- URL input: `translated_transcript.json`

### 12. SRT generation

Unless `--no-srt` is set, `generate_srt_file()`:

- Reads `translated_transcript["segments"]`.
- Formats SRT timestamps as `HH:MM:SS,mmm`.
- Writes bilingual entries when `zh` exists.
- Falls back to original text only when translation is missing.

Default SRT location:

- local file input: next to the input file
- URL input: next to the most recently modified downloaded video in `downloads/`

### 13. Completion and cleanup

On success:

- The pipeline prints coverage stats.
- `translation_pipeline()` marks the state as `completed`.
- `main.py` optionally deletes the artifact output directory unless `--keep-artifacts` is set.

On failure:

- download, extraction, transcription, or translation errors can mark the pipeline state as `failed`.

## Important saved artifacts

Common files you will see in the output folder:

- `state.json`
- `00_whisper_original.json`
- `00_whisper_transcription.json`
- `01_refined_transcript.json`
- `04_summary.txt`
- `08_translated_segments.json`
- `11_final_translated_transcript.json`
- `PIPELINE_SUMMARY.txt`

Note that some filenames overlap conceptually across stages. For example, there is both a raw Whisper save from `main.py` and a translation-pipeline copy of the Whisper input.

## Edge cases covered by code and tests

### Main pipeline and orchestration

- Auto-resume is enabled whenever `state.json` already exists in the output directory.
- Explicit `--resume` fails if the output directory or state file does not exist.
- Video inputs produce `extracted_audio.wav`.
- Audio and video end-to-end tests expect translated JSON output and valid subtitles.

### Resource management

- Existing running tasks block new work until they finish or timeout.
- Missing `psutil` does not block execution; RAM checks are skipped.

### Download and ingestion

- URL download state can say completed while the actual file is missing; in that case the video is downloaded again.
- Chrome cookies are optional.
- If the exact downloaded filename cannot be matched by title, the code falls back to scanning common video extensions.

### Audio extraction

- Missing input video raises `FileNotFoundError`.
- Invalid output directory raises `FileNotFoundError`.
- Existing output files are overwritten.
- Tests assert the extracted WAV is readable, mono, 16 kHz, 16-bit, and non-empty.

### Transcription

- Missing audio file raises `FileNotFoundError`.
- Missing `mlx_whisper` raises `ImportError`.
- Segment output is constrained to exactly `id`, `start`, `end`, and `text`.

### Refinement edge cases

- Empty segments are removed.
- Punctuation-only segments like `!`, `?`, `...`, or `@#$` are removed.
- Leading spaces are stripped.
- Spaces before `!` and `?` are removed, while other punctuation spacing is preserved.
- Fragments merge until sentence-ending punctuation is found.
- A final incomplete segment is not merged if there is no next segment.
- Long multi-sentence segments split only when the sentences are long enough, or when quote delimiters justify splitting.
- Duplicate segments are removed when the same text appears 4 or more times, case-insensitively.
- Repeated words inside one segment are collapsed when they appear 4 or more times.
- Character-level repeated patterns can remove or clean a segment.
- ID gaps caused by merges or removals are fixed by sequential ID regeneration.

### Translation validation edge cases

- Empty `zh` fails validation.
- JSON that is malformed is retried after sanitization or JSON extraction.
- Output count must exactly match input batch size.
- Output order and indexes must exactly match input order.
- A response item must be a dictionary and contain both `index` and `zh`.
- Chinese output identical to the French input is rejected.
- Numbers-only output is allowed.
- Short unchanged names or expressions are allowed.
- Batch failures trigger recursive splitting.
- Single-sentence failures eventually raise a hard error.

### SRT edge cases

- Timestamp formatting handles milliseconds and hour rollover.
- Missing translation falls back to original text only.
- Bilingual SRT output places French on one line and Chinese on the next.

## Gaps or quirks visible in the current implementation

- `state["steps"]` includes a `validation` step, but the pipeline never executes a standalone validation phase.
- `translate_transcript()` and `translation_pipeline()` produce similar artifacts but are not identical flows.
- `main.py` may delete the artifact directory on success unless `--keep-artifacts` is set, even though that directory contains state and debug files.
- The completion message under `--keep-artifacts` says "Use --keep-artifacts option to preserve them" even though that branch is already the preserve-artifacts path.
