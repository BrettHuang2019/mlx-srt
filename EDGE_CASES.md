# Edge cases and operational limits

## Files and outputs

- Any local format understood by ffmpeg is accepted; URL inputs are not.
- Audio and video are both normalized to mono 16 kHz signed-16-bit WAV.
- Missing input files and missing output parents fail before model loading.
- ffmpeg output must exist and be non-empty.
- The final SRT is always next to the input unless `--srt-output` is used. `--output` only relocates artifacts.
- A run silently overwrites an existing final SRT. Hand-edited subtitles should be copied elsewhere before rerunning.
- Successful runs remove artifacts unless `--keep-artifacts`; failures preserve them.

## Text and cue construction

- Empty alignment text is rejected.
- Empty punctuation input returns empty text without loading a model.
- The punctuator accepts raw and `LABEL_`-prefixed classifier labels, maps token spans back to words, rolls incomplete chunk tails into the next chunk, normalizes repeated punctuation and spacing, and restores sentence capitalization.
- Word matching uses a three-word timestamp lookahead. An unmatched punctuated word inherits the previous matched start (or zero initially) with zero duration.
- Long cues prefer audio gaps plus punctuation, then gaps, punctuation, and French conjunctions. They penalize dangling French function words and splits between a number and its unit.
- Short, brief cues merge with the next cue when possible, then the previous cue, without exceeding `merge.max_chars`.
- SRT timestamp rounding carries correctly into the next second/minute and clamps negative values to zero.

## Translation responses

- Smart quotes are normalized and a JSON array may be extracted from surrounding model prose.
- Response count, order, integer IDs, and `zh` fields must match the batch exactly.
- Empty translations and unchanged multi-word French are rejected.
- Numbers/statistics, unchanged names of at most three words, Chinese output, and different non-Chinese output are accepted.
- A failed batch retries according to config and then halves recursively. A failed single cue aborts while keeping pipeline artifacts for resume.
- Ellipsis-only cues are removed and remaining cues are sorted and renumbered before translation.

## State and resume

- Only schema version 2 state matching the absolute input path is trusted.
- Missing, corrupt, old, or input-mismatched state starts fresh; explicit `--resume` instead errors.
- A stage runs only when all of its inputs exist. Missing artifacts cause the resolver to walk backward to the nearest runnable stage.
- `last_completed_step` takes priority over filesystem inference. A `current_step` without completion is treated as interrupted and rerun.
- Completion is idempotent only while the final SRT still exists.
- A French-only preserved run can later resume directly at translation.

## Concurrency

- Separate invocations serialize on `~/.mlx-srt/run.lock` using `fcntl.flock`.
- The persistent file itself is not the lock and is never deleted.
- Holder metadata can be empty or torn without affecting correctness.
- `SIGKILL` and crashes release the kernel lock immediately; there is no stale-lock recovery heuristic.
- Waiters report holder metadata and fail after the configured timeout.

## Long media risk

STT and forced alignment each process the whole clip in one model call. The code deliberately does not invent chunking absent evidence from the upstream flow. A real 15-second Apple Silicon check with `mlx-audio 0.4.7` and `mlx 0.32.0` passed: STT took 7.72 seconds with 772,636,672 bytes maximum RSS, and alignment took 4.50 seconds with 1,741,570,048 bytes maximum RSS. It produced flat French text and 10 valid word timestamp objects.

The planned 30-second/5-minute/20-minute/60-minute sweep has not yet been recorded. Until that measurement exists, hour-long media is an acknowledged unbounded-memory and runtime risk.

If a ceiling appears, prefer adding alignment chunking upstream, then silence-based chunks; falling back to timestamped mlx-whisper is the last option.
