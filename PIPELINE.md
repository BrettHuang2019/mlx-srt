# MLX-SRT pipeline

MLX-SRT turns a local audio or video file into French/Chinese subtitles. It is a single installed package and process; model stages communicate through Python values and pipeline artifacts, not subprocesses.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
bin/mlx-srt /path/to/clip.mp4
```

The macOS Service calls the same `bin/mlx-srt` shim. URL ingestion is not supported.

The default and explicit forms are equivalent:

```bash
mlx-srt clip.mp4
mlx-srt run clip.mp4
```

Useful pipeline options:

- `--output DIR`: relocate artifacts only.
- `--srt-output PATH`: relocate the final SRT.
- `--no-translate`: stop after French cue generation and still write the final SRT.
- `--keep-artifacts`: preserve artifacts and `state.json` after success.
- `--resume`: require compatible state; auto-resume does not need this flag.
- `--config PATH`, `--prompt-file PATH`, and `--debug`.

The final output defaults to `<input_dir>/<input_stem>.srt`. Artifacts default to `<input_dir>/<input_stem>_output/`. Existing final SRT files are overwritten.

## Stages

1. `audio`: ffmpeg converts every input, including audio inputs, to mono 16 kHz signed-16-bit WAV.
2. `stt`: mlx-audio Whisper emits `{"text": str}` with no timing data.
3. `punctuate`: `kredor/punctuate-all` punctuates the flat transcript. It can be disabled in config, in which case the unmodified text is still saved as the normal stage artifact.
4. `align`: mlx-audio's Qwen forced aligner assigns timestamps to words in one whole-file call.
5. `merge`: punctuation is matched back to timestamps, then sentences are split and short cues merged using gap, punctuation, French conjunction, dangling-word, and number-split scores.
6. `translate`: mlx-lm translates French cues in batches with three preceding cues of context. Responses require exact ID fidelity and valid non-empty translations; failed batches retry and recursively halve.

Artifacts are:

| Stage | Artifact |
|---|---|
| audio | `00_audio.wav` |
| stt | `01_transcript.json` |
| punctuate | `02_punctuated.json` |
| align | `03_words.json` |
| merge | `04_fr.srt` |
| translate | final `<input_stem>.srt` |

## Standalone stages

Standalone commands write to an explicit `--output-file` (`-o`) or to their stage filename in the current directory. `--stdout` emits JSON and writes no file.

```bash
mlx-srt audio clip.mp4 -o 00_audio.wav
mlx-srt stt clip.mp4 -o 01_transcript.json
mlx-srt punctuate --input-file 01_transcript.json -o 02_punctuated.json
mlx-srt align clip.mp4 --text-file 02_punctuated.json -o 03_words.json
mlx-srt merge --words 03_words.json --text 02_punctuated.json -o 04_fr.srt
mlx-srt translate 04_fr.srt -o bilingual.srt
```

A file named exactly like a command is disambiguated with the explicit form, for example `mlx-srt run ./stt`.

## Configuration

The complete default schema ships in `mlx_srt/defaults.yaml`. An optional user file is selected in this order:

1. `--config PATH`
2. `$MLX_SRT_CONFIG`
3. `~/.mlx-srt/config.yaml`
4. the first `config.yaml` found while walking upward from the input file

Overrides merge per section. Relative paths declared in a user config resolve against that config's directory; the packaged prompt resolves against package data. CLI paths resolve against the current directory. `--debug` reports the selected override.

## Resume and cleanup

State schema version 2 records one `last_completed_step`, the currently running step, and a failed step. Before every stage, the current state is written; after success, the completed stage is written atomically.

When state exists, the runner trusts `last_completed_step`, chooses the next intended stage, and walks backward only if that stage's input artifacts are missing. Interrupted stages rerun. Old or mismatched state starts fresh. `--resume` turns the absence of valid state into an error.

The terminal stage depends on the invocation. A preserved French-only run resumes at translation on a later full invocation. A repeated completed invocation exits without loading models when the final SRT still exists. Deleting the final SRT regenerates it from the nearest usable artifact; deleting the artifact directory forces a fresh run.

On normal success the complete artifact directory, including state, is removed. Failures preserve it.

## Concurrency and resources

Only one pipeline may load models at a time. `~/.mlx-srt/run.lock` is protected by `fcntl.flock`; its JSON content identifies the holder but never determines ownership. Waiters poll to the configured timeout. The kernel releases the lock after normal exit, exceptions, or process death. The file is intentionally never unlinked.

After acquiring the lock, the process checks available RAM using `psutil`. Missing `psutil` skips that check.

## Tests

`pytest` runs fast mocked unit tests and deselects `integration` tests. Real model validation is opt-in:

```bash
.venv/bin/pytest
.venv/bin/pytest -m integration
```
