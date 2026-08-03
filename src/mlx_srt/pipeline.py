"""Artifact-oriented six-stage runner and resume policy."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import align as align_stage
from . import audio, merge, punctuate, stt, translate
from .config import Config
from .lock import RunLock, check_available_memory

logger = logging.getLogger(__name__)

STEPS = ["audio", "stt", "punctuate", "align", "merge", "translate"]
FRESH = "FRESH"
DONE = "DONE"
SCHEMA_VERSION = 2
ARTIFACT_NAMES = {
    "audio": "00_audio.wav",
    "stt": "01_transcript.json",
    "punctuate": "02_punctuated.json",
    "align": "03_words.json",
    "merge": "04_fr.srt",
}


def artifact_paths(artifact_dir: str | Path) -> dict[str, Path]:
    root = Path(artifact_dir)
    return {step: root / filename for step, filename in ARTIFACT_NAMES.items()}


def _inputs_present(step: str, paths: dict[str, Path], input_file: Path) -> bool:
    required = {
        "audio": [input_file],
        "stt": [paths["audio"]],
        "punctuate": [paths["stt"]],
        "align": [paths["audio"], paths["punctuate"]],
        "merge": [paths["align"], paths["punctuate"]],
        "translate": [paths["merge"]],
    }[step]
    return all(path.is_file() for path in required)


def _runnable_at_or_before(
    want: str,
    paths: dict[str, Path],
    input_file: Path,
) -> str:
    for index in range(STEPS.index(want), -1, -1):
        if _inputs_present(STEPS[index], paths, input_file):
            return STEPS[index]
    return "audio"


def resolve_start_step(
    state: dict[str, Any] | None,
    artifact_dir: str | Path,
    terminal: str,
    *,
    input_file: str | Path | None = None,
    final_srt: str | Path | None = None,
) -> str:
    """Resolve FRESH, DONE, or the nearest runnable stage at/before intent."""
    if terminal not in STEPS:
        raise ValueError(f"Unknown terminal step: {terminal}")
    if not state or state.get("schema_version") != SCHEMA_VERSION:
        return FRESH
    actual_input = Path(input_file or state.get("input_file", "")).resolve()
    if not actual_input.is_file() or state.get("input_file") != str(actual_input):
        return FRESH
    paths = artifact_paths(artifact_dir)
    final_path_value = final_srt or state.get("pipeline_info", {}).get("final_srt")
    final_path = Path(final_path_value) if final_path_value else None
    last = state.get("last_completed_step")
    if last in STEPS:
        if STEPS.index(last) >= STEPS.index(terminal):
            if final_path is not None and final_path.is_file():
                return DONE
            return _runnable_at_or_before(terminal, paths, actual_input)
        return _runnable_at_or_before(STEPS[STEPS.index(last) + 1], paths, actual_input)
    current = state.get("current_step")
    if current in STEPS:
        return _runnable_at_or_before(current, paths, actual_input)
    return _runnable_at_or_before(terminal, paths, actual_input)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _clean_artifacts(artifact_dir: Path, *, preserve: tuple[Path, ...] = ()) -> None:
    """Remove only pipeline-owned files, never unrelated contents of --output."""
    preserved = {path.resolve() for path in preserve}
    owned = [*artifact_paths(artifact_dir).values(), artifact_dir / "state.json"]
    for path in owned:
        if path.resolve() not in preserved:
            path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        artifact_dir.rmdir()


def _new_state(input_file: Path, final_srt: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_info": {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "final_srt": str(final_srt),
        },
        "input_file": str(input_file),
        "last_completed_step": None,
        "current_step": None,
        "failed_step": None,
    }


def run_pipeline(
    input_file: str | Path,
    config: Config,
    *,
    output_dir: str | Path | None = None,
    srt_output: str | Path | None = None,
    keep_artifacts: bool = False,
    no_translate: bool = False,
    resume: bool = False,
    prompt_file: str | Path | None = None,
    use_lock: bool = True,
) -> Path:
    input_file = Path(input_file).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    artifact_dir = Path(output_dir).expanduser().resolve() if output_dir else input_file.parent / f"{input_file.stem}_output"
    final_srt = Path(srt_output).expanduser().resolve() if srt_output else input_file.with_suffix(".srt")
    terminal = "merge" if no_translate else "translate"

    def execute() -> Path:
        state_file = artifact_dir / "state.json"
        existing_state = None
        if state_file.is_file():
            try:
                existing_state = _read_json(state_file)
            except (OSError, json.JSONDecodeError):
                existing_state = None
        start = resolve_start_step(
            existing_state, artifact_dir, terminal,
            input_file=input_file, final_srt=final_srt,
        )
        if resume and start == FRESH:
            raise RuntimeError("--resume requested, but no compatible resumable state exists")
        if start == DONE:
            logger.info("Already complete, nothing to do: %s", final_srt)
            return final_srt
        if start == FRESH:
            if artifact_dir.exists():
                _clean_artifacts(artifact_dir, preserve=(input_file, final_srt))
            artifact_dir.mkdir(parents=True, exist_ok=True)
            state = _new_state(input_file, final_srt)
            start = "audio"
        else:
            state = existing_state
            state["pipeline_info"]["status"] = "running"
            state["pipeline_info"]["final_srt"] = str(final_srt)
            state["failed_step"] = None
        paths = artifact_paths(artifact_dir)
        _write_json(state_file, state)

        try:
            start_index = STEPS.index(start)
            terminal_index = STEPS.index(terminal)
            for step in STEPS[start_index : terminal_index + 1]:
                state["current_step"] = step
                _write_json(state_file, state)
                logger.info("Running stage: %s", step)
                if step == "audio":
                    audio.extract_audio(input_file, paths["audio"])
                elif step == "stt":
                    _write_json(paths["stt"], stt.transcribe(paths["audio"], config.stt.model_path))
                elif step == "punctuate":
                    raw_text = _read_json(paths["stt"])["text"]
                    result = (
                        punctuate.punctuate(raw_text, model_id=config.punctuation.model_path,
                                            chunk_words=config.punctuation.chunk_words)
                        if config.punctuation.enabled else {"text": raw_text}
                    )
                    _write_json(paths["punctuate"], result)
                elif step == "align":
                    text = _read_json(paths["punctuate"])["text"]
                    _write_json(paths["align"], align_stage.align(paths["audio"], text, config.align.model_path))
                elif step == "merge":
                    merged = merge.merge_srt(
                        _read_json(paths["align"]), _read_json(paths["punctuate"])["text"],
                        max_chars=config.merge.max_chars,
                        min_chars=config.merge.min_chars,
                        min_duration=config.merge.min_duration,
                    )
                    paths["merge"].write_text(merged, encoding="utf-8")
                    if no_translate:
                        final_srt.parent.mkdir(parents=True, exist_ok=True)
                        if paths["merge"].resolve() != final_srt:
                            shutil.copyfile(paths["merge"], final_srt)
                elif step == "translate":
                    bilingual = translate.translate_srt(
                        paths["merge"].read_text(encoding="utf-8"),
                        settings=translate.TranslationSettings.from_config(config.translate),
                        prompt_file=Path(prompt_file).expanduser().resolve() if prompt_file else config.translate.prompt_file,
                    )
                    final_srt.parent.mkdir(parents=True, exist_ok=True)
                    final_srt.write_text(bilingual, encoding="utf-8")
                state["last_completed_step"] = step
                state["current_step"] = None
                _write_json(state_file, state)
            state["pipeline_info"]["status"] = "completed"
            state["pipeline_info"]["ended_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(state_file, state)
        except Exception:
            state["pipeline_info"]["status"] = "failed"
            state["failed_step"] = state["current_step"]
            _write_json(state_file, state)
            raise
        if not keep_artifacts:
            _clean_artifacts(artifact_dir, preserve=(input_file, final_srt))
        return final_srt

    if not use_lock:
        return execute()
    with RunLock(
        input_file,
        interval=config.system.task_check_interval,
        timeout=config.system.max_wait_time_minutes * 60,
        on_wait=lambda holder: logger.info(
            "Waiting for pid=%s input=%s", holder.get("pid", "unknown"), holder.get("input", "unknown")
        ),
    ):
        enough, available = check_available_memory(config.system.min_ram_gb)
        if not enough:
            raise RuntimeError(
                f"Insufficient available RAM: {available:.2f} GB; requires {config.system.min_ram_gb:.2f} GB"
            )
        return execute()
