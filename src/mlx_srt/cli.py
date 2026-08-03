"""Click command-line interface for pipeline and standalone stages."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import click

from . import __version__, align as align_stage, audio, merge, punctuate, stt, translate
from .config import load_config
from .pipeline import run_pipeline


class DefaultGroup(click.Group):
    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["run", *args]
        return super().resolve_command(ctx, args)


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def _write_result(value, output_file: Path, stdout: bool) -> None:
    if stdout:
        click.echo(json.dumps(value, ensure_ascii=False))
        return
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        output_file.write_text(value, encoding="utf-8")
    else:
        output_file.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(str(output_file))


@click.group(cls=DefaultGroup, no_args_is_help=True)
@click.version_option(__version__)
def main() -> None:
    """Turn local media into French/Chinese subtitles."""


def _pipeline_options(function):
    options = [
        click.option("--debug", is_flag=True, help="Enable debug logging."),
        click.option("--prompt-file", type=click.Path(path_type=Path), help="Override translation prompt."),
        click.option("--config", "config_file", type=click.Path(path_type=Path), help="Configuration override."),
        click.option("--resume", is_flag=True, help="Require compatible resume state."),
        click.option("--no-translate", is_flag=True, help="Produce French-only SRT."),
        click.option("--keep-artifacts", is_flag=True, help="Keep stage artifacts and state."),
        click.option("--srt-output", type=click.Path(path_type=Path), help="Final SRT file path."),
        click.option("--output", type=click.Path(path_type=Path), help="Artifact directory."),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_pipeline_options
def run(
    input_file: Path,
    output: Path | None,
    srt_output: Path | None,
    keep_artifacts: bool,
    no_translate: bool,
    resume: bool,
    config_file: Path | None,
    prompt_file: Path | None,
    debug: bool,
) -> None:
    """Run the complete staged pipeline."""
    _configure_logging(debug)
    config = load_config(input_file, config_file)
    if debug:
        logging.getLogger(__name__).debug("Config override: %s", config.source or "none (packaged defaults)")
    try:
        output_path = run_pipeline(
            input_file, config,
            output_dir=output,
            srt_output=srt_output,
            keep_artifacts=keep_artifacts,
            no_translate=no_translate,
            resume=resume,
            prompt_file=prompt_file,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(output_path))


def _common_stage_options(function):
    function = click.option("--debug", is_flag=True)(function)
    function = click.option("--stdout", "to_stdout", is_flag=True, help="Print JSON; do not write a file.")(function)
    return function


@main.command("audio")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-file", type=click.Path(path_type=Path), default=Path("00_audio.wav"), show_default=True)
@click.option("--debug", is_flag=True)
def audio_command(input_file: Path, output_file: Path, debug: bool) -> None:
    """Normalize media to 16 kHz mono WAV."""
    _configure_logging(debug)
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    click.echo(str(audio.extract_audio(input_file, output_file)))


@main.command("stt")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-file", type=click.Path(path_type=Path), default=Path("01_transcript.json"), show_default=True)
@click.option("--config", "config_file", type=click.Path(path_type=Path))
@_common_stage_options
def stt_command(input_file: Path, output_file: Path, config_file: Path | None, to_stdout: bool, debug: bool) -> None:
    """Transcribe media to plain text JSON."""
    _configure_logging(debug)
    config = load_config(input_file, config_file)
    with tempfile.TemporaryDirectory() as temporary:
        wav = audio.extract_audio(input_file, Path(temporary) / "audio.wav")
        result = stt.transcribe(wav, config.stt.model_path)
    _write_result(result, output_file, to_stdout)


@main.command("punctuate")
@click.option("--input-file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-file", type=click.Path(path_type=Path), default=Path("02_punctuated.json"), show_default=True)
@click.option("--config", "config_file", type=click.Path(path_type=Path))
@_common_stage_options
def punctuate_command(input_file: Path, output_file: Path, config_file: Path | None, to_stdout: bool, debug: bool) -> None:
    """Punctuate transcript JSON."""
    _configure_logging(debug)
    config = load_config(input_file, config_file)
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    result = punctuate.punctuate(
        payload["text"], model_id=config.punctuation.model_path,
        chunk_words=config.punctuation.chunk_words,
    ) if config.punctuation.enabled else payload
    _write_result(result, output_file, to_stdout)


@main.command("align")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--text-file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-file", type=click.Path(path_type=Path), default=Path("03_words.json"), show_default=True)
@click.option("--config", "config_file", type=click.Path(path_type=Path))
@_common_stage_options
def align_command(input_file: Path, text_file: Path, output_file: Path, config_file: Path | None, to_stdout: bool, debug: bool) -> None:
    """Force-align transcript text to media."""
    _configure_logging(debug)
    config = load_config(input_file, config_file)
    text = json.loads(text_file.read_text(encoding="utf-8"))["text"]
    with tempfile.TemporaryDirectory() as temporary:
        wav = audio.extract_audio(input_file, Path(temporary) / "audio.wav")
        result = align_stage.align(wav, text, config.align.model_path)
    _write_result(result, output_file, to_stdout)


@main.command("merge")
@click.option("--words", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--text", "text_file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-file", type=click.Path(path_type=Path), default=Path("04_fr.srt"), show_default=True)
@click.option("--config", "config_file", type=click.Path(path_type=Path))
@_common_stage_options
def merge_command(words: Path, text_file: Path, output_file: Path, config_file: Path | None, to_stdout: bool, debug: bool) -> None:
    """Build French SRT cues from timestamps and punctuated text."""
    _configure_logging(debug)
    config = load_config(text_file, config_file)
    result = merge.merge_srt(
        json.loads(words.read_text(encoding="utf-8")),
        json.loads(text_file.read_text(encoding="utf-8"))["text"],
        max_chars=config.merge.max_chars,
        min_chars=config.merge.min_chars,
        min_duration=config.merge.min_duration,
    )
    _write_result(result, output_file, to_stdout)


@main.command("translate")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-file", type=click.Path(path_type=Path), default=Path("translated.srt"), show_default=True)
@click.option("--prompt-file", type=click.Path(path_type=Path))
@click.option("--config", "config_file", type=click.Path(path_type=Path))
@_common_stage_options
def translate_command(input_file: Path, output_file: Path, prompt_file: Path | None, config_file: Path | None, to_stdout: bool, debug: bool) -> None:
    """Translate a French SRT to bilingual French/Chinese SRT."""
    _configure_logging(debug)
    config = load_config(input_file, config_file)
    settings = translate.TranslationSettings(
        config.translate.model_path, config.translate.batch_size, config.translate.max_tokens,
        config.translate.temperature, config.translate.max_retries, config.translate.retry_delay,
    )
    result = translate.translate_srt(
        input_file.read_text(encoding="utf-8"), settings=settings,
        prompt_file=prompt_file.expanduser().resolve() if prompt_file else config.translate.prompt_file,
    )
    _write_result(result, output_file, to_stdout)


if __name__ == "__main__":
    main()
