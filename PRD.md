# Project Requirement Document: Multi-Language Subtitle Generation

## Overview

Create a Mac Mini tool that generates bilingual SRT files (FR + ZH) from any video/Audio for French language learning.

---

## Functional Decomposition

### 1. Media Ingestion
**What it does**: Accept and normalize input media into processable audio format

#### Feature: Media Type Detection
- **Description**: Identify if input is video or audio file
- **Inputs**: File path
- **Outputs**: Media type (video/audio), codec info, duration
- **Behavior**: Probe file metadata using ffmpeg/ffprobe

#### Feature: Audio Extraction
- **Description**: Extract audio stream from video files
- **Inputs**: Video file path, target audio format (wav/mp3)
- **Outputs**: Audio file path, sample rate, channels
- **Behavior**: Use ffmpeg to extract audio track, normalize to 16kHz mono for optimal whisper performance

---

### 2. Transcription Processing
**What it does**: Convert audio to timestamped text segments

#### Feature: MLX Whisper Transcription
- **Description**: Generate timestamped transcription using mlx_whisper with streaming output
- **Inputs**: Audio file path, language hint (optional), model size
- **Outputs**: Stream of segments (text, start_time, end_time)
- **Behavior**: Process audio through MLX Whisper, yield segments as they're generated, persist intermediate results to allow resume on failure

#### Feature: Segment Refinement
- **Description**: Merge over-segmented sentences into natural phrase boundaries
- **Inputs**: Raw segment list with timestamps
- **Outputs**: Refined segment list with improved grouping
- **Behavior**: 
  - Detect fragments (< 3 words, no punctuation)
  - Merge with adjacent segments if gap < 0.5s
  - Respect punctuation boundaries (., !, ?)
  - Preserve original timestamps

---

### 3. Translation Pipeline
**What it does**: Translate transcribed text into target languages

#### Feature: Translation Payload Preparation
- **Description**: Format transcription segments into translation-ready JSON structure
- **Inputs**: Refined segment list, target language codes
- **Outputs**: JSON payload with segments indexed for batch processing
- **Behavior**: Structure as `[{id, text, start, end}, ...]` with metadata (source_lang, target_lang)

#### Feature: MLX Batch Translation
- **Description**: Translate segments using mlx_lm with resumable streaming
- **Inputs**: Translation payload, target language, model name, checkpoint path (optional)
- **Outputs**: Stream of translated segments, checkpoint state
- **Behavior**:
  - Process in batches (5-10 segments per prompt)
  - Stream outputs as they generate
  - Save checkpoint every N batches
  - Allow cancellation and resume from last checkpoint

---

### 4. Subtitle Generation
**What it does**: Export timestamped translations as standard subtitle files

#### Feature: SRT File Creation
- **Description**: Format translated segments into SRT subtitle format
- **Inputs**: Translated segment list (per language), output directory
- **Outputs**: SRT file path per language
- **Behavior**: 
  - Format: `index\ntimestamp --> timestamp\ntext\n\n`
  - Convert timestamps to SRT format (HH:MM:SS,mmm)
  - Handle special characters and line breaks
  - Validate segment ordering

---

### 5. State Management (Resilience)
**What it does**: Enable pause/resume across all processing stages

#### Feature: Progress Checkpointing
- **Description**: Save intermediate state to allow resume after interruption
- **Inputs**: Processing stage, completed items, pending items
- **Outputs**: Checkpoint file (JSON)
- **Behavior**: Persist after each major step (transcription complete, translation batch N complete)

#### Feature: Resume Detection
- **Description**: Detect existing checkpoints and continue from last successful step
- **Inputs**: Input file path, output directory
- **Outputs**: Resume point (stage + item index) or null if starting fresh
- **Behavior**: Look for checkpoint files, validate integrity, return resume state

---

## Structural Decomposition

### Project Structure

```
subtitle-generator/
├── src/
│   ├── models/                     # Data classes
│   │   ├── media.py                # MediaInfo, AudioFile
│   │   ├── segment.py              # Segment, TranslatedSegment
│   │   ├── checkpoint.py           # CheckpointState, ResumePoint
│   │   └── __init__.py
│   │
│   ├── config/                     # Configuration
│   │   ├── models.py               # Model paths and sizes
│   │   ├── languages.py            # Language code mappings
│   │   ├── settings.py             # Processing parameters
│   │   ├── prompts.py              # Whisper & LLM prompts
│   │   └── __init__.py
│   │
│   ├── ingestion/                  # Media processing
│   │   ├── media_detector.py       # Media Type Detection
│   │   ├── audio_extractor.py      # Audio Extraction
│   │   └── __init__.py
│   │
│   ├── transcription/              # Speech-to-text
│   │   ├── whisper_transcriber.py  # MLX Whisper Transcription
│   │   ├── segment_refiner.py      # Segment Refinement
│   │   └── __init__.py
│   │
│   ├── translation/                # Text translation
│   │   ├── payload_builder.py      # Translation Payload Preparation
│   │   ├── mlx_translator.py       # MLX Batch Translation
│   │   └── __init__.py
│   │
│   ├── subtitle/                   # Output generation
│   │   ├── srt_writer.py           # SRT File Creation
│   │   └── __init__.py
│   │
│   ├── state/                      # Checkpoint management
│   │   ├── checkpoint_manager.py   # Progress Checkpointing
│   │   ├── resume_handler.py       # Resume Detection
│   │   └── __init__.py
│   │
│   └── main.py                     # Orchestration layer
│
├── tests/
│   ├── fixtures/                   # Test data
│   │   ├── sample_10s.mp4
│   │   ├── sample_30s.wav
│   │   ├── sample_segments.json
│   │   └── sample_checkpoint.json
│   ├── test_models.py
│   ├── test_state.py
│   ├── test_ingestion.py
│   ├── test_transcription.py
│   ├── test_translation.py
│   ├── test_subtitle.py
│   └── test_integration.py
│
├── checkpoints/                    # Runtime checkpoint storage
├── output/                         # Generated subtitles
├── requirements.txt
└── README.md
```

### Module Exports

#### `models/`
**Exports:**
- `MediaInfo` - Data class for media file metadata
- `AudioFile` - Data class for extracted audio info
- `Segment` - Data class for timestamped text segment
- `TranslatedSegment` - Data class for translated segment
- `CheckpointState` - Data class for checkpoint data
- `ResumePoint` - Data class for resume information

#### `config/`
**Exports:**
- `get_model_config(model_name: str) -> dict` - Model paths and parameters
- `get_language_name(code: str) -> str` - Language code to name mapping
- `get_settings() -> dict` - Processing parameters
- `get_whisper_prompt(language: str) -> str` - Whisper system prompt
- `get_translation_prompt(source_lang: str, target_lang: str) -> str` - Translation prompt

#### `ingestion/`
**Exports:**
- `detect_media_type(file_path: str) -> MediaInfo`
- `extract_audio(video_path: str, output_path: str) -> AudioFile`

#### `transcription/`
**Exports:**
- `transcribe_stream(audio_path: str, model_size: str) -> Iterator[Segment]`
- `refine_segments(segments: List[Segment]) -> List[Segment]`

#### `translation/`
**Exports:**
- `build_translation_payload(segments: List[Segment], target_langs: List[str]) -> dict`
- `translate_batch_stream(payload: dict, checkpoint_path: str) -> Iterator[TranslatedSegment]`

#### `subtitle/`
**Exports:**
- `write_srt(segments: List[TranslatedSegment], output_path: str, language: str) -> str`

#### `state/`
**Exports:**
- `save_checkpoint(stage: str, data: dict, file_path: str) -> None`
- `load_checkpoint(file_path: str) -> Optional[CheckpointState]`
- `find_resume_point(input_file: str, output_dir: str) -> Optional[ResumePoint]`

#### `main.py`
**Orchestrates** the pipeline:
- Calls modules in dependency order
- Handles checkpointing between stages
- Provides CLI interface

---

## Dependency Graph

### Phase 0: Foundation Layer (no dependencies)

#### `models/`
- **No dependencies**
- Defines: MediaInfo, AudioFile, Segment, TranslatedSegment, CheckpointState, ResumePoint

#### `config/`
- **No dependencies**
- Defines: Model configs, language mappings, processing settings, prompts

---

### Phase 1: State & Utilities Layer

#### `state/`
- **Depends on**: `models/` (needs CheckpointState, ResumePoint)
- Provides checkpoint save/load/resume logic

---

### Phase 2: Processing Layer (parallel - no inter-dependencies)

#### `ingestion/`
- **Depends on**: `models/` (returns MediaInfo, AudioFile), `config/` (audio format settings)
- Converts video/audio → normalized audio

#### `transcription/`
- **Depends on**: `models/` (returns Segment list), `config/` (whisper prompts, model sizes), `state/` (checkpoint during streaming)
- Converts audio → timestamped text segments

#### `translation/`
- **Depends on**: `models/` (takes Segment, returns TranslatedSegment), `config/` (translation prompts, target languages), `state/` (checkpoint during batch processing)
- Converts segments → translated segments

#### `subtitle/`
- **Depends on**: `models/` (takes TranslatedSegment), `config/` (output format settings)
- Converts translated segments → SRT files

---

### Phase 3: Orchestration Layer

#### `main.py`
- **Depends on**: ALL modules
- Orchestrates the pipeline:
  1. `ingestion/` → audio file
  2. `transcription/` → segments
  3. `translation/` → translated segments
  4. `subtitle/` → SRT files
- Uses `state/` for resume logic between steps

---

### Dependency Diagram

```
Phase 0:  [models]  [config]
            ↓         ↓
Phase 1:    [state]
            ↓  ↓  ↓  ↓
Phase 2:  [ingestion] [transcription] [translation] [subtitle]
            ↓           ↓                ↓            ↓
Phase 3:              [main.py]
```

**Key Design**: Phase 2 modules are independent - can be developed/tested in parallel

---

## Implementation Roadmap

### Phase 0: Foundation
**Entry**: Clean repository, Python environment setup

**Tasks** (parallelizable):
- [ ] Define data classes in `models/` (MediaInfo, AudioFile, Segment, TranslatedSegment, CheckpointState, ResumePoint)
- [ ] Create `config/` structure (models.py, languages.py, settings.py, prompts.py)
- [ ] Setup project structure (folders, __init__.py files, requirements.txt)

**Exit Criteria**:
- All data classes importable and type-checkable
- Config files return valid values
- `pytest tests/test_models.py` passes

**Usable Output**: Foundation that other modules can import

---

### Phase 1: State Management
**Entry**: Phase 0 complete

**Tasks**:
- [ ] Implement checkpoint save/load in `state/checkpoint_manager.py`
- [ ] Implement resume detection in `state/resume_handler.py`
- [ ] Write unit tests for checkpoint integrity

**Exit Criteria**:
- Can save arbitrary stage data to JSON checkpoint
- Can load and validate checkpoint files
- Can detect resume point from file system
- `pytest tests/test_state.py` passes

**Usable Output**: Reusable checkpoint system for all pipeline stages

---

### Phase 2: Media Ingestion
**Entry**: Phase 0 complete

**Tasks**:
- [ ] Implement media detection in `ingestion/media_detector.py` (ffprobe wrapper)
- [ ] Implement audio extraction in `ingestion/audio_extractor.py` (ffmpeg wrapper)
- [ ] Test with sample video/audio files

**Exit Criteria**:
- Can detect video vs audio input
- Can extract audio from video → 16kHz mono WAV
- Handles corrupted/unsupported files gracefully
- `pytest tests/test_ingestion.py` passes

**Usable Output**: Standalone tool to prepare audio for transcription

---

### Phase 3: Transcription Pipeline
**Entry**: Phase 0, Phase 1 complete

**Tasks**:
- [ ] Implement mlx_whisper streaming in `transcription/whisper_transcriber.py`
- [ ] Implement segment refinement in `transcription/segment_refiner.py`
- [ ] Integrate checkpointing during streaming
- [ ] Test with 5-10 minute audio samples

**Exit Criteria**:
- Produces timestamped segments from audio
- Can resume after interruption
- Merges over-segmented sentences correctly
- `pytest tests/test_transcription.py` passes

**Usable Output**: Audio → timestamped transcription (usable even without translation)

---

### Phase 4: Translation Pipeline
**Entry**: Phase 0, Phase 1 complete

**Tasks** (can start parallel with Phase 3):
- [ ] Implement payload builder in `translation/payload_builder.py`
- [ ] Implement mlx_lm batch translation in `translation/mlx_translator.py`
- [ ] Integrate checkpointing for batch processing
- [ ] Test with sample segment lists

**Exit Criteria**:
- Translates segment batches via mlx_lm
- Can cancel and resume mid-translation
- Preserves timestamps during translation
- `pytest tests/test_translation.py` passes

**Usable Output**: Segments → translated segments (can test with mock segments)

---

### Phase 5: Subtitle Generation
**Entry**: Phase 0 complete

**Tasks** (can start parallel with Phase 3-4):
- [ ] Implement SRT formatter in `subtitle/srt_writer.py`
- [ ] Handle timestamp conversion and special characters
- [ ] Test output with VLC/video players

**Exit Criteria**:
- Generates valid SRT files from segments
- Timestamps display correctly in video players
- Handles edge cases (overlapping times, special chars)
- `pytest tests/test_subtitle.py` passes

**Usable Output**: Segments → playable subtitle files

---

### Phase 6: Pipeline Orchestration
**Entry**: All previous phases complete

**Tasks**:
- [ ] Implement `main.py` pipeline orchestration
- [ ] Add CLI argument parsing (input file, target languages, model options)
- [ ] Integrate all modules in correct order
- [ ] Add progress logging between stages
- [ ] End-to-end testing with real videos

**Exit Criteria**:
- Single command processes video → multi-language SRT files
- Resume works across all stages
- Error handling and user feedback clear
- `pytest tests/test_integration.py` passes

**Usable Output**: Complete working tool

---

### Phase 7: Polish & Optimization
**Entry**: Phase 6 complete, tool is functional

**Tasks**:
- [ ] Add batch processing (multiple videos)
- [ ] Optimize memory usage for long videos
- [ ] Add quality metrics (WER for transcription)
- [ ] Create README with examples
- [ ] Performance benchmarking

**Exit Criteria**:
- Can process 1hr video in < 15 minutes
- Memory usage stays under 8GB
- Documentation complete

**Usable Output**: Production-ready tool

---

**Key parallelization opportunities:**
- Phase 2, 3, 4, 5 can have parallel development (different developers)
- Phase 3 and 4 only need Phase 0 + 1, not each other

---

## Test Strategy

### Test Pyramid

```
        /\
       /E2E\      ← 10% (Full pipeline, real files)
      /------\
     /Integration\ ← 20% (Module interactions, mocked I/O)
    /------------\
   /  Unit Tests  \ ← 70% (Fast, isolated, deterministic)
  /----------------\
```

---

### Coverage Requirements

- **Line coverage**: 85% minimum
- **Branch coverage**: 80% minimum (especially error paths)
- **Function coverage**: 90% minimum
- **Statement coverage**: 85% minimum

**Critical modules requiring 95%+ coverage**:
- `state/` (checkpoint corruption = data loss)
- `subtitle/srt_writer.py` (format errors = unusable output)

---

### Test Scenarios by Module

#### `models/` (Unit - 95% coverage)
**Happy path**:
- Data classes instantiate with valid data
- Serialization/deserialization works (for checkpoints)

**Edge cases**:
- Optional fields are None
- Timestamps are zero or negative
- Empty text strings

**Error cases**:
- Invalid types raise TypeError
- Missing required fields raise ValueError

---

#### `config/` (Unit - 90% coverage)
**Happy path**:
- All config values return expected types
- Prompt lookups return strings

**Edge cases**:
- Unknown language codes fall back to defaults
- Missing config files use hardcoded defaults

**Error cases**:
- Invalid model paths logged as warnings

---

#### `state/` (Unit 90% + Integration 10%)
**Unit - Happy path**:
- Save checkpoint writes valid JSON
- Load checkpoint returns correct CheckpointState
- Resume detection finds latest checkpoint

**Unit - Edge cases**:
- Empty checkpoint directory returns None
- Partial checkpoint file (interrupted write) detected

**Unit - Error cases**:
- Corrupted JSON raises specific error
- Write permission denied handled gracefully

**Integration**:
- Save → Load roundtrip preserves all data
- Resume after mock transcription interruption

---

#### `ingestion/` (Unit 70% + Integration 30%)
**Unit - Happy path**:
- `detect_media_type()` identifies MP4, MOV, MP3, WAV
- `extract_audio()` calls ffmpeg with correct args

**Unit - Edge cases**:
- Detect handles files without extensions
- Extract handles mono audio (no conversion needed)

**Unit - Error cases**:
- Unsupported codec returns error
- Missing ffmpeg raises clear exception

**Integration**:
- Extract real 10s video → playable WAV
- Extracted audio has correct sample rate (16kHz)

---

#### `transcription/` (Unit 60% + Integration 40%)
**Unit - Happy path**:
- `refine_segments()` merges fragments correctly
- Segment streaming yields items progressively

**Unit - Edge cases**:
- Single-word segments preserved if punctuated
- Segments with 0.0s duration handled
- Empty audio returns empty segment list

**Unit - Error cases**:
- Invalid audio format caught early
- mlx_whisper failure logged with context

**Integration**:
- Mock mlx_whisper → refine_segments → valid output
- Checkpoint save mid-stream → resume works
- Real 30s audio → reasonable transcription

---

#### `translation/` (Unit 60% + Integration 40%)
**Unit - Happy path**:
- `build_translation_payload()` formats JSON correctly
- Batch processing groups segments properly

**Unit - Edge cases**:
- Single segment batch
- Empty text segments skipped
- Very long text (>500 chars) split

**Unit - Error cases**:
- Invalid target language returns error
- mlx_lm timeout handled with retry

**Integration**:
- Mock mlx_lm → batch translate → preserves timestamps
- Checkpoint every 10 batches → resume mid-translation
- Real 20 segments → French translation quality check

---

#### `subtitle/` (Unit 80% + Integration 20%)
**Unit - Happy path**:
- `write_srt()` formats timestamps correctly (HH:MM:SS,mmm)
- Segment indexing starts at 1
- Newlines between entries

**Unit - Edge cases**:
- Overlapping timestamps (end > next start) → warning
- Special characters escaped (&, <, >)
- Empty text segments skipped

**Unit - Error cases**:
- Write permission denied
- Invalid timestamp (negative) raises error

**Integration**:
- Generated SRT loads in VLC without errors
- Timestamps sync with video

---

#### `main.py` (Integration 50% + E2E 50%)
**Integration**:
- Mock all modules → pipeline orchestration logic works
- Checkpoint between each stage
- CLI argument parsing correct

**E2E - Happy path**:
- 30s video → English transcription → French SRT (full pipeline)
- Resume after killing process mid-translation

**E2E - Edge cases**:
- Audio-only input (skips extraction)
- Multiple target languages (2-3 SRT files)

**E2E - Error cases**:
- Corrupted video file → clear error message
- Disk full during processing → checkpoint preserves progress

---

### Test Generation Guidelines (for Surgical Test Generator)

#### Priority Order:
1. **Error paths first** - Most bugs hide here
2. **Edge cases** - Boundary conditions, empty inputs
3. **Happy path** - Should work if above pass

#### Test Isolation:
- **Unit tests**: Mock all external dependencies (ffmpeg, mlx_whisper, mlx_lm, file I/O)
- **Integration tests**: Mock only slow/external services (actual file I/O okay)
- **E2E tests**: Real files, real models (use small test files <1min)

#### Fixture Strategy:
- `tests/fixtures/` contains:
  - `sample_10s.mp4` (video)
  - `sample_30s.wav` (audio)
  - `sample_segments.json` (mock transcription)
  - `sample_checkpoint.json` (mock state)

#### Assertions:
- **Always assert** on both success AND error messages
- **Check types** explicitly (isinstance checks)
- **Verify side effects** (files created, checkpoints saved)

---

### Critical Test Scenarios (Must Pass Before Release)

1. **Resilience**: Kill process during each stage → resume completes successfully
2. **Correctness**: Known audio → transcription WER < 10%
3. **Performance**: 1hr video processes in < 15 minutes
4. **Multi-language**: Single video → 3 languages, all SRTs load in VLC
5. **Edge case**: Silent audio → empty segments, no crash

---

## Appendix

### Technology Stack
- **Language**: Python 3.10+
- **Transcription**: mlx_whisper (Apple Silicon optimized)
- **Translation**: mlx_lm (Apple Silicon optimized)
- **Media Processing**: ffmpeg/ffprobe
- **Testing**: pytest
- **Type Checking**: mypy (optional)

### External Dependencies
```
mlx_whisper>=0.1.0
mlx_lm>=0.1.0
ffmpeg-python>=0.2.0
pytest>=7.0.0
```

### Development Notes
- Target platform: macOS with Apple Silicon (M1/M2/M3)
- MLX framework leverages unified memory for efficient processing
- All processing happens locally - no cloud services required
- Checkpoint files use JSON for human readability and debugging