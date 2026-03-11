

import json
import re
import yaml
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from mlx_lm import load, generate
except ImportError:
    print("Warning: mlx_lm not available. Translation functions will not work.")
    load = None
    generate = None

# Import segment refiner for preprocessing
import sys
sys.path.append(str(Path(__file__).parent.parent))
from transcription.segment_refiner import refine_segments

# State Management Functions
def load_state(output_dir: str) -> Dict[str, Any]:
    """Load existing state from state.json file"""
    state_file = Path(output_dir) / "state.json"
    if state_file.exists():
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_state(state: Dict[str, Any], output_dir: str):
    """Save current state to state.json file"""
    state_file = Path(output_dir) / "state.json"
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def prepare_state_for_resume(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize persisted state so a failed pipeline can resume cleanly."""
    if not state:
        return state

    pipeline_info = state.setdefault("pipeline_info", {})
    pipeline_info["status"] = "running"
    pipeline_info["last_checkpoint"] = datetime.now().isoformat()
    pipeline_info.pop("end_time", None)
    pipeline_info.pop("error", None)

    return state

def create_initial_state(whisper_transcript: str, output_dir: str, url: str = None, downloaded_file: str = None, video_title: str = None) -> Dict[str, Any]:
    """Create initial state structure for a new translation pipeline"""
    import uuid
    state = {
        "pipeline_info": {
            "pipeline_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "last_checkpoint": datetime.now().isoformat()
        },
        "input_files": {
            "whisper_transcript": whisper_transcript,
            "output_directory": output_dir
        },
        "completed_steps": [],
        "current_step": None,
        "steps": {
            "refinement": {"status": "pending", "file": "01_refined_transcript.json"},
            "summary": {"status": "pending", "file": "04_summary.txt"},
            "preprocessing": {"status": "pending", "files": ["02_filtered_segments.json", "03_ordered_segments.json", "03b_regenerated_ids_segments.json"]},
            "translation": {
                "status": "pending",
                "total_batches": 0,
                "completed_batches": [],
                "failed_batches": [],
                "current_batch": None
            },
            "validation": {"status": "pending"},
            "merging": {"status": "pending"},
            "final_output": {"status": "pending"}
        }
    }

    # Add download info if URL is provided
    if url:
        state["download_info"] = {
            "url": url,
            "downloaded_file": downloaded_file,
            "video_title": video_title,
            "download_completed": bool(downloaded_file),
            "download_time": datetime.now().isoformat() if downloaded_file else None
        }
        state["steps"]["download"] = {
            "status": "completed" if downloaded_file else "pending",
            "url": url,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat() if downloaded_file else None
        }
        if downloaded_file and "download" not in state["completed_steps"]:
            state["completed_steps"].append("download")

    return state

def update_download_status(state: Dict[str, Any], url: str, status: str, downloaded_file: str = None, video_title: str = None, **kwargs):
    """Update download status in state"""
    if "download_info" not in state:
        state["download_info"] = {
            "url": url,
            "downloaded_file": None,
            "video_title": None,
            "download_completed": False,
            "download_time": None
        }

    if "download" not in state["steps"]:
        state["steps"]["download"] = {
            "status": "pending",
            "url": url
        }

    # Update download_info
    state["download_info"].update({
        "url": url,
        "downloaded_file": downloaded_file,
        "video_title": video_title,
        "download_completed": status == "completed"
    })

    if status == "completed":
        state["download_info"]["download_time"] = datetime.now().isoformat()

    # Update step status
    update_step_status(state, "download", status, **kwargs)

    # Add download-specific info to step
    if downloaded_file:
        state["steps"]["download"]["downloaded_file"] = downloaded_file
    if video_title:
        state["steps"]["download"]["video_title"] = video_title

def update_step_status(state: Dict[str, Any], step_name: str, status: str, **kwargs):
    """Update status of a pipeline step"""
    state["steps"][step_name]["status"] = status

    if status == "completed":
        if step_name not in state["completed_steps"]:
            state["completed_steps"].append(step_name)
        state["steps"][step_name]["end_time"] = datetime.now().isoformat()
    elif status == "running":
        state["steps"][step_name]["start_time"] = datetime.now().isoformat()
        state["current_step"] = step_name

    # Add any additional parameters
    for key, value in kwargs.items():
        state["steps"][step_name][key] = value

    state["pipeline_info"]["last_checkpoint"] = datetime.now().isoformat()

def update_batch_status(state: Dict[str, Any], batch_id: str, status: str, **kwargs):
    """Update status of a translation batch"""
    if "translation" not in state["steps"]:
        return

    translation_step = state["steps"]["translation"]

    if status == "completed":
        if batch_id not in translation_step["completed_batches"]:
            translation_step["completed_batches"].append(batch_id)
        # Remove from failed batches if it was previously failed
        if batch_id in translation_step["failed_batches"]:
            translation_step["failed_batches"].remove(batch_id)
    elif status == "failed":
        if batch_id in translation_step["completed_batches"]:
            translation_step["completed_batches"].remove(batch_id)
        if batch_id not in translation_step["failed_batches"]:
            translation_step["failed_batches"].append(batch_id)
    elif status == "running":
        translation_step["current_batch"] = batch_id

    # Add any additional parameters
    if "batch_details" not in translation_step:
        translation_step["batch_details"] = {}

    if batch_id not in translation_step["batch_details"]:
        translation_step["batch_details"][batch_id] = {}

    for key, value in kwargs.items():
        translation_step["batch_details"][batch_id][key] = value

    state["pipeline_info"]["last_checkpoint"] = datetime.now().isoformat()

def load_cached_batch_translation(output_dir: Optional[str], batch_id: str,
                                  batch_segments: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Load the newest valid cached translation result for a batch or its split descendants."""
    if not output_dir:
        return None

    output_path = Path(output_dir)
    candidate_files = sorted(
        output_path.glob(f"07_llm_response_{batch_id}*.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )

    exact_name_pattern = re.compile(rf"^07_llm_response_{re.escape(batch_id)}(?:_retry_\d+)?\.json$")

    for candidate_file in candidate_files:
        if not exact_name_pattern.match(candidate_file.name):
            continue

        try:
            with open(candidate_file, 'r', encoding='utf-8') as f:
                cached_response = json.load(f)

            return validate_and_parse_batch_response(
                cached_response["raw_response"],
                batch_segments,
                -1,
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    if len(batch_segments) <= 1:
        return None

    first_half_id = f"{batch_id}_a"
    second_half_id = f"{batch_id}_b"
    first_half = batch_segments[:len(batch_segments) // 2]
    second_half = batch_segments[len(batch_segments) // 2:]

    first_half_cached = load_cached_batch_translation(output_dir, first_half_id, first_half)
    second_half_cached = load_cached_batch_translation(output_dir, second_half_id, second_half)

    if first_half_cached is not None and second_half_cached is not None:
        return first_half_cached + second_half_cached

    return None

def has_split_batch_artifacts(output_dir: Optional[str], batch_id: str) -> bool:
    """Return True when recursive split artifacts exist for this batch."""
    if not output_dir:
        return False

    output_path = Path(output_dir)
    return any(output_path.glob(f"07_llm_response_{batch_id}_a*.json")) or any(
        output_path.glob(f"07_llm_response_{batch_id}_b*.json")
    )

def get_resume_point(state: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Determine where to resume the pipeline based on state and existing files"""
    if not state:
        return None

    # Check each step in order
    step_order = ["refinement", "summary", "preprocessing", "translation", "validation", "merging", "final_output"]

    for step in step_order:
        step_status = state["steps"][step]["status"]

        if step_status == "completed":
            # Verify files exist
            if "file" in state["steps"][step]:
                file_path = Path(output_dir) / state["steps"][step]["file"]
                if not file_path.exists():
                    print(f"⚠️  Missing file for completed step {step}: {file_path}")
                    return step  # Resume from this step
            elif "files" in state["steps"][step]:
                for file_name in state["steps"][step]["files"]:
                    file_path = Path(output_dir) / file_name
                    if not file_path.exists():
                        print(f"⚠️  Missing file for completed step {step}: {file_path}")
                        return step  # Resume from this step

        elif step_status in ["failed", "running"]:
            return step  # Resume from failed or running step

        elif step_status == "pending":
            return step  # Start this step

    # All steps completed
    return None

def validate_completed_files(state: Dict[str, Any], output_dir: str) -> bool:
    """Validate that all files for completed steps exist"""
    if not state:
        return True

    for step_name in state["completed_steps"]:
        step = state["steps"][step_name]

        if "file" in step:
            file_path = Path(output_dir) / step["file"]
            if not file_path.exists():
                print(f"❌ Missing file for step {step_name}: {file_path}")
                return False

        elif "files" in step:
            for file_name in step["files"]:
                file_path = Path(output_dir) / file_name
                if not file_path.exists():
                    print(f"❌ Missing file for step {step_name}: {file_path}")
                    return False

    return True

def convert_segments_to_translation_format(segments: List[Dict[str, Any]],
                                          output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert transcript segments to LLM input format.
    Extract only id (as 'index') and text (as 'fr' field).
    Filter out empty segments and ellipsis.
    """
    translation_format = []
    for segment in segments:
        text = segment.get('text', '').strip()
        if text and text != '...' and not text.startswith('...'):
            translation_format.append({
                'index': segment.get('id'),
                'fr': text
            })

    # Save to output file if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        segments_file = output_path / "01_converted_segments.json"
        with open(segments_file, 'w', encoding='utf-8') as f:
            json.dump(translation_format, f, ensure_ascii=False, indent=2)

        print(f"Converted segments saved to: {segments_file}")

    return translation_format

def filter_out_empty_and_ellipsis_segments(segments: List[Dict[str, Any]],
                                           output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Remove segments with no meaningful content.
    """
    filtered = []
    for segment in segments:
        text = segment.get('text', '').strip()
        if text and text != '...' and not text.startswith('...'):
            filtered.append(segment)

    # Save to output file if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filtered_file = output_path / "02_filtered_segments.json"
        with open(filtered_file, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

        print(f"Filtered segments saved to: {filtered_file}")

    return filtered

def preserve_segment_order(segments: List[Dict[str, Any]],
                          output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Ensure segment order is maintained during preprocessing.
    """
    ordered = sorted(segments, key=lambda x: x.get('id', 0))

    # Save to output file if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ordered_file = output_path / "03_ordered_segments.json"
        with open(ordered_file, 'w', encoding='utf-8') as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)

        print(f"Ordered segments saved to: {ordered_file}")

    return ordered

def regenerate_sequential_ids(segments: List[Dict[str, Any]],
                             output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Regenerate sequential IDs for segments after refinement to ensure
    continuous indexing without gaps caused by merged segments.
    """
    regenerated_segments = []
    for i, segment in enumerate(segments, start=1):
        new_segment = segment.copy()
        new_segment['id'] = i
        regenerated_segments.append(new_segment)

    # Save to output file if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        regenerated_file = output_path / "03b_regenerated_ids_segments.json"
        with open(regenerated_file, 'w', encoding='utf-8') as f:
            json.dump(regenerated_segments, f, ensure_ascii=False, indent=2)

        print(f"Regenerated ID segments saved to: {regenerated_file}")

    return regenerated_segments

def is_valid_translation(zh: str, original_fr: str = "") -> bool:
    """
    Validate translation: either contains Chinese characters, contains only numbers/punctuation,
    or is a name/expression that should remain in original language, or is a different translation.
    Returns False if empty or identical to original French (except for numbers-only content and names).
    """
    if not zh or not zh.strip():
        return False

    zh = zh.strip()

    # Check if contains Chinese characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    if chinese_pattern.search(zh):
        # For Chinese content, it should not be identical to French
        return zh != original_fr

    # Check if contains only numbers, decimals, commas, and spaces (statistical data)
    numbers_pattern = re.compile(r'^[\d\s,.\-:;]+$')
    if numbers_pattern.match(zh):
        # For numbers-only content, allow it to be identical to French (statistical data)
        return True

    # Check if it's a name, proper noun, or expression that should remain in original language
    # Allow single words or short phrases that are likely names/proper nouns
    words = zh.split()
    if len(words) <= 3 and zh.lower() == original_fr.lower():
        # Likely a name or expression that should remain unchanged
        return True

    # If it's different from the original, it's a valid translation (even without Chinese characters)
    if zh.lower() != original_fr.lower():
        return True

    return False

def verify_translation_contains_chinese_characters(translated_items: List[Dict[str, Any]]) -> bool:
    """
    Verify that translations contain valid content (Chinese characters, numbers-only, or names/expressions).
    """
    for item in translated_items:
        zh = item.get('zh', '').strip()
        fr = item.get('fr', '')
        if not is_valid_translation(zh, fr):
            return False
    return True

def merge_translations_back_to_segments(original_segments: List[Dict[str, Any]],
                                      translations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge translated text back into original transcript structure.
    """
    # Handle both 'id' and 'index' keys in translations
    translation_map = {}
    for t in translations:
        if 'id' in t:
            translation_map[t['id']] = t['zh']
        elif 'index' in t:
            translation_map[t['index']] = t['zh']

    merged_segments = []
    for segment in original_segments:
        merged_segment = segment.copy()
        segment_id = segment.get('id')
        if segment_id in translation_map:
            merged_segment['zh'] = translation_map[segment_id]
        merged_segments.append(merged_segment)

    return merged_segments

def regenerate_full_transcript_text(segments: List[Dict[str, Any]]) -> str:
    """
    Regenerate top-level 'text' field from segments.
    """
    return ' '.join(segment.get('text', '') for segment in segments)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_summary_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return summary config, supporting both nested and legacy flat keys."""
    translation_config = config.get('translation', {})
    summary_config = translation_config.get('summary', {}).copy()

    if 'model_path' not in summary_config:
        summary_config['model_path'] = translation_config.get(
            'summary_model_path',
            translation_config.get('model_path', 'mlx-community/Qwen3.5-4B-4bit')
        )
    if 'prompt' not in summary_config:
        summary_config['prompt'] = translation_config.get(
            'summary_prompt',
            '用中文简短总结以下内容(150字以内)： {text}'
        )
    if 'chunk_prompt' not in summary_config:
        summary_config['chunk_prompt'] = summary_config['prompt']
    if 'reduce_prompt' not in summary_config:
        summary_config['reduce_prompt'] = '用中文将以下分块摘要压缩整合为一个简短总结(150字以内)： {text}'
    if 'verbose' not in summary_config:
        summary_config['verbose'] = translation_config.get('verbose', False)
    if 'enable_thinking' not in summary_config:
        summary_config['enable_thinking'] = translation_config.get('summary_enable_thinking', False)
    if 'chunk_max_input_tokens' not in summary_config:
        summary_config['chunk_max_input_tokens'] = 60000
    if 'chunk_overlap_tokens' not in summary_config:
        summary_config['chunk_overlap_tokens'] = 5000

    return summary_config


def get_translation_step_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return translation-step config, supporting both nested and legacy flat keys."""
    translation_config = config.get('translation', {})
    step_config = translation_config.get('translate', {}).copy()

    legacy_defaults = {
        'model_path': 'mlx-community/Qwen2.5-7B-Instruct-4bit',
        'batch_size': 10,
        'max_tokens': 2048,
        'temperature': 0.1,
        'translation_prompt': '',
        'verbose': False,
        'max_retries': 3,
        'retry_delay': 1.0,
    }

    for key, default in legacy_defaults.items():
        if key not in step_config:
            step_config[key] = translation_config.get(key, default)

    return step_config

def _fallback_summary(text: str) -> str:
    return f"这是一个关于文稿的总结。原文总长度：{len(text)}字符。"


def _generate_with_temperature(model: Any, tokenizer: Any, *, prompt: str,
                               verbose: bool, max_tokens: int,
                               temperature: Optional[float] = None) -> str:
    """Use MLX generation with temp-style sampling configuration."""
    base_kwargs = {
        "prompt": prompt,
        "verbose": verbose,
        "max_tokens": max_tokens,
    }

    if temperature is None:
        return generate(model, tokenizer, **base_kwargs)

    try:
        return generate(
            model,
            tokenizer,
            temp=temperature,
            **base_kwargs,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'temp'" not in str(exc):
            raise

    return generate(
        model,
        tokenizer,
        temperature=temperature,
        **base_kwargs,
    )


def _encode_text(tokenizer: Any, text: str) -> List[Any]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        if isinstance(encoded, list):
            return encoded
        return list(encoded)
    return text.split()


def _decode_tokens(tokenizer: Any, tokens: List[Any]) -> str:
    if tokens and isinstance(tokens[0], int) and hasattr(tokenizer, "decode"):
        return tokenizer.decode(tokens)
    return " ".join(str(token) for token in tokens)


def _count_tokens(tokenizer: Any, text: str) -> int:
    return len(_encode_text(tokenizer, text))


def _chunk_text_by_tokens(text: str, tokenizer: Any, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    tokens = _encode_text(tokenizer, text)
    if not tokens:
        return [{"index": 1, "token_start": 0, "token_end": 0, "token_count": 0, "text": ""}]

    if chunk_size <= 0:
        chunk_size = len(tokens)
    overlap = max(0, min(overlap, max(chunk_size - 1, 0)))
    step = max(chunk_size - overlap, 1)

    chunks = []
    start = 0
    chunk_index = 1
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append({
            "index": chunk_index,
            "token_start": start,
            "token_end": end,
            "token_count": len(chunk_tokens),
            "text": _decode_tokens(tokenizer, chunk_tokens),
        })
        if end >= len(tokens):
            break
        start += step
        chunk_index += 1

    return chunks


def _generate_summary_text(
    model: Any,
    tokenizer: Any,
    prompt_template: str,
    text: str,
    verbose: bool,
    enable_thinking: bool,
) -> Tuple[str, str]:
    prompt = prompt_template.replace("{text}", text)
    messages = [{"role": "user", "content": prompt}]
    chat_template_kwargs = {"add_generation_prompt": True, "enable_thinking": enable_thinking}
    formatted_prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    summary = generate(model, tokenizer, prompt=formatted_prompt, verbose=verbose)
    return summary, prompt


def summarize(transcript_file: str, output_dir: Optional[str] = None, return_metadata: bool = False):
    """
    Generate summary of transcript text.
    Takes refined transcript JSON and generates 150-200 words summary.
    """
    config = load_config()
    summary_config = get_summary_config(config)

    # Load transcript
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    # Extract full text
    full_text = transcript.get('text', '')
    if not full_text:
        # Combine segment texts if no full text
        full_text = ' '.join(seg.get('text', '') for seg in transcript.get('segments', []))

    # Save original text for examination
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save original transcript
        original_file = output_path / "00_original_transcript.json"
        with open(original_file, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        # Save extracted text
        text_file = output_path / "00_extracted_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(full_text)

        print(f"Original transcript saved to: {original_file}")
        print(f"Extracted text saved to: {text_file}")

    summary = ""
    prompt = ""
    summary_tokenizer = None
    map_reduce_details = {
        "strategy": "mock",
        "full_text_token_count": None,
        "chunk_max_input_tokens": summary_config.get('chunk_max_input_tokens', 60000),
        "chunk_overlap_tokens": summary_config.get('chunk_overlap_tokens', 5000),
        "chunk_count": 0,
        "reduce_passes": 0,
        "chunks": [],
        "reduce_steps": [],
    }

    # Generate summary
    if load and generate:
        try:
            model_path = summary_config.get('model_path', 'mlx-community/Qwen3.5-4B-4bit')
            model, tokenizer = load(model_path)
            summary_tokenizer = tokenizer
            chunk_limit = summary_config.get('chunk_max_input_tokens', 60000)
            overlap = summary_config.get('chunk_overlap_tokens', 5000)
            verbose = summary_config.get('verbose', False)
            enable_thinking = summary_config.get('enable_thinking', False)
            summary_prompt = summary_config.get('prompt', '用中文简短总结以下内容(150字以内)： {text}')
            chunk_prompt = summary_config.get('chunk_prompt', summary_prompt)
            reduce_prompt = summary_config.get('reduce_prompt', '用中文将以下分块摘要压缩整合为一个简短总结(150字以内)： {text}')

            full_text_token_count = _count_tokens(tokenizer, full_text)
            map_reduce_details["full_text_token_count"] = full_text_token_count

            if full_text_token_count > chunk_limit:
                chunks = _chunk_text_by_tokens(full_text, tokenizer, chunk_limit, overlap)
                chunk_summaries = []
                for chunk in chunks:
                    chunk_summary, chunk_prompt_text = _generate_summary_text(
                        model,
                        tokenizer,
                        chunk_prompt,
                        chunk["text"],
                        verbose,
                        enable_thinking,
                    )
                    chunk_details = {
                        **chunk,
                        "prompt": chunk_prompt_text,
                        "summary": chunk_summary,
                        "summary_token_count": _count_tokens(tokenizer, chunk_summary),
                    }
                    map_reduce_details["chunks"].append(chunk_details)
                    chunk_summaries.append(chunk_summary)

                summary = "\n\n".join(chunk_summaries)
                prompt = "\n\n".join(chunk["prompt"] for chunk in map_reduce_details["chunks"])
                map_reduce_details["strategy"] = "map_reduce"
                map_reduce_details["chunk_count"] = len(map_reduce_details["chunks"])

                while _count_tokens(tokenizer, summary) > chunk_limit:
                    reduced_summary, reduce_prompt_text = _generate_summary_text(
                        model,
                        tokenizer,
                        reduce_prompt,
                        summary,
                        verbose,
                        enable_thinking,
                    )
                    map_reduce_details["reduce_passes"] += 1
                    prompt = reduce_prompt_text
                    map_reduce_details["reduce_steps"].append({
                        "pass": map_reduce_details["reduce_passes"],
                        "input_token_count": _count_tokens(tokenizer, summary),
                        "output_token_count": _count_tokens(tokenizer, reduced_summary),
                        "prompt": reduce_prompt_text,
                        "summary": reduced_summary,
                    })
                    summary = reduced_summary
            else:
                summary, prompt = _generate_summary_text(
                    model,
                    tokenizer,
                    summary_prompt,
                    full_text,
                    verbose,
                    enable_thinking,
                )
                map_reduce_details["strategy"] = "single_pass"
                map_reduce_details["chunk_count"] = 1

        except Exception as e:
            print(f"Warning: Could not generate summary with MLX-LM: {e}")
            summary = _fallback_summary(full_text)
            map_reduce_details["strategy"] = "fallback"
    else:
        summary = _fallback_summary(full_text)

    map_reduce_details["final_summary_token_count"] = None
    if summary_tokenizer is not None and map_reduce_details["strategy"] not in {"mock", "fallback"}:
        try:
            map_reduce_details["final_summary_token_count"] = _count_tokens(summary_tokenizer, summary)
        except Exception:
            map_reduce_details["final_summary_token_count"] = None

    # Save summary and comprehensive report if output directory specified
    if output_dir:
        summary_file = Path(output_dir) / "04_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        details_file = Path(output_dir) / "04_summary_map_reduce.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(map_reduce_details, f, ensure_ascii=False, indent=2)

        # Save comprehensive summary report
        report_file = Path(output_dir) / "04_summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Summary Generation Report ===\n\n")

            # Model information
            if load and generate:
                model_path = summary_config.get('model_path', 'mlx-community/Qwen3.5-4B-4bit')
                f.write(f"Model Used: {model_path}\n")
                f.write("Status: MLX-LM model loaded successfully\n")
            else:
                f.write("Model Used: None (mlx_lm not available)\n")
                f.write("Status: Using mock summary\n")

            f.write(f"\nInput Statistics:\n")
            f.write(f"- Original text length: {len(full_text)} characters\n")
            f.write(f"- Original text words: {len(full_text.split())} words\n")
            f.write(f"- Number of segments: {len(transcript.get('segments', []))}\n")
            f.write(f"- Summary strategy: {map_reduce_details['strategy']}\n")
            if map_reduce_details["full_text_token_count"] is not None:
                f.write(f"- Estimated input tokens: {map_reduce_details['full_text_token_count']}\n")
            f.write(f"- Chunk max input tokens: {map_reduce_details['chunk_max_input_tokens']}\n")
            f.write(f"- Chunk overlap tokens: {map_reduce_details['chunk_overlap_tokens']}\n")
            f.write(f"- Chunk count: {map_reduce_details['chunk_count']}\n")
            f.write(f"- Reduce passes: {map_reduce_details['reduce_passes']}\n")

            f.write(f"\nFinal Prompt Sent to Model:\n")
            f.write("-" * 50 + "\n")
            if load and generate:
                f.write(prompt)
            else:
                f.write("Mock mode - no prompt sent to model")
            f.write("\n" + "-" * 50 + "\n")

            f.write(f"\nGenerated Result:\n")
            f.write("-" * 50 + "\n")
            f.write(summary)
            f.write("\n" + "-" * 50 + "\n")

            f.write(f"\nResult Statistics:\n")
            f.write(f"- Summary length: {len(summary)} characters\n")
            f.write(f"- Summary words: {len(summary.split())} words\n")
            if map_reduce_details["final_summary_token_count"] is not None:
                f.write(f"- Summary tokens: {map_reduce_details['final_summary_token_count']}\n")

            # Validate if summary contains Chinese characters
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
            has_chinese = bool(chinese_pattern.search(summary))
            f.write(f"- Contains Chinese characters: {has_chinese}\n")

            if map_reduce_details["chunks"]:
                f.write(f"\nChunk Summaries:\n")
                for chunk in map_reduce_details["chunks"]:
                    f.write(f"- Chunk {chunk['index']}: tokens {chunk['token_start']}-{chunk['token_end']} ")
                    f.write(f"({chunk['token_count']} input, {chunk['summary_token_count']} summary)\n")

            if map_reduce_details["reduce_steps"]:
                f.write(f"\nReduce Passes:\n")
                for reduce_step in map_reduce_details["reduce_steps"]:
                    f.write(
                        f"- Pass {reduce_step['pass']}: "
                        f"{reduce_step['input_token_count']} -> {reduce_step['output_token_count']} tokens\n"
                    )

            f.write(f"\nTimestamp: {Path.cwd()}\n")

        print(f"Summary saved to: {summary_file}")
        print(f"Summary report saved to: {report_file}")
        print(f"Summary map-reduce details saved to: {details_file}")

    if return_metadata:
        return summary, map_reduce_details
    return summary

SMART_DOUBLE_QUOTES = {'\u201c', '\u201d', '\u201f', '\u00ab', '\u00bb', '\u2039', '\u203a', '\u300c', '\u300d', '\u300e', '\u300f'}
SMART_SINGLE_QUOTES = {'\u2018', '\u2019', '\u201b'}
JSON_STRING_END_CHARS = {',', '}', ']', ':'}


def _next_non_whitespace_char(text: str, start: int) -> str:
    """Return the next non-whitespace character after start, or an empty string."""
    while start < len(text):
        if not text[start].isspace():
            return text[start]
        start += 1
    return ''


def sanitize_model_output(output: str) -> str:
    """Normalize smart-quote JSON delimiters without corrupting valid string content."""
    sanitized = []
    in_string = False
    escape = False

    for i, char in enumerate(output):
        next_char = _next_non_whitespace_char(output, i + 1)

        if escape:
            sanitized.append(char)
            escape = False
            continue

        if char == '\\':
            sanitized.append(char)
            if in_string:
                escape = True
            continue

        if not in_string:
            if char == '"':
                in_string = True
                sanitized.append(char)
            elif char in SMART_DOUBLE_QUOTES or char in SMART_SINGLE_QUOTES:
                in_string = True
                sanitized.append('"')
            else:
                sanitized.append(char)
            continue

        if char == '"':
            in_string = False
            sanitized.append(char)
            continue

        if char in SMART_DOUBLE_QUOTES or char in SMART_SINGLE_QUOTES:
            if next_char in JSON_STRING_END_CHARS or not next_char:
                in_string = False
                sanitized.append('"')
            else:
                sanitized.append(char)
            continue

        sanitized.append(char)

    output = ''.join(sanitized)

    # Fix common JSON structure mistakes made by the model
    output = output.replace("'}]", "\"}]")
    output = output.replace("'}", "\"}")

    return output

def validate_and_parse_batch_response(response: str, batch_segments: List[Dict[str, Any]],
                                    batch_num: int) -> List[Dict[str, Any]]:
    """
    Validate and parse LLM response for a batch.
    Raises ValueError for validation failures (Error Types 3-6).
    """
    # Sanitize response to handle smart quotes and other Unicode issues
    response = sanitize_model_output(response)

    # Error Type 3: JSON Parsing Errors
    try:
        batch_translated = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                batch_translated = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse extracted JSON from batch {batch_num + 1}: {e}")
        else:
            raise ValueError(f"Could not extract JSON from response for batch {batch_num + 1}")

    # Error Type 4: Batch Structure Validation
    if not isinstance(batch_translated, list):
        raise ValueError(f"Expected list from batch {batch_num + 1}, got {type(batch_translated)}")

    if len(batch_translated) != len(batch_segments):
        raise ValueError(f"Batch {batch_num + 1} output count mismatch: expected {len(batch_segments)}, got {len(batch_translated)}")

    # Error Type 5: Field Validation
    for i, translated_item in enumerate(batch_translated):
        if not isinstance(translated_item, dict):
            raise ValueError(f"Batch {batch_num + 1} item {i} is not a dictionary")

        expected_index = batch_segments[i]['index']
        if 'index' not in translated_item:
            raise ValueError(f"Batch {batch_num + 1} item {i} missing 'index' field")

        if translated_item['index'] != expected_index:
            raise ValueError(f"Batch {batch_num + 1} item {i} index mismatch: expected {expected_index}, got {translated_item['index']}")

        if 'zh' not in translated_item:
            raise ValueError(f"Batch {batch_num + 1} item {i} missing 'zh' field")

    # Error Type 6: Translation Quality Validation
    for i, translated_item in enumerate(batch_translated):
        zh = translated_item.get('zh', '').strip()
        original_fr = batch_segments[i].get('fr', '')

        if not zh:
            raise ValueError(f"Batch {batch_num + 1} item {i} has empty translation")

        # Use the updated validation logic that allows numbers-only translations and names
        if not is_valid_translation(zh, original_fr):
            # Determine the specific error type for better error messages
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
            numbers_pattern = re.compile(r'^[\d\s,.\-:;]+$')
            words = zh.split()

            if chinese_pattern.search(zh):
                raise ValueError(f"Batch {batch_num + 1} item {i} Chinese translation is identical to original French: '{zh}'")
            elif numbers_pattern.match(zh):
                # This shouldn't happen with our updated logic, but just in case
                pass  # Numbers-only should always be valid
            elif len(words) <= 3 and zh.lower() == original_fr.lower():
                # This shouldn't happen with our updated logic, but just in case
                pass  # Names/expressions should always be valid
            else:
                raise ValueError(f"Batch {batch_num + 1} item {i} translation has invalid content: '{zh}'")

    return batch_translated

def process_batch_recursive(segments: List[Dict[str, Any]], summary: str, context_text: str,
                           model, tokenizer, translation_prompt: str, max_tokens: int,
                           temperature: float, verbose: bool, max_retries: int,
                           retry_delay: float, output_dir: Optional[str],
                           batch_id: str, depth: int = 0) -> List[Dict[str, Any]]:
    """
    Process a batch with recursive splitting strategy.
    Keeps splitting until success or single sentences remain.
    """
    indent = "  " * depth
    print(f"{indent}📦 Processing batch ({len(segments)} segments)...")

    cached_result = load_cached_batch_translation(output_dir, batch_id, segments)
    if cached_result is not None:
        print(f"{indent}♻️  Using cached result for {batch_id}")
        return cached_result

    # Base case: single sentence - if it fails, we cannot split further
    if len(segments) == 1:
        print(f"{indent}🔍 Single sentence batch - attempting with retries...")
        return process_single_batch_with_retries(
            segments, summary, context_text, model, tokenizer,
            translation_prompt, max_tokens, temperature, verbose,
            max_retries, retry_delay, output_dir, batch_id, depth, indent
        )

    if has_split_batch_artifacts(output_dir, batch_id):
        print(f"{indent}♻️  Resuming from split sub-batches for {batch_id}")

        mid_point = len(segments) // 2
        first_half = segments[:mid_point]
        second_half = segments[mid_point:]

        first_half_id = f"{batch_id}_a"
        second_half_id = f"{batch_id}_b"

        first_result = process_batch_recursive(
            first_half, summary, context_text, model, tokenizer,
            translation_prompt, max_tokens, temperature, verbose,
            max_retries, retry_delay, output_dir, first_half_id, depth + 1
        )
        second_result = process_batch_recursive(
            second_half, summary, context_text, model, tokenizer,
            translation_prompt, max_tokens, temperature, verbose,
            max_retries, retry_delay, output_dir, second_half_id, depth + 1
        )

        combined_result = first_result + second_result
        print(f"{indent}✅ Combined resumed result: {len(combined_result)} segments")
        return combined_result

    # Try processing current batch first
    try:
        result = process_single_batch_with_retries(
            segments, summary, context_text, model, tokenizer,
            translation_prompt, max_tokens, temperature, verbose,
            max_retries, retry_delay, output_dir, batch_id, depth, indent
        )
        print(f"{indent}✅ Batch completed successfully ({len(segments)} segments)")
        return result
    except RuntimeError as e:
        print(f"{indent}⚠️  Batch failed after retries: {e}")
        print(f"{indent}🔄 Splitting batch into smaller batches...")

        # Split batch into two smaller batches
        mid_point = len(segments) // 2
        first_half = segments[:mid_point]
        second_half = segments[mid_point:]

        print(f"{indent}📂 Split into: {len(first_half)} + {len(second_half)} segments")

        # Create sub-batch IDs
        first_half_id = f"{batch_id}_a"
        second_half_id = f"{batch_id}_b"

        # Recursively process each half
        print(f"{indent}Processing first half...")
        first_result = process_batch_recursive(
            first_half, summary, context_text, model, tokenizer,
            translation_prompt, max_tokens, temperature, verbose,
            max_retries, retry_delay, output_dir, first_half_id, depth + 1
        )

        print(f"{indent}Processing second half...")
        second_result = process_batch_recursive(
            second_half, summary, context_text, model, tokenizer,
            translation_prompt, max_tokens, temperature, verbose,
            max_retries, retry_delay, output_dir, second_half_id, depth + 1
        )

        # Combine results
        combined_result = first_result + second_result
        print(f"{indent}✅ Combined result: {len(combined_result)} segments")
        return combined_result


def process_single_batch_with_retries(segments: List[Dict[str, Any]], summary: str, context_text: str,
                                    model, tokenizer, translation_prompt: str, max_tokens: int,
                                    temperature: float, verbose: bool, max_retries: int,
                                    retry_delay: float, output_dir: Optional[str],
                                    batch_id: str, depth: int, indent: str) -> List[Dict[str, Any]]:
    """
    Process a single batch with retry logic (no splitting).
    """
    # Update context for this batch (use previous segments)
    batch_context_segments = []
    if len(segments) > 1:
        batch_context_segments = segments[:-1]
    batch_context_text = " ".join([seg['fr'] for seg in batch_context_segments]) if batch_context_segments else ""

    # Convert batch segments to JSON
    segments_json = json.dumps(segments, ensure_ascii=False, indent=2)

    # Create prompt for this batch
    prompt = translation_prompt.replace("{summary}", summary).replace("{context}", batch_context_text).replace("{segments}", segments_json)

    # Prepare messages for LLM
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # Retry loop for this batch
    last_error = None

    for retry_count in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Generate response
            generation_start_time = time.time()
            response = _generate_with_temperature(
                model,
                tokenizer,
                prompt=formatted_prompt,
                verbose=verbose,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            generation_time = time.time() - generation_start_time

            # Validate and parse response
            batch_translated = validate_and_parse_batch_response(response, segments, -1)  # Use -1 for split batches

            # Save response for this attempt
            if output_dir:
                retry_suffix = f"_retry_{retry_count}" if retry_count > 0 else ""
                output_path = Path(output_dir)
                response_file = output_path / f"07_llm_response_{batch_id}{retry_suffix}.json"
                response_data = {
                    "batch_id": batch_id,
                    "depth": depth,
                    "segments_count": len(segments),
                    "attempt": retry_count + 1,
                    "max_attempts": max_retries + 1,
                    "generation_time_seconds": generation_time,
                    "raw_response": response
                }
                with open(response_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=2)

            if depth == 0:
                print(f"    ✅ Batch completed successfully on attempt {retry_count + 1}")
            else:
                print(f"{indent}✅ Batch completed successfully on attempt {retry_count + 1}")

            return batch_translated

        except ValueError as e:
            last_error = e
            if retry_count < max_retries:
                if depth == 0:
                    print(f"    ⚠️  Batch failed (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                else:
                    print(f"{indent}⚠️  Batch failed (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                time.sleep(retry_delay)
            else:
                if len(segments) == 1:
                    # Single sentence failed completely - raise error and exit
                    error_msg = f"❌ Single sentence failed after {max_retries + 1} attempts: {e}"
                    print(f"{indent}{error_msg}")
                    raise RuntimeError(error_msg)
                else:
                    # Multi-sentence batch failed - this should be handled by recursive splitting
                    error_msg = f"❌ Batch failed after {max_retries + 1} attempts: {e}"
                    print(f"{indent}{error_msg}")
                    raise RuntimeError(error_msg)

def batch_translate(segments: List[Dict[str, Any]], summary: str,
                   output_dir: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Translate segments using LLM with contextual summary.
    Processes segments in batches based on config.batch_size.
    Requires MLX-LM to be available.
    """
    config = load_config()
    translation_config = get_translation_step_config(config)

    # Extract configuration parameters
    model_path = translation_config.get('model_path', 'mlx-community/Qwen2.5-7B-Instruct-4bit')
    batch_size = translation_config.get('batch_size', 10)
    max_tokens = translation_config.get('max_tokens', 2048)
    temperature = translation_config.get('temperature', 0.1)
    translation_prompt = translation_config.get('translation_prompt', '')
    verbose = translation_config.get('verbose', False)
    max_retries = translation_config.get('max_retries', 3)
    retry_delay = translation_config.get('retry_delay', 1.0)

    # Validate MLX-LM availability
    if not load or not generate:
        raise RuntimeError("MLX-LM is not available. Please install mlx_lm to use translation functionality.")

    # Save prompt and summary for examination
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary with context
        context_file = output_path / "05_translation_context.txt"
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write("=== Translation Context (Summary) ===\n")
            f.write(summary)
            f.write(f"\n\n=== Configuration ===\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Max Tokens: {max_tokens}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Total Segments: {len(segments)}\n")
            f.write(f"Number of Batches: {(len(segments) + batch_size - 1) // batch_size}\n")
            f.write(f"Verbose Mode: {verbose}\n")
            f.write("\n=== Segments to Translate ===\n")
            f.write(json.dumps(segments, ensure_ascii=False, indent=2))

        print(f"Translation context saved to: {context_file}")

    # Load model once for all batches
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    # Process in batches
    translated_segments = []
    total_batches = (len(segments) + batch_size - 1) // batch_size

    # Initialize state tracking for translation step
    if state:
        state["steps"]["translation"]["total_batches"] = total_batches
        # Clear previous failed batches for fresh start
        if state["steps"]["translation"].get("status") != "running":
            state["steps"]["translation"]["completed_batches"] = []
            state["steps"]["translation"]["failed_batches"] = []

    print(f"Processing {len(segments)} segments in {total_batches} batches of size {batch_size}")

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(segments))
        batch_segments = segments[start_idx:end_idx]
        batch_start_time = time.time()

        print(f"Processing batch {batch_num + 1}/{total_batches} (segments {start_idx + 1}-{end_idx})")

        # Create unique identifier for this batch
        batch_id = f"batch_{batch_num + 1:02d}_segments_{start_idx + 1:03d}_{end_idx:03d}"

        # Check if batch already completed (for resume functionality)
        if state and batch_id in state["steps"]["translation"].get("completed_batches", []):
            print(f"  ✅ Batch {batch_id} already completed, skipping...")
            batch_translated = load_cached_batch_translation(output_dir, batch_id, batch_segments)
            if batch_translated is not None:
                translated_segments.extend(batch_translated)
                continue

            print(f"  ⚠️  No valid cached response found for completed batch {batch_id}, reprocessing...")
            if state:
                update_batch_status(state, batch_id, "failed", error="Cached response invalid or missing")
                save_state(state, output_dir)

        # Update batch status to running
        if state:
            update_batch_status(state, batch_id, "running",
                              batch_num=batch_num + 1,
                              segments_count=len(batch_segments),
                              start_index=start_idx + 1,
                              end_index=end_idx)
            save_state(state, output_dir)

        # Extract context from previous segments (up to 3 sentences before this batch)
        context_segments = []
        if start_idx > 0:
            # Get up to 3 segments before the current batch
            context_start_idx = max(0, start_idx - 3)
            context_segments = segments[context_start_idx:start_idx]

        # Format context as a readable string of French sentences
        context_text = " ".join([seg['fr'] for seg in context_segments]) if context_segments else ""

        # Convert batch segments to JSON string for prompt
        segments_json = json.dumps(batch_segments, ensure_ascii=False, indent=2)

        # Replace placeholders in the prompt template
        prompt = translation_prompt.replace("{summary}", summary).replace("{context}", context_text).replace("{segments}", segments_json)

        # Create comprehensive batch report
        if output_dir:
            report_file = output_path / f"06_batch_report_{batch_id}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"=== BATCH TRANSLATION REPORT: {batch_id.upper()} ===\n\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Batch: {batch_num + 1}/{total_batches}\n")
                f.write(f"Segments: {start_idx + 1}-{end_idx} (total: {len(batch_segments)})\n")
                f.write(f"Batch Size Config: {batch_size}\n")
                f.write(f"Max Tokens Config: {max_tokens}\n")
                f.write(f"Temperature Config: {temperature}\n")
                f.write(f"Verbose Mode: {verbose}\n\n")

                f.write("=== INPUT SUMMARY ===\n")
                f.write(f"Context Summary: {summary}\n\n")

                f.write("=== PREVIOUS CONTEXT (French) ===\n")
                if context_segments:
                    for i, seg in enumerate(context_segments):
                        f.write(f"  C{i+1}. Index {seg['index']}: '{seg['fr']}'\n")
                    f.write(f"\nContext Text: {context_text}\n")
                else:
                    f.write("  (No previous context - this is the first batch)\n")
                f.write("\n")

                f.write("=== CURRENT BATCH SEGMENTS ===\n")
                for i, seg in enumerate(batch_segments):
                    f.write(f"  {i+1}. Index {seg['index']}: '{seg['fr']}'\n")
                f.write("\n")

                f.write("=== FORMATTED PROMPT ===\n")
                f.write("-" * 80 + "\n")
                f.write(prompt)
                f.write("\n" + "-" * 80 + "\n")

            print(f"Batch report saved to: {report_file}")

        # Prepare messages for LLM
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Generate translation
        generation_start_time = time.time()
        response = _generate_with_temperature(
            model,
            tokenizer,
            prompt=formatted_prompt,
            verbose=verbose,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generation_time = time.time() - generation_start_time
        batch_total_time = time.time() - batch_start_time

        # Save raw LLM response for each batch with unique name
        if output_dir:
            response_file = output_path / f"07_llm_response_{batch_id}.json"
            response_data = {
                "batch_id": batch_id,
                "generation_time_seconds": generation_time,
                "total_batch_time_seconds": batch_total_time,
                "segments_processed": len(batch_segments),
                "raw_response": response
            }
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            print(f"LLM response saved to: {response_file}")

            # Update batch report with results
            with open(report_file, 'a', encoding='utf-8') as f:
                f.write("\n=== LLM RESPONSE ===\n")
                f.write(f"Generation Time: {generation_time:.3f} seconds\n")
                f.write(f"Total Batch Time: {batch_total_time:.3f} seconds\n")
                f.write("=== RAW RESPONSE ===\n")
                f.write("-" * 80 + "\n")
                f.write(response)
                f.write("\n" + "-" * 80 + "\n")

        # Initial validation attempt
        try:
            # Validate and parse the response
            batch_translated = validate_and_parse_batch_response(response, batch_segments, batch_num)

            # Update batch report with successful parsing
            if output_dir:
                with open(report_file, 'a', encoding='utf-8') as f:
                    f.write("\n=== PARSED TRANSLATIONS ===\n")
                    f.write(f"Successfully parsed {len(batch_translated)} translated segments\n\n")
                    for i, item in enumerate(batch_translated):
                        f.write(f"  {i+1}. Index {item['index']}: '{item['zh']}'\n")
                    f.write("\n")

                    f.write("=== VALIDATION ===\n")
                    f.write(f"✓ Passed validation on first attempt (no splitting needed)\n")
                    f.write(f"✓ Parsed JSON successfully\n")
                    f.write(f"✓ Output count matches input ({len(batch_translated)} segments)\n")
                    f.write(f"✓ All items have 'index' field\n")
                    f.write(f"✓ All indexes match input\n")
                    f.write(f"✓ All items have 'zh' field\n")
                    f.write(f"✓ All translations contain valid content (Chinese characters, numbers-only, or names/expressions)\n")
                    f.write(f"✓ No translations identical to French\n")
                    f.write(f"✓ Batch {batch_num + 1} completed successfully\n")

            print(f"✅ Batch {batch_num + 1} completed successfully ({len(batch_translated)} segments): {len(batch_translated)} segments translated in {batch_total_time:.3f}s")

            # Update batch status to completed
            if state:
                update_batch_status(state, batch_id, "completed",
                                  generation_time_seconds=generation_time,
                                  total_time_seconds=batch_total_time,
                                  translated_segments=len(batch_translated))
                save_state(state, output_dir)

        except ValueError as e:
            print(f"⚠️  Batch {batch_num + 1} failed validation: {e}")

            # Update batch status to failed
            if state:
                update_batch_status(state, batch_id, "failed", error=str(e))
                save_state(state, output_dir)

            print(f"🔄 Starting recursive splitting for batch {batch_num + 1} ({len(batch_segments)} segments)...")

            # Log the split decision
            if output_dir:
                with open(report_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== BATCH SPLIT DECISION ===\n")
                    f.write(f"Original batch {batch_num + 1} failed with error: {e}\n")
                    f.write(f"Starting recursive splitting of {len(batch_segments)} segments\n")
                    f.write(f"Original response saved to: {response_file}\n")

            # Use recursive processing
            batch_start_recursive = time.time()
            batch_translated = process_batch_recursive(
                batch_segments, summary, context_text, model, tokenizer,
                translation_prompt, max_tokens, temperature, verbose,
                max_retries, retry_delay, output_dir, batch_id
            )
            batch_recursive_time = time.time() - batch_start_recursive

            # Update batch report with recursive success
            if output_dir:
                with open(report_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== RECURSIVE SPLITTING SUCCESS ===\n")
                    f.write(f"Total segments processed: {len(batch_translated)}\n")
                    f.write(f"Recursive processing time: {batch_recursive_time:.3f} seconds\n")
                    f.write(f"✓ All sub-batches completed successfully\n")

            print(f"✅ Batch {batch_num + 1} completed successfully after recursive splitting: {len(batch_translated)} segments translated in {batch_recursive_time:.3f}s (recursive)")

            # Update batch status to completed (after recursive splitting)
            if state:
                update_batch_status(state, batch_id, "completed",
                                  generation_time_seconds=batch_recursive_time,
                                  total_time_seconds=batch_recursive_time,
                                  translated_segments=len(batch_translated),
                                  recursive_splitting=True)
                save_state(state, output_dir)

        # If we got here, validation passed
        translated_segments.extend(batch_translated)

    print(f"All batches completed. Total segments translated: {len(translated_segments)}")

    # Save translations if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        translations_file = output_path / "08_translated_segments.json"

        with open(translations_file, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, ensure_ascii=False, indent=2)

        print(f"Translated segments saved to: {translations_file}")

        # Save validation results
        validation_result = verify_translation_contains_chinese_characters(translated_segments)
        validation_file = output_path / "09_translation_validation.txt"
        with open(validation_file, 'w', encoding='utf-8') as f:
            f.write(f"Translation validation: {'PASSED' if validation_result else 'FAILED'}\n")
            f.write(f"Number of segments: {len(translated_segments)}\n")

            for i, item in enumerate(translated_segments):
                zh = item.get('zh', '')
                fr = item.get('fr', '')
                chinese_chars = bool(re.search(r'[\u4e00-\u9fff]', zh))
                numbers_only = bool(re.match(r'^[\d\s,.\-:;]+$', zh))
                words = zh.split()
                is_name = len(words) <= 3 and zh.lower() == fr.lower() and not chinese_chars and not numbers_only
                is_valid = is_valid_translation(zh, fr)
                content_type = "Chinese" if chinese_chars else ("Numbers-only" if numbers_only else ("Name/Expression" if is_name else "Other"))
                f.write(f"Segment {item.get('index', i)}: zh='{zh}', valid={is_valid}, type={content_type}\n")

        print(f"Translation validation saved to: {validation_file}")

    return translated_segments

def translate_transcript(transcript_file: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Main orchestrator for transcript translation.

    Args:
        transcript_file: Path to refined transcript JSON file
        output_dir: Directory to save output files (optional)

    Returns:
        Translated transcript with zh fields added to segments
    """
    if output_dir is None:
        output_dir = Path(transcript_file).parent / "output"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load original transcript
    with open(transcript_file, 'r', encoding='utf-8') as f:
        original_transcript = json.load(f)

    segments = original_transcript.get('segments', [])

    # Step 1: Generate summary
    print("Step 1: Generating summary...")
    summary, _ = summarize(transcript_file, output_dir, return_metadata=True)

    # Step 2: Preprocess segments for translation
    print("Step 2: Preprocessing segments...")
    filtered_segments = filter_out_empty_and_ellipsis_segments(segments, output_dir)
    ordered_segments = preserve_segment_order(filtered_segments, output_dir)
    regenerated_segments = regenerate_sequential_ids(ordered_segments, output_dir)
    translation_segments = convert_segments_to_translation_format(regenerated_segments, output_dir)

    # Step 3: Batch translate segments
    print("Step 3: Translating segments...")
    translated_segments = batch_translate(translation_segments, summary, output_dir)

    # Step 4: Merge translations back to original segments
    print("Step 4: Merging translations...")
    final_segments = merge_translations_back_to_segments(regenerated_segments, translated_segments)

    # Save merged segments before final transcript
    merged_segments_file = output_path / "10_merged_segments.json"
    with open(merged_segments_file, 'w', encoding='utf-8') as f:
        json.dump(final_segments, f, ensure_ascii=False, indent=2)
    print(f"Merged segments saved to: {merged_segments_file}")

    # Step 5: Create final transcript
    print("Step 5: Creating final transcript...")
    final_transcript = original_transcript.copy()
    final_transcript['segments'] = final_segments
    final_transcript['text'] = regenerate_full_transcript_text(final_segments)

    # Save final transcript
    transcript_path = Path(transcript_file)
    final_file = output_path / f"11_{transcript_path.stem}_translated.json"

    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_transcript, f, ensure_ascii=False, indent=2)

    print(f"Final translated transcript saved to: {final_file}")

    # Save final statistics
    stats_file = output_path / "12_translation_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== Translation Statistics ===\n")
        f.write(f"Original segments: {len(segments)}\n")
        f.write(f"Filtered segments: {len(filtered_segments)}\n")
        f.write(f"Translation segments: {len(translation_segments)}\n")
        f.write(f"Translated segments: {len(translated_segments)}\n")
        f.write(f"Final segments with zh: {len([s for s in final_segments if 'zh' in s])}\n")
        f.write(f"Segments without zh: {len([s for s in final_segments if 'zh' not in s])}\n")

        # Translation coverage percentage
        with_zh = len([s for s in final_segments if 'zh' in s])
        coverage = (with_zh / len(final_segments)) * 100 if final_segments else 0
        f.write(f"Translation coverage: {coverage:.1f}%\n")

    print(f"Translation statistics saved to: {stats_file}")

    # Create process summary
    process_summary_file = output_path / "README.txt"
    with open(process_summary_file, 'w', encoding='utf-8') as f:
        f.write("=== Translation Process Output Files ===\n\n")
        f.write("This directory contains all intermediate files from the translation process:\n\n")
        f.write("00_original_transcript.json - Original input transcript\n")
        f.write("00_extracted_text.txt - Extracted full text from transcript\n")
        f.write("01_converted_segments.json - Segments converted to LLM format\n")
        f.write("02_filtered_segments.json - Segments after filtering empty content\n")
        f.write("03_ordered_segments.json - Segments ordered by ID\n")
        f.write("04_summary.txt - Generated summary of the transcript\n")
        f.write("04_summary_map_reduce.json - Chunking and reduce-pass metadata for summary generation\n")
        f.write("05_translation_context.txt - Summary and segments sent for translation\n")
        f.write("06_translation_prompt.txt - Full prompt sent to LLM\n")
        f.write("07_llm_raw_response.json - Raw response from LLM (JSON format)\n")
        f.write("08_translated_segments.json - Parsed translated segments\n")
        f.write("09_translation_validation.txt - Validation of Chinese characters\n")
        f.write("10_merged_segments.json - Translations merged back to original segments\n")
        f.write("11_*_translated.json - Final translated transcript\n")
        f.write("12_translation_statistics.txt - Process statistics and coverage\n")

    print(f"Process summary saved to: {process_summary_file}")
    print(f"\nTranslation complete! Check the output directory: {output_path}")

    return final_transcript


def translation_pipeline(whisper_transcript: Dict[str, Any],
                        output_dir: Optional[str] = None,
                        resume: bool = False,
                        url: str = None,
                        downloaded_file: str = None,
                        video_title: str = None) -> Dict[str, Any]:
    """
    Complete pipeline from Whisper transcription to translated transcript.

    This function orchestrates all steps:
    1. Segment refinement using segment_refiner
    2. Summary generation
    3. Translation using LLM
    4. Merging translations back

    Args:
        whisper_transcript: Raw Whisper transcription output with 'text' and 'segments' fields
        output_dir: Directory to save intermediate and final files
        resume: If True, resume from the last checkpoint instead of starting fresh

    Returns:
        Translated transcript with zh fields added to segments
    """
    if output_dir is None:
        output_dir = Path.cwd() / "translation_output"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize or load state
    if resume:
        state = load_state(output_dir)
        if state is None:
            print("No existing state found, starting fresh pipeline")
            state = create_initial_state("whisper_transcript", output_dir, url, downloaded_file, video_title)
        else:
            print(f"Resuming pipeline from state: {state['pipeline_info']['pipeline_id']}")
            print(f"Status: {state['pipeline_info']['status']}")
            print(f"Completed steps: {len(state['completed_steps'])}/{len(state['steps'])}")

            # Validate completed files
            if not validate_completed_files(state, output_dir):
                print("❌ Some files for completed steps are missing. Cannot resume safely.")
                return None

            # Check if already completed
            resume_point = get_resume_point(state, output_dir)
            if resume_point is None:
                print("✅ Pipeline already completed!")
                # Load final transcript
                final_file = output_path / "11_final_translated_transcript.json"
                if final_file.exists():
                    with open(final_file, 'r', encoding='utf-8') as f:
                        return json.load(f)

            state = prepare_state_for_resume(state)
    else:
        state = create_initial_state("whisper_transcript", output_dir, url, downloaded_file, video_title)

    print("=== TRANSLATION PIPELINE ===")
    print(f"Pipeline ID: {state['pipeline_info']['pipeline_id']}")
    print(f"Mode: {'Resuming' if resume else 'Fresh start'}")
    print(f"Input: Whisper transcription with {len(whisper_transcript.get('segments', []))} segments")

    # Save original whisper output
    original_file = output_path / "00_whisper_original.json"
    with open(original_file, 'w', encoding='utf-8') as f:
        json.dump(whisper_transcript, f, ensure_ascii=False, indent=2)
    print(f"Original Whisper output saved to: {original_file}")

    # Save initial state
    save_state(state, output_dir)

    # Step 1: Refine segments using segment_refiner
    resume_point = get_resume_point(state, output_dir) if resume else None
    if resume and resume_point != "refinement":
        print("\nStep 1: Skipping refinement (already completed)")
        # Load refined transcript
        refined_file = output_path / "01_refined_transcript.json"
        with open(refined_file, 'r', encoding='utf-8') as f:
            refined_transcript = json.load(f)
    else:
        print("\nStep 1: Refining Whisper segments...")
        update_step_status(state, "refinement", "running")
        save_state(state, output_dir)

        refined_result = refine_segments(whisper_transcript, output_dir)

        # Create refined transcript structure
        refined_transcript = {
            "text": refined_result["text"],
            "segments": refined_result["segments"]
        }

        # Save refined transcript
        refined_file = output_path / "01_refined_transcript.json"
        with open(refined_file, 'w', encoding='utf-8') as f:
            json.dump(refined_transcript, f, ensure_ascii=False, indent=2)
        print(f"Refined transcript saved to: {refined_file}")

        # Update state
        stats = refined_result["statistics"]
        update_step_status(state, "refinement", "completed",
                          total_input_segments=stats['total_input_segments'],
                          total_output_segments=stats['total_output_segments'],
                          segments_merged=stats['segments_merged'],
                          segments_split=stats['segments_split'],
                          punctuation_fixed=stats['punctuation_fixed'])
        save_state(state, output_dir)

    # Initialize stats with default value
    stats = {}

    # Get refinement statistics
    if resume and resume_point != "refinement":
        # Get stats from state when resuming
        if "refinement" in state["completed_steps"]:
            stats = state["steps"]["refinement"]
        print("  - Refinement already completed")
    else:
        # Get stats from fresh refinement result
        if not resume:
            stats = refined_result["statistics"]
        else:
            stats = state["steps"]["refinement"]

    # Always print refinement stats if available
    if stats:
        print(f"Refinement stats:")
        print(f"  - Input segments: {stats.get('total_input_segments', 'N/A')}")
        print(f"  - Output segments: {stats.get('total_output_segments', 'N/A')}")
        print(f"  - Segments merged: {stats.get('segments_merged', 'N/A')}")
        print(f"  - Segments split: {stats.get('segments_split', 'N/A')}")
        print(f"  - Punctuation fixes: {stats.get('punctuation_fixed', 'N/A')}")

    # Step 2: Generate summary
    if resume and resume_point not in ["refinement", "summary"]:
        print("\nStep 2: Skipping summary generation (already completed)")
        # Load summary
        summary_file = output_path / "04_summary.txt"
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = f.read()
    else:
        print("\nStep 2: Generating summary...")
        update_step_status(state, "summary", "running")
        save_state(state, output_dir)

        summary, summary_details = summarize(str(refined_file), output_dir, return_metadata=True)
        update_step_status(
            state,
            "summary",
            "completed",
            strategy=summary_details.get("strategy"),
            chunk_count=summary_details.get("chunk_count"),
            reduce_passes=summary_details.get("reduce_passes"),
            input_tokens=summary_details.get("full_text_token_count"),
            output_tokens=summary_details.get("final_summary_token_count"),
        )
        save_state(state, output_dir)

    # Step 3: Preprocess refined segments for translation
    if resume and resume_point not in ["refinement", "summary", "preprocessing"]:
        print("\nStep 3: Skipping preprocessing (already completed)")
        # Load preprocessed segments
        translation_segments_file = output_path / "01_converted_segments.json"
        with open(translation_segments_file, 'r', encoding='utf-8') as f:
            translation_segments = json.load(f)
        # Load regenerated segments for merging step
        regenerated_file = output_path / "03b_regenerated_ids_segments.json"
        with open(regenerated_file, 'r', encoding='utf-8') as f:
            regenerated_segments = json.load(f)
    else:
        print("\nStep 3: Preprocessing refined segments for translation...")
        update_step_status(state, "preprocessing", "running")
        save_state(state, output_dir)

        segments = refined_transcript.get('segments', [])

        # Filter out ellipsis and empty segments from refined segments
        filtered_segments = filter_out_empty_and_ellipsis_segments(segments, output_dir)
        ordered_segments = preserve_segment_order(filtered_segments, output_dir)
        regenerated_segments = regenerate_sequential_ids(ordered_segments, output_dir)
        translation_segments = convert_segments_to_translation_format(regenerated_segments, output_dir)

        update_step_status(state, "preprocessing", "completed",
                          input_segments=len(segments),
                          output_segments=len(translation_segments))
        save_state(state, output_dir)

    # Step 4: Translate segments
    if resume and resume_point not in ["refinement", "summary", "preprocessing", "translation"]:
        print("\nStep 4: Skipping translation (already completed)")
        # Load translated segments
        translated_file = output_path / "08_translated_segments.json"
        with open(translated_file, 'r', encoding='utf-8') as f:
            translated_segments = json.load(f)
    else:
        print("\nStep 4: Translating segments...")
        update_step_status(state, "translation", "running")
        save_state(state, output_dir)

        # Pass state to batch_translate for batch-level tracking
        translated_segments = batch_translate(translation_segments, summary, output_dir, state)

        update_step_status(state, "translation", "completed",
                          total_segments=len(translation_segments),
                          translated_segments=len([s for s in translated_segments if 'zh' in s and s['zh']]))
        save_state(state, output_dir)

    # Step 5: Merge translations back to refined segments
    if resume and resume_point not in ["refinement", "summary", "preprocessing", "translation", "merging"]:
        print("\nStep 5: Skipping merging (already completed)")
        # Load final transcript
        final_file = output_path / "11_final_translated_transcript.json"
        with open(final_file, 'r', encoding='utf-8') as f:
            final_transcript = json.load(f)
        final_segments = final_transcript['segments']
    else:
        print("\nStep 5: Merging translations back to refined segments...")
        update_step_status(state, "merging", "running")
        save_state(state, output_dir)

        final_segments = merge_translations_back_to_segments(regenerated_segments, translated_segments)

        # Create final transcript
        final_transcript = refined_transcript.copy()
        final_transcript['segments'] = final_segments
        final_transcript['text'] = regenerate_full_transcript_text(final_segments)

        # Save final transcript
        final_file = output_path / "11_final_translated_transcript.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_transcript, f, ensure_ascii=False, indent=2)
        print(f"Final translated transcript saved to: {final_file}")

        update_step_status(state, "merging", "completed",
                          merged_segments=len(final_segments))
        save_state(state, output_dir)

    # Final step: Mark pipeline as completed
    update_step_status(state, "final_output", "running")
    save_state(state, output_dir)

    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Final transcript with {len(final_transcript.get('segments', []))} segments")

    # Count segments with translations
    segments_with_zh = len([s for s in final_segments if 'zh' in s and s['zh']])
    print(f"Segments with translations: {segments_with_zh}/{len(final_segments)}")

    # Create pipeline summary
    pipeline_summary_file = output_path / "PIPELINE_SUMMARY.txt"
    with open(pipeline_summary_file, 'w', encoding='utf-8') as f:
        f.write("=== TRANSLATION PIPELINE SUMMARY ===\n\n")
        f.write(f"Input: Whisper transcription\n")
        f.write(f"Output: Translated transcript with Chinese translations\n\n")

        f.write("Pipeline Steps:\n")
        f.write("1. Segment Refinement (segment_refiner.py)\n")
        f.write("   - Remove empty segments\n")
        f.write("   - Fix punctuation spacing\n")
        f.write("   - Merge fragmented sentences\n")
        f.write("   - Split long blocks\n")
        f.write("2. Summary Generation\n")
        summary_state = state["steps"].get("summary", {})
        f.write(f"   - Strategy: {summary_state.get('strategy', 'unknown')}\n")
        if summary_state.get("chunk_count") is not None:
            f.write(f"   - Chunks: {summary_state.get('chunk_count')}\n")
        if summary_state.get("reduce_passes") is not None:
            f.write(f"   - Reduce passes: {summary_state.get('reduce_passes')}\n")
        f.write("3. LLM Translation\n")
        f.write("4. Merging Translations\n\n")

        f.write("Refinement Statistics:\n")
        if stats:
            for key, value in stats.items():
                f.write(f"  - {key.replace('_', ' ').title()}: {value}\n")
        else:
            f.write("  - No refinement statistics available\n")

        f.write(f"\nFinal Output:\n")
        f.write(f"  - Total refined segments: {len(final_segments)}\n")
        f.write(f"  - Segments with translations: {segments_with_zh}\n")
        f.write(f"  - Translation coverage: {segments_with_zh/len(final_segments)*100:.1f}%\n")
        f.write(f"  - Final transcript file: 11_final_translated_transcript.json\n")

    print(f"Pipeline summary saved to: {pipeline_summary_file}")

    # Mark pipeline as completed
    state["pipeline_info"]["status"] = "completed"
    state["pipeline_info"]["end_time"] = datetime.now().isoformat()
    update_step_status(state, "final_output", "completed",
                      total_segments=len(final_segments),
                      segments_with_translations=segments_with_zh,
                      translation_coverage=segments_with_zh/len(final_segments)*100)
    save_state(state, output_dir)

    print(f"✅ Pipeline {state['pipeline_info']['pipeline_id']} completed successfully!")
    print(f"State saved to: {output_path / 'state.json'}")

    return final_transcript
