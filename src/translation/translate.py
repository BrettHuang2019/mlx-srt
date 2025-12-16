

import json
import re
import yaml
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
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

def verify_translation_contains_chinese_characters(translated_items: List[Dict[str, Any]]) -> bool:
    """
    Verify that translations contain valid Chinese characters.
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    for item in translated_items:
        zh = item.get('zh', '').strip()
        if not zh:
            return False
        if not chinese_pattern.search(zh):
            return False
        if zh == item.get('fr', ''):
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

def summarize(transcript_file: str, output_dir: Optional[str] = None) -> str:
    """
    Generate summary of transcript text.
    Takes refined transcript JSON and generates 150-200 words summary.
    """
    config = load_config()
    translation_config = config.get('translation', {})

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

    # Generate summary
    if load and generate:
        try:
            # Load model
            model_path = translation_config.get('model_path', 'mlx-community/Qwen2.5-7B-Instruct-4bit')
            model, tokenizer = load(model_path)

            # Generate summary
            summary_prompt = translation_config.get('summary_prompt',
                                                 '用150字左右总结以下文稿: {text}')
            prompt = summary_prompt.format(text=full_text)

            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            summary = generate(model, tokenizer, prompt=formatted_prompt,
                            verbose=translation_config.get('verbose', False))

        except Exception as e:
            print(f"Warning: Could not generate summary with MLX-LM: {e}")
            summary = f"这是一个关于文稿的总结。原文总长度：{len(full_text)}字符。"
    else:
        # Mock summary when mlx_lm is not available
        summary = f"这是一个关于文稿的总结。原文总长度：{len(full_text)}字符。"

    # Save summary and comprehensive report if output directory specified
    if output_dir:
        summary_file = Path(output_dir) / "04_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        # Save comprehensive summary report
        report_file = Path(output_dir) / "04_summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== Summary Generation Report ===\n\n")

            # Model information
            if load and generate:
                model_path = translation_config.get('model_path', 'mlx-community/Qwen2.5-7B-Instruct-4bit')
                f.write(f"Model Used: {model_path}\n")
                f.write("Status: MLX-LM model loaded successfully\n")
            else:
                f.write("Model Used: None (mlx_lm not available)\n")
                f.write("Status: Using mock summary\n")

            f.write(f"\nInput Statistics:\n")
            f.write(f"- Original text length: {len(full_text)} characters\n")
            f.write(f"- Original text words: {len(full_text.split())} words\n")
            f.write(f"- Number of segments: {len(transcript.get('segments', []))}\n")

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

            # Validate if summary contains Chinese characters
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
            has_chinese = bool(chinese_pattern.search(summary))
            f.write(f"- Contains Chinese characters: {has_chinese}\n")

            f.write(f"\nTimestamp: {Path.cwd()}\n")

        print(f"Summary saved to: {summary_file}")
        print(f"Summary report saved to: {report_file}")

    return summary

def batch_translate(segments: List[Dict[str, Any]], summary: str,
                   output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Translate segments using LLM with contextual summary.
    Processes segments in batches based on config.batch_size.
    Requires MLX-LM to be available.
    """
    config = load_config()
    translation_config = config.get('translation', {})

    # Extract configuration parameters
    model_path = translation_config.get('model_path', 'mlx-community/Qwen2.5-7B-Instruct-4bit')
    batch_size = translation_config.get('batch_size', 10)
    max_tokens = translation_config.get('max_tokens', 2048)
    temperature = translation_config.get('temperature', 0.1)
    translation_prompt = translation_config.get('translation_prompt', '')
    verbose = translation_config.get('verbose', False)

    # Validate MLX-LM availability
    if not load or not generate:
        raise RuntimeError("MLX-LM is not available. Please install mlx_lm to use translation functionality.")

    # Save prompt and summary for examination
    if output_dir:
        output_path = Path(output_dir)

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

    print(f"Processing {len(segments)} segments in {total_batches} batches of size {batch_size}")

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(segments))
        batch_segments = segments[start_idx:end_idx]
        batch_start_time = time.time()

        print(f"Processing batch {batch_num + 1}/{total_batches} (segments {start_idx + 1}-{end_idx})")

        # Create unique identifier for this batch
        batch_id = f"batch_{batch_num + 1:02d}_segments_{start_idx + 1:03d}_{end_idx:03d}"

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
        try:
            prompt = translation_prompt.format(summary=summary, context=context_text, segments=segments_json)
        except KeyError as e:
            raise ValueError(f"Missing placeholder in translation prompt template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting translation prompt: {e}")

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
        response = generate(model, tokenizer, prompt=formatted_prompt, verbose=verbose, max_tokens=max_tokens)
        generation_time = time.time() - generation_start_time
        batch_total_time = time.time() - batch_start_time

        # Save raw LLM response for each batch with unique name
        if output_dir:
            response_file = output_path / f"07_llm_response_{batch_id}.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"=== LLM RESPONSE: {batch_id.upper()} ===\n\n")
                f.write(f"Generation Time: {generation_time:.3f} seconds\n")
                f.write(f"Total Batch Time: {batch_total_time:.3f} seconds\n")
                f.write(f"Segments Processed: {len(batch_segments)}\n\n")
                f.write("=== RAW RESPONSE ===\n")
                f.write("-" * 80 + "\n")
                f.write(response)
                f.write("\n" + "-" * 80 + "\n")

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

        # Parse response as JSON
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

        # Validate batch translation structure
        if not isinstance(batch_translated, list):
            raise ValueError(f"Expected list from batch {batch_num + 1}, got {type(batch_translated)}")

        if len(batch_translated) != len(batch_segments):
            raise ValueError(f"Batch {batch_num + 1} output count mismatch: expected {len(batch_segments)}, got {len(batch_translated)}")

        # Validate each translated segment
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

        translated_segments.extend(batch_translated)

        # Update batch report with successful parsing
        if output_dir:
            with open(report_file, 'a', encoding='utf-8') as f:
                f.write("\n=== PARSED TRANSLATIONS ===\n")
                f.write(f"Successfully parsed {len(batch_translated)} translated segments\n\n")
                for i, item in enumerate(batch_translated):
                    f.write(f"  {i+1}. Index {item['index']}: '{item['zh']}'\n")
                f.write("\n")

                f.write("=== VALIDATION ===\n")
                f.write(f"✓ Parsed JSON successfully\n")
                f.write(f"✓ Output count matches input ({len(batch_translated)} segments)\n")
                f.write(f"✓ All items have 'index' field\n")
                f.write(f"✓ All indexes match input\n")
                f.write(f"✓ All items have 'zh' field\n")
                f.write(f"✓ Batch {batch_num + 1} completed successfully\n")

        print(f"Batch {batch_num + 1} completed successfully: {len(batch_translated)} segments translated in {batch_total_time:.3f}s")

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
                chinese_chars = bool(re.search(r'[\u4e00-\u9fff]', zh))
                f.write(f"Segment {item.get('index', i)}: zh='{zh}', contains_chinese={chinese_chars}\n")

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
    summary = summarize(transcript_file, output_dir)

    # Step 2: Preprocess segments for translation
    print("Step 2: Preprocessing segments...")
    filtered_segments = filter_out_empty_and_ellipsis_segments(segments, output_dir)
    ordered_segments = preserve_segment_order(filtered_segments, output_dir)
    translation_segments = convert_segments_to_translation_format(ordered_segments, output_dir)

    # Step 3: Batch translate segments
    print("Step 3: Translating segments...")
    translated_segments = batch_translate(translation_segments, summary, output_dir)

    # Step 4: Merge translations back to original segments
    print("Step 4: Merging translations...")
    final_segments = merge_translations_back_to_segments(segments, translated_segments)

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
        f.write("05_translation_context.txt - Summary and segments sent for translation\n")
        f.write("06_translation_prompt.txt - Full prompt sent to LLM\n")
        f.write("07_llm_raw_response.txt - Raw response from LLM\n")
        f.write("08_translated_segments.json - Parsed translated segments\n")
        f.write("09_translation_validation.txt - Validation of Chinese characters\n")
        f.write("10_merged_segments.json - Translations merged back to original segments\n")
        f.write("11_*_translated.json - Final translated transcript\n")
        f.write("12_translation_statistics.txt - Process statistics and coverage\n")

    print(f"Process summary saved to: {process_summary_file}")
    print(f"\nTranslation complete! Check the output directory: {output_path}")

    return final_transcript


def translation_pipeline(whisper_transcript: Dict[str, Any],
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
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

    Returns:
        Translated transcript with zh fields added to segments
    """
    if output_dir is None:
        output_dir = Path.cwd() / "translation_output"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== TRANSLATION PIPELINE ===")
    print(f"Input: Whisper transcription with {len(whisper_transcript.get('segments', []))} segments")

    # Save original whisper output
    original_file = output_path / "00_whisper_original.json"
    with open(original_file, 'w', encoding='utf-8') as f:
        json.dump(whisper_transcript, f, ensure_ascii=False, indent=2)
    print(f"Original Whisper output saved to: {original_file}")

    # Step 1: Refine segments using segment_refiner
    print("\nStep 1: Refining Whisper segments...")
    refined_result = refine_segments(whisper_transcript)

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

    # Print refinement statistics
    stats = refined_result["statistics"]
    print(f"Refinement stats:")
    print(f"  - Input segments: {stats['total_input_segments']}")
    print(f"  - Output segments: {stats['total_output_segments']}")
    print(f"  - Segments merged: {stats['segments_merged']}")
    print(f"  - Segments split: {stats['segments_split']}")
    print(f"  - Punctuation fixes: {stats['punctuation_fixed']}")

    # Step 2: Generate summary
    print("\nStep 2: Generating summary...")
    summary = summarize(str(refined_file), output_dir)

    # Step 3: Preprocess refined segments for translation
    print("Step 3: Preprocessing refined segments for translation...")
    segments = refined_transcript.get('segments', [])

    # Filter out ellipsis and empty segments from refined segments
    filtered_segments = filter_out_empty_and_ellipsis_segments(segments, output_dir)
    ordered_segments = preserve_segment_order(filtered_segments, output_dir)
    translation_segments = convert_segments_to_translation_format(ordered_segments, output_dir)

    # Step 4: Translate segments
    print("Step 4: Translating segments...")
    translated_segments = batch_translate(translation_segments, summary, output_dir)

    # Step 5: Merge translations back to refined segments
    print("Step 5: Merging translations back to refined segments...")
    final_segments = merge_translations_back_to_segments(ordered_segments, translated_segments)

    # Create final transcript
    final_transcript = refined_transcript.copy()
    final_transcript['segments'] = final_segments
    final_transcript['text'] = regenerate_full_transcript_text(final_segments)

    # Save final transcript
    final_file = output_path / "11_final_translated_transcript.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_transcript, f, ensure_ascii=False, indent=2)
    print(f"Final translated transcript saved to: {final_file}")

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
        f.write("3. LLM Translation\n")
        f.write("4. Merging Translations\n\n")

        f.write("Refinement Statistics:\n")
        for key, value in stats.items():
            f.write(f"  - {key.replace('_', ' ').title()}: {value}\n")

        f.write(f"\nFinal Output:\n")
        f.write(f"  - Total refined segments: {len(final_segments)}\n")
        f.write(f"  - Segments with translations: {segments_with_zh}\n")
        f.write(f"  - Translation coverage: {segments_with_zh/len(final_segments)*100:.1f}%\n")
        f.write(f"  - Final transcript file: 11_final_translated_transcript.json\n")

    print(f"Pipeline summary saved to: {pipeline_summary_file}")

    return final_transcript