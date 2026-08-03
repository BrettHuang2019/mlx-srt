from mlx_srt.merge import align_words, find_best_split, merge_short, merge_srt, split_long
from mlx_srt.srt import Segment, format_srt_time, parse_srt, render_srt


def _words(texts):
    return [(text, i * 0.5, i * 0.5 + 0.4) for i, text in enumerate(texts)]


def test_timestamp_carry_rounding():
    assert format_srt_time(59.9996) == "00:01:00,000"
    assert format_srt_time(-1) == "00:00:00,000"


def test_srt_round_trip_multiline():
    content = render_srt([Segment(1, "00:00:00,000", "00:00:01,000", "Bonjour\n你好")])
    assert parse_srt(content)[0].text == "Bonjour\n你好"


def test_align_words_lookahead_and_unmatched_fallback():
    stamps = [
        {"text": "bruit", "start": 0, "end": 0.1},
        {"text": "bonjour", "start": 1, "end": 1.4},
        {"text": "monde", "start": 1.5, "end": 2},
    ]
    aligned = align_words(stamps, "Bonjour intrus monde.")
    assert aligned[0][1:] == (1.0, 1.4)
    assert aligned[1][1:] == (1.0, 1.0)
    assert aligned[2][1:] == (1.5, 2.0)


def test_split_scoring_avoids_number_and_dangling_word():
    words = _words(["Ceci", "est", "de", "30", "minutes", "mais", "très", "bien."])
    split = find_best_split(words)
    assert split not in {3, 4}


def test_split_long_and_merge_short_are_configurable():
    long = _words(["abcdefghij"] * 8)
    assert len(split_long([long], max_chars=30)) > 1
    short = [[("Oui.", 0, 0.2)], [("Bonjour", 0.3, 0.8), ("ici.", 0.8, 1.1)]]
    assert len(merge_short(short, max_chars=100, min_chars=30, min_duration=1)) == 1


def test_merge_srt_constructs_cues():
    stamps = [
        {"text": "Bonjour", "start": 0, "end": 0.5},
        {"text": "monde", "start": 0.6, "end": 1.1},
    ]
    output = merge_srt(stamps, "Bonjour monde.")
    assert "00:00:00,000 --> 00:00:01,100" in output
    assert "Bonjour monde." in output
