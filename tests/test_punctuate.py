from mlx_srt.punctuate import (
    compute_word_spans,
    find_second_to_last_sentence_end,
    normalize_label,
    punctuate_text,
)


def test_punctuate_maps_predictions_to_word_spans():
    def classifier(text):
        return [
            {"entity_group": ",", "start": 0, "end": 7},
            {"entity_group": ".", "start": 8, "end": 13},
        ]

    assert punctuate_text("bonjour monde", classifier, chunk_words=180) == "Bonjour, monde."


def test_punctuate_empty_does_not_load_model():
    assert punctuate_text("... !!!", classifier=lambda _: (_ for _ in ()).throw(AssertionError())) == ""


def test_helpers():
    assert compute_word_spans(["un", "mot"]) == [(0, 2), (3, 6)]
    assert normalize_label("LABEL_0") == "0"
    assert find_second_to_last_sentence_end("Une. Deux? Trois.") == 10
