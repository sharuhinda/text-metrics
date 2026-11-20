import math
from pathlib import Path

import pytest

from text_metrics.analysis import analyze_directory, lexical_entropy
from text_metrics.english import EnglishStanzaLemmatizer
from text_metrics.ppl import TransformersPerplexityScorer


class MappingLemmatizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def lemmatize(self, text):
        return self.mapping[text]


class FixedScorer:
    def __init__(self, value=7.25):
        self.value = value

    def perplexity(self, text):
        return self.value




def test_perplexity_rejects_blank_text_without_loading_a_model():
    scorer = TransformersPerplexityScorer()

    with pytest.raises(ValueError, match="non-whitespace"):
        scorer.perplexity(" \n\t ")


def test_english_stanza_lemmatizer_marks_stopwords_and_numbers():
    class Word:
        def __init__(self, text, lemma, upos):
            self.text = text
            self.lemma = lemma
            self.upos = upos

    class Sentence:
        words = [
            Word("Dogs", "dog", "NOUN"),
            Word("and", "and", "CCONJ"),
            Word("42", "42", "NUM"),
            Word("!", "!", "PUNCT"),
        ]

    class Pipeline:
        def __call__(self, text):
            assert text == "Dogs and 42!"
            return type("Document", (), {"sentences": [Sentence()]})()

    assert EnglishStanzaLemmatizer(Pipeline()).lemmatize("Dogs and 42!") == [
        ("Dogs", "dog", False, False),
        ("and", "and", True, False),
        ("42", "42", False, True),
        ("!", "!", False, False),
    ]

    lemmatizer = MappingLemmatizer(
        {
            "ignored": [
                ("Dog", "dog", False, False),
                ("dogs", "dog", False, False),
                ("and", "and", True, False),
                ("42", "42", False, True),
                ("!", "!", False, False),
                ("cat", "cat", False, False),
            ]
        }
    )

    result = lexical_entropy("ignored", lemmatizer)

    assert result.token_count == 3
    assert result.vocabulary_size == 2
    assert result.entropy_bits == pytest.approx(-(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3))
    assert result.effective_vocabulary_size == pytest.approx(2 ** result.entropy_bits)


def test_analyze_directory_scores_each_text_file_recursively_and_writes_csv(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested" / "b.TXT").write_text("beta", encoding="utf-8")
    (tmp_path / "skip.md").write_text("not included", encoding="utf-8")

    lemmatizer = MappingLemmatizer(
        {
            "alpha": [("alpha", "alpha", False, False)],
            "beta": [("beta", "beta", False, False)],
        }
    )
    output = tmp_path / "results.csv"

    results = analyze_directory(tmp_path, lemmatizer, FixedScorer(), output)

    assert [item.file for item in results] == ["a.txt", "nested/b.TXT"]
    assert [item.perplexity for item in results] == [7.25, 7.25]
    assert output.read_text(encoding="utf-8").splitlines() == [
        "file,perplexity,lexical_token_count,lexical_vocabulary_size,lexical_entropy_bits,effective_vocabulary_size",
        "a.txt,7.25,1,1,0.0,1.0",
        "nested/b.TXT,7.25,1,1,0.0,1.0",
    ]


def test_analyze_directory_rejects_output_inside_input_as_text_file(tmp_path: Path):
    lemmatizer = MappingLemmatizer({"alpha": [("alpha", "alpha", False, False)]})
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")

    results = analyze_directory(tmp_path, lemmatizer, FixedScorer(), tmp_path / "results.txt")

    assert [item.file for item in results] == ["a.txt"]
