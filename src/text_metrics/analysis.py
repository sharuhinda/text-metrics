from __future__ import annotations

import csv
import math
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class Lemmatizer(Protocol):
    def lemmatize(self, text: str) -> list[tuple[str, str, bool, bool]]:
        """Return (surface, lemma, is_stopword, is_number) entries."""
        ...


class PerplexityScorer(Protocol):
    def perplexity(self, text: str) -> float:
        """Return model-specific perplexity for text."""
        ...


@dataclass(frozen=True)
class LexicalEntropy:
    token_count: int
    vocabulary_size: int
    entropy_bits: float
    effective_vocabulary_size: float


@dataclass(frozen=True)
class FileMetrics:
    file: str
    perplexity: float
    lexical_token_count: int
    lexical_vocabulary_size: int
    lexical_entropy_bits: float
    effective_vocabulary_size: float


def _is_punctuation(surface: str) -> bool:
    return bool(surface) and all(unicodedata.category(character).startswith("P") for character in surface)


def lexical_entropy(text: str, lemmatizer: Lemmatizer) -> LexicalEntropy:
    """Calculate Shannon entropy in bits over retained lemma frequencies."""
    lemmas = [
        lemma.casefold()
        for surface, lemma, is_stopword, is_number in lemmatizer.lemmatize(text)
        if lemma and not is_stopword and not is_number and not _is_punctuation(surface)
    ]
    counts = Counter(lemmas)
    token_count = len(lemmas)
    if token_count == 0:
        return LexicalEntropy(0, 0, 0.0, 0.0)

    entropy_bits = sum(
        -(count / token_count) * math.log2(count / token_count)
        for count in counts.values()
    )
    return LexicalEntropy(
        token_count=token_count,
        vocabulary_size=len(counts),
        entropy_bits=entropy_bits,
        effective_vocabulary_size=2**entropy_bits,
    )


def analyze_directory(
    input_dir: Path,
    lemmatizer: Lemmatizer,
    scorer: PerplexityScorer,
    output_csv: Path,
) -> list[FileMetrics]:
    """Score all .txt files under input_dir and write a deterministic CSV."""
    input_dir = input_dir.resolve()
    output_csv = output_csv.resolve()
    files = sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.casefold() == ".txt" and path.resolve() != output_csv),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )

    results: list[FileMetrics] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        entropy = lexical_entropy(text, lemmatizer)
        results.append(
            FileMetrics(
                file=path.relative_to(input_dir).as_posix(),
                perplexity=scorer.perplexity(text),
                lexical_token_count=entropy.token_count,
                lexical_vocabulary_size=entropy.vocabulary_size,
                lexical_entropy_bits=entropy.entropy_bits,
                effective_vocabulary_size=entropy.effective_vocabulary_size,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FileMetrics.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    return results
