from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A stable built-in list avoids an extra corpus download and keeps runs reproducible.
ENGLISH_STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves""".split()
)


class EnglishStanzaLemmatizer:
    """Adapter from a Stanza English pipeline to the project's lexical filter."""

    def __init__(self, pipeline: Callable[[str], Any]):
        self._pipeline = pipeline

    def lemmatize(self, text: str) -> list[tuple[str, str, bool, bool]]:
        document = self._pipeline(text)
        result = []
        for sentence in document.sentences:
            for word in sentence.words:
                lemma = word.lemma or word.text
                result.append((word.text, lemma, lemma.casefold() in ENGLISH_STOPWORDS, word.upos == "NUM"))
        return result


def load_stanza_lemmatizer(download: bool = False) -> EnglishStanzaLemmatizer:
    """Create the English tokenize/POS/lemma pipeline, optionally downloading its model."""
    import stanza

    if download:
        stanza.download("en", processors="tokenize,pos,lemma", verbose=False)
    pipeline = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma",
        use_gpu=False,
        verbose=False,
    )
    return EnglishStanzaLemmatizer(pipeline)
