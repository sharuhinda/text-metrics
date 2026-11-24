# text-metrics

Batch scoring for English `.txt` files:

- **Perplexity (PPL)** from Hugging Face [`distilgpt2`](https://huggingface.co/distilgpt2), a compact 82M-parameter causal English model.
- **Unconditional lexical entropy** in bits over **Stanza lemmas**, after removing punctuation, numeric tokens, and a fixed built-in English stop-word list.

`distilgpt2` is deliberately CPU-friendly enough: its checkpoint is roughly 330 MB, while full GPT-2-class models would impose needless memory and latency costs. Scoring is explicitly CPU-only and uses one Torch thread by default so it does not monopolize the machine. It is useful for *within-protocol comparisons*, not an absolute quality score.

## Install

Use a project-local virtual environment. Install the CPU-only PyTorch wheel first; ordinary PyPI resolution may choose much larger CUDA packages.

```bash
cd /home/eviza/workspace/projects/text-metrics
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python -m pip install -e '.[dev]'
```

## Score a directory

Place UTF-8 `.txt` files anywhere under an input directory. Files are found recursively.

```bash
.venv/bin/text-metrics samples \
  --output results/metrics.csv \
  --download-stanza
```

`--download-stanza` is needed only on the first run to fetch the English Stanza model. The first PPL run also downloads `distilgpt2` from Hugging Face into the local cache.

Useful options:

```bash
.venv/bin/text-metrics samples --output results/metrics.csv \
  --model distilgpt2 --stride 512 --cpu-threads 1
```

## CSV columns

- `file`: path relative to the input directory.
- `perplexity`: PPL under the selected model/tokenizer. Lower means more predictable **to that model**.
- `lexical_token_count`: retained lemma occurrences.
- `lexical_vocabulary_size`: distinct retained lemmas.
- `lexical_entropy_bits`: Shannon entropy `−Σ p(lemma) log₂ p(lemma)`.
- `effective_vocabulary_size`: `2^entropy_bits`, an intuitive diversity equivalent.

## Interpretation and limitations

PPL and entropy answer different questions. PPL is conditional next-token predictability under a named receiver-model; lexical entropy is the unconditional diversity of the retained lemma frequency distribution. Compare documents only using the same model, version, preprocessing, and ideally comparable genre/length. PPL is not a truthfulness, originality, or “how interesting a person is” score.

The stop-word list is fixed in `src/text_metrics/english.py` for reproducibility. Proper nouns are retained. Empty files and samples tokenized to fewer than two model tokens cannot receive PPL and cause a clear error rather than a fabricated score.

## Tests

```bash
.venv/bin/python -m pytest -q
```
