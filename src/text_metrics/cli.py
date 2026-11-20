from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_directory
from .english import load_stanza_lemmatizer
from .ppl import TransformersPerplexityScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score .txt files with distilgpt2 PPL and Stanza lemma lexical entropy."
    )
    parser.add_argument("input_dir", type=Path, help="Directory searched recursively for .txt files.")
    parser.add_argument("--output", type=Path, default=Path("metrics.csv"), help="CSV output path (default: metrics.csv).")
    parser.add_argument("--model", default="distilgpt2", help="Hugging Face causal model ID (default: distilgpt2).")
    parser.add_argument("--stride", type=int, default=512, help="PPL sliding-window stride in model tokens.")
    parser.add_argument("--cpu-threads", type=int, default=1, help="Torch CPU threads while scoring (default: 1).")
    parser.add_argument("--download-stanza", action="store_true", help="Download the Stanza English model before scoring.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {args.input_dir}")
    if args.stride < 1 or args.cpu_threads < 1:
        raise SystemExit("--stride and --cpu-threads must both be positive.")

    lemmatizer = load_stanza_lemmatizer(download=args.download_stanza)
    scorer = TransformersPerplexityScorer(
        model_id=args.model,
        stride=args.stride,
        cpu_threads=args.cpu_threads,
    )
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    results = analyze_directory(args.input_dir, lemmatizer, scorer, output)
    print(f"Wrote {len(results)} file metrics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
