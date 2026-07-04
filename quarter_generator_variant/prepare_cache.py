"""
Build tokenized train/dev chunk caches for CKTN-ELECTRA training.

Usage:
    python quarter_generator_variant/prepare_cache.py
    python quarter_generator_variant/prepare_cache.py --rebuild_cache
"""

import argparse

from training import (
    DEV_FILES,
    SEQ_LEN,
    TRAIN_FILES,
    load_or_build_chunks,
    load_tokenizer_and_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tokenized CKTN-ELECTRA train/dev chunk caches"
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild caches even when matching cache files already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = load_tokenizer_and_check()

    load_or_build_chunks(
        "train", TRAIN_FILES, tokenizer, SEQ_LEN, args.rebuild_cache
    )
    load_or_build_chunks(
        "dev", DEV_FILES, tokenizer, SEQ_LEN, args.rebuild_cache
    )

    print("[Cache] Ready.")


if __name__ == "__main__":
    main()
