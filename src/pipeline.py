"""End-to-end pipeline to scrape, filter, normalize, and label subreddit resume data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from src.classify import JobClassifier
from src.clean import filter_and_extract_posts, normalize_posts
from src.comments import CommentAdviceClassifier, clean_and_classify_comments
from src.scraper import scrape_comments, scrape_posts


def run_pipeline(
    from_dt: str,
    to_dt: str,
    subreddit: str = "resumes",
    output_dir: str | Path = "data/final",
    raw_dir: str | Path = "data/raw",
    intermediate_dir: str | Path = "data/intermediate",
    taxonomy_path: str | Path = "data/reference/amended_soc_structure.csv",
    model: str = "gemma4:e4b-mlx",
    post_workers: int = 4,
    shard_days: int = 30,
    chunk_size: int = 500,
    skip_scrape: bool = False,
    skip_comments: bool = False,
    raw_posts_path: Optional[str | Path] = None,
    raw_comments_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run the full scraping, filtering, normalization, and comment labeling pipeline."""
    load_dotenv()

    output_dir = Path(output_dir)
    raw_dir = Path(raw_dir)
    intermediate_dir = Path(intermediate_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    posts_raw = (
        Path(raw_posts_path)
        if raw_posts_path
        else raw_dir / f"subreddit_{subreddit}_posts.parquet"
    )
    comments_raw = (
        Path(raw_comments_path)
        if raw_comments_path
        else raw_dir / f"subreddit_{subreddit}_comments.parquet"
    )

    if not skip_scrape:
        print(
            f"[1/4] Scraping r/{subreddit} posts from {from_dt} to {to_dt}..."
        )
        post_ids = scrape_posts(
            subreddit=subreddit,
            from_dt=from_dt,
            to_dt=to_dt,
            output_path=posts_raw,
            post_workers=post_workers,
            shard_days=shard_days,
        )
        print(f"      Scraped {len(post_ids)} raw posts -> {posts_raw}")
    else:
        print(f"[1/4] Skipping post scrape; reading {posts_raw}")

    raw_posts_df = pd.read_parquet(posts_raw)

    print("[2/4] Filtering posts and extracting structured metadata...")
    filtered_posts = filter_and_extract_posts(raw_posts_df)
    print(
        f"      Kept {len(filtered_posts)} / {len(raw_posts_df)} posts after filtering"
    )

    print("[3/4] Normalizing roles (SOC) and countries...")
    role_classifier = JobClassifier(
        taxonomy_path=str(taxonomy_path),
        model=model,
    )
    cleaned_posts = normalize_posts(
        filtered_posts,
        role_classifier=role_classifier,
        role_cache_path=intermediate_dir / "normalized_roles.csv",
        country_cache_path=intermediate_dir / "normalized_countries.csv",
    )

    final_posts_path = output_dir / "posts.parquet"
    cleaned_posts.to_parquet(final_posts_path, index=False)
    print(f"      Saved {len(cleaned_posts)} cleaned posts -> {final_posts_path}")

    if skip_comments:
        print("[4/4] Skipping comments.")
        return cleaned_posts, None

    if not skip_scrape:
        post_ids = cleaned_posts["id"].dropna().astype(str).tolist()
        print(
            f"[4/4] Scraping comments for {len(post_ids)} cleaned posts..."
        )
        num_comments = scrape_comments(
            post_ids=post_ids,
            output_path=comments_raw,
            chunk_size=chunk_size,
        )
        print(f"      Scraped {num_comments} raw comments -> {comments_raw}")
    else:
        print(f"[4/4] Skipping comment scrape; reading {comments_raw}")

    raw_comments_df = pd.read_parquet(comments_raw)
    comment_classifier = CommentAdviceClassifier(model=model)
    cleaned_comments = clean_and_classify_comments(
        raw_comments_df, classifier=comment_classifier
    )

    final_comments_path = output_dir / "comments.parquet"
    cleaned_comments.to_parquet(final_comments_path, index=False)
    print(
        f"      Saved {len(cleaned_comments)} classified comments -> {final_comments_path}"
    )

    return cleaned_posts, cleaned_comments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape, filter, normalize, and label Reddit resume data."
    )
    parser.add_argument(
        "--from-dt",
        required=True,
        help="Start date (inclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--to-dt",
        required=True,
        help="End date (exclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--subreddit",
        default="resumes",
        help="Target subreddit name without r/ (default: resumes).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/final",
        help="Directory for final posts.parquet and comments.parquet.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for raw scraped parquet files.",
    )
    parser.add_argument(
        "--intermediate-dir",
        default="data/intermediate",
        help="Directory for role/country normalization lookup caches.",
    )
    parser.add_argument(
        "--taxonomy-path",
        default="data/reference/amended_soc_structure.csv",
        help="Path to amended SOC taxonomy CSV.",
    )
    parser.add_argument(
        "--model",
        default="gemma4:e4b-mlx",
        help="Ollama model name for role and comment classification.",
    )
    parser.add_argument(
        "--post-workers",
        type=int,
        default=4,
        help="Parallel workers for deepsift post scraping (default: 4).",
    )
    parser.add_argument(
        "--shard-days",
        type=int,
        default=30,
        help="Date shard size in days for deepsift (default: 30).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Post ID chunk size for comment scraping (default: 500).",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip Reddit scraping and process existing raw parquet files.",
    )
    parser.add_argument(
        "--skip-comments",
        action="store_true",
        help="Skip comment scraping and classification.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_pipeline(
        from_dt=args.from_dt,
        to_dt=args.to_dt,
        subreddit=args.subreddit,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        intermediate_dir=args.intermediate_dir,
        taxonomy_path=args.taxonomy_path,
        model=args.model,
        post_workers=args.post_workers,
        shard_days=args.shard_days,
        chunk_size=args.chunk_size,
        skip_scrape=args.skip_scrape,
        skip_comments=args.skip_comments,
    )


if __name__ == "__main__":
    main()
