"""Reddit scraping wrappers using deepsift."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from deepsift.filters import Filters
from deepsift.models import CommentField, PostField
from deepsift.scraper import RedditScraper


def scrape_posts(
    subreddit: str,
    from_dt: str,
    to_dt: str,
    output_path: str | Path,
    post_workers: int = 4,
    shard_days: int = 30,
) -> list[str]:
    """Scrape subreddit posts with >=1 comment in [from_dt, to_dt] to Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filters = Filters(
        post_include_fields=[
            PostField.ID,
            PostField.TITLE,
            PostField.SELFTEXT,
            PostField.SCORE,
            PostField.NUM_COMMENTS,
            PostField.CREATED_UTC,
            PostField.SUBREDDIT,
            PostField.URL,
        ],
        post_filter=lambda post: post.num_comments >= 1,
    )

    scraper = RedditScraper(
        filters=filters,
        post_workers=post_workers,
        shard_days=shard_days,
    )

    return scraper.scrape_posts(
        subreddits=[subreddit],
        from_dt=from_dt,
        to_dt=to_dt,
        output_path=str(output_path),
    )


def scrape_comments(
    post_ids: Sequence[str],
    output_path: str | Path,
    chunk_size: int = 500,
) -> int:
    """Scrape non-deleted comments for the given post IDs to Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filters = Filters(
        comment_include_fields=[
            CommentField.ID,
            CommentField.POST_ID,
            CommentField.PARENT_ID,
            CommentField.AUTHOR,
            CommentField.BODY,
            CommentField.SCORE,
            CommentField.CREATED_UTC,
        ],
        comment_filter=lambda c: c.body not in ("[deleted]", "[removed]"),
    )

    scraper = RedditScraper(filters=filters)
    return scraper.scrape_comments(
        post_ids=list(post_ids),
        output_path=str(output_path),
        chunk_size=chunk_size,
    )
