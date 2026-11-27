"""Post filtering, metadata extraction, and normalization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.classify import JobClassifier
from src.country import build_country_mapping

RELEVANT_TITLE_REGEX = r"\[([^,]+),\s*([^,]+),\s*([^,]+),\s*([^\]]+)\]"
INVALID_TEXTS = {"[removed]", "[deleted]"}
VALID_IMAGE_ENDINGS = {"png", "jpeg"}
YOE_PHRASES = ("yoe", "years", "yrs", "yr", "year")


def filter_and_extract_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw subreddit posts and extract structured title fields."""
    posts = posts_df.copy()

    is_text_valid = (posts["selftext"].str.len() > 0) & (
        ~posts["selftext"].isin(INVALID_TEXTS)
    )
    url_ending = posts["url"].str.split(".").str[-1]
    is_url_valid = url_ending.isin(VALID_IMAGE_ENDINGS)

    posts[["yoe", "current_role", "target_role", "country"]] = posts[
        "title"
    ].str.extract(RELEVANT_TITLE_REGEX)

    yoe_lower = posts["yoe"].str.lower()
    contains_yoe_phrase = pd.Series(False, index=posts.index)
    for phrase in YOE_PHRASES:
        contains_yoe_phrase = contains_yoe_phrase | yoe_lower.str.contains(
            phrase, na=False
        )

    yoe_len = posts["yoe"].str.len()
    is_yoe_valid = (yoe_len > 3.5) & (yoe_len < 7) & contains_yoe_phrase
    is_country_valid = (posts["country"].str.len() > 0) & (
        posts["country"].str.len() < 26
    )

    filtered = posts[
        is_text_valid & is_url_valid & is_yoe_valid & is_country_valid
    ].dropna()

    filtered = filtered.copy()
    filtered["yoe"] = pd.to_numeric(
        filtered["yoe"].str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    return filtered


def normalize_posts(
    filtered_posts: pd.DataFrame,
    role_classifier: JobClassifier,
    role_cache_path: Optional[str | Path] = None,
    country_cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Normalize roles (via SOC classifier + cache) and countries on posts."""
    posts = filtered_posts.copy()

    # 1. Roles (load cache if available, classify only unseen roles)
    role_map: dict[str, str] = {}
    cached_df: Optional[pd.DataFrame] = None
    if role_cache_path and Path(role_cache_path).exists():
        cached_df = pd.read_csv(role_cache_path)
        role_map = dict(
            zip(cached_df["raw_role"], cached_df["detailed_code"])
        )

    all_roles = pd.concat(
        [posts["current_role"], posts["target_role"]], ignore_index=True
    ).dropna()
    unseen_roles = [r for r in all_roles.unique() if r not in role_map]

    if unseen_roles and role_classifier is not None:
        new_rows = []
        for raw_role in tqdm(unseen_roles, desc="Classifying roles"):
            res = role_classifier.classify(raw_role)
            new_rows.append(res.__dict__)
            role_map[raw_role] = res.detailed_code

        if role_cache_path:
            new_df = pd.DataFrame(new_rows)
            combined = (
                pd.concat([cached_df, new_df], ignore_index=True)
                if cached_df is not None
                else new_df
            )
            Path(role_cache_path).parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(role_cache_path, index=False)

    posts["current_role"] = posts["current_role"].map(role_map)
    posts["target_role"] = posts["target_role"].map(role_map)

    # 2. Countries
    country_map = build_country_mapping(posts["country"])
    if country_cache_path:
        Path(country_cache_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            country_map.items(), columns=["raw_country", "cleaned_code"]
        ).to_csv(country_cache_path, index=False)

    posts["country"] = posts["country"].map(country_map)
    return posts.dropna().reset_index(drop=True)
