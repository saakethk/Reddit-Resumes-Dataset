# Reddit Resume Dataset

An empirical, structured dataset of resume submissions and peer critique comments curated from Reddit (`r/resumes`) using [deepsift](https://github.com/saakethk/deepsift).

---

## Purpose

High-quality, publicly available resume datasets paired with real-world feedback are difficult to find due to privacy concerns, paywalled platforms, and unstructured formatting. This repository provides a cleaned, normalized, and reproducible dataset built from public `r/resumes` posts and comments to support:

1. **Resume Evaluation & Feedback Modeling** — Training and benchmarking models that critique resumes or predict actionable feedback.
2. **Labor Market & Career-Transition Research** — Analyzing trends across years of experience, current-to-target role transitions, and geographic distribution.
3. **Pipeline Benchmarking** — Evaluating and stress-testing the `deepsift` Reddit scraping and filtering library.

---

## Dataset Overview

The final processed dataset is stored in [`data/final/`](data/final/) and spans **January 1, 2026 – October 31, 2026**:

| File | Rows | Columns | Description |
| :--- | ---: | ---: | :--- |
| [`data/final/posts.parquet`](data/final/posts.parquet) | 3,037 | 12 | Cleaned and normalized resume posts (Parquet) |
| [`data/final/posts.csv`](data/final/posts.csv) | 3,037 | 12 | Cleaned and normalized resume posts (CSV) |
| [`data/final/comments.parquet`](data/final/comments.parquet) | 8,101 | 8 | Non-AutoModerator comments across 1,941 posts with advice relevance labels |

### `posts.parquet` / `posts.csv` Schema

| Column | Type | Description |
| :--- | :--- | :--- |
| `created_utc` | int64 | Unix timestamp of post creation |
| `id` | string | Reddit submission ID |
| `num_comments` | int64 | Total comment count on Reddit at scrape time |
| `score` | int64 | Net Reddit upvotes |
| `selftext` | string | Post body text (context provided by the author) |
| `subreddit` | string | Source subreddit (`resumes`) |
| `title` | string | Original Reddit post title |
| `url` | string | Direct URL to the resume image (`.png` or `.jpeg`) |
| `yoe` | float64 | Parsed years of experience (median: `2.0`, mean: `2.72`, range: `0.0–45.0`) |
| `current_role` | string | Standardized 2018 SOC (or amended `99-*`) detailed occupation code |
| `target_role` | string | Standardized 2018 SOC (or amended `99-*`) detailed occupation code |
| `country` | string | Normalized country or work-mode label (e.g., `United States`, `India`, `Canada`, `Remote`) |

### `comments.parquet` Schema

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Reddit comment ID |
| `post_id` | string | Parent submission ID (joins with `posts.id`) |
| `parent_id` | string | Parent Reddit thing ID (post or parent comment) |
| `author` | string | Comment author username (`AutoModerator` excluded) |
| `body` | string | Comment text |
| `score` | int64 | Net Reddit upvotes |
| `created_utc` | int64 | Unix timestamp of comment creation |
| `is_relevant` | bool | `True` (4,070 rows) if actionable resume advice; `False` (4,031 rows) otherwise |

---

## Methodology

### 1. Subreddit Selection & Raw Collection
- Initial subreddit discovery was performed with PRAW, identifying `r/resumes` (1,328,408 subscribers) as the primary source due to its volume and structured posting conventions.
- Using `deepsift`, **20,827** posts with at least 1 comment published between January 1, 2026 and November 1, 2026 were collected into [`data/raw/subreddit_resumes_posts.parquet`](data/raw/subreddit_resumes_posts.parquet).

### 2. Post Filtering & Metadata Extraction
Posts were filtered through a multi-stage pipeline in [`notebooks/02_exploring_sample_data.ipynb`](notebooks/02_exploring_sample_data.ipynb) and [`notebooks/06_finalizing_sample_data.ipynb`](notebooks/06_finalizing_sample_data.ipynb):
1. **Valid Body Text**: Removed posts with empty, `[removed]`, or `[deleted]` `selftext` (9,658 remaining).
2. **Resume Image URLs**: Retained only posts linking directly to `.png` or `.jpeg` resume images (9,215 remaining).
3. **Structured Title Parsing**: Extracted `[X YoE, Current Role/Unemployed, Target Role, Country]` using regex matching `r/resumes` submission guidelines (8,512 matched).
4. **Numeric YoE & Country Validation**: Extracted numeric `yoe`, filtered `country` strings under 26 characters, and dropped rows with unmapped/missing fields, yielding **3,037** complete posts.

### 3. Country Normalization
Raw location strings (`data/intermediate/countries.csv`) were cleaned using regex rules and a lookup table in [`notebooks/03_developing_country_classifications.ipynb`](notebooks/03_developing_country_classifications.ipynb), mapping 1,408 unique entries to standardized country names and work-mode tags (`Remote`, `Europe / EU`, `UNMAPPED`, etc.).

### 4. Role Standardization (Amended SOC Taxonomy + LLM Classifier)
- **Amended Taxonomy**: Starting from the 2018 U.S. BLS Standard Occupational Classification ([`data/reference/soc_structure_2018.xlsx`](data/reference/soc_structure_2018.xlsx)), we added Major Group `99-0000` (*Non-Occupational and Transition Statuses*) with 13 detailed codes (`99-1011` through `99-5011`) in [`data/reference/amended_soc_structure.csv`](data/reference/amended_soc_structure.csv) to capture statuses common in resume posts (e.g., *Unemployed*, *Laid Off*, *Enrolled Student*, *Intern*, *Recent Graduate*, *Career Break*, *Freelancer*).
- **Hierarchical Classification**: [`src/classify.py`](src/classify.py) (`JobClassifier`) implements a 3-node LangGraph workflow backed by Ollama (`gemma4:e4b-mlx`) with Pydantic-enforced structured outputs:
  1. `step1_major` — Classifies the raw role into a SOC Major Group.
  2. `step2_detailed` — Selects the best Detailed Occupation within that Major Group.
  3. `step3_lookup` — Deterministically resolves Minor and Broad group codes from the taxonomy table.
- This pipeline standardized **7,784** unique role strings into [`data/intermediate/normalized_roles.csv`](data/intermediate/normalized_roles.csv).

### 5. Comment Extraction & Advice Filtering
- **11,141** comments associated with the filtered posts were scraped via `deepsift` into [`data/raw/subreddit_resumes_comments.parquet`](data/raw/subreddit_resumes_comments.parquet).
- `AutoModerator` comments were removed, leaving **8,101** comments.
- Each comment was classified (`is_relevant`) in [`notebooks/08_exploring_comments.ipynb`](notebooks/08_exploring_comments.ipynb) using `gemma4:e4b-mlx` structured output to distinguish actionable resume feedback (wording, formatting, sections, projects, skills, targeting) from general discussion, questions, or solicitations.

---

## Repository Structure

```text
resume_dataset/
├── .env.example                 # Template for Reddit API credentials
├── .gitignore
├── README.md
├── data/
│   ├── final/                   # Final cleaned & labeled dataset files
│   │   ├── posts.parquet
│   │   ├── posts.csv
│   │   └── comments.parquet
│   ├── raw/                     # Raw scraped Reddit posts and comments
│   │   ├── subreddit_resumes_posts.parquet
│   │   └── subreddit_resumes_comments.parquet
│   ├── intermediate/            # Extracted and normalized lookup tables
│   │   ├── roles.csv
│   │   ├── normalized_roles.csv
│   │   ├── countries.csv
│   │   ├── normalized_countries.csv
│   │   └── comment_texts.csv
│   └── reference/               # Official & amended 2018 SOC taxonomies
│       ├── soc_structure_2018.xlsx
│       ├── soc_structure_2018.pdf
│       └── amended_soc_structure.csv
├── notebooks/                   # Ordered exploration and processing notebooks
│   ├── 01_exploring_subreddits.ipynb
│   ├── 02_exploring_sample_data.ipynb
│   ├── 03_developing_country_classifications.ipynb
│   ├── 04_developing_role_classifications.ipynb
│   ├── 05_testing_role_classifier.ipynb
│   ├── 06_finalizing_sample_data.ipynb
│   ├── 07_extracting_comments.ipynb
│   └── 08_exploring_comments.ipynb
└── src/                         # Reusable Python modules
    ├── __init__.py
    └── classify.py              # LangGraph + Ollama SOC role classifier
```

---

## Setup & Usage

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) with `gemma4:e4b-mlx` (required only if re-running LLM role or comment classification)
- Reddit API credentials in `.env` (see [`.env.example`](.env.example), required only if re-scraping with PRAW / `deepsift`)

### Quick Start

```bash
# Activate the virtual environment
source venv/bin/activate
```

All notebooks assume the **repository root** as the current working directory.

To classify a job title programmatically using `JobClassifier`:

```python
from src.classify import JobClassifier

classifier = JobClassifier()
result = classifier.classify("Software Engineering Intern")
print(result.detailed_code, result.detailed_name)
```
