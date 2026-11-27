"""Comment filtering and actionable resume-advice classification."""

from __future__ import annotations

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tqdm import tqdm


class ResumeAdviceCheck(BaseModel):
    """Structured output schema for resume-advice classification."""

    is_resume_advice: bool = Field(
        description="True if the comment gives actionable resume advice, else False."
    )


class CommentAdviceClassifier:
    """Classifies Reddit comments as actionable resume advice or not."""

    def __init__(
        self,
        model: str = "gemma4:e4b-mlx",
        num_ctx: int = 2048,
        num_predict: int = 16,
    ) -> None:
        llm = ChatOllama(
            model=model,
            temperature=0.0,
            reasoning=False,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        self._structured_llm = llm.with_structured_output(ResumeAdviceCheck)

    def is_resume_advice(self, comment: str) -> bool:
        """Return True if a comment provides actionable resume feedback."""
        prompt = (
            "Does this Reddit comment give actionable resume advice "
            "(feedback on wording, formatting, sections, projects, skills, "
            "or targeting)?\n"
            "Return False for mod notices, paid service solicitations, "
            "questions, general job-market talk, or vague reactions.\n\n"
            f'Comment:\n"""{str(comment)[:1500]}"""'
        )
        result: ResumeAdviceCheck = self._structured_llm.invoke(
            [HumanMessage(content=prompt)]
        )
        return result.is_resume_advice


COMMENT_COLUMNS = [
    "id",
    "post_id",
    "parent_id",
    "author",
    "body",
    "score",
    "created_utc",
]


def clean_and_classify_comments(
    comments_df: pd.DataFrame,
    classifier: CommentAdviceClassifier,
) -> pd.DataFrame:
    """Filter AutoModerator comments and label actionable resume advice."""
    cols = [c for c in COMMENT_COLUMNS if c in comments_df.columns]
    filtered = comments_df[cols].copy()
    filtered = filtered[filtered["author"] != "AutoModerator"].copy()

    tqdm.pandas(desc="Classifying comments")
    filtered["is_relevant"] = filtered["body"].progress_apply(
        classifier.is_resume_advice
    )
    return filtered
