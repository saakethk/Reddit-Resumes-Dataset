"""Script to classify a role into the amended SOC classification table."""

from typing import Optional

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class JobState(BaseModel):
    """State passed through the SOC classification workflow."""

    raw_role: str
    major_code: Optional[str] = None
    major_name: Optional[str] = None
    detailed_code: Optional[str] = None
    detailed_name: Optional[str] = None
    broad_code: Optional[str] = None
    broad_name: Optional[str] = None
    minor_code: Optional[str] = None
    minor_name: Optional[str] = None


class MajorSelection(BaseModel):
    """Structured output for Major Group selection."""

    major_code: str = Field(
        description="The exact Major Group code selected, e.g. '13-0000'"
    )


class DetailedSelection(BaseModel):
    """Structured output for Detailed Occupation selection."""

    detailed_code: str = Field(
        description="The exact Detailed Occupation code selected, e.g. '13-2011'"
    )


class JobClassifier:
    """Classifies raw job titles into the amended SOC hierarchy."""

    def __init__(
        self,
        taxonomy_path: str = "data/sample/amended_soc_structure.csv",
        model: str = "gemma4:e4b-mlx",
    ) -> None:
        llm = ChatOllama(model=model, temperature=0.0, format="json")
        self._major_llm = llm.with_structured_output(MajorSelection)
        self._detailed_llm = llm.with_structured_output(DetailedSelection)

        soc_taxonomy = pd.read_csv(taxonomy_path)
        self._major_map = self._build_name_map(soc_taxonomy, "Major Group")
        self._minor_map = self._build_name_map(soc_taxonomy, "Minor Group")
        self._broad_map = self._build_name_map(soc_taxonomy, "Broad Group")
        self._major_options = "\n".join(
            f"- {code}: {name}" for code, name in self._major_map.items()
        )

        df_hier = soc_taxonomy.copy()
        for col in ("Major Group", "Minor Group", "Broad Group"):
            df_hier[col] = df_hier[col].ffill()
        self._detailed_df = df_hier[
            df_hier["Detailed Occupation"].notna()
        ].copy()

        self.pipeline = self._build_workflow()

    @staticmethod
    def _build_name_map(df: pd.DataFrame, column: str) -> dict[str, str]:
        subset = df[df[column].notna()][[column, "Name"]].dropna()
        return dict(zip(subset[column], subset["Name"]))

    def _build_workflow(self):
        workflow = StateGraph(JobState)
        workflow.add_node("step1_major", self._choose_major_node)
        workflow.add_node("step2_detailed", self._choose_detailed_node)
        workflow.add_node("step3_lookup", self._lookup_hierarchy_node)

        workflow.add_edge(START, "step1_major")
        workflow.add_edge("step1_major", "step2_detailed")
        workflow.add_edge("step2_detailed", "step3_lookup")
        workflow.add_edge("step3_lookup", END)

        return workflow.compile()

    def _choose_major_node(self, state: JobState) -> dict:
        prompt = (
            "You are an official labor classification system.\n"
            f'Role: "{state.raw_role}"\n\n'
            "Choose the SINGLE best Major Group code from this list:\n"
            f"{self._major_options}\n"
        )
        result: MajorSelection = self._major_llm.invoke(
            [HumanMessage(content=prompt)]
        )
        code = result.major_code.strip()

        if code not in self._major_map:
            raise ValueError(
                f"LLM returned invalid major_code '{code}' "
                f"for role '{state.raw_role}'."
            )

        return {"major_code": code, "major_name": self._major_map[code]}

    def _choose_detailed_node(self, state: JobState) -> dict:
        selected_major = state.major_code
        sub_df = self._detailed_df[
            self._detailed_df["Major Group"] == selected_major
        ]

        if sub_df.empty:
            raise ValueError(
                "No detailed occupations found in table for Major Group "
                f"'{selected_major}'."
            )

        candidates_text = "\n".join(
            f"- {row['Detailed Occupation']}: {row['Name']}"
            for _, row in sub_df.iterrows()
        )
        prompt = (
            f'For the role "{state.raw_role}", choose the single most accurate '
            f"Detailed Occupation from Major Group {selected_major}:\n"
            f"{candidates_text}\n"
        )
        result: DetailedSelection = self._detailed_llm.invoke(
            [HumanMessage(content=prompt)]
        )

        det_code = result.detailed_code.strip()
        match = sub_df[sub_df["Detailed Occupation"] == det_code]

        if match.empty:
            raise ValueError(
                f"LLM returned invalid detailed_code '{det_code}' for Major Group "
                f"'{selected_major}'.\n"
                f"Valid options: {sub_df['Detailed Occupation'].tolist()}"
            )

        row = match.iloc[0]
        return {
            "detailed_code": row["Detailed Occupation"],
            "detailed_name": row["Name"],
        }

    def _lookup_hierarchy_node(self, state: JobState) -> dict:
        det_code = state.detailed_code
        match = self._detailed_df[
            self._detailed_df["Detailed Occupation"] == det_code
        ]

        if match.empty:
            raise ValueError(
                f"Detailed code '{det_code}' does not exist in hierarchy table."
            )

        row = match.iloc[0]
        broad_code = row["Broad Group"]
        minor_code = row["Minor Group"]

        return {
            "broad_code": broad_code,
            "broad_name": self._broad_map.get(broad_code),
            "minor_code": minor_code,
            "minor_name": self._minor_map.get(minor_code),
        }

    def classify(self, role: str) -> JobState:
        """Classify a single raw role string into the SOC hierarchy."""
        result = self.pipeline.invoke({"raw_role": role})
        return JobState(**result)