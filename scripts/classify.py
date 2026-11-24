"""Script to classify the role into our amended SOC classification table"""

from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
import pandas as pd
from pydantic import BaseModel, Field


# State schema
class JobState(BaseModel):
    raw_role: str
    major_code: Optional[str] = None
    major_name: Optional[str] = None
    detailed_code: Optional[str] = None
    detailed_name: Optional[str] = None
    broad_code: Optional[str] = None
    broad_name: Optional[str] = None
    minor_code: Optional[str] = None
    minor_name: Optional[str] = None


# LLM output schemas
class MajorSelection(BaseModel):
    major_code: str = Field(
        description="The exact Major Group code selected, e.g. '13-0000'"
    )


class DetailedSelection(BaseModel):
    detailed_code: str = Field(
        description="The exact Detailed Occupation code selected, e.g. '13-2011'"
    )


major_structured_llm = llm.with_structured_output(MajorSelection)
detailed_structured_llm = llm.with_structured_output(DetailedSelection)

# Prepare taxonomy hierarchy
df_hier = soc_taxonomy.copy()
df_hier["Major Group"] = df_hier["Major Group"].ffill()
df_hier["Minor Group"] = df_hier["Minor Group"].ffill()
df_hier["Broad Group"] = df_hier["Broad Group"].ffill()

major_df = soc_taxonomy[soc_taxonomy["Major Group"].notna()][
    ["Major Group", "Name"]
].dropna()
MAJOR_OPTIONS = "\n".join(
    [
        f"- {row['Major Group']}: {row['Name']}"
        for _, row in major_df.iterrows()
    ]
)

minor_df = soc_taxonomy[soc_taxonomy["Minor Group"].notna()][
    ["Minor Group", "Name"]
].dropna()
minor_name_map = dict(zip(minor_df["Minor Group"], minor_df["Name"]))

broad_df = soc_taxonomy[soc_taxonomy["Broad Group"].notna()][
    ["Broad Group", "Name"]
].dropna()
broad_name_map = dict(zip(broad_df["Broad Group"], broad_df["Name"]))

detailed_df = df_hier[df_hier["Detailed Occupation"].notna()].copy()


# Step 1: Select Major Group
def choose_major_node(state: JobState) -> dict:
    prompt = f"""You are an official labor classification system.
Role: "{state.raw_role}"

Choose the SINGLE best Major Group code from this list:
{MAJOR_OPTIONS}
"""
    result: MajorSelection = major_structured_llm.invoke(
        [HumanMessage(content=prompt)]
    )

    code = result.major_code.strip()
    matched = major_df[major_df["Major Group"] == code]

    if matched.empty:
        raise ValueError(
            f"LLM returned invalid major_code '{code}' for role"
            f" '{state.raw_role}'."
        )

    return {"major_code": code, "major_name": matched["Name"].values[0]}


# Step 2: Select Detailed Occupation
def choose_detailed_node(state: JobState) -> dict:
    selected_major = state.major_code
    sub_df = detailed_df[detailed_df["Major Group"] == selected_major]

    if sub_df.empty:
        raise ValueError(
            f"No detailed occupations found in table for Major Group"
            f" '{selected_major}'."
        )

    candidates = [
        f"- {row['Detailed Occupation']}: {row['Name']}"
        for _, row in sub_df.iterrows()
    ]
    candidates_text = "\n".join(candidates)

    prompt = f"""For the role "{state.raw_role}", choose the single most accurate Detailed Occupation from Major Group {selected_major}:
{candidates_text}
"""
    result: DetailedSelection = detailed_structured_llm.invoke(
        [HumanMessage(content=prompt)]
    )

    det_code = result.detailed_code.strip()
    match = sub_df[sub_df["Detailed Occupation"] == det_code]

    if match.empty:
        raise ValueError(
            f"LLM returned invalid detailed_code '{det_code}' for Major Group"
            f" '{selected_major}'.\nValid options:"
            f" {sub_df['Detailed Occupation'].tolist()}"
        )

    return {
        "detailed_code": match.iloc[0]["Detailed Occupation"],
        "detailed_name": match.iloc[0]["Name"],
    }


# Step 3: Lookup Parent Hierarchy
def lookup_hierarchy_node(state: JobState) -> dict:
    det_code = state.detailed_code
    match = detailed_df[detailed_df["Detailed Occupation"] == det_code]

    if match.empty:
        raise ValueError(
            f"Detailed code '{det_code}' does not exist in hierarchy table."
        )

    row = match.iloc[0]
    broad_code = row["Broad Group"]
    minor_code = row["Minor Group"]

    return {
        "broad_code": broad_code,
        "broad_name": broad_name_map.get(broad_code),
        "minor_code": minor_code,
        "minor_name": minor_name_map.get(minor_code),
    }


class JobClassifier:

    def __init__(self):
        # initialize the workflow

        self.workflow = StateGraph(JobState)

        self.workflow.add_node("step1_major", choose_major_node)
        self.workflow.add_node("step2_detailed", choose_detailed_node)
        self.workflow.add_node("step3_lookup", lookup_hierarchy_node)

        self.workflow.add_edge(START, "step1_major")
        self.workflow.add_edge("step1_major", "step2_detailed")
        self.workflow.add_edge("step2_detailed", "step3_lookup")
        self.workflow.add_edge("step3_lookup", END)

        self.pipeline = self.workflow.compile()

    def classify(self, role: str) -> JobState:
        return self.pipeline({"raw_role": role})

    