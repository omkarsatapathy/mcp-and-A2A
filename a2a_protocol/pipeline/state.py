"""ResearchState — Centralized state for the research agent LangGraph pipeline."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ResearchState(BaseModel):
    """
    State model for the single-topic research pipeline.
    Every node accepts ResearchState as input and returns dict to merge into state.
    """

    # ——— Input (set by input_parser) ————————————————————————————————————————
    user_query: str = Field(default="", description="Raw user query string")
    topic: str = Field(default="", description="Extracted research topic")
    desired_length: int = Field(default=500, description="Desired output word count")

    # ——— Research Phase ——————————————————————————————————————————————————————
    raw_research_data: str = Field(default="", description="Assembled corpus from web scraping")

    # ——— Processing Phase ————————————————————————————————————————————————————
    ranked_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-k semantically deduplicated and reranked chunks",
    )

    # ——— Output ——————————————————————————————————————————————————————————————
    research_output: str = Field(default="", description="Final summarized research output")

    # ——— Error Tracking ——————————————————————————————————————————————————————
    last_error: Optional[str] = Field(default=None, description="Latest error message (if any)")

    class Config:
        arbitrary_types_allowed = True
