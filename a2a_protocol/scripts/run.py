"""Minimal execution script — invoke the research agent graph with a one-liner."""

import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the a2a_protocol root (one level up from scripts/)
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from a2a_protocol.pipeline.graph import compiled_graph

query = input("Enter your research query: ").strip()

result = compiled_graph.invoke({"user_query": query})

print("\n" + "=" * 80)
print(f"RESEARCH OUTPUT ({len(result['research_output'].split())} words)")
print("=" * 80)
print(result["research_output"])
