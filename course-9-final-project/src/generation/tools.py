from smolagents import Tool
from src.retrieval.retrieve import search_knowledge_base
from src.utils.logger import logger


class SearchKnowledgeBase(Tool):
    name = "search_knowledge_base"
    description = "Semantic search over the knowledge base. Returns the most relevant document excerpts."
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query in affirmative form – use descriptive, natural language.",
        },
        "top_k": {
            "type": "integer",
            "description": (
                "How many results to return. Usually 1 for questions about a single specific entity; "
                "3–8 for comparisons, multiple examples, interactions, overviews or broader questions."
            ),
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query: str, top_k: int) -> str:
        logger.info(f"[Tool] search_knowledge_base called with query: {query!r}")
        results = search_knowledge_base(query, top_k)
        logger.info(f"[Tool] Returning {len(results)} documents")
        return results


tools = [SearchKnowledgeBase()]
