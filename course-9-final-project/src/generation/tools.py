from smolagents import Tool
from src.retrieval.retrieve import search_knowledge_base
from src.utils.logger import logger


class SearchKnowledgeBase(Tool):
    name = "search_knowledge_base"
    description = "Semantic search in the Persian drug database and return related texts. Note that each drug can include the following information: English name of the drug, Persian name of the drug, indications for use, mechanism of action, contraindications, pharmacokinetics, warnings, side effects, drug interactions, recommended tips, dosage, pharmaceutical forms, product ingredients, active ingredients, drug interactions."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of documents to be returned. If the question is about a specific drug, one document should be returned because each drug has only one document, otherwise this number should be adjusted depending on the type of question. For example, for drug prescription, drug comparison, interactions, and the like, this number should definitely be greater than one.",
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
