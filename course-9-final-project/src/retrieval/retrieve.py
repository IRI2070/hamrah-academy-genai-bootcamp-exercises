from src.config import config
from src.retrieval.indexing import load_index
from src.retrieval.rerank import rerank_documents
from src.utils.logger import logger

index = load_index()


def search_knowledge_base(query, top_k=config.FINAL_K):
    logger.info("Retrieving documents...")
    candidates = index.similarity_search(query, k=config.TOP_K)
    logger.info(f"Found {len(candidates)} candidates")
    docs = [doc.page_content for doc in candidates]
    logger.info("Reranking documents...")
    reranked_docs = rerank_documents(query, docs, top_k)
    logger.info(f"Found {len(reranked_docs)} reranked documents")
    return '\n\n'.join(reranked_docs)
