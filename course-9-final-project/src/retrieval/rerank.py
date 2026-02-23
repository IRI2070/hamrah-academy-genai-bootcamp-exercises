import requests
from src.config import config
from src.utils.logger import logger

headers = {
    "Authorization": f"Bearer {config.API_KEY}",
    "Content-Type": "application/json"
}


def rerank_documents(query, docs, top_k=config.FINAL_K):
    payload = {
        "model": config.RERANK_MODEL,
        "query": query,
        "documents": docs,
        "top_n": top_k
    }

    try:
        response = requests.post(
            f"{config.BASE_URL}/rerank",
            headers=headers,
            json=payload,
            timeout=15
        )

        if response.status_code != 200:
            logger.warning(f"Reranking failed (status {response.status_code}). Returning original order.")

        results = response.json().get("results", [])
        logger.info("Reranking completed successfully")
        return [docs[item["index"]] for item in results]

    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning original order.")
        return docs
