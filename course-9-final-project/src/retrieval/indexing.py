from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.config import config
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from src.utils.logger import logger


def load_index():
    embeddings = OpenAIEmbeddings(
        model=config.RETRIEVAL_MODEL,
        base_url=config.BASE_URL,
        api_key=config.API_KEY,
        max_retries=10,
        request_timeout=60
    )

    if os.path.exists(config.FAISS_DIRECTORY_PATH):
        logger.info("FAISS index exist")
        return FAISS.load_local(
            config.FAISS_DIRECTORY_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    logger.info("FAISS index not exist. An index must be created.")
    loader = DirectoryLoader(
        config.CHUNKS_DIRECTORY,
        glob="./*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )

    documents = loader.load()

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(config.FAISS_DIRECTORY_PATH)

    return vector_store
