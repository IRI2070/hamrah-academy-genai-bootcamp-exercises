from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    API_KEY = os.getenv("AVALAI_API_KEY")
    BASE_URL = "https://api.avalai.ir/v1"

    BOT_TOKEN = os.getenv("BALE_TOKEN")
    BOT_BASE_URL = "https://tapi.bale.ai/bot"

    RETRIEVAL_MODEL = "text-embedding-3-small"
    RERANK_MODEL = "cohere-rerank-v4.0-fast"
    LLM_MODEL = "gpt-4o-mini"
    LLM_AS_A_JUDGE_MODEL = "gpt-4o"

    TOP_K = 100
    FINAL_K = 10

    FAISS_DIRECTORY_PATH = BASE_DIR / "artifacts/faiss_index"
    CHUNKS_DIRECTORY = BASE_DIR / 'chunks'
    TESTCASE_CSV_PATH = BASE_DIR / 'evaluation/testset.csv'

    AGENT_MAX_STEPS = 5
    AGENT_VERBOSITY_LEVEL = 2


config = Config()
