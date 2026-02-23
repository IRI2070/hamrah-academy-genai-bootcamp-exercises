from src.generation.generate import generate_answer
from src.generation.prompt import system_prompt
from src.utils.logger import logger


class RAGPipeline:
    def __init__(self):
        self.system_prompt = system_prompt
        self.histories = {}

    def _get_session_history(self, session_id: str):
        if session_id not in self.histories:
            self.histories[session_id] = []
        return self.histories[session_id]

    def reset_session(self, session_id: str) -> None:
        self.histories[session_id] = []
        logger.info(f"[RAGPipeline] Reset history for session {session_id}")

    def ask(self, query: str, session_id: str) -> str:
        logger.info(f"[RAGPipeline] Received query from {session_id}: {query!r}")

        history = self._get_session_history(session_id)
        history.append(f"User: {query}")

        full_query = (
                self.system_prompt
                + "\n\nConversation history:\n"
                + "\n".join(history)
                + "\n\nAnswer the last user question ONLY."
        )

        answer = generate_answer(full_query, False)

        history.append(f"Assistant: {answer}")

        logger.info("[RAGPipeline] Returning final answer")
        return answer
