from smolagents import OpenAIModel, ToolCallingAgent

from src.config import config
from src.generation.tools import tools
from src.utils.logger import logger

model = OpenAIModel(
    model_id=config.LLM_MODEL,
    api_base=config.BASE_URL,
    api_key=config.API_KEY,
)

agent = ToolCallingAgent(
    tools=tools,
    model=model,
    max_steps=config.AGENT_MAX_STEPS,
    verbosity_level=config.AGENT_VERBOSITY_LEVEL,
)


def generate_answer(query: str, return_full_result=False) -> str:
    logger.info(f"[generate_answer] Running agent for query: {query!r}")
    return agent.run(query, return_full_result=return_full_result)
