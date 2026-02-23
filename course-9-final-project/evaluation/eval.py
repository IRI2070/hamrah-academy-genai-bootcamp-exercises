import os

import pandas as pd
from deepeval.evaluate import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI
from src.config import config
from src.generation.generate import generate_answer
from deepeval.models import DeepEvalBaseLLM


class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(
            self,
            model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return f"Custom OpenAI Model: {self.model.model_name}"


testcases_df = pd.read_csv(filepath_or_buffer=config.TESTCASE_CSV_PATH, encoding='utf-8')

test_cases = []
for index, golden in testcases_df.iterrows():
    # answer = generate_answer(golden.input, return_full_result=False)
    test_case = LLMTestCase(
        input=golden.input,
        # actual_output=answer,
        actual_output=golden.actual_output,
        expected_output=golden.expected_output
    )
    test_cases.append(test_case)

eval_model = ChatOpenAI(
    model=config.LLM_AS_A_JUDGE_MODEL,
    api_key=os.getenv('AVALAI_API_KEY'),
    base_url=config.BASE_URL
)

eval_model_openai_wrapper = CustomOpenAI(model=eval_model)

generator_metrics = [
    AnswerRelevancyMetric(model=eval_model_openai_wrapper),
    # FaithfulnessMetric(model=eval_model_openai_wrapper)
]

eval_results = evaluate(test_cases, generator_metrics)

print(eval_results)
