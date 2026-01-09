import os, time, logging
from mistralai import Mistral
from dotenv import load_dotenv
from logger import setup_logger

load_dotenv()
logger = setup_logger()

class LLM:
    def __init__(self, model="mistral-large-latest", retries=3):
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.model = model
        self.retries = retries
        logger.info("LLM initialized with model %s", model)

    def call(self, messages, temperature=0.3):
        for i in range(self.retries):
            try:
                logger.info("LLM call attempt %d", i + 1)
                res = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                return res.choices[0].message.content
            except Exception as e:
                logger.error("LLM call failed on attempt %d: %s", i + 1, str(e))
                if i == self.retries - 1:
                    raise
                time.sleep(2 ** i)
    
    def embed(self, text: str):
        """
        Returns embedding vector for semantic similarity
        """
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=text
        )
        return response.data[0].embedding
