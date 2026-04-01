"""
Z.AI Embedder for MemVault v2.

Uses Z.AI's embedding-3 model (2048D) via their OpenAI-compatible API.
Handles empty/whitespace-only inputs gracefully.
"""

from collections.abc import Iterable
from typing import Any
import logging

from openai import AsyncOpenAI

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'embedding-3'
DEFAULT_EMBEDDING_DIM = 2048
FALLBACK_TEXT = "empty"
logger = logging.getLogger(__name__)


class ZAIEmbedderConfig(EmbedderConfig):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


class ZAIEmbedder(EmbedderClient):
    def __init__(self, config: ZAIEmbedderConfig | None = None, client: AsyncOpenAI | None = None):
        if config is None:
            config = ZAIEmbedderConfig()
        self.config = config
        if client is not None:
            self.client = client
        else:
            import os
            api_key = config.api_key or os.getenv('ZAI_API_KEY')
            if api_key is None:
                raise ValueError("ZAI_API_KEY must be provided either in config or as environment variable")
            self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)

    def _sanitize_input(self, input_data: str | list[str]) -> str | list[str]:
        """Replace empty/whitespace strings with fallback to avoid API errors."""
        if isinstance(input_data, str):
            return input_data.strip() or FALLBACK_TEXT
        return [s.strip() or FALLBACK_TEXT for s in input_data]

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        input_data = self._sanitize_input(input_data)
        result = await self.client.embeddings.create(input=input_data, model=self.config.embedding_model)
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        input_data_list = self._sanitize_input(input_data_list)
        result = await self.client.embeddings.create(input=input_data_list, model=self.config.embedding_model)
        return [e.embedding[: self.config.embedding_dim] for e in result.data]
