"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'embedding-3'
# Z.AI embedding-3 outputs 2048 dimensions
DEFAULT_EMBEDDING_DIM = 2048


class ZAIEmbedderConfig(EmbedderConfig):
    """Configuration for Z.AI Embedder client."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    # CRITICAL: Set to 2048 to match Z.AI's output dimension
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


class ZAIEmbedder(EmbedderClient):
    """
    Z.AI Embedder Client for Graphiti.

    This client uses Z.AI's embedding-3 model via their OpenAI-compatible API.
    The model outputs 2048-dimensional embeddings, which is why we set
    embedding_dim to 2048 instead of the default 1024.
    """

    def __init__(
        self,
        config: ZAIEmbedderConfig | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """
        Initialize the ZAIEmbedder with the provided configuration.

        Args:
            config (ZAIEmbedderConfig | None): The configuration for the embedder.
            client (AsyncOpenAI | None): An optional async client instance.
        """
        if config is None:
            config = ZAIEmbedderConfig()

        self.config = config

        if client is not None:
            self.client = client
        else:
            # Read API key from environment variable if not provided
            import os

            api_key = config.api_key or os.getenv('ZAI_API_KEY')
            if api_key is None:
                raise ValueError(
                    "ZAI_API_KEY must be provided either in config or as environment variable"
                )

            self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for the given input data.

        Args:
            input_data: Text to embed (string, list of strings, or token IDs).

        Returns:
            list[float]: Embedding vector truncated to config.embedding_dim.
        """
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        # Truncate to configured embedding_dim (though we set it to 2048)
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for multiple texts in a single batch request.

        Args:
            input_data_list: List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        return [
            embedding.embedding[: self.config.embedding_dim] for embedding in result.data
        ]
