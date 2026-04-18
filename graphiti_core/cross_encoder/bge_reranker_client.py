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

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
else:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            'sentence-transformers is required for BGERerankerClient. '
            'Install it with: pip install graphiti-core[sentence-transformers]'
        ) from None

from graphiti_core.cross_encoder.client import CrossEncoderClient


DEFAULT_BGE_MODEL = 'BAAI/bge-reranker-v2-m3'


class BGERerankerClient(CrossEncoderClient):
    def __init__(self, model: str | None = None):
        import logging

        logger = logging.getLogger(__name__)
        model_name = model or DEFAULT_BGE_MODEL
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            logger.error(f'Failed to load CrossEncoder model "{model_name}": {e}')
            raise ImportError(
                f'Failed to load CrossEncoder model "{model_name}". '
                f'Ensure sentence-transformers is installed and the model name is valid. '
                f'Original error: {e}'
            ) from e

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        input_pairs = [[query, passage] for passage in passages]

        # Run the synchronous predict method in an executor
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self.model.predict, input_pairs)

        ranked_passages = sorted(
            [(passage, float(score)) for passage, score in zip(passages, scores, strict=False)],
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked_passages
