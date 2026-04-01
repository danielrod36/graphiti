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

import json
import typing
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .openai_base_client import BaseOpenAIClient

DEFAULT_MODEL = 'mistral-small-latest'
DEFAULT_SMALL_MODEL = 'mistral-tiny'


class MistralClientConfig(LLMConfig):
    """Configuration for Mistral LLM client."""

    base_url: str = "https://api.mistral.ai/v1"
    model: str = DEFAULT_MODEL
    small_model: str = DEFAULT_SMALL_MODEL
    api_key: str | None = None


class MistralClient(BaseOpenAIClient):
    """
    Mistral Client for Graphiti.

    This client uses Mistral's Chat Completions API (which is OpenAI-compatible)
    for both regular and structured output. Mistral does not support the OpenAI
    Responses API, so we use Chat Completions with JSON schema for structured output.
    """

    def __init__(
        self,
        config: MistralClientConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """
        Initialize the MistralClient with the provided configuration.

        Args:
            config (MistralClientConfig | None): The configuration for the LLM client.
            cache (bool): Whether to use caching. Not implemented for Mistral.
            client (Any | None): An optional async client instance.
            max_tokens (int): Maximum tokens for generation.
            reasoning (str | None): Not used by Mistral (kept for compatibility).
            verbosity (str | None): Not used by Mistral (kept for compatibility).
        """
        if config is None:
            config = MistralClientConfig()

        # Set default models from config if not provided
        if config.model is None:
            config.model = DEFAULT_MODEL
        if config.small_model is None:
            config.small_model = DEFAULT_SMALL_MODEL

        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """
        Create a structured completion using Mistral's Chat Completions API.

        Mistral does not support the Responses API, so we use Chat Completions
        with JSON schema format for structured output.
        """
        # Convert Pydantic model to JSON schema
        schema_name = getattr(response_model, '__name__', 'structured_response')
        json_schema = response_model.model_json_schema()

        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'name': schema_name,
                'schema': json_schema,
            },
        }

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,  # type: ignore[arg-type]
        )

        return response

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """
        Create a regular completion with JSON format.

        Uses the same Chat Completions API as structured completion
        for consistency in behavior.
        """
        response_format: dict[str, Any] = {'type': 'json_object'}

        # If a response_model is provided, use JSON schema (optional)
        # This is for consistency, though response_model is typically None here
        if response_model is not None:
            schema_name = getattr(response_model, '__name__', 'response')
            json_schema = response_model.model_json_schema()
            response_format = {
                'type': 'json_schema',
                'json_schema': {
                    'name': schema_name,
                    'schema': json_schema,
                },
            }

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,  # type: ignore[arg-type]
        )

        return response
