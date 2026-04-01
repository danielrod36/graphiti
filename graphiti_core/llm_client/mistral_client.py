import json
import typing
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .openai_base_client import BaseOpenAIClient

DEFAULT_MODEL = 'mistral-small-latest'
DEFAULT_SMALL_MODEL = 'mistral-small-latest'


class MistralClientConfig(LLMConfig):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            base_url=base_url or 'https://api.mistral.ai/v1',
            temperature=temperature,
            max_tokens=max_tokens,
            small_model=small_model or DEFAULT_SMALL_MODEL,
        )


class MistralClient(BaseOpenAIClient):
    def __init__(
        self,
        config: MistralClientConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        if config is None:
            config = MistralClientConfig()
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
            response_format=response_format,
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
        response_format: dict[str, Any] = {'type': 'json_object'}
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
            response_format=response_format,
        )
        return response

    def _handle_structured_response(self, response):
        content_str = response.choices[0].message.content or ""
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        if content_str:
            return json.loads(content_str), input_tokens, output_tokens
        else:
            raise Exception(f"Invalid response from Mistral: {response}")

    def _handle_json_response(self, response):
        content_str = response.choices[0].message.content or "{}"
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        return json.loads(content_str), input_tokens, output_tokens
