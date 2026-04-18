from .bge_reranker_client import BGERerankerClient
from .client import CrossEncoderClient
from .noop_reranker import NoopReranker
from .openai_reranker_client import OpenAIRerankerClient
from .zai_reranker import ZAIReranker

__all__ = [
    'BGERerankerClient',
    'CrossEncoderClient',
    'NoopReranker',
    'OpenAIRerankerClient',
    'ZAIReranker',
]

# Optional imports — only available if dependencies are installed
try:
    from .gemini_reranker_client import GeminiRerankerClient  # requires google-genai

    __all__.append('GeminiRerankerClient')
except ImportError:
    pass
