"""Z.AI Reranker — cross-encoder reranking via Z.AI's rerank model."""

import httpx
from .client import CrossEncoderClient

DEFAULT_RERANK_MODEL = "rerank"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_TOP_N = 50


class ZAIReranker(CrossEncoderClient):
    """Cross-encoder reranker using Z.AI's rerank model.

    Uses the same Z.AI API key as the embedder. No additional keys needed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def rank(
        self, query: str, passages: list[str]
    ) -> list[tuple[str, float]]:
        """Rerank passages by relevance to query.

        Returns list of (passage, score) sorted descending by score.
        """
        if not passages:
            return []

        resp = await self._client.post(
            f"{self.base_url}/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": passages,
                "top_n": len(passages),
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[tuple[str, float]] = []
        for r in data.get("results", []):
            idx = r["index"]
            score = r["relevance_score"]
            if 0 <= idx < len(passages):
                results.append((passages[idx], score))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    async def close(self):
        await self._client.aclose()
