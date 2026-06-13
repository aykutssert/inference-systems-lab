from collections.abc import AsyncIterator

import httpx
from fastapi import HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse


class InferenceProxy:
    def __init__(
        self,
        client: httpx.AsyncClient,
        upstream_base_url: str,
        upstream_api_key: str | None = None,
        health_timeout_seconds: float = 2.0,
    ) -> None:
        self._client = client
        upstream_root = upstream_base_url.rstrip("/")
        self._upstream_url = f"{upstream_root}/v1/chat/completions"
        self._health_url = f"{upstream_root}/health"
        self._upstream_api_key = upstream_api_key
        self._health_timeout_seconds = health_timeout_seconds

    def upstream_headers(self) -> dict[str, str]:
        if self._upstream_api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._upstream_api_key}"}

    async def is_ready(self) -> bool:
        try:
            response = await self._client.get(
                self._health_url,
                headers=self.upstream_headers(),
                timeout=self._health_timeout_seconds,
            )
        except httpx.RequestError:
            return False
        return response.status_code == status.HTTP_200_OK

    async def forward(self, request: Request) -> Response:
        headers = {
            "Content-Type": "application/json",
            **self.upstream_headers(),
        }

        upstream_request = self._client.build_request(
            "POST",
            self._upstream_url,
            headers=headers,
            content=await request.body(),
        )
        try:
            upstream_response = await self._client.send(
                upstream_request,
                stream=True,
            )
        except httpx.RequestError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference upstream is unavailable",
            ) from error

        content_type = upstream_response.headers.get(
            "content-type",
            "application/json",
        )
        if content_type.startswith("text/event-stream"):
            return StreamingResponse(
                stream_response(upstream_response),
                status_code=upstream_response.status_code,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            content = await upstream_response.aread()
        finally:
            await upstream_response.aclose()
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            media_type=content_type,
        )


async def stream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()
