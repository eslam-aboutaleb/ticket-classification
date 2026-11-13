"""Bound request buffering at the ASGI receive boundary, including chunked bodies."""

from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: Callable[[], int]) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_bytes()
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                if not value.isdigit():
                    await JSONResponse(
                        {"detail": "Invalid Content-Length"}, status_code=400
                    )(scope, receive, send)
                    return
                declared = value.lstrip(b"0") or b"0"
                ceiling = str(limit).encode()
                if len(declared) > len(ceiling) or (
                    len(declared) == len(ceiling) and declared > ceiling
                ):
                    await self._reject(scope, receive, send, limit)
                    return
        body_buffer = bytearray()
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                # Do not consume the rest of an oversized stream, even without Content-Length.
                await self._reject(scope, receive, send, limit)
                return
            body_buffer.extend(chunk)
            if not message.get("more_body", False):
                break
        body = bytes(body_buffer)
        del body_buffer
        delivered = False

        async def buffered_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, buffered_receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        await JSONResponse(
            {"detail": f"Request exceeds {limit} byte limit"}, status_code=413
        )(scope, receive, send)
