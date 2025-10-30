from typing import Callable


class UploadSizeLimitMiddleware:
	def __init__(self, app, max_bytes: int) -> None:
		self.app = app
		self.max_bytes = max_bytes

	async def __call__(self, scope, receive, send):
		if scope.get("type") in ("http", "websocket"):
			headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
			cl = headers.get("content-length")
			if cl is not None:
				try:
					if int(cl) > self.max_bytes:
						await self._reject_413(send)
						return
				except Exception:
					pass
		return await self.app(scope, receive, send)

	async def _reject_413(self, send: Callable) -> None:
		await send({
			"type": "http.response.start",
			"status": 413,
			"headers": [(b"content-type", b"application/json")],
		})
		await send({
			"type": "http.response.body",
			"body": b'{"detail":"Request entity too large"}',
		})


