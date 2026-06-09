"""Static file serving: SPA mount and image serving."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

IMAGES_ROOT = "/images"


class NoCacheStatic(StaticFiles):
    """StaticFiles subclass that adds no-store Cache-Control and handles SPA routing.

    For SPA routes (paths without a file extension), falls back to index.html
    instead of returning 404, enabling client-side routing.
    """

    async def get_response(self, path: str, scope: dict) -> object:
        """Return the response for the given path.

        Args:
            path: Requested path relative to the mount root.
            scope: ASGI scope dict.

        Returns:
            Response with no-store Cache-Control header.
        """
        is_spa = "." not in path.rsplit("/", 1)[-1]
        try:
            resp = await super().get_response(path, scope)
            if resp.status_code == 404 and is_spa:
                resp = await super().get_response("index.html", scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and is_spa:
                resp = await super().get_response("index.html", scope)
            else:
                raise
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp


def mount_static(app: FastAPI) -> None:
    """Mount the SPA static files and image serving on the app.

    Image endpoint is registered as a route before the static mount so it
    takes precedence. The SPA catch-all is mounted last.

    Args:
        app: The FastAPI application instance.
    """

    @app.get("/api/images/{path:path}")
    async def image(path: str) -> FileResponse:
        if ".." in path:
            raise HTTPException(status_code=400, detail="bad path")
        full = os.path.join(IMAGES_ROOT, path)
        if not os.path.isfile(full):
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(full)

    if os.path.isdir("/frontend"):
        app.mount("/", NoCacheStatic(directory="/frontend", html=True), name="frontend")
