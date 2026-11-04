"""Interactive API docs (``/docs``, ``/redoc``) under a path-scoped CSP.

Swagger UI and ReDoc load their bundles from a CDN, which the service-wide
policy (``script-src 'self'``) forbids, and Swagger UI starts from one inline
script. Rather than weakening the policy for the whole site, these two pages
get their own Content-Security-Policy:

* scripts: ``'self'``, the *exact pinned package directory* on jsDelivr
  (``https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/`` for ``/docs``,
  ``https://cdn.jsdelivr.net/npm/redoc@2.1.5/`` for ``/redoc``; not the whole
  CDN) and, for Swagger UI, the SHA-256 hash of the exact inline bootstrap
  script (computed here from the generated HTML, so no ``'unsafe-inline'`` for
  scripts);
* styles: the same package directory plus ``'unsafe-inline'`` (both UIs set
  inline styles at runtime; styles cannot execute code);
* every CDN ``<script>`` and ``<link>`` carries a Subresource Integrity hash
  (``integrity="sha384-..."``, ``crossorigin="anonymous"``), so a modified file
  on the CDN is refused by the browser instead of executed;
* no third-party fonts, favicons or validator badges (the page uses the web
  UI's own favicon, ReDoc's Google Fonts link is disabled, Swagger's online
  validator is switched off).

Every other path keeps the strict policy of
:data:`battery_soh.api.middleware.CONTENT_SECURITY_POLICY`. Updating a pinned
version means updating its URL and its hash in :data:`_SRI` together (the hashes
are of the files as published on npm, which jsDelivr serves byte for byte).
"""

from __future__ import annotations

import base64
import hashlib
import re

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse

__all__ = ["DOCS_PATH", "REDOC_PATH", "install_docs"]

DOCS_PATH = "/docs"
REDOC_PATH = "/redoc"
_CDN = "https://cdn.jsdelivr.net"
_SWAGGER_DIR = f"{_CDN}/npm/swagger-ui-dist@5.17.14/"
_REDOC_DIR = f"{_CDN}/npm/redoc@2.1.5/"
_SWAGGER_JS = f"{_SWAGGER_DIR}swagger-ui-bundle.js"
_SWAGGER_CSS = f"{_SWAGGER_DIR}swagger-ui.css"
_REDOC_JS = f"{_REDOC_DIR}bundles/redoc.standalone.js"

#: Subresource Integrity of every CDN file (sha384 of the npm package files).
_SRI = {
    _SWAGGER_JS: "sha384-wmyclcVGX/WhUkdkATwhaK1X1JtiNrr2EoYJ+diV3vj4v6OC5yCeSu+yW13SYJep",
    _SWAGGER_CSS: "sha384-wxLW6kwyHktdDGr6Pv1zgm/VGJh99lfUbzSn6HNHBENZlCN7W602k9VkGdxuFvPn",
    _REDOC_JS: "sha384-0GrsyTQc9Oqd8h+b2dbc4XdR2T/DYpy0tLNNstyx+LBMUyiBbcWPbEs9aRmUcaxD",
}
_INLINE_SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


def _script_hashes(html: str) -> list[str]:
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode() + "'"
        for body in _INLINE_SCRIPT.findall(html)
    ]


def _with_integrity(html: str, urls: tuple[str, ...]) -> str:
    """Add ``integrity`` and ``crossorigin`` to the tag that loads each of ``urls``.

    Raises:
        RuntimeError: If a URL does not appear exactly once as ``src``/``href`` (the
            FastAPI template changed): failing at startup beats serving an unpinned file.
    """
    for url in urls:
        pattern = re.compile(r'((?:src|href)="' + re.escape(url) + '")')
        html, count = pattern.subn(rf'\1 integrity="{_SRI[url]}" crossorigin="anonymous"', html)
        if count != 1:
            raise RuntimeError(f"expected one tag loading {url} in the docs page, found {count}")
    return html


def _policy(package_dir: str, script_hashes: list[str]) -> str:
    scripts = " ".join(["'self'", package_dir, *script_hashes])
    return (
        f"default-src 'self'; script-src {scripts}; "
        f"style-src 'self' {package_dir} 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self' data:; connect-src 'self'; "
        "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )


def install_docs(app: FastAPI, *, title: str, favicon_url: str) -> dict[str, str]:
    """Add ``/docs`` and ``/redoc`` to ``app``.

    Args:
        app: Application created with ``docs_url=None, redoc_url=None``.
        title: Page title.
        favicon_url: Same-origin favicon URL.

    Returns:
        ``{path: Content-Security-Policy}`` for the middleware's per-path overrides.
    """
    openapi_url = app.openapi_url or "/openapi.json"
    swagger_html_body = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=f"{title} - Swagger UI",
        swagger_js_url=_SWAGGER_JS,
        swagger_css_url=_SWAGGER_CSS,
        swagger_favicon_url=favicon_url,
        oauth2_redirect_url=None,
        swagger_ui_parameters={"validatorUrl": None},
    ).body
    redoc_body = get_redoc_html(
        openapi_url=openapi_url,
        title=f"{title} - ReDoc",
        redoc_js_url=_REDOC_JS,
        redoc_favicon_url=favicon_url,
        with_google_fonts=False,
    ).body
    swagger_html = _with_integrity(
        bytes(swagger_html_body).decode("utf-8"), (_SWAGGER_JS, _SWAGGER_CSS)
    )
    redoc_html = _with_integrity(bytes(redoc_body).decode("utf-8"), (_REDOC_JS,))

    @app.get(DOCS_PATH, include_in_schema=False)
    def swagger_ui() -> HTMLResponse:
        return HTMLResponse(swagger_html)

    @app.get(REDOC_PATH, include_in_schema=False)
    def redoc() -> HTMLResponse:
        return HTMLResponse(redoc_html)

    return {
        DOCS_PATH: _policy(_SWAGGER_DIR, _script_hashes(swagger_html)),
        REDOC_PATH: _policy(_REDOC_DIR, _script_hashes(redoc_html)),
    }
