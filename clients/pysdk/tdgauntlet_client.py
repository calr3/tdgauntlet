"""Write a tdgauntlet client in Python.

Standard library only, so it drops into any environment - including one whose JAX install must
not be disturbed. Give `serve` a function that chooses a move and it handles the protocol:

    from tdgauntlet_client import serve

    def choose_move(game, state, budget, context):
        ...
        return {"lead": 17, "rear": 3, "dest": 45}

    serve(name="my-agent", games=["epaminondas/1"], choose_move=choose_move, port=9001)

`choose_move` may also return `(move, info)`, where `info` is a dict of anything worth recording
(`eval`, `depth`, `nodes`, `pv`, ...). It is called concurrently from several threads, up to
`max_concurrency` at once, for positions from any number of games in any order: keep nothing
about "the current game" between calls. See PROTOCOL.md.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PROTOCOL_VERSION = 1

Move = Any
Info = dict[str, Any]
ChooseMove = Callable[[str, Any, dict[str, Any], dict[str, Any]], Move | tuple[Move, Info]]


class ClientError(Exception):
    """Raise from `choose_move` to refuse a request with a message, e.g. an unknown position."""


def _handler(
    name: str,
    version: str,
    games: list[str],
    max_concurrency: int,
    choose_move: ChooseMove,
    ready: Callable[[], bool],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # one line per move is noise; errors are reported below

        def _send(self, status: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/identify":
                self._send(
                    200,
                    {
                        "protocol": PROTOCOL_VERSION,
                        "name": name,
                        "version": version,
                        "games": games,
                        "max_concurrency": max_concurrency,
                    },
                )
            elif self.path == "/health":
                if ready():
                    self._send(200, "ok")
                else:
                    self._send(503, {"error": "starting up"})
            else:
                self._send(404, {"error": f"no such endpoint: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/move":
                self._send(404, {"error": f"no such endpoint: {self.path}"})
                return
            started = time.monotonic()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length))
                if req.get("protocol") != PROTOCOL_VERSION:
                    raise ValueError(f"unsupported protocol version {req.get('protocol')!r}")
                request_id = req["request_id"]
                game, state, context = req["game"], req["state"], req["context"]
                budget = req.get("budget") or {}
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": f"bad request: {e}"})
                return
            if game not in games:
                self._send(422, {"error": f"unsupported game: {game}"})
                return
            try:
                answer = choose_move(game, state, budget, context)
            except ClientError as e:
                self._send(422, {"error": str(e)})
                return
            except Exception as e:  # noqa: BLE001 - report it, keep serving
                traceback.print_exc(file=sys.stderr)
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return
            move, info = answer if isinstance(answer, tuple) else (answer, {})
            info = dict(info or {})
            info.setdefault("elapsed_ms", round(1000 * (time.monotonic() - started)))
            self._send(200, {"request_id": request_id, "move": move, "info": info})

    return Handler


def make_server(
    *,
    name: str,
    games: list[str],
    choose_move: ChooseMove,
    port: int = 9001,
    host: str = "127.0.0.1",
    version: str = "",
    max_concurrency: int = 1,
    ready: Callable[[], bool] | None = None,
) -> ThreadingHTTPServer:
    """A server, not yet running. `port=0` picks a free one (see `server.server_port`)."""
    handler = _handler(
        name, version, list(games), max_concurrency, choose_move, ready or (lambda: True)
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(**kwargs: Any) -> None:
    """Serve until interrupted. Takes the same arguments as `make_server`."""
    server = make_server(**kwargs)
    host = kwargs.get("host", "127.0.0.1")
    print(
        f"{kwargs['name']} listening on http://{host}:{server.server_port}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve_in_background(**kwargs: Any) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start a server on a thread and return it: for tests, and for clients that load a model
    after the port is open (pass `ready` so /health reports when they are done)."""
    server = make_server(**kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
