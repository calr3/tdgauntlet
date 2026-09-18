"""The Python SDK serves a conformant client."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdgauntlet_client import ClientError, serve_in_background  # noqa: E402


def request(game: str = "g/1", state: Any = None, request_id: str = "r-1") -> dict[str, Any]:
    return {
        "protocol": 1,
        "request_id": request_id,
        "game": game,
        "state": state if state is not None else {"say": "hi"},
        "budget": {"ms": 50},
        "context": {"match_id": "m", "game_id": "g", "ply": 0, "side": 0},
    }


def call(url: str, path: str, body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture()
def client():
    seen: list[str] = []
    lock = threading.Lock()
    concurrent = {"now": 0, "peak": 0}

    def choose(game: str, state: Any, budget: dict[str, Any], context: dict[str, Any]) -> Any:
        with lock:
            concurrent["now"] += 1
            concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
            seen.append(state.get("say"))
        try:
            time.sleep(0.02)
            if state.get("say") == "refuse":
                raise ClientError("cannot play that")
            if state.get("say") == "crash":
                raise RuntimeError("boom")
            if state.get("say") == "with-info":
                return {"echo": state}, {"depth": 3}
            return {"echo": state, "budget": budget}
        finally:
            with lock:
                concurrent["now"] -= 1

    server, _ = serve_in_background(
        name="echo", version="1.0", games=["g/1"], choose_move=choose, port=0, max_concurrency=4
    )
    yield f"http://127.0.0.1:{server.server_port}", seen, concurrent
    server.shutdown()


def test_identify_and_health(client: Any) -> None:
    url, _, _ = client
    status, body = call(url, "/identify")
    assert status == 200
    assert body == {
        "protocol": 1,
        "name": "echo",
        "version": "1.0",
        "games": ["g/1"],
        "max_concurrency": 4,
    }
    assert call(url, "/health")[0] == 200


def test_a_move_echoes_the_request_id_and_records_time(client: Any) -> None:
    url, _, _ = client
    status, body = call(url, "/move", request("g/1", {"say": "hi"}, "abc-7"))
    assert status == 200
    assert body["request_id"] == "abc-7"
    assert body["move"] == {"echo": {"say": "hi"}, "budget": {"ms": 50}}
    assert body["info"]["elapsed_ms"] >= 20


def test_info_is_passed_through(client: Any) -> None:
    url, _, _ = client
    _, body = call(url, "/move", request(state={"say": "with-info"}))
    assert body["info"]["depth"] == 3
    assert "elapsed_ms" in body["info"]


def test_refusals_and_errors(client: Any) -> None:
    url, _, _ = client
    status, body = call(url, "/move", request(game="chess/1"))
    assert (status, body["error"]) == (422, "unsupported game: chess/1")
    status, body = call(url, "/move", request(state={"say": "refuse"}))
    assert (status, body["error"]) == (422, "cannot play that")
    status, body = call(url, "/move", request(state={"say": "crash"}))
    assert status == 500 and "boom" in body["error"]
    bad = request()
    bad["protocol"] = 2
    assert call(url, "/move", bad)[0] == 400
    missing = request()
    del missing["state"]
    assert call(url, "/move", missing)[0] == 400
    # Still serving afterwards.
    assert call(url, "/move", request())[0] == 200


def test_requests_are_served_concurrently(client: Any) -> None:
    url, seen, concurrent = client
    threads = [
        threading.Thread(target=call, args=(url, "/move", request(state={"say": str(i)})))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen, key=int) == [str(i) for i in range(8)]
    assert concurrent["peak"] > 1


def test_health_waits_for_ready() -> None:
    flag = {"ready": False}
    server, _ = serve_in_background(
        name="slow",
        games=["g/1"],
        choose_move=lambda *a: {},
        port=0,
        ready=lambda: flag["ready"],
    )
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert call(url, "/health")[0] == 503
        flag["ready"] = True
        assert call(url, "/health")[0] == 200
    finally:
        server.shutdown()
