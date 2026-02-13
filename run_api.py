from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = (os.environ.get("AGENT_API_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port_raw = (os.environ.get("AGENT_API_PORT") or "58452").strip() or "58452"
    try:
        port = int(port_raw)
    except ValueError:
        port = 58452
    uvicorn.run("api.app:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()

