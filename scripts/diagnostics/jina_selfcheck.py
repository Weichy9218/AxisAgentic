"""Standalone pre-run Jina egress check for AxisAgentic.

Loads a repo ``.env`` if present, then probes the Jina Reader path and prints an
UP/DOWN verdict. Exit code 0 = up, 1 = down. Run this before a full offline batch
so a dead jina egress (e.g. the ``JINA_PROXY`` tunnel is not up) is caught before
the run, not one Cloudflare/403 page at a time.

    python scripts/diagnostics/jina_selfcheck.py [probe_url]

On tyyun_4 the only route to ``r.jina.ai`` is the reverse tunnel; export
``JINA_PROXY=http://127.0.0.1:17890`` (alongside ``JINA_API_KEY``) before running.
"""
import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def _load_env() -> None:
    env = _REPO_ROOT / ".env"
    if not env.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env)
        return
    except Exception:
        pass
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> None:
    _load_env()
    from agentic.tools.web_search.scrape import jina_self_check

    probe_url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com/"
    res = asyncio.run(jina_self_check(probe_url))
    verdict = "UP  " if res.get("ok") else "DOWN"
    print(
        f"jina {verdict}  proxy={res.get('proxy')}  kind={res.get('kind')}  "
        f"latency={res.get('latency_s')}s"
    )
    if not res.get("ok") and res.get("detail"):
        print("detail:", res["detail"])
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
