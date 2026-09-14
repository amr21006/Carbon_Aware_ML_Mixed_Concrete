"""Retrieve ready-mixed concrete EPD records from two open digital registries.

Both registries run soda4LCA nodes that serve ILCD process data sets as JSON
without credentials:

  epdnorge     EPD-Norge digital library, class "Bygg / Ferdig betong"
  environdec   EPD International digital library, records matching "concrete"

Usage:  python fetch_external_registries.py epdnorge
        python fetch_external_registries.py environdec

Records are written one JSON file per data set under
data/external/<registry>_raw/ and can then be mapped with extract_epdnorge.py
or extract_environdec.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}

REGISTRIES = {
    "epdnorge": {
        "base": "https://epdnorway.lca-data.com/resource/processes",
        "params": {"search": "true", "classId": "6cf99d9f-7e16-4747-83c3-6e86a60b2ff3"},
    },
    "environdec": {
        "base": "https://data.environdec.com/resource/processes",
        "params": {"search": "true", "name": "concrete"},
    },
}


def index(base: str, params: dict) -> list[dict]:
    rows, start = [], 0
    while True:
        r = requests.get(base, params={**params, "format": "json", "pageSize": 500, "startIndex": start},
                         headers=HEADERS, timeout=120)
        r.raise_for_status()
        page = r.json()
        rows += page["data"]
        if start + 500 >= page["totalCount"]:
            return rows
        start += 500


def fetch_one(base: str, uuid: str, out_dir: Path) -> str:
    out = out_dir / f"{uuid}.json"
    if out.exists():
        return "cached"
    for _ in range(3):
        try:
            r = requests.get(f"{base}/{uuid}", params={"format": "json", "view": "extended"},
                             headers=HEADERS, timeout=120)
            if r.status_code == 200 and r.text.startswith("{"):
                out.write_text(r.text, encoding="utf-8")
                return "ok"
        except requests.RequestException:
            time.sleep(2)
    return "failed"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "epdnorge"
    reg = REGISTRIES[name]
    out_dir = OUT / f"{name}_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = index(reg["base"], reg["params"])
    (OUT / f"{name}_index.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {len(rows)} records in the index")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda d: fetch_one(reg["base"], d["uuid"], out_dir), rows))
    print(f"{name}: ok {results.count('ok')}, cached {results.count('cached')}, failed {results.count('failed')} "
          f"in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
