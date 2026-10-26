from __future__ import annotations

"""Download product-matched IMERG V07 Late Run half-hour controls (2021-2024)."""

import base64
import csv
import http.cookiejar
import json
import netrc
import os
from pathlib import Path
import queue
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(r"D:\MathorCup_Data\External_Gyirong\P0-2_IMERG")
CATALOG = ROOT / "imerg_late_history_2021-2024_aug20-27_catalog.csv"
DATA_DIR = ROOT / "data_late_history"
YEARS = (2021, 2022, 2023, 2024)
WORKERS = 8
socket.setdefaulttimeout(300)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_catalog() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for year in YEARS:
        params = urllib.parse.urlencode(
            {
                "short_name": "GPM_3IMERGHHL",
                "version": "07",
                "temporal": f"{year}-08-20T00:00:00Z,{year}-08-27T23:59:59Z",
                "page_size": "2000",
            }
        )
        req = urllib.request.Request(
            "https://cmr.earthdata.nasa.gov/search/granules.json?" + params,
            headers={"User-Agent": "MathorCup-IMERG-Late-controls/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            entries = json.load(response)["feed"]["entry"]
        for entry in entries:
            candidates = [
                link.get("href", "")
                for link in entry.get("links", [])
                if link.get("href", "").endswith(".HDF5")
                and "opendap" not in link.get("href", "").lower()
            ]
            if not candidates:
                continue
            url = candidates[0]
            rows.append(
                {
                    "year": str(year),
                    "granule_id": entry.get("producer_granule_id", entry.get("title", "")),
                    "time_start": entry.get("time_start", ""),
                    "time_end": entry.get("time_end", ""),
                    "url": url,
                    "filename": url.rsplit("/", 1)[-1],
                }
            )
        print(f"CMR {year}: {len(entries)} entries", flush=True)
    rows.sort(key=lambda row: (row["year"], row["time_start"]))
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"catalog total={len(rows)} -> {CATALOG}", flush=True)
    return rows


def credentials() -> str:
    path = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "_netrc"
    login, _, password = netrc.netrc(path).authenticators("urs.earthdata.nasa.gov")
    return base64.b64encode(f"{login}:{password}".encode()).decode()


def make_opener():
    return urllib.request.build_opener(
        NoRedirect, urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )


def open_auth(opener, url: str, auth: str, max_hops: int = 12):
    for _ in range(max_hops):
        req = urllib.request.Request(
            url,
            headers={"Authorization": "Basic " + auth, "User-Agent": "Mozilla/5.0"},
        )
        try:
            return opener.open(req)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                url = urllib.parse.urljoin(url, exc.headers["Location"].replace(" ", "%20"))
                continue
            raise
    raise RuntimeError("too many redirects")


def main() -> None:
    rows = build_catalog()
    expected = len(YEARS) * 8 * 48
    if len(rows) != expected:
        raise RuntimeError(f"IMERG catalog incomplete: {len(rows)}/{expected}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    auth = credentials()
    jobs: queue.Queue[dict[str, str]] = queue.Queue()
    for row in rows:
        jobs.put(row)
    completed = 0
    failures: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        nonlocal completed
        opener = make_opener()
        while True:
            try:
                row = jobs.get_nowait()
            except queue.Empty:
                return
            target = DATA_DIR / row["filename"]
            ok = False
            for attempt in range(6):
                try:
                    response = open_auth(opener, row["url"], auth)
                    length = int(response.headers.get("Content-Length", 0))
                    if target.exists() and length and target.stat().st_size == length:
                        response.close()
                        ok = True
                        break
                    partial = target.with_suffix(target.suffix + ".part")
                    with partial.open("wb") as handle:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                    response.close()
                    if length and partial.stat().st_size != length:
                        raise RuntimeError(f"size mismatch {partial.stat().st_size}/{length}")
                    partial.replace(target)
                    ok = True
                    break
                except Exception:
                    time.sleep(3 + 5 * attempt)
            with lock:
                completed += 1
                if not ok:
                    failures.append(row["filename"])
                if completed % 25 == 0 or completed == len(rows):
                    print(f"IMERG Late history {completed}/{len(rows)} failed={len(failures)}", flush=True)
            jobs.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"IMERG failures={len(failures)}; first={failures[:5]}")
    print("IMERG Late history complete", flush=True)


if __name__ == "__main__":
    main()
