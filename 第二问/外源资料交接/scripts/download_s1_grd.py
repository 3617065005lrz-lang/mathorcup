#!/usr/bin/env python
"""P0-4 Sentinel-1 GRD downloader for the Copernicus Data Space Ecosystem.

Authentication model
--------------------
The CDSE OData download service requires a Bearer access token.  This script
never stores, prints or logs a token: it reads the token that the already
logged-in Copernicus Browser SPA attaches to its own XHR requests, using the
Chrome DevTools Protocol, keeps it in memory only, and discards it at exit.

Because CDSE products are distributed as "multi-object downloads", HTTP Range
requests are rejected (DAT-ZIP-116), so each scene is streamed in a single
pass and verified against the catalogue `ContentLength`.

Usage
-----
    python download_s1_grd.py --dest <folder> [--log <file>]
"""

import argparse
import csv
import hashlib
import io
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp  # noqa: E402

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) sentinel1-grd-fetch"
CHUNK = 1 << 20


class Logger:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, msg):
        line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()


def ssl_context():
    return ssl.create_default_context()


def get_token(log):
    """Read the Bearer token from the browser's own request headers (memory only)."""
    ctx_free = None
    pages = [t for t in cdp.list_targets() if t.get("type") == "page"]
    if not pages:
        raise RuntimeError("no browser page target; is Chrome running with --remote-debugging-port?")
    page = cdp.Page(pages[0]["id"])
    try:
        page.ws.call("Network.enable")
        page.ws.events.clear()
        page.ws.call("Page.reload")
        page.ws.drain(10)
        for ev in page.ws.events:
            if ev.get("method") != "Network.requestWillBeSent":
                continue
            hdrs = {k.lower(): v for k, v in (ev["params"]["request"].get("headers") or {}).items()}
            auth = hdrs.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                tok = auth.split(" ", 1)[1].strip()
                if tok:
                    log("token refreshed from browser session (length %d, value not stored)" % len(tok))
                    return tok
    finally:
        page.close()
    raise RuntimeError("could not observe an Authorization header; log in to the Copernicus Browser first")


def product_meta(pid, ctx):
    url = "%s(%s)?$select=Id,Name,ContentLength,Online,S3Path" % (CATALOGUE, pid)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, token, target, expected, log, label):
    part = target + ".part"
    if os.path.exists(part):
        os.remove(part)
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Authorization": "Bearer " + token}
    )
    t0 = time.time()
    done = 0
    last_log = 0.0
    log("%s: opening stream -> %s" % (label, os.path.basename(target)))
    with urllib.request.urlopen(req, timeout=180, context=ssl_context()) as resp:
        length = int(resp.headers.get("Content-Length") or 0)
        if length and expected and length != expected:
            raise RuntimeError("server Content-Length %d != catalogue %d" % (length, expected))
        with open(part, "wb") as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_log >= 15:
                    last_log = now
                    rate = done / max(1e-6, now - t0) / 1e6
                    pct = 100.0 * done / expected if expected else 0.0
                    log(
                        "%s: %5.1f%%  %8.1f MiB / %8.1f MiB  %6.2f MiB/s  elapsed %ds"
                        % (label, pct, done / 2.0 ** 20, expected / 2.0 ** 20, rate, now - t0)
                    )
    size = os.path.getsize(part)
    if expected and size != expected:
        raise RuntimeError("size mismatch: got %d, expected %d" % (size, expected))
    os.replace(part, target)
    dt = time.time() - t0
    log(
        "%s: DONE %.1f MiB in %.1fs (%.2f MiB/s), size verified == %d"
        % (label, size / 2.0 ** 20, dt, size / 2.0 ** 20 / max(1e-6, dt), size)
    )
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--result", default=None)
    args = ap.parse_args()

    log = Logger(args.log)
    ctx = ssl_context()
    os.makedirs(args.dest, exist_ok=True)

    with open(args.catalog, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    results = []
    token = None
    for row in rows:
        pid = row["product_id"].strip()
        name = row["name"].strip()
        label = name.split("_")[5] if name.count("_") > 5 else name
        target = os.path.join(args.dest, name if name.endswith(".zip") else name + ".zip")

        meta = product_meta(pid, ctx)
        expected = int(meta.get("ContentLength") or 0)
        log("---- %s | %s | catalogue ContentLength=%d | online=%s"
            % (name, pid, expected, meta.get("Online")))

        if os.path.exists(target) and expected and os.path.getsize(target) == expected:
            digest = md5_file(target)
            log("%s: already complete (%d bytes), md5=%s -> skipped_download" % (name, expected, digest))
            results.append(
                {"product_id": pid, "name": name, "path": target, "bytes": expected,
                 "md5": digest, "status": "already_present"}
            )
            continue

        attempt = 0
        while True:
            attempt += 1
            if token is None:
                token = get_token(log)
            try:
                size = download("%s(%s)/$value" % (DOWNLOAD, pid), token, target, expected, log, name)
                digest = md5_file(target)
                log("%s: md5=%s" % (name, digest))
                results.append(
                    {"product_id": pid, "name": name, "path": target, "bytes": size,
                     "md5": digest, "status": "downloaded", "attempts": attempt}
                )
                break
            except urllib.error.HTTPError as exc:
                body = exc.read()[:300]
                log("%s: HTTP %s %s" % (name, exc.code, body))
                if exc.code in (401, 403):
                    token = None
                    if attempt >= 4:
                        raise
                    time.sleep(3)
                    continue
                if attempt >= 5:
                    raise
                time.sleep(5 * attempt)
            except Exception as exc:  # network hiccup, truncated stream ...
                log("%s: attempt %d failed: %s: %s" % (name, attempt, type(exc).__name__, exc))
                if attempt >= 5:
                    raise
                time.sleep(5 * attempt)

    if args.result:
        with open(args.result, "w", encoding="utf-8") as fh:
            json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "scenes": results},
                      fh, indent=2, ensure_ascii=False)
        log("result written to %s" % args.result)
    log("ALL TASKS COMPLETE")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
